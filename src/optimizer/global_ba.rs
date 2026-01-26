//! Global Bundle Adjustment for Loop Closing.
//!
//! After pose graph optimization corrects keyframe poses, Global BA refines
//! both poses and map points to minimize reprojection error across the entire map.
//!
//! Global BA runs in a background thread to avoid blocking the main SLAM pipeline.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x6, Vector2, Vector3};
use opencv::prelude::*;
use parking_lot::RwLock;

use crate::atlas::atlas::Atlas;
use crate::atlas::map::{KeyFrameId, Map, MapPointId};
use crate::geometry::SE3;
use crate::tracking::frame::CameraModel;

/// Configuration for Global Bundle Adjustment.
#[derive(Debug, Clone)]
pub struct GlobalBAConfig {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,

    /// Convergence threshold on parameter change.
    pub param_tolerance: f64,

    /// Convergence threshold on gradient norm.
    pub gradient_tolerance: f64,

    /// Huber kernel threshold (pixels).
    pub huber_threshold: f64,
}

impl Default for GlobalBAConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            param_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            huber_threshold: 5.991_f64.sqrt(),
        }
    }
}

/// Data for Global BA problem.
pub struct GlobalBAProblemData {
    /// All keyframe poses (T_cw, world-to-camera).
    pub kf_poses: HashMap<KeyFrameId, SE3>,

    /// All map point positions.
    pub mp_positions: HashMap<MapPointId, Vector3<f64>>,

    /// Visual observations.
    pub observations: Vec<GlobalBAObservation>,

    /// Ordered keyframe IDs for parameter indexing.
    pub kf_ids: Vec<KeyFrameId>,

    /// Ordered map point IDs for parameter indexing.
    pub mp_ids: Vec<MapPointId>,

    /// Fixed keyframe ID (anchor).
    pub fixed_kf_id: KeyFrameId,
}

/// A visual observation for Global BA.
#[derive(Clone)]
pub struct GlobalBAObservation {
    /// Keyframe ID.
    pub kf_id: KeyFrameId,

    /// Map point ID.
    pub mp_id: MapPointId,

    /// Observed 2D pixel coordinates.
    pub observed_uv: Vector2<f64>,
}

/// Result of Global BA.
pub struct GlobalBAResult {
    /// Optimized keyframe poses (T_wc).
    pub optimized_poses: HashMap<KeyFrameId, SE3>,

    /// Optimized map point positions.
    pub optimized_points: HashMap<MapPointId, Vector3<f64>>,

    /// Number of iterations.
    pub iterations: usize,

    /// Initial error.
    pub initial_error: f64,

    /// Final error.
    pub final_error: f64,
}

/// PHASE 1: Collect data for Global BA.
pub fn collect_global_ba_data(map: &Map) -> Option<GlobalBAProblemData> {
    let mut kf_ids = Vec::new();
    let mut kf_poses = HashMap::new();
    let mut mp_ids = Vec::new();
    let mut mp_positions = HashMap::new();
    let mut observations = Vec::new();

    // Collect all keyframes
    for kf in map.keyframes() {
        if kf.is_bad {
            continue;
        }
        kf_ids.push(kf.id);
        kf_poses.insert(kf.id, kf.pose.inverse()); // T_cw
    }

    if kf_ids.is_empty() {
        return None;
    }

    // Sort by ID for consistent ordering
    kf_ids.sort_by_key(|id| id.0);

    // First keyframe is anchor (fixed)
    let fixed_kf_id = kf_ids[0];

    // Collect all map points
    let kf_set: HashSet<KeyFrameId> = kf_ids.iter().copied().collect();

    for mp in map.map_points() {
        if mp.is_bad {
            continue;
        }

        // Only include points observed by at least one valid keyframe
        let has_valid_obs = mp.observations.keys().any(|kf_id| kf_set.contains(kf_id));
        if !has_valid_obs {
            continue;
        }

        mp_ids.push(mp.id);
        mp_positions.insert(mp.id, mp.position);
    }

    if mp_ids.is_empty() {
        return None;
    }

    // Collect observations
    let mp_set: HashSet<MapPointId> = mp_ids.iter().copied().collect();

    for &kf_id in &kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            for (feat_idx, mp_id_opt) in kf.map_point_ids.iter().enumerate() {
                if let Some(mp_id) = mp_id_opt {
                    if mp_set.contains(mp_id) {
                        if let Ok(kp) = kf.keypoints.get(feat_idx) {
                            observations.push(GlobalBAObservation {
                                kf_id,
                                mp_id: *mp_id,
                                observed_uv: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                            });
                        }
                    }
                }
            }
        }
    }

    if observations.is_empty() {
        return None;
    }

    Some(GlobalBAProblemData {
        kf_poses,
        mp_positions,
        observations,
        kf_ids,
        mp_ids,
        fixed_kf_id,
    })
}

/// PHASE 2: Solve Global BA.
pub fn solve_global_ba(
    problem: &GlobalBAProblemData,
    camera: &CameraModel,
    config: &GlobalBAConfig,
    should_stop: &dyn Fn() -> bool,
) -> Option<GlobalBAResult> {
    let n_kfs = problem.kf_ids.len();
    let n_mps = problem.mp_ids.len();

    if n_kfs < 2 || n_mps == 0 {
        return None;
    }

    // Build index mappings (skip fixed keyframe)
    let fixed_idx = problem
        .kf_ids
        .iter()
        .position(|&id| id == problem.fixed_kf_id)?;

    let n_optimized_kfs = n_kfs - 1;
    let n_params = n_optimized_kfs * 6 + n_mps * 3;
    let n_residuals = problem.observations.len() * 2;

    if n_params == 0 {
        return None;
    }

    // Map keyframe ID to parameter index (skipping fixed)
    let mut kf_to_param: HashMap<KeyFrameId, usize> = HashMap::new();
    let mut param_idx = 0;
    for (idx, &kf_id) in problem.kf_ids.iter().enumerate() {
        if idx != fixed_idx {
            kf_to_param.insert(kf_id, param_idx);
            param_idx += 1;
        }
    }

    // Map point to parameter index
    let mp_to_param: HashMap<MapPointId, usize> = problem
        .mp_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Initialize parameters
    let mut params = DVector::zeros(n_params);

    // Keyframe poses (axis-angle + translation)
    for (&kf_id, &param_idx) in &kf_to_param {
        if let Some(pose_cw) = problem.kf_poses.get(&kf_id) {
            let rot = pose_cw.rotation.scaled_axis();
            let base = param_idx * 6;
            params[base] = rot.x;
            params[base + 1] = rot.y;
            params[base + 2] = rot.z;
            params[base + 3] = pose_cw.translation.x;
            params[base + 4] = pose_cw.translation.y;
            params[base + 5] = pose_cw.translation.z;
        }
    }

    // Map point positions
    for (&mp_id, &mp_idx) in &mp_to_param {
        if let Some(pos) = problem.mp_positions.get(&mp_id) {
            let base = n_optimized_kfs * 6 + mp_idx * 3;
            params[base] = pos.x;
            params[base + 1] = pos.y;
            params[base + 2] = pos.z;
        }
    }

    // Get fixed pose
    let fixed_pose = problem
        .kf_poses
        .get(&problem.fixed_kf_id)
        .cloned()
        .unwrap_or_else(SE3::identity);

    // Compute initial error
    let initial_residuals = compute_global_ba_residuals(
        &params,
        &problem.observations,
        &kf_to_param,
        &mp_to_param,
        &fixed_pose,
        &problem.fixed_kf_id,
        camera,
        config.huber_threshold,
        n_optimized_kfs,
    );
    let initial_error = initial_residuals.norm() / (n_residuals as f64).sqrt();

    // LM optimization
    let mut current_params = params;
    let mut lambda = 1e-3;
    let lambda_up = 10.0;
    let lambda_down = 0.1;
    let min_lambda = 1e-10;
    let max_lambda = 1e10;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        if should_stop() {
            break;
        }
        iterations = iter + 1;

        let residuals = compute_global_ba_residuals(
            &current_params,
            &problem.observations,
            &kf_to_param,
            &mp_to_param,
            &fixed_pose,
            &problem.fixed_kf_id,
            camera,
            config.huber_threshold,
            n_optimized_kfs,
        );

        let jacobian = compute_global_ba_jacobian(
            &current_params,
            &problem.observations,
            &kf_to_param,
            &mp_to_param,
            &fixed_pose,
            &problem.fixed_kf_id,
            camera,
            config.huber_threshold,
            n_optimized_kfs,
            n_mps,
        );

        let current_error_sq = residuals.norm_squared();

        let gradient = jacobian.transpose() * &residuals;
        let jtj = jacobian.transpose() * &jacobian;

        if gradient.norm() < config.gradient_tolerance {
            break;
        }

        let mut damped_jtj = jtj.clone();
        for i in 0..n_params {
            damped_jtj[(i, i)] += lambda * damped_jtj[(i, i)].max(1e-6);
        }

        let delta = match damped_jtj.clone().lu().solve(&(-&gradient)) {
            Some(d) => d,
            None => break,
        };

        if delta.norm() < config.param_tolerance * (current_params.norm() + config.param_tolerance) {
            break;
        }

        let trial_params = &current_params + &delta;
        let trial_residuals = compute_global_ba_residuals(
            &trial_params,
            &problem.observations,
            &kf_to_param,
            &mp_to_param,
            &fixed_pose,
            &problem.fixed_kf_id,
            camera,
            config.huber_threshold,
            n_optimized_kfs,
        );
        let trial_error_sq = trial_residuals.norm_squared();

        if trial_error_sq < current_error_sq {
            current_params = trial_params;
            lambda = (lambda * lambda_down).max(min_lambda);
        } else {
            lambda = (lambda * lambda_up).min(max_lambda);
        }
    }

    // Compute final error
    let final_residuals = compute_global_ba_residuals(
        &current_params,
        &problem.observations,
        &kf_to_param,
        &mp_to_param,
        &fixed_pose,
        &problem.fixed_kf_id,
        camera,
        config.huber_threshold,
        n_optimized_kfs,
    );
    let final_error = final_residuals.norm() / (n_residuals as f64).sqrt();

    // Extract results
    let mut optimized_poses = HashMap::new();

    // Add fixed pose
    optimized_poses.insert(problem.fixed_kf_id, fixed_pose.inverse());

    // Extract optimized poses
    for (&kf_id, &param_idx) in &kf_to_param {
        let base = param_idx * 6;
        let rot = Vector3::new(
            current_params[base],
            current_params[base + 1],
            current_params[base + 2],
        );
        let trans = Vector3::new(
            current_params[base + 3],
            current_params[base + 4],
            current_params[base + 5],
        );
        let pose_cw = se3_from_params(&rot, &trans);
        optimized_poses.insert(kf_id, pose_cw.inverse());
    }

    // Extract optimized points
    let mut optimized_points = HashMap::new();
    for (&mp_id, &mp_idx) in &mp_to_param {
        let base = n_optimized_kfs * 6 + mp_idx * 3;
        let pos = Vector3::new(
            current_params[base],
            current_params[base + 1],
            current_params[base + 2],
        );
        optimized_points.insert(mp_id, pos);
    }

    Some(GlobalBAResult {
        optimized_poses,
        optimized_points,
        iterations,
        initial_error,
        final_error,
    })
}

/// PHASE 3: Apply Global BA results.
pub fn apply_global_ba_results(map: &mut Map, results: &GlobalBAResult) -> usize {
    let mut updated = 0;

    for (kf_id, pose) in &results.optimized_poses {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            if !kf.is_bad {
                kf.pose = pose.clone();
                updated += 1;
            }
        }
    }

    for (mp_id, pos) in &results.optimized_points {
        if let Some(mp) = map.get_map_point_mut(*mp_id) {
            if !mp.is_bad {
                mp.position = *pos;
                updated += 1;
            }
        }
    }

    updated
}

/// Run Global BA synchronously.
///
/// Note: Background execution requires careful handling of thread safety.
/// For now, this runs synchronously. To run in background, the caller should
/// spawn a thread with appropriate safety considerations.
pub fn run_global_ba(
    atlas: &RwLock<Atlas>,
    camera: &CameraModel,
    config: &GlobalBAConfig,
    running_flag: &AtomicBool,
) -> Option<GlobalBAResult> {
    running_flag.store(true, Ordering::SeqCst);

    // Phase 1: Collect
    let problem = {
        let atlas_guard = atlas.read();
        collect_global_ba_data(atlas_guard.active_map())
    };

    let problem = match problem {
        Some(p) => p,
        None => {
            running_flag.store(false, Ordering::SeqCst);
            return None;
        }
    };

    // Phase 2: Solve
    let should_stop = || !running_flag.load(Ordering::SeqCst);
    let result = solve_global_ba(&problem, camera, config, &should_stop);

    let result = match result {
        Some(r) => r,
        None => {
            running_flag.store(false, Ordering::SeqCst);
            return None;
        }
    };

    // Phase 3: Apply
    {
        let mut atlas_guard = atlas.write();
        apply_global_ba_results(atlas_guard.active_map_mut(), &result);
    }

    tracing::info!(
        "Global BA complete: {} iterations, error {:.4} -> {:.4}",
        result.iterations,
        result.initial_error,
        result.final_error
    );

    running_flag.store(false, Ordering::SeqCst);
    Some(result)
}

/// Compute residuals for Global BA.
fn compute_global_ba_residuals(
    params: &DVector<f64>,
    observations: &[GlobalBAObservation],
    kf_to_param: &HashMap<KeyFrameId, usize>,
    mp_to_param: &HashMap<MapPointId, usize>,
    fixed_pose: &SE3,
    fixed_kf_id: &KeyFrameId,
    camera: &CameraModel,
    huber_threshold: f64,
    n_optimized_kfs: usize,
) -> DVector<f64> {
    let mut residuals = DVector::zeros(observations.len() * 2);

    for (i, obs) in observations.iter().enumerate() {
        let pose_cw = get_pose(params, &obs.kf_id, kf_to_param, fixed_pose, fixed_kf_id);
        let point = get_point(params, &obs.mp_id, mp_to_param, n_optimized_kfs);

        let p_cam = pose_cw.transform_point(&point);

        if p_cam.z <= 0.001 {
            residuals[i * 2] = 100.0;
            residuals[i * 2 + 1] = 100.0;
            continue;
        }

        let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
        let v = camera.fy * p_cam.y / p_cam.z + camera.cy;

        let error = Vector2::new(obs.observed_uv.x - u, obs.observed_uv.y - v);
        let error_norm = error.norm();
        let weight = if error_norm <= huber_threshold {
            1.0
        } else {
            huber_threshold / error_norm
        };

        residuals[i * 2] = error.x * weight.sqrt();
        residuals[i * 2 + 1] = error.y * weight.sqrt();
    }

    residuals
}

/// Compute Jacobian for Global BA (simplified numerical version for large problems).
fn compute_global_ba_jacobian(
    params: &DVector<f64>,
    observations: &[GlobalBAObservation],
    kf_to_param: &HashMap<KeyFrameId, usize>,
    mp_to_param: &HashMap<MapPointId, usize>,
    fixed_pose: &SE3,
    fixed_kf_id: &KeyFrameId,
    camera: &CameraModel,
    huber_threshold: f64,
    n_optimized_kfs: usize,
    n_mps: usize,
) -> DMatrix<f64> {
    let n_params = n_optimized_kfs * 6 + n_mps * 3;
    let n_residuals = observations.len() * 2;
    let mut jacobian = DMatrix::zeros(n_residuals, n_params);

    for (i, obs) in observations.iter().enumerate() {
        let pose_cw = get_pose(params, &obs.kf_id, kf_to_param, fixed_pose, fixed_kf_id);
        let point = get_point(params, &obs.mp_id, mp_to_param, n_optimized_kfs);

        let p_cam = pose_cw.transform_point(&point);

        if p_cam.z <= 0.001 {
            continue;
        }

        let error = compute_reprojection_error(&pose_cw, &point, &obs.observed_uv, camera);
        let error_norm = error.norm();
        let weight_sqrt = if error_norm <= huber_threshold {
            1.0
        } else {
            (huber_threshold / error_norm).sqrt()
        };

        // Jacobian w.r.t. pose (if not fixed)
        if let Some(&kf_param) = kf_to_param.get(&obs.kf_id) {
            let j_pose = jacobian_pose(&pose_cw, &point, camera) * weight_sqrt;
            let col_base = kf_param * 6;
            for row in 0..2 {
                for col in 0..6 {
                    jacobian[(i * 2 + row, col_base + col)] = j_pose[(row, col)];
                }
            }
        }

        // Jacobian w.r.t. point
        if let Some(&mp_param) = mp_to_param.get(&obs.mp_id) {
            let j_point = jacobian_point(&pose_cw, &point, camera) * weight_sqrt;
            let col_base = n_optimized_kfs * 6 + mp_param * 3;
            for row in 0..2 {
                for col in 0..3 {
                    jacobian[(i * 2 + row, col_base + col)] = j_point[(row, col)];
                }
            }
        }
    }

    jacobian
}

fn get_pose(
    params: &DVector<f64>,
    kf_id: &KeyFrameId,
    kf_to_param: &HashMap<KeyFrameId, usize>,
    fixed_pose: &SE3,
    fixed_kf_id: &KeyFrameId,
) -> SE3 {
    if kf_id == fixed_kf_id {
        return fixed_pose.clone();
    }

    if let Some(&param_idx) = kf_to_param.get(kf_id) {
        let base = param_idx * 6;
        let rot = Vector3::new(params[base], params[base + 1], params[base + 2]);
        let trans = Vector3::new(params[base + 3], params[base + 4], params[base + 5]);
        se3_from_params(&rot, &trans)
    } else {
        SE3::identity()
    }
}

fn get_point(
    params: &DVector<f64>,
    mp_id: &MapPointId,
    mp_to_param: &HashMap<MapPointId, usize>,
    n_optimized_kfs: usize,
) -> Vector3<f64> {
    if let Some(&mp_idx) = mp_to_param.get(mp_id) {
        let base = n_optimized_kfs * 6 + mp_idx * 3;
        Vector3::new(params[base], params[base + 1], params[base + 2])
    } else {
        Vector3::zeros()
    }
}

fn se3_from_params(axis_angle: &Vector3<f64>, translation: &Vector3<f64>) -> SE3 {
    let angle = axis_angle.norm();
    let rotation = if angle > 1e-10 {
        nalgebra::UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(*axis_angle / angle),
            angle,
        )
    } else {
        nalgebra::UnitQuaternion::identity()
    };
    SE3 {
        rotation,
        translation: *translation,
    }
}

fn compute_reprojection_error(
    pose_cw: &SE3,
    point: &Vector3<f64>,
    observed: &Vector2<f64>,
    camera: &CameraModel,
) -> Vector2<f64> {
    let p_cam = pose_cw.transform_point(point);

    if p_cam.z <= 0.001 {
        return Vector2::new(100.0, 100.0);
    }

    let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
    let v = camera.fy * p_cam.y / p_cam.z + camera.cy;

    Vector2::new(observed.x - u, observed.y - v)
}

fn jacobian_pose(pose_cw: &SE3, point: &Vector3<f64>, camera: &CameraModel) -> Matrix2x6<f64> {
    let p_cam = pose_cw.transform_point(point);

    let x = p_cam.x;
    let y = p_cam.y;
    let z = p_cam.z;

    if z.abs() < 1e-6 {
        return Matrix2x6::zeros();
    }

    let invz = 1.0 / z;
    let invz2 = invz * invz;
    let fx = camera.fx;
    let fy = camera.fy;

    Matrix2x6::new(
        x * y * invz2 * fx,
        -(1.0 + x * x * invz2) * fx,
        y * invz * fx,
        -invz * fx,
        0.0,
        x * invz2 * fx,
        (1.0 + y * y * invz2) * fy,
        -x * y * invz2 * fy,
        -x * invz * fy,
        0.0,
        -invz * fy,
        y * invz2 * fy,
    )
}

fn jacobian_point(pose_cw: &SE3, point: &Vector3<f64>, camera: &CameraModel) -> Matrix2x3<f64> {
    let p_cam = pose_cw.transform_point(point);

    let x = p_cam.x;
    let y = p_cam.y;
    let z = p_cam.z;

    if z.abs() < 1e-6 {
        return Matrix2x3::zeros();
    }

    let invz = 1.0 / z;
    let fx = camera.fx;
    let fy = camera.fy;

    let r_cw = pose_cw.rotation.to_rotation_matrix().into_inner();

    let tmp = Matrix2x3::new(fx, 0.0, -fx * x * invz, 0.0, fy, -fy * y * invz);

    (-invz) * tmp * r_cw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_ba_config_default() {
        let config = GlobalBAConfig::default();
        assert_eq!(config.max_iterations, 10);
    }

    #[test]
    fn test_se3_from_params() {
        let rot = Vector3::new(0.0, 0.0, 0.0);
        let trans = Vector3::new(1.0, 2.0, 3.0);
        let pose = se3_from_params(&rot, &trans);

        assert_eq!(pose.translation, trans);
    }
}
