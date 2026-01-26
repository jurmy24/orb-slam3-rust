//! Local Inertial Bundle Adjustment.
//!
//! Optimizes keyframe poses, velocities, and biases along with map points
//! using both visual reprojection errors and IMU preintegration constraints.
//!
//! # State Variables (per keyframe)
//!
//! - Pose (6): axis-angle rotation + translation (T_wc, camera-to-world)
//! - Velocity (3): velocity in world frame
//! - Gyro bias (3): gyroscope bias
//! - Accel bias (3): accelerometer bias
//!
//! # Residuals
//!
//! 1. Visual reprojection error (2D per observation)
//! 2. IMU preintegration residual (9D per consecutive KF pair)
//! 3. Bias random walk prior (6D per consecutive KF pair)

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, Vector2, Vector3};
use opencv::prelude::*;
use tracing::debug;

use crate::atlas::map::{KeyFrameId, Map, MapPointId};
use crate::geometry::SE3;
use crate::imu::{ImuBias, PreintegratedState};
use crate::tracking::frame::CameraModel;

use super::imu_factors::compute_imu_residual;

// ============================================================================
// THREE-PHASE BA: Data structures for lock-free solving
// ============================================================================

/// Extracted snapshot of data for lock-free inertial BA solving.
///
/// This contains all the information needed to run Inertial Bundle Adjustment
/// without holding any locks on the map.
#[derive(Clone)]
pub struct InertialBAProblemData {
    /// Poses of keyframes as T_wc (camera-to-world).
    /// All keyframes in the temporal window.
    pub kf_poses: HashMap<KeyFrameId, SE3>,
    /// Velocities of keyframes in world frame.
    pub kf_velocities: HashMap<KeyFrameId, Vector3<f64>>,
    /// IMU biases of keyframes.
    pub kf_biases: HashMap<KeyFrameId, ImuBias>,
    /// Positions of map points to optimize.
    pub mp_positions: HashMap<MapPointId, Vector3<f64>>,
    /// Poses of fixed keyframes (outside optimization window) as T_cw.
    pub fixed_kf_poses: HashMap<KeyFrameId, SE3>,
    /// All visual observations.
    pub visual_observations: Vec<InertialVisualObs>,
    /// IMU edges between consecutive keyframes.
    pub imu_edges: Vec<ImuEdgeData>,
    /// Ordered list of keyframe IDs in temporal window (oldest first).
    pub opt_kf_ids: Vec<KeyFrameId>,
    /// Ordered list of map point IDs.
    pub mp_ids: Vec<MapPointId>,
}

/// A single visual observation for inertial BA.
#[derive(Clone)]
pub struct InertialVisualObs {
    /// Keyframe ID where this observation was made.
    pub kf_id: KeyFrameId,
    /// Map point ID being observed.
    pub mp_id: MapPointId,
    /// Observed 2D pixel coordinates.
    pub observed_uv: Vector2<f64>,
    /// Whether this is a stereo observation.
    pub is_stereo: bool,
    /// Whether this keyframe is in the optimization window.
    pub is_kf_in_window: bool,
}

/// IMU edge data between consecutive keyframes.
#[derive(Clone)]
pub struct ImuEdgeData {
    /// First keyframe ID (earlier in time).
    pub kf_i_id: KeyFrameId,
    /// Second keyframe ID (later in time).
    pub kf_j_id: KeyFrameId,
    /// Preintegrated IMU measurement.
    pub preint: PreintegratedState,
}

/// Result data from inertial BA, ready to be applied to the map.
#[derive(Clone)]
pub struct InertialBAResultData {
    /// Optimized poses (T_wc) indexed by keyframe ID.
    pub optimized_poses: HashMap<KeyFrameId, SE3>,
    /// Optimized velocities indexed by keyframe ID.
    pub optimized_velocities: HashMap<KeyFrameId, Vector3<f64>>,
    /// Optimized biases indexed by keyframe ID.
    pub optimized_biases: HashMap<KeyFrameId, ImuBias>,
    /// Optimized 3D positions indexed by map point ID.
    pub optimized_points: HashMap<MapPointId, Vector3<f64>>,
    /// Number of LM iterations performed.
    pub iterations: usize,
    /// Initial total error.
    pub initial_error: f64,
    /// Final total error.
    pub final_error: f64,
}

/// Configuration for Local Inertial Bundle Adjustment.
pub struct LocalInertialBAConfig {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,
    /// Number of keyframes in temporal window.
    pub window_size: usize,
    /// Huber threshold for visual observations (pixels).
    pub huber_threshold_mono: f64,
    /// Huber threshold for stereo observations.
    pub huber_threshold_stereo: f64,
    /// Initial LM damping factor.
    pub initial_lambda: f64,
    /// Information weight for gyro bias random walk.
    pub gyro_rw_info: f64,
    /// Information weight for accel bias random walk.
    pub accel_rw_info: f64,
}

impl Default for LocalInertialBAConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            window_size: 10,
            huber_threshold_mono: 5.991_f64.sqrt(),  // ~2.45 pixels
            huber_threshold_stereo: 7.815_f64.sqrt(), // ~2.79 pixels
            initial_lambda: 1e-2,
            // From ORB-SLAM3: Cov_bg = dt * sigma_gw^2, sigma_gw ~ 1e-3
            gyro_rw_info: 1e6,
            accel_rw_info: 1e4,
        }
    }
}

/// Result of Local Inertial Bundle Adjustment.
#[derive(Debug)]
pub struct LocalInertialBAResult {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Initial total error.
    pub initial_error: f64,
    /// Final total error.
    pub final_error: f64,
    /// Number of optimized keyframes.
    pub num_keyframes: usize,
    /// Number of optimized map points.
    pub num_map_points: usize,
    /// Number of visual observations.
    pub num_visual_obs: usize,
    /// Number of IMU edges.
    pub num_imu_edges: usize,
}

/// State layout for inertial BA (15D per keyframe + 3D per map point).
#[derive(Debug, Clone)]
struct InertialStateLayout {
    /// Number of optimized keyframes.
    num_keyframes: usize,
    /// Number of map points.
    num_map_points: usize,
}

impl InertialStateLayout {
    const KF_POSE_DIM: usize = 6;
    const KF_VEL_DIM: usize = 3;
    const KF_BIAS_DIM: usize = 6; // 3 gyro + 3 accel
    const KF_TOTAL_DIM: usize = 15; // pose + vel + bias
    const MP_DIM: usize = 3;

    fn new(num_keyframes: usize, num_map_points: usize) -> Self {
        Self { num_keyframes, num_map_points }
    }

    fn total_params(&self) -> usize {
        self.num_keyframes * Self::KF_TOTAL_DIM + self.num_map_points * Self::MP_DIM
    }

    fn pose_start(&self, kf_idx: usize) -> usize {
        kf_idx * Self::KF_TOTAL_DIM
    }

    fn vel_start(&self, kf_idx: usize) -> usize {
        kf_idx * Self::KF_TOTAL_DIM + Self::KF_POSE_DIM
    }

    fn gyro_bias_start(&self, kf_idx: usize) -> usize {
        kf_idx * Self::KF_TOTAL_DIM + Self::KF_POSE_DIM + Self::KF_VEL_DIM
    }

    fn accel_bias_start(&self, kf_idx: usize) -> usize {
        kf_idx * Self::KF_TOTAL_DIM + Self::KF_POSE_DIM + Self::KF_VEL_DIM + 3
    }

    fn mp_start(&self, mp_idx: usize) -> usize {
        self.num_keyframes * Self::KF_TOTAL_DIM + mp_idx * Self::MP_DIM
    }
}

/// Visual observation for BA.
struct VisualObs {
    kf_param_idx: Option<usize>, // None if fixed
    mp_param_idx: usize,
    observed_uv: Vector2<f64>,
    kf_id: KeyFrameId,
    is_stereo: bool,
}

/// IMU edge between consecutive keyframes.
struct ImuEdge {
    kf_i_idx: usize,
    kf_j_idx: usize,
    preint: PreintegratedState,
}

/// Run Local Inertial Bundle Adjustment.
///
/// Optimizes poses, velocities, biases of recent keyframes and map point positions.
/// Uses temporal chain (prev_kf links) instead of covisibility graph.
pub fn local_inertial_ba(
    map: &mut Map,
    current_kf_id: KeyFrameId,
    camera: &CameraModel,
    config: &LocalInertialBAConfig,
    should_stop: &dyn Fn() -> bool,
) -> Option<LocalInertialBAResult> {
    // Step 1: Collect temporal chain of keyframes
    let opt_kf_ids = collect_temporal_keyframes(map, current_kf_id, config.window_size);
    if opt_kf_ids.len() < 2 {
        return None; // Need at least 2 KFs for IMU constraint
    }

    // Step 2: Collect map points observed by optimized keyframes
    let mp_ids = collect_map_points(map, &opt_kf_ids);
    if mp_ids.is_empty() {
        return None;
    }

    // Step 3: Collect fixed keyframes (observe local map points but not in opt window)
    let fixed_kf_ids = collect_fixed_keyframes(map, &opt_kf_ids, &mp_ids);

    // Step 4: Build index mappings
    let kf_to_idx: HashMap<KeyFrameId, usize> = opt_kf_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mp_to_idx: HashMap<MapPointId, usize> = mp_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let layout = InertialStateLayout::new(opt_kf_ids.len(), mp_ids.len());

    // Step 5: Build observations and edges
    let (visual_obs, fixed_poses) = build_visual_observations(
        map, &opt_kf_ids, &fixed_kf_ids, &mp_ids, &kf_to_idx, &mp_to_idx,
    );
    let imu_edges = build_imu_edges(map, &opt_kf_ids, &kf_to_idx);

    if visual_obs.is_empty() {
        return None;
    }

    // Step 6: Initialize parameter vector
    let params = initialize_params(map, &opt_kf_ids, &mp_ids, &layout);

    // Compute initial error
    let initial_residuals = compute_all_residuals(
        &params, &layout, &visual_obs, &imu_edges, &fixed_poses, camera, config,
    );
    let initial_error = initial_residuals.norm();

    // Step 7: LM optimization loop
    let mut current_params = params.clone();
    let mut lambda = config.initial_lambda;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        if should_stop() {
            break;
        }
        iterations = iter + 1;

        // Compute residuals and Jacobian
        let residuals = compute_all_residuals(
            &current_params, &layout, &visual_obs, &imu_edges, &fixed_poses, camera, config,
        );
        let jacobian = compute_all_jacobians(
            &current_params, &layout, &visual_obs, &imu_edges, &fixed_poses, camera, config,
        );

        let current_error_sq = residuals.norm_squared();

        // Normal equations
        let gradient = jacobian.transpose() * &residuals;
        let jtj = jacobian.transpose() * &jacobian;

        // Check convergence
        if gradient.norm() < 1e-8 {
            break;
        }

        // Damped Gauss-Newton step
        let num_params = layout.total_params();
        let mut damped_jtj = jtj.clone();
        for i in 0..num_params {
            damped_jtj[(i, i)] += lambda * damped_jtj[(i, i)].max(1e-6);
        }

        let delta = match damped_jtj.lu().solve(&(-&gradient)) {
            Some(d) => d,
            None => break,
        };

        // Trial step
        let trial_params = &current_params + &delta;
        let trial_residuals = compute_all_residuals(
            &trial_params, &layout, &visual_obs, &imu_edges, &fixed_poses, camera, config,
        );
        let trial_error_sq = trial_residuals.norm_squared();

        // Accept or reject
        if trial_error_sq < current_error_sq {
            current_params = trial_params;
            lambda = (lambda * 0.1).max(1e-10);
        } else {
            lambda = (lambda * 10.0).min(1e10);
        }
    }

    // Compute final error
    let final_residuals = compute_all_residuals(
        &current_params, &layout, &visual_obs, &imu_edges, &fixed_poses, camera, config,
    );
    let final_error = final_residuals.norm();

    // Step 8: Write back results
    write_back_results(map, &opt_kf_ids, &mp_ids, &current_params, &layout);

    debug!(
        "[LocalInertialBA] {} iters, error: {:.2} -> {:.2}, kfs={}, mps={}, vis={}, imu={}",
        iterations, initial_error, final_error,
        opt_kf_ids.len(), mp_ids.len(), visual_obs.len(), imu_edges.len()
    );

    Some(LocalInertialBAResult {
        iterations,
        initial_error,
        final_error,
        num_keyframes: opt_kf_ids.len(),
        num_map_points: mp_ids.len(),
        num_visual_obs: visual_obs.len(),
        num_imu_edges: imu_edges.len(),
    })
}

/// Collect keyframes in temporal chain via prev_kf links.
fn collect_temporal_keyframes(
    map: &Map,
    current_kf_id: KeyFrameId,
    window_size: usize,
) -> Vec<KeyFrameId> {
    let mut kf_ids = Vec::with_capacity(window_size);
    kf_ids.push(current_kf_id);

    let mut prev_id = map.get_keyframe(current_kf_id).and_then(|kf| kf.prev_kf);
    while kf_ids.len() < window_size && prev_id.is_some() {
        let pid = prev_id.unwrap();
        kf_ids.push(pid);
        prev_id = map.get_keyframe(pid).and_then(|kf| kf.prev_kf);
    }

    // Reverse so oldest is first (anchor)
    kf_ids.reverse();
    kf_ids
}

/// Collect map points observed by keyframes.
fn collect_map_points(map: &Map, kf_ids: &[KeyFrameId]) -> Vec<MapPointId> {
    let mut mp_set = HashSet::new();
    for &kf_id in kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            for mp_id_opt in &kf.map_point_ids {
                if let Some(mp_id) = mp_id_opt {
                    if let Some(mp) = map.get_map_point(*mp_id) {
                        if !mp.is_bad {
                            mp_set.insert(*mp_id);
                        }
                    }
                }
            }
        }
    }
    mp_set.into_iter().collect()
}

/// Collect fixed keyframes that observe local map points.
fn collect_fixed_keyframes(
    map: &Map,
    opt_kf_ids: &[KeyFrameId],
    mp_ids: &[MapPointId],
) -> HashSet<KeyFrameId> {
    let opt_set: HashSet<_> = opt_kf_ids.iter().copied().collect();
    let mut fixed = HashSet::new();

    for &mp_id in mp_ids {
        if let Some(mp) = map.get_map_point(mp_id) {
            for &obs_kf_id in mp.observations.keys() {
                if !opt_set.contains(&obs_kf_id) {
                    if let Some(kf) = map.get_keyframe(obs_kf_id) {
                        if !kf.is_bad {
                            fixed.insert(obs_kf_id);
                        }
                    }
                }
            }
        }
    }
    fixed
}

/// Build visual observations.
fn build_visual_observations(
    map: &Map,
    opt_kf_ids: &[KeyFrameId],
    fixed_kf_ids: &HashSet<KeyFrameId>,
    mp_ids: &[MapPointId],
    kf_to_idx: &HashMap<KeyFrameId, usize>,
    mp_to_idx: &HashMap<MapPointId, usize>,
) -> (Vec<VisualObs>, HashMap<KeyFrameId, SE3>) {
    let mut observations = Vec::new();
    let mut fixed_poses = HashMap::new();

    // Cache fixed poses
    for &kf_id in fixed_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            fixed_poses.insert(kf_id, kf.pose.inverse()); // T_cw
        }
    }

    // Also add first KF (anchor) as reference
    if let Some(&first_kf_id) = opt_kf_ids.first() {
        if let Some(kf) = map.get_keyframe(first_kf_id) {
            fixed_poses.insert(first_kf_id, kf.pose.inverse());
        }
    }

    // Build observations from all keyframes to map points
    let all_kf_ids: Vec<_> = opt_kf_ids.iter()
        .chain(fixed_kf_ids.iter())
        .copied()
        .collect();

    for &mp_id in mp_ids {
        let mp_idx = mp_to_idx[&mp_id];

        if let Some(mp) = map.get_map_point(mp_id) {
            for (&obs_kf_id, &feat_idx) in &mp.observations {
                if !all_kf_ids.contains(&obs_kf_id) {
                    continue;
                }

                if let Some(kf) = map.get_keyframe(obs_kf_id) {
                    if let Ok(kp) = kf.keypoints.get(feat_idx) {
                        let uv = Vector2::new(kp.pt().x as f64, kp.pt().y as f64);
                        let is_stereo = kf.points_cam.get(feat_idx)
                            .map_or(false, |p| p.is_some());

                        // Check if this KF is optimized or fixed
                        let kf_param_idx = kf_to_idx.get(&obs_kf_id).copied();
                        // Skip first KF (anchor) - treated as fixed
                        let kf_param_idx = if kf_param_idx == Some(0) { None } else { kf_param_idx };

                        observations.push(VisualObs {
                            kf_param_idx,
                            mp_param_idx: mp_idx,
                            observed_uv: uv,
                            kf_id: obs_kf_id,
                            is_stereo,
                        });
                    }
                }
            }
        }
    }

    (observations, fixed_poses)
}

/// Build IMU edges between consecutive keyframes.
fn build_imu_edges(
    map: &Map,
    opt_kf_ids: &[KeyFrameId],
    kf_to_idx: &HashMap<KeyFrameId, usize>,
) -> Vec<ImuEdge> {
    let mut edges = Vec::new();

    // Iterate through consecutive pairs
    for i in 0..opt_kf_ids.len().saturating_sub(1) {
        let kf_i_id = opt_kf_ids[i];
        let kf_j_id = opt_kf_ids[i + 1];

        // Get preintegration from KF j (it stores preint from its prev_kf)
        if let Some(kf_j) = map.get_keyframe(kf_j_id) {
            if let Some(ref preint) = kf_j.imu_preintegrated {
                if preint.dt > 0.0 {
                    edges.push(ImuEdge {
                        kf_i_idx: kf_to_idx[&kf_i_id],
                        kf_j_idx: kf_to_idx[&kf_j_id],
                        preint: preint.clone(),
                    });
                }
            }
        }
    }

    edges
}

/// Initialize parameter vector from map.
fn initialize_params(
    map: &Map,
    opt_kf_ids: &[KeyFrameId],
    mp_ids: &[MapPointId],
    layout: &InertialStateLayout,
) -> DVector<f64> {
    let mut params = DVector::zeros(layout.total_params());

    // Initialize keyframe states
    for (i, &kf_id) in opt_kf_ids.iter().enumerate() {
        if let Some(kf) = map.get_keyframe(kf_id) {
            // Pose (as axis-angle + translation, T_wc)
            let pose_idx = layout.pose_start(i);
            let rot = kf.pose.rotation.scaled_axis();
            let trans = &kf.pose.translation;
            params[pose_idx] = rot.x;
            params[pose_idx + 1] = rot.y;
            params[pose_idx + 2] = rot.z;
            params[pose_idx + 3] = trans.x;
            params[pose_idx + 4] = trans.y;
            params[pose_idx + 5] = trans.z;

            // Velocity
            let vel_idx = layout.vel_start(i);
            params[vel_idx] = kf.velocity.x;
            params[vel_idx + 1] = kf.velocity.y;
            params[vel_idx + 2] = kf.velocity.z;

            // Biases
            let bg_idx = layout.gyro_bias_start(i);
            params[bg_idx] = kf.imu_bias.gyro.x;
            params[bg_idx + 1] = kf.imu_bias.gyro.y;
            params[bg_idx + 2] = kf.imu_bias.gyro.z;

            let ba_idx = layout.accel_bias_start(i);
            params[ba_idx] = kf.imu_bias.accel.x;
            params[ba_idx + 1] = kf.imu_bias.accel.y;
            params[ba_idx + 2] = kf.imu_bias.accel.z;
        }
    }

    // Initialize map points
    for (i, &mp_id) in mp_ids.iter().enumerate() {
        if let Some(mp) = map.get_map_point(mp_id) {
            let mp_idx = layout.mp_start(i);
            params[mp_idx] = mp.position.x;
            params[mp_idx + 1] = mp.position.y;
            params[mp_idx + 2] = mp.position.z;
        }
    }

    params
}

/// Extract pose from parameters.
fn extract_pose(params: &DVector<f64>, layout: &InertialStateLayout, kf_idx: usize) -> SE3 {
    let pose_idx = layout.pose_start(kf_idx);
    let rot = Vector3::new(params[pose_idx], params[pose_idx + 1], params[pose_idx + 2]);
    let trans = Vector3::new(params[pose_idx + 3], params[pose_idx + 4], params[pose_idx + 5]);

    let rotation = nalgebra::UnitQuaternion::from_scaled_axis(rot);
    SE3 { rotation, translation: trans }
}

/// Extract velocity from parameters.
fn extract_velocity(params: &DVector<f64>, layout: &InertialStateLayout, kf_idx: usize) -> Vector3<f64> {
    let vel_idx = layout.vel_start(kf_idx);
    Vector3::new(params[vel_idx], params[vel_idx + 1], params[vel_idx + 2])
}

/// Extract biases from parameters.
fn extract_bias(params: &DVector<f64>, layout: &InertialStateLayout, kf_idx: usize) -> ImuBias {
    let bg_idx = layout.gyro_bias_start(kf_idx);
    let ba_idx = layout.accel_bias_start(kf_idx);
    ImuBias {
        gyro: Vector3::new(params[bg_idx], params[bg_idx + 1], params[bg_idx + 2]),
        accel: Vector3::new(params[ba_idx], params[ba_idx + 1], params[ba_idx + 2]),
    }
}

/// Extract map point position from parameters.
fn extract_point(params: &DVector<f64>, layout: &InertialStateLayout, mp_idx: usize) -> Vector3<f64> {
    let idx = layout.mp_start(mp_idx);
    Vector3::new(params[idx], params[idx + 1], params[idx + 2])
}

/// Compute all residuals (visual + IMU + bias RW).
fn compute_all_residuals(
    params: &DVector<f64>,
    layout: &InertialStateLayout,
    visual_obs: &[VisualObs],
    imu_edges: &[ImuEdge],
    fixed_poses: &HashMap<KeyFrameId, SE3>,
    camera: &CameraModel,
    config: &LocalInertialBAConfig,
) -> DVector<f64> {
    let num_visual = visual_obs.len() * 2;
    let num_imu = imu_edges.len() * 9;
    let num_bias_rw = imu_edges.len() * 6; // 3 gyro + 3 accel per edge
    let total = num_visual + num_imu + num_bias_rw;

    let mut residuals = DVector::zeros(total);
    let mut idx = 0;

    // Visual residuals
    for obs in visual_obs {
        let pose_cw = if let Some(kf_idx) = obs.kf_param_idx {
            extract_pose(params, layout, kf_idx).inverse()
        } else {
            fixed_poses.get(&obs.kf_id).cloned().unwrap_or_else(SE3::identity)
        };

        let point = extract_point(params, layout, obs.mp_param_idx);
        let p_cam = pose_cw.transform_point(&point);

        if p_cam.z > 0.001 {
            let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
            let v = camera.fy * p_cam.y / p_cam.z + camera.cy;
            let err = Vector2::new(obs.observed_uv.x - u, obs.observed_uv.y - v);

            // Apply Huber weighting
            let threshold = if obs.is_stereo { config.huber_threshold_stereo } else { config.huber_threshold_mono };
            let err_norm = err.norm();
            let weight = if err_norm <= threshold { 1.0 } else { threshold / err_norm };

            residuals[idx] = err.x * weight.sqrt();
            residuals[idx + 1] = err.y * weight.sqrt();
        } else {
            residuals[idx] = 100.0;
            residuals[idx + 1] = 100.0;
        }
        idx += 2;
    }

    // IMU residuals
    for edge in imu_edges {
        let pose_i = extract_pose(params, layout, edge.kf_i_idx);
        let vel_i = extract_velocity(params, layout, edge.kf_i_idx);
        let pose_j = extract_pose(params, layout, edge.kf_j_idx);
        let vel_j = extract_velocity(params, layout, edge.kf_j_idx);

        let imu_res = compute_imu_residual(&pose_i, &vel_i, &pose_j, &vel_j, &edge.preint);
        let res_vec = imu_res.as_vector();

        // Weight by information matrix if available
        for k in 0..9 {
            residuals[idx + k] = res_vec[k];
        }
        idx += 9;
    }

    // Bias random walk residuals
    for edge in imu_edges {
        let bias_i = extract_bias(params, layout, edge.kf_i_idx);
        let bias_j = extract_bias(params, layout, edge.kf_j_idx);

        // Gyro bias RW: bg_j - bg_i
        let bg_diff = bias_j.gyro - bias_i.gyro;
        residuals[idx] = bg_diff.x * config.gyro_rw_info.sqrt();
        residuals[idx + 1] = bg_diff.y * config.gyro_rw_info.sqrt();
        residuals[idx + 2] = bg_diff.z * config.gyro_rw_info.sqrt();

        // Accel bias RW: ba_j - ba_i
        let ba_diff = bias_j.accel - bias_i.accel;
        residuals[idx + 3] = ba_diff.x * config.accel_rw_info.sqrt();
        residuals[idx + 4] = ba_diff.y * config.accel_rw_info.sqrt();
        residuals[idx + 5] = ba_diff.z * config.accel_rw_info.sqrt();

        idx += 6;
    }

    residuals
}

/// Compute Jacobian of all residuals using analytical Jacobians for visual terms.
///
/// This hybrid approach uses:
/// - Analytical Jacobians for visual residuals (dominant cost, 5-10x speedup)
/// - Numerical Jacobians for IMU and bias residuals (smaller, more complex)
fn compute_all_jacobians(
    params: &DVector<f64>,
    layout: &InertialStateLayout,
    visual_obs: &[VisualObs],
    imu_edges: &[ImuEdge],
    fixed_poses: &HashMap<KeyFrameId, SE3>,
    camera: &CameraModel,
    config: &LocalInertialBAConfig,
) -> DMatrix<f64> {
    let num_visual = visual_obs.len() * 2;
    let num_imu = imu_edges.len() * 9;
    let num_bias_rw = imu_edges.len() * 6;
    let num_residuals = num_visual + num_imu + num_bias_rw;
    let num_params = layout.total_params();

    let mut jacobian = DMatrix::zeros(num_residuals, num_params);

    // ========== VISUAL JACOBIANS (ANALYTICAL) ==========
    // This is the critical optimization: visual obs dominate the residuals
    let mut vis_res_idx = 0;
    for obs in visual_obs {
        let pose_cw = if let Some(kf_idx) = obs.kf_param_idx {
            extract_pose(params, layout, kf_idx).inverse()
        } else {
            fixed_poses.get(&obs.kf_id).cloned().unwrap_or_else(SE3::identity)
        };

        let point = extract_point(params, layout, obs.mp_param_idx);
        let p_cam = pose_cw.transform_point(&point);

        if p_cam.z > 0.001 {
            let x = p_cam.x;
            let y = p_cam.y;
            let z = p_cam.z;
            let z_inv = 1.0 / z;
            let z_inv_sq = z_inv * z_inv;

            // Compute Huber weight
            let u = camera.fx * x * z_inv + camera.cx;
            let v = camera.fy * y * z_inv + camera.cy;
            let err = Vector2::new(obs.observed_uv.x - u, obs.observed_uv.y - v);
            let threshold = if obs.is_stereo { config.huber_threshold_stereo } else { config.huber_threshold_mono };
            let err_norm = err.norm();
            let weight_sqrt = if err_norm <= threshold { 1.0 } else { (threshold / err_norm).sqrt() };

            // Jacobian w.r.t. point (3 params)
            // d(error)/d(point) = d(error)/d(p_cam) * d(p_cam)/d(point)
            // d(p_cam)/d(point) = R_cw
            // d(u)/d(p_cam) = [fx/z, 0, -fx*x/z^2]
            // d(v)/d(p_cam) = [0, fy/z, -fy*y/z^2]
            let du_dp = Vector3::new(camera.fx * z_inv, 0.0, -camera.fx * x * z_inv_sq);
            let dv_dp = Vector3::new(0.0, camera.fy * z_inv, -camera.fy * y * z_inv_sq);

            // Transform to world frame: d(error)/d(point_world) = d(error)/d(p_cam) * R_cw
            let r_cw = pose_cw.rotation.to_rotation_matrix();
            let du_dpoint = r_cw.matrix().transpose() * du_dp;
            let dv_dpoint = r_cw.matrix().transpose() * dv_dp;

            // Fill Jacobian for point (negative because error = obs - proj)
            let mp_start = layout.mp_start(obs.mp_param_idx);
            jacobian[(vis_res_idx, mp_start)] = -du_dpoint.x * weight_sqrt;
            jacobian[(vis_res_idx, mp_start + 1)] = -du_dpoint.y * weight_sqrt;
            jacobian[(vis_res_idx, mp_start + 2)] = -du_dpoint.z * weight_sqrt;
            jacobian[(vis_res_idx + 1, mp_start)] = -dv_dpoint.x * weight_sqrt;
            jacobian[(vis_res_idx + 1, mp_start + 1)] = -dv_dpoint.y * weight_sqrt;
            jacobian[(vis_res_idx + 1, mp_start + 2)] = -dv_dpoint.z * weight_sqrt;

            // Jacobian w.r.t. pose (if not fixed)
            if let Some(kf_idx) = obs.kf_param_idx {
                let xy = x * y;
                let x_sq = x * x;
                let y_sq = y * y;

                // Jacobian w.r.t. pose parameters [rot(3), trans(3)]
                let pose_start = layout.pose_start(kf_idx);

                // d(error)/d(rot) - standard pinhole camera Jacobian
                jacobian[(vis_res_idx, pose_start)] = -camera.fx * xy * z_inv_sq * weight_sqrt;
                jacobian[(vis_res_idx, pose_start + 1)] = camera.fx * (1.0 + x_sq * z_inv_sq) * weight_sqrt;
                jacobian[(vis_res_idx, pose_start + 2)] = -camera.fx * y * z_inv * weight_sqrt;

                jacobian[(vis_res_idx + 1, pose_start)] = -camera.fy * (1.0 + y_sq * z_inv_sq) * weight_sqrt;
                jacobian[(vis_res_idx + 1, pose_start + 1)] = camera.fy * xy * z_inv_sq * weight_sqrt;
                jacobian[(vis_res_idx + 1, pose_start + 2)] = camera.fy * x * z_inv * weight_sqrt;

                // d(error)/d(trans)
                jacobian[(vis_res_idx, pose_start + 3)] = camera.fx * z_inv * weight_sqrt;
                jacobian[(vis_res_idx, pose_start + 4)] = 0.0;
                jacobian[(vis_res_idx, pose_start + 5)] = -camera.fx * x * z_inv_sq * weight_sqrt;

                jacobian[(vis_res_idx + 1, pose_start + 3)] = 0.0;
                jacobian[(vis_res_idx + 1, pose_start + 4)] = camera.fy * z_inv * weight_sqrt;
                jacobian[(vis_res_idx + 1, pose_start + 5)] = -camera.fy * y * z_inv_sq * weight_sqrt;
            }
        }
        vis_res_idx += 2;
    }

    // ========== IMU AND BIAS JACOBIANS (NUMERICAL) ==========
    // These are more complex and fewer in number, keep numerical for now
    let eps = 1e-6;
    let imu_start_idx = num_visual;
    let bias_start_idx = num_visual + num_imu;

    // Compute base residuals for IMU and bias terms only
    for (edge_idx, edge) in imu_edges.iter().enumerate() {
        let pose_i = extract_pose(params, layout, edge.kf_i_idx);
        let vel_i = extract_velocity(params, layout, edge.kf_i_idx);
        let pose_j = extract_pose(params, layout, edge.kf_j_idx);
        let vel_j = extract_velocity(params, layout, edge.kf_j_idx);

        let imu_res_base = compute_imu_residual(&pose_i, &vel_i, &pose_j, &vel_j, &edge.preint);
        let imu_vec_base = imu_res_base.as_vector();

        // Numerical Jacobian for IMU residuals w.r.t. involved KF states
        for kf_idx in [edge.kf_i_idx, edge.kf_j_idx] {
            // Pose parameters
            let pose_start = layout.pose_start(kf_idx);
            for j in 0..6 {
                let mut params_plus = params.clone();
                params_plus[pose_start + j] += eps;

                let pose_i_p = extract_pose(&params_plus, layout, edge.kf_i_idx);
                let vel_i_p = extract_velocity(&params_plus, layout, edge.kf_i_idx);
                let pose_j_p = extract_pose(&params_plus, layout, edge.kf_j_idx);
                let vel_j_p = extract_velocity(&params_plus, layout, edge.kf_j_idx);

                let imu_res_plus = compute_imu_residual(&pose_i_p, &vel_i_p, &pose_j_p, &vel_j_p, &edge.preint);
                let imu_vec_plus = imu_res_plus.as_vector();

                for k in 0..9 {
                    jacobian[(imu_start_idx + edge_idx * 9 + k, pose_start + j)] =
                        (imu_vec_plus[k] - imu_vec_base[k]) / eps;
                }
            }

            // Velocity parameters
            let vel_start = layout.vel_start(kf_idx);
            for j in 0..3 {
                let mut params_plus = params.clone();
                params_plus[vel_start + j] += eps;

                let pose_i_p = extract_pose(&params_plus, layout, edge.kf_i_idx);
                let vel_i_p = extract_velocity(&params_plus, layout, edge.kf_i_idx);
                let pose_j_p = extract_pose(&params_plus, layout, edge.kf_j_idx);
                let vel_j_p = extract_velocity(&params_plus, layout, edge.kf_j_idx);

                let imu_res_plus = compute_imu_residual(&pose_i_p, &vel_i_p, &pose_j_p, &vel_j_p, &edge.preint);
                let imu_vec_plus = imu_res_plus.as_vector();

                for k in 0..9 {
                    jacobian[(imu_start_idx + edge_idx * 9 + k, vel_start + j)] =
                        (imu_vec_plus[k] - imu_vec_base[k]) / eps;
                }
            }
        }

        // Bias random walk Jacobians (analytical - simple difference)
        let bg_i_start = layout.gyro_bias_start(edge.kf_i_idx);
        let bg_j_start = layout.gyro_bias_start(edge.kf_j_idx);
        let ba_i_start = layout.accel_bias_start(edge.kf_i_idx);
        let ba_j_start = layout.accel_bias_start(edge.kf_j_idx);

        let gyro_weight = config.gyro_rw_info.sqrt();
        let accel_weight = config.accel_rw_info.sqrt();

        // d(bg_j - bg_i)/d(bg_i) = -I, d(bg_j - bg_i)/d(bg_j) = I
        for k in 0..3 {
            jacobian[(bias_start_idx + edge_idx * 6 + k, bg_i_start + k)] = -gyro_weight;
            jacobian[(bias_start_idx + edge_idx * 6 + k, bg_j_start + k)] = gyro_weight;
            jacobian[(bias_start_idx + edge_idx * 6 + 3 + k, ba_i_start + k)] = -accel_weight;
            jacobian[(bias_start_idx + edge_idx * 6 + 3 + k, ba_j_start + k)] = accel_weight;
        }
    }

    jacobian
}

/// Write optimized results back to map.
fn write_back_results(
    map: &mut Map,
    opt_kf_ids: &[KeyFrameId],
    mp_ids: &[MapPointId],
    params: &DVector<f64>,
    layout: &InertialStateLayout,
) {
    // Update keyframe states (skip first which is anchor)
    for (i, &kf_id) in opt_kf_ids.iter().enumerate().skip(1) {
        if let Some(kf) = map.get_keyframe_mut(kf_id) {
            // Update pose
            kf.pose = extract_pose(params, layout, i);

            // Update velocity
            kf.velocity = extract_velocity(params, layout, i);

            // Update bias
            kf.imu_bias = extract_bias(params, layout, i);
        }
    }

    // Update map points
    for (i, &mp_id) in mp_ids.iter().enumerate() {
        if let Some(mp) = map.get_map_point_mut(mp_id) {
            mp.position = extract_point(params, layout, i);
        }
    }
}

// ============================================================================
// THREE-PHASE BA: Public functions for lock-free optimization
// ============================================================================

/// PHASE 1: COLLECT - Extract data snapshot for inertial bundle adjustment.
///
/// This function extracts all necessary data from the map to run Inertial BA.
/// It should be called while holding a **read lock** on the map.
/// After calling this, the lock can be released and BA can run lock-free.
///
/// # Arguments
/// * `map` - The map to extract data from (read access)
/// * `current_kf_id` - The current keyframe triggering BA
/// * `config` - BA configuration
///
/// # Returns
/// `Some(InertialBAProblemData)` if enough data exists, `None` otherwise.
pub fn collect_inertial_ba_data(
    map: &Map,
    current_kf_id: KeyFrameId,
    config: &LocalInertialBAConfig,
) -> Option<InertialBAProblemData> {
    // Step 1: Collect temporal chain of keyframes
    let opt_kf_ids = collect_temporal_keyframes(map, current_kf_id, config.window_size);
    if opt_kf_ids.len() < 2 {
        return None; // Need at least 2 KFs for IMU constraint
    }

    // Step 2: Collect map points observed by optimized keyframes
    let mp_ids = collect_map_points(map, &opt_kf_ids);
    if mp_ids.is_empty() {
        return None;
    }

    // Step 3: Collect fixed keyframes (observe local map points but not in opt window)
    let fixed_kf_ids = collect_fixed_keyframes(map, &opt_kf_ids, &mp_ids);
    let opt_kf_set: HashSet<_> = opt_kf_ids.iter().copied().collect();

    // Step 4: Extract keyframe states
    let mut kf_poses = HashMap::new();
    let mut kf_velocities = HashMap::new();
    let mut kf_biases = HashMap::new();
    for &kf_id in &opt_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            kf_poses.insert(kf_id, kf.pose.clone());
            kf_velocities.insert(kf_id, kf.velocity);
            kf_biases.insert(kf_id, kf.imu_bias.clone());
        }
    }

    // Step 5: Extract fixed keyframe poses (as T_cw for projection)
    let mut fixed_kf_poses = HashMap::new();
    for &kf_id in &fixed_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            fixed_kf_poses.insert(kf_id, kf.pose.inverse()); // T_cw
        }
    }
    // Also add first KF (anchor) pose for visual observations
    if let Some(&first_kf_id) = opt_kf_ids.first() {
        if let Some(kf) = map.get_keyframe(first_kf_id) {
            fixed_kf_poses.insert(first_kf_id, kf.pose.inverse());
        }
    }

    // Step 6: Extract map point positions
    let mut mp_positions = HashMap::new();
    for &mp_id in &mp_ids {
        if let Some(mp) = map.get_map_point(mp_id) {
            mp_positions.insert(mp_id, mp.position);
        }
    }

    // Step 7: Build visual observations
    let mut visual_observations = Vec::new();

    let all_kf_ids: Vec<_> = opt_kf_ids.iter()
        .chain(fixed_kf_ids.iter())
        .copied()
        .collect();

    for &mp_id in &mp_ids {
        if let Some(mp) = map.get_map_point(mp_id) {
            for (&obs_kf_id, &feat_idx) in &mp.observations {
                if !all_kf_ids.contains(&obs_kf_id) {
                    continue;
                }

                if let Some(kf) = map.get_keyframe(obs_kf_id) {
                    if let Ok(kp) = kf.keypoints.get(feat_idx) {
                        let uv = Vector2::new(kp.pt().x as f64, kp.pt().y as f64);
                        let is_stereo = kf.points_cam.get(feat_idx)
                            .map_or(false, |p| p.is_some());

                        // KF is in window if it's in opt_kf_ids (but first one is anchor/fixed)
                        let is_in_window = opt_kf_set.contains(&obs_kf_id);
                        // Skip first KF index (anchor) - it's treated as fixed
                        let is_kf_in_window = is_in_window &&
                            opt_kf_ids.first() != Some(&obs_kf_id);

                        visual_observations.push(InertialVisualObs {
                            kf_id: obs_kf_id,
                            mp_id,
                            observed_uv: uv,
                            is_stereo,
                            is_kf_in_window,
                        });
                    }
                }
            }
        }
    }

    // Step 8: Build IMU edges between consecutive keyframes
    let mut imu_edges = Vec::new();
    for i in 0..opt_kf_ids.len().saturating_sub(1) {
        let kf_i_id = opt_kf_ids[i];
        let kf_j_id = opt_kf_ids[i + 1];

        // Get preintegration from KF j (it stores preint from its prev_kf)
        if let Some(kf_j) = map.get_keyframe(kf_j_id) {
            if let Some(ref preint) = kf_j.imu_preintegrated {
                if preint.dt > 0.0 {
                    imu_edges.push(ImuEdgeData {
                        kf_i_id,
                        kf_j_id,
                        preint: preint.clone(),
                    });
                }
            }
        }
    }

    Some(InertialBAProblemData {
        kf_poses,
        kf_velocities,
        kf_biases,
        mp_positions,
        fixed_kf_poses,
        visual_observations,
        imu_edges,
        opt_kf_ids,
        mp_ids,
    })
}

/// PHASE 2: SOLVE - Run LM optimization on extracted inertial BA data.
///
/// This function runs Inertial Bundle Adjustment on the extracted data snapshot.
/// It does NOT require any locks on the map - all data is in the problem struct.
///
/// # Arguments
/// * `problem` - The extracted problem data from `collect_inertial_ba_data`
/// * `camera` - Camera intrinsics
/// * `config` - BA configuration
/// * `should_stop` - Callback to check if BA should abort early
///
/// # Returns
/// `Some(InertialBAResultData)` with optimized values, `None` if optimization failed.
pub fn solve_inertial_ba(
    problem: &InertialBAProblemData,
    camera: &CameraModel,
    config: &LocalInertialBAConfig,
    should_stop: &dyn Fn() -> bool,
) -> Option<InertialBAResultData> {
    if problem.opt_kf_ids.len() < 2 {
        return None;
    }

    // Build index mappings
    let kf_to_idx: HashMap<KeyFrameId, usize> = problem
        .opt_kf_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mp_to_idx: HashMap<MapPointId, usize> = problem
        .mp_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let layout = InertialStateLayout::new(problem.opt_kf_ids.len(), problem.mp_ids.len());

    // Build internal observations (with param indices)
    let visual_obs: Vec<VisualObs> = problem
        .visual_observations
        .iter()
        .filter_map(|obs| {
            let mp_param_idx = *mp_to_idx.get(&obs.mp_id)?;
            let kf_param_idx = if obs.is_kf_in_window {
                kf_to_idx.get(&obs.kf_id).copied()
            } else {
                None
            };
            Some(VisualObs {
                kf_param_idx,
                mp_param_idx,
                observed_uv: obs.observed_uv,
                kf_id: obs.kf_id,
                is_stereo: obs.is_stereo,
            })
        })
        .collect();

    // Build IMU edges with indices
    let imu_edges: Vec<ImuEdge> = problem
        .imu_edges
        .iter()
        .filter_map(|edge| {
            let kf_i_idx = *kf_to_idx.get(&edge.kf_i_id)?;
            let kf_j_idx = *kf_to_idx.get(&edge.kf_j_id)?;
            Some(ImuEdge {
                kf_i_idx,
                kf_j_idx,
                preint: edge.preint.clone(),
            })
        })
        .collect();

    // Initialize parameter vector
    let mut params = DVector::zeros(layout.total_params());

    for (i, &kf_id) in problem.opt_kf_ids.iter().enumerate() {
        // Pose (as axis-angle + translation, T_wc)
        if let Some(pose) = problem.kf_poses.get(&kf_id) {
            let pose_idx = layout.pose_start(i);
            let rot = pose.rotation.scaled_axis();
            params[pose_idx] = rot.x;
            params[pose_idx + 1] = rot.y;
            params[pose_idx + 2] = rot.z;
            params[pose_idx + 3] = pose.translation.x;
            params[pose_idx + 4] = pose.translation.y;
            params[pose_idx + 5] = pose.translation.z;
        }

        // Velocity
        if let Some(vel) = problem.kf_velocities.get(&kf_id) {
            let vel_idx = layout.vel_start(i);
            params[vel_idx] = vel.x;
            params[vel_idx + 1] = vel.y;
            params[vel_idx + 2] = vel.z;
        }

        // Biases
        if let Some(bias) = problem.kf_biases.get(&kf_id) {
            let bg_idx = layout.gyro_bias_start(i);
            params[bg_idx] = bias.gyro.x;
            params[bg_idx + 1] = bias.gyro.y;
            params[bg_idx + 2] = bias.gyro.z;

            let ba_idx = layout.accel_bias_start(i);
            params[ba_idx] = bias.accel.x;
            params[ba_idx + 1] = bias.accel.y;
            params[ba_idx + 2] = bias.accel.z;
        }
    }

    // Initialize map points
    for (i, &mp_id) in problem.mp_ids.iter().enumerate() {
        if let Some(pos) = problem.mp_positions.get(&mp_id) {
            let mp_idx = layout.mp_start(i);
            params[mp_idx] = pos.x;
            params[mp_idx + 1] = pos.y;
            params[mp_idx + 2] = pos.z;
        }
    }

    // Compute initial error
    let initial_residuals = compute_all_residuals(
        &params, &layout, &visual_obs, &imu_edges, &problem.fixed_kf_poses, camera, config,
    );
    let initial_error = initial_residuals.norm();

    // LM optimization loop
    let mut current_params = params;
    let mut lambda = config.initial_lambda;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        if should_stop() {
            break;
        }
        iterations = iter + 1;

        let residuals = compute_all_residuals(
            &current_params, &layout, &visual_obs, &imu_edges, &problem.fixed_kf_poses, camera, config,
        );
        let jacobian = compute_all_jacobians(
            &current_params, &layout, &visual_obs, &imu_edges, &problem.fixed_kf_poses, camera, config,
        );

        let current_error_sq = residuals.norm_squared();

        let gradient = jacobian.transpose() * &residuals;
        let jtj = jacobian.transpose() * &jacobian;

        if gradient.norm() < 1e-8 {
            break;
        }

        let num_params = layout.total_params();
        let mut damped_jtj = jtj.clone();
        for i in 0..num_params {
            damped_jtj[(i, i)] += lambda * damped_jtj[(i, i)].max(1e-6);
        }

        let delta = match damped_jtj.lu().solve(&(-&gradient)) {
            Some(d) => d,
            None => break,
        };

        let trial_params = &current_params + &delta;
        let trial_residuals = compute_all_residuals(
            &trial_params, &layout, &visual_obs, &imu_edges, &problem.fixed_kf_poses, camera, config,
        );
        let trial_error_sq = trial_residuals.norm_squared();

        if trial_error_sq < current_error_sq {
            current_params = trial_params;
            lambda = (lambda * 0.1).max(1e-10);
        } else {
            lambda = (lambda * 10.0).min(1e10);
        }
    }

    // Compute final error
    let final_residuals = compute_all_residuals(
        &current_params, &layout, &visual_obs, &imu_edges, &problem.fixed_kf_poses, camera, config,
    );
    let final_error = final_residuals.norm();

    // Extract optimized results (skip first KF which is anchor)
    let mut optimized_poses = HashMap::new();
    let mut optimized_velocities = HashMap::new();
    let mut optimized_biases = HashMap::new();

    for (i, &kf_id) in problem.opt_kf_ids.iter().enumerate().skip(1) {
        optimized_poses.insert(kf_id, extract_pose(&current_params, &layout, i));
        optimized_velocities.insert(kf_id, extract_velocity(&current_params, &layout, i));
        optimized_biases.insert(kf_id, extract_bias(&current_params, &layout, i));
    }

    // Extract optimized map points
    let mut optimized_points = HashMap::new();
    for (i, &mp_id) in problem.mp_ids.iter().enumerate() {
        optimized_points.insert(mp_id, extract_point(&current_params, &layout, i));
    }

    Some(InertialBAResultData {
        optimized_poses,
        optimized_velocities,
        optimized_biases,
        optimized_points,
        iterations,
        initial_error,
        final_error,
    })
}

/// PHASE 3: APPLY - Write optimized inertial BA results back to the map.
///
/// This function writes the optimized poses, velocities, biases and positions back to the map.
/// It should be called while holding a **write lock** on the map.
/// Entities that were deleted during optimization are silently skipped.
///
/// # Arguments
/// * `map` - The map to write to (write access)
/// * `results` - The optimization results from `solve_inertial_ba`
///
/// # Returns
/// Number of entities successfully updated.
pub fn apply_inertial_ba_results(map: &mut Map, results: &InertialBAResultData) -> usize {
    let mut updated = 0;

    // Update keyframe states
    for (kf_id, pose) in &results.optimized_poses {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            if !kf.is_bad {
                kf.pose = pose.clone();
                updated += 1;
            }
        }
    }

    for (kf_id, vel) in &results.optimized_velocities {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            if !kf.is_bad {
                kf.velocity = *vel;
            }
        }
    }

    for (kf_id, bias) in &results.optimized_biases {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            if !kf.is_bad {
                kf.imu_bias = bias.clone();
            }
        }
    }

    // Update map point positions
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inertial_state_layout() {
        let layout = InertialStateLayout::new(3, 10);

        assert_eq!(layout.total_params(), 3 * 15 + 10 * 3); // 45 + 30 = 75
        assert_eq!(layout.pose_start(0), 0);
        assert_eq!(layout.vel_start(0), 6);
        assert_eq!(layout.gyro_bias_start(0), 9);
        assert_eq!(layout.accel_bias_start(0), 12);
        assert_eq!(layout.pose_start(1), 15);
        assert_eq!(layout.mp_start(0), 45);
    }
}
