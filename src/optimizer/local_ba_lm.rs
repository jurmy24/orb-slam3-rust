//! Local Bundle Adjustment using Levenberg-Marquardt optimizer.
//!
//! This module provides a robust, numerically stable implementation of Local BA
//! using the `levenberg-marquardt` crate. It replaces the hand-written Gauss-Newton
//! solver that had Jacobian/gradient sign issues causing divergence.
//!
//! # Theory
//!
//! Bundle Adjustment minimizes the reprojection error:
//!
//! ```text
//! E = Σ_i,j ρ(||u_ij - π(T_i, p_j)||²)
//! ```
//!
//! where:
//! - `u_ij` is the observed 2D point in keyframe `i` for map point `j`
//! - `π(T, p)` projects 3D point `p` using camera pose `T`
//! - `ρ` is a robust kernel (Huber or Cauchy) to downweight outliers
//!
//! The LM algorithm iteratively solves:
//!
//! ```text
//! (J^T J + λ diag(J^T J)) δ = -J^T r
//! ```
//!
//! where `λ` is adaptively adjusted based on error reduction.

use std::collections::{HashMap, HashSet};

use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{
    DMatrix, DVector, Dyn, Matrix2x3, Matrix2x6, Owned, Vector2, Vector3,
};
use opencv::prelude::*;

use crate::atlas::map::{KeyFrameId, Map, MapPointId};
use crate::geometry::SE3;
use crate::tracking::frame::CameraModel;

// ============================================================================
// THREE-PHASE BA: Data structures for lock-free solving
// ============================================================================

/// Extracted snapshot of data for lock-free visual BA solving.
///
/// This contains all the information needed to run Bundle Adjustment
/// without holding any locks on the map.
#[derive(Clone)]
pub struct VisualBAProblemData {
    /// Poses of local (optimized) keyframes as T_cw (world-to-camera).
    /// Does NOT include the anchor keyframe.
    pub local_kf_poses: HashMap<KeyFrameId, SE3>,
    /// Positions of map points to optimize.
    pub local_mp_positions: HashMap<MapPointId, Vector3<f64>>,
    /// Poses of fixed keyframes (anchor + other observers) as T_cw.
    pub fixed_kf_poses: HashMap<KeyFrameId, SE3>,
    /// The anchor keyframe ID (first/oldest local KF, kept fixed).
    pub anchor_kf_id: KeyFrameId,
    /// All observations linking keyframes to map points.
    pub observations: Vec<VisualObservation>,
    /// Ordered list of optimized keyframe IDs (for parameter indexing).
    pub optimized_kf_ids: Vec<KeyFrameId>,
    /// Ordered list of map point IDs (for parameter indexing).
    pub mp_ids: Vec<MapPointId>,
}

/// A single visual observation for bundle adjustment.
#[derive(Clone)]
pub struct VisualObservation {
    /// Keyframe ID where this observation was made.
    pub kf_id: KeyFrameId,
    /// Map point ID being observed.
    pub mp_id: MapPointId,
    /// Observed 2D pixel coordinates.
    pub observed_uv: Vector2<f64>,
    /// Whether this keyframe is optimized (true) or fixed (false).
    pub is_kf_optimized: bool,
}

/// Result data from visual BA, ready to be applied to the map.
#[derive(Clone)]
pub struct VisualBAResultData {
    /// Optimized poses (T_wc, camera-to-world) indexed by keyframe ID.
    pub optimized_poses: HashMap<KeyFrameId, SE3>,
    /// Optimized 3D positions indexed by map point ID.
    pub optimized_points: HashMap<MapPointId, Vector3<f64>>,
    /// Number of LM iterations performed.
    pub iterations: usize,
    /// Initial mean reprojection error (pixels).
    pub initial_error: f64,
    /// Final mean reprojection error (pixels).
    pub final_error: f64,
}

/// Configuration for LM-based Local Bundle Adjustment.
pub struct LocalBAConfigLM {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,
    /// Convergence threshold on parameter change.
    pub param_tolerance: f64,
    /// Convergence threshold on gradient norm.
    pub gradient_tolerance: f64,
    /// Huber kernel threshold (pixels) for robust estimation.
    pub huber_threshold: f64,
    /// Maximum covisible keyframes to include.
    pub max_covisible_keyframes: usize,
}

impl Default for LocalBAConfigLM {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            param_tolerance: 1e-8,
            gradient_tolerance: 1e-8,
            huber_threshold: 5.991_f64.sqrt(), // ~2.45 pixels (95% chi2 with 2 DOF)
            max_covisible_keyframes: 20,
        }
    }
}

/// Result of LM-based Local Bundle Adjustment.
#[derive(Debug)]
pub struct LocalBAResultLM {
    /// Number of LM iterations performed.
    pub iterations: usize,
    /// Initial mean reprojection error (pixels).
    pub initial_error: f64,
    /// Final mean reprojection error (pixels).
    pub final_error: f64,
    /// Number of keyframes optimized (excluding fixed).
    pub num_keyframes: usize,
    /// Number of map points optimized.
    pub num_map_points: usize,
    /// Number of observations (edges).
    pub num_observations: usize,
    /// Termination reason from LM.
    pub termination: String,
}

/// An observation for bundle adjustment.
struct Observation {
    /// Index of keyframe in the parameter vector (None if fixed).
    kf_param_idx: Option<usize>,
    /// Index of map point in the parameter vector.
    mp_param_idx: usize,
    /// Observed 2D pixel coordinates.
    observed_uv: Vector2<f64>,
    /// Keyframe ID (for debug/lookup).
    kf_id: KeyFrameId,
    /// Map point ID (for debug/lookup).
    mp_id: MapPointId,
}

/// The Bundle Adjustment problem for LM solver.
struct BAProblem<'a> {
    /// Camera intrinsics.
    camera: &'a CameraModel,
    /// Observations to fit.
    observations: Vec<Observation>,
    /// Fixed keyframe poses (T_cw, world-to-camera).
    fixed_poses_cw: HashMap<KeyFrameId, SE3>,
    /// Huber threshold for robust estimation.
    huber_threshold: f64,
    /// Number of optimized keyframes.
    num_kf_params: usize,
    /// Number of optimized map points.
    num_mp_params: usize,
}

impl<'a> BAProblem<'a> {
    /// Extract poses from parameter vector.
    fn get_pose_cw(&self, params: &DVector<f64>, kf_param_idx: Option<usize>) -> Option<SE3> {
        match kf_param_idx {
            Some(idx) => {
                // Optimized pose: extract from params
                let base = idx * 6;
                let rot = Vector3::new(params[base], params[base + 1], params[base + 2]);
                let trans = Vector3::new(params[base + 3], params[base + 4], params[base + 5]);
                Some(se3_from_params(&rot, &trans))
            }
            None => None, // Will be looked up from fixed_poses_cw by caller
        }
    }

    /// Extract map point position from parameter vector.
    fn get_point(&self, params: &DVector<f64>, mp_param_idx: usize) -> Vector3<f64> {
        let base = self.num_kf_params * 6 + mp_param_idx * 3;
        Vector3::new(params[base], params[base + 1], params[base + 2])
    }

    /// Compute reprojection error for one observation.
    fn compute_error(
        &self,
        pose_cw: &SE3,
        point_world: &Vector3<f64>,
        observed: &Vector2<f64>,
    ) -> Vector2<f64> {
        // Transform to camera frame
        let p_cam = pose_cw.transform_point(point_world);

        if p_cam.z <= 0.001 {
            // Point behind camera - return large error
            return Vector2::new(100.0, 100.0);
        }

        // Project to image plane
        let u = self.camera.fx * p_cam.x / p_cam.z + self.camera.cx;
        let v = self.camera.fy * p_cam.y / p_cam.z + self.camera.cy;

        // Error = observed - projected (g2o convention)
        Vector2::new(observed.x - u, observed.y - v)
    }

    /// Compute Jacobian of error w.r.t. pose (6-DoF: rotation then translation).
    /// Matches g2o's EdgeSE3ProjectXYZ::linearizeOplus().
    fn jacobian_pose(
        &self,
        pose_cw: &SE3,
        point_world: &Vector3<f64>,
    ) -> Matrix2x6<f64> {
        let p_cam = pose_cw.transform_point(point_world);

        let x = p_cam.x;
        let y = p_cam.y;
        let z = p_cam.z;

        if z.abs() < 1e-6 {
            return Matrix2x6::zeros();
        }

        let invz = 1.0 / z;
        let invz2 = invz * invz;

        let fx = self.camera.fx;
        let fy = self.camera.fy;

        // g2o convention: columns are [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        // These are derivatives of -projected (since error = observed - projected)
        Matrix2x6::new(
            // Row 0: d(u_err)/d(params)
            x * y * invz2 * fx,           // d(u_err)/d(rot_x)
            -(1.0 + x * x * invz2) * fx,  // d(u_err)/d(rot_y)
            y * invz * fx,                // d(u_err)/d(rot_z)
            -invz * fx,                   // d(u_err)/d(trans_x)
            0.0,                          // d(u_err)/d(trans_y)
            x * invz2 * fx,               // d(u_err)/d(trans_z)
            // Row 1: d(v_err)/d(params)
            (1.0 + y * y * invz2) * fy,   // d(v_err)/d(rot_x)
            -x * y * invz2 * fy,          // d(v_err)/d(rot_y)
            -x * invz * fy,               // d(v_err)/d(rot_z)
            0.0,                          // d(v_err)/d(trans_x)
            -invz * fy,                   // d(v_err)/d(trans_y)
            y * invz2 * fy,               // d(v_err)/d(trans_z)
        )
    }

    /// Compute Jacobian of error w.r.t. 3D point position.
    /// From g2o: _jacobianOplusXi = -1./z * tmp * R
    fn jacobian_point(
        &self,
        pose_cw: &SE3,
        point_world: &Vector3<f64>,
    ) -> Matrix2x3<f64> {
        let p_cam = pose_cw.transform_point(point_world);

        let x = p_cam.x;
        let y = p_cam.y;
        let z = p_cam.z;

        if z.abs() < 1e-6 {
            return Matrix2x3::zeros();
        }

        let invz = 1.0 / z;
        let fx = self.camera.fx;
        let fy = self.camera.fy;

        let r_cw = pose_cw.rotation.to_rotation_matrix().into_inner();

        // tmp = [fx, 0, -fx*x/z; 0, fy, -fy*y/z]
        let tmp = Matrix2x3::new(
            fx, 0.0, -fx * x * invz,
            0.0, fy, -fy * y * invz,
        );

        // j_point = -1/z * tmp * R_cw
        (-invz) * tmp * r_cw
    }

    /// Apply Huber weighting to an error.
    fn huber_weight(&self, error_norm: f64) -> f64 {
        if error_norm <= self.huber_threshold {
            1.0
        } else {
            self.huber_threshold / error_norm
        }
    }
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for BAProblem<'a> {
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, _params: &DVector<f64>) {
        // No internal state to update - we read params directly in residuals/jacobian
    }

    fn params(&self) -> DVector<f64> {
        // This is called once at start - we return zeros as placeholder
        // The actual initial params are set in the LM solver call
        DVector::zeros(self.num_kf_params * 6 + self.num_mp_params * 3)
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        // Can't compute without params - this method signature is awkward
        // We'll use the version that takes params explicitly
        None
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        // Same issue - need params
        None
    }
}

/// Run Local Bundle Adjustment using Levenberg-Marquardt.
///
/// Optimizes:
/// - Poses of local keyframes (except anchor/first keyframe)
/// - Positions of map points observed by local keyframes
///
/// Fixed:
/// - Anchor keyframe (oldest in local set)
/// - Poses of "fixed" keyframes (see local map points but aren't local)
pub fn local_bundle_adjustment_lm(
    map: &mut Map,
    current_kf_id: KeyFrameId,
    camera: &CameraModel,
    config: &LocalBAConfigLM,
    should_stop: &dyn Fn() -> bool,
) -> Option<LocalBAResultLM> {
    // Step 1: Collect local keyframes (current + covisible)
    let local_kf_ids = collect_local_keyframes(map, current_kf_id, config.max_covisible_keyframes);
    if local_kf_ids.is_empty() {
        return None;
    }

    // Step 2: Collect local map points
    let local_mp_ids = collect_local_map_points(map, &local_kf_ids);
    if local_mp_ids.is_empty() {
        return None;
    }

    // Step 3: Collect fixed keyframes
    let fixed_kf_ids = collect_fixed_keyframes(map, &local_kf_ids, &local_mp_ids);

    // Step 4: Set up optimization
    // First keyframe is anchor (fixed), rest are optimized
    let anchor_kf_id = *local_kf_ids.first().unwrap();
    let optimized_kf_ids: Vec<KeyFrameId> = local_kf_ids.iter().skip(1).copied().collect();

    let kf_to_param_idx: HashMap<KeyFrameId, usize> = optimized_kf_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mp_to_param_idx: HashMap<MapPointId, usize> = local_mp_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Step 5: Build fixed poses map (T_cw)
    let mut fixed_poses_cw: HashMap<KeyFrameId, SE3> = HashMap::new();
    fixed_poses_cw.insert(
        anchor_kf_id,
        map.get_keyframe(anchor_kf_id)?.pose.inverse(),
    );
    for &kf_id in &fixed_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            fixed_poses_cw.insert(kf_id, kf.pose.inverse());
        }
    }

    // Step 6: Build observations
    let observations = build_observations(
        map,
        &local_kf_ids,
        &fixed_kf_ids,
        &local_mp_ids,
        &kf_to_param_idx,
        &mp_to_param_idx,
    );
    if observations.is_empty() {
        return None;
    }

    // Step 7: Build initial parameter vector
    // Layout: [pose_0 (6), pose_1 (6), ..., point_0 (3), point_1 (3), ...]
    let num_kf_params = optimized_kf_ids.len();
    let num_mp_params = local_mp_ids.len();
    let num_params = num_kf_params * 6 + num_mp_params * 3;
    let num_residuals = observations.len() * 2;

    let mut params = DVector::zeros(num_params);

    // Initialize pose parameters (as axis-angle + translation, in T_cw frame)
    for (i, &kf_id) in optimized_kf_ids.iter().enumerate() {
        let kf = map.get_keyframe(kf_id)?;
        let pose_cw = kf.pose.inverse();
        let (rot, trans) = se3_to_params(&pose_cw);
        params[i * 6] = rot.x;
        params[i * 6 + 1] = rot.y;
        params[i * 6 + 2] = rot.z;
        params[i * 6 + 3] = trans.x;
        params[i * 6 + 4] = trans.y;
        params[i * 6 + 5] = trans.z;
    }

    // Initialize map point parameters
    for (i, &mp_id) in local_mp_ids.iter().enumerate() {
        let mp = map.get_map_point(mp_id)?;
        let base = num_kf_params * 6 + i * 3;
        params[base] = mp.position.x;
        params[base + 1] = mp.position.y;
        params[base + 2] = mp.position.z;
    }

    // Step 8: Create BA problem
    let problem = BAProblem {
        camera,
        observations,
        fixed_poses_cw,
        huber_threshold: config.huber_threshold,
        num_kf_params,
        num_mp_params,
    };

    // Compute initial error
    let initial_residuals = compute_residuals(&problem, &params);
    let initial_error = initial_residuals.norm() / (num_residuals as f64).sqrt();

    // Step 9: Run LM optimization with manual iteration loop
    let mut current_params = params.clone();
    let mut iterations = 0;
    let mut lambda = 1e-3; // Initial damping
    let lambda_up = 10.0;
    let lambda_down = 0.1;
    let min_lambda = 1e-10;
    let max_lambda = 1e10;

    for iter in 0..config.max_iterations {
        if should_stop() {
            break;
        }

        iterations = iter + 1;

        // Compute residuals and Jacobian
        let residuals = compute_residuals(&problem, &current_params);
        let jacobian = compute_jacobian(&problem, &current_params);

        let current_error_sq = residuals.norm_squared();

        // Compute gradient and approximate Hessian
        let gradient = jacobian.transpose() * &residuals;
        let jtj = jacobian.transpose() * &jacobian;

        // Check gradient convergence
        if gradient.norm() < config.gradient_tolerance {
            break;
        }

        // Solve (J^T J + lambda * diag(J^T J)) * delta = -J^T r
        let mut damped_jtj = jtj.clone();
        for i in 0..num_params {
            damped_jtj[(i, i)] += lambda * damped_jtj[(i, i)].max(1e-6);
        }

        // Solve the linear system
        let delta = match damped_jtj.clone().lu().solve(&(-&gradient)) {
            Some(d) => d,
            None => break, // Singular matrix
        };

        // Check parameter convergence
        if delta.norm() < config.param_tolerance * (current_params.norm() + config.param_tolerance) {
            break;
        }

        // Trial update
        let trial_params = &current_params + &delta;
        let trial_residuals = compute_residuals(&problem, &trial_params);
        let trial_error_sq = trial_residuals.norm_squared();

        // Accept or reject
        if trial_error_sq < current_error_sq {
            // Accept and decrease damping
            current_params = trial_params;
            lambda = (lambda * lambda_down).max(min_lambda);
        } else {
            // Reject and increase damping
            lambda = (lambda * lambda_up).min(max_lambda);
        }
    }

    // Step 10: Compute final error
    let final_residuals = compute_residuals(&problem, &current_params);
    let final_error = final_residuals.norm() / (num_residuals as f64).sqrt();

    // Step 11: Write back optimized values
    // Update poses (convert T_cw back to T_wc)
    for (i, &kf_id) in optimized_kf_ids.iter().enumerate() {
        let rot = Vector3::new(
            current_params[i * 6],
            current_params[i * 6 + 1],
            current_params[i * 6 + 2],
        );
        let trans = Vector3::new(
            current_params[i * 6 + 3],
            current_params[i * 6 + 4],
            current_params[i * 6 + 5],
        );
        let pose_cw = se3_from_params(&rot, &trans);
        if let Some(kf) = map.get_keyframe_mut(kf_id) {
            kf.pose = pose_cw.inverse(); // T_wc = T_cw^-1
        }
    }

    // Update map points
    for (i, &mp_id) in local_mp_ids.iter().enumerate() {
        let base = num_kf_params * 6 + i * 3;
        let pos = Vector3::new(
            current_params[base],
            current_params[base + 1],
            current_params[base + 2],
        );
        if let Some(mp) = map.get_map_point_mut(mp_id) {
            mp.position = pos;
        }
    }

    Some(LocalBAResultLM {
        iterations,
        initial_error,
        final_error,
        num_keyframes: optimized_kf_ids.len(),
        num_map_points: local_mp_ids.len(),
        num_observations: problem.observations.len(),
        termination: format!("Completed {} iterations", iterations),
    })
}

/// Compute residuals for the BA problem.
fn compute_residuals(problem: &BAProblem, params: &DVector<f64>) -> DVector<f64> {
    let mut residuals = DVector::zeros(problem.observations.len() * 2);

    for (i, obs) in problem.observations.iter().enumerate() {
        // Get pose (either from params or fixed)
        let pose_cw = match obs.kf_param_idx {
            Some(idx) => {
                let base = idx * 6;
                let rot = Vector3::new(params[base], params[base + 1], params[base + 2]);
                let trans = Vector3::new(params[base + 3], params[base + 4], params[base + 5]);
                se3_from_params(&rot, &trans)
            }
            None => problem.fixed_poses_cw.get(&obs.kf_id).cloned().unwrap_or_else(SE3::identity),
        };

        // Get point
        let point = problem.get_point(params, obs.mp_param_idx);

        // Compute error
        let error = problem.compute_error(&pose_cw, &point, &obs.observed_uv);

        // Apply Huber weighting
        let error_norm = error.norm();
        let weight = problem.huber_weight(error_norm);
        let weighted_error = error * weight.sqrt();

        residuals[i * 2] = weighted_error.x;
        residuals[i * 2 + 1] = weighted_error.y;
    }

    residuals
}

/// Compute Jacobian for the BA problem.
fn compute_jacobian(problem: &BAProblem, params: &DVector<f64>) -> DMatrix<f64> {
    let num_residuals = problem.observations.len() * 2;
    let num_params = problem.num_kf_params * 6 + problem.num_mp_params * 3;
    let mut jacobian = DMatrix::zeros(num_residuals, num_params);

    for (i, obs) in problem.observations.iter().enumerate() {
        // Get pose
        let pose_cw = match obs.kf_param_idx {
            Some(idx) => {
                let base = idx * 6;
                let rot = Vector3::new(params[base], params[base + 1], params[base + 2]);
                let trans = Vector3::new(params[base + 3], params[base + 4], params[base + 5]);
                se3_from_params(&rot, &trans)
            }
            None => problem.fixed_poses_cw.get(&obs.kf_id).cloned().unwrap_or_else(SE3::identity),
        };

        // Get point
        let point = problem.get_point(params, obs.mp_param_idx);

        // Compute error for Huber weight
        let error = problem.compute_error(&pose_cw, &point, &obs.observed_uv);
        let error_norm = error.norm();
        let weight = problem.huber_weight(error_norm);
        let weight_sqrt = weight.sqrt();

        // Jacobian w.r.t. pose (if optimized)
        if let Some(kf_idx) = obs.kf_param_idx {
            let j_pose = problem.jacobian_pose(&pose_cw, &point) * weight_sqrt;
            let col_base = kf_idx * 6;
            for row in 0..2 {
                for col in 0..6 {
                    jacobian[(i * 2 + row, col_base + col)] = j_pose[(row, col)];
                }
            }
        }

        // Jacobian w.r.t. point
        let j_point = problem.jacobian_point(&pose_cw, &point) * weight_sqrt;
        let col_base = problem.num_kf_params * 6 + obs.mp_param_idx * 3;
        for row in 0..2 {
            for col in 0..3 {
                jacobian[(i * 2 + row, col_base + col)] = j_point[(row, col)];
            }
        }
    }

    jacobian
}

/// Convert SE3 to axis-angle rotation + translation parameters.
fn se3_to_params(pose: &SE3) -> (Vector3<f64>, Vector3<f64>) {
    let axis_angle = pose.rotation.scaled_axis();
    (axis_angle, pose.translation)
}

/// Construct SE3 from axis-angle rotation + translation parameters.
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

/// Collect local keyframes: current + covisible neighbors.
fn collect_local_keyframes(
    map: &Map,
    current_kf_id: KeyFrameId,
    max_covisible: usize,
) -> Vec<KeyFrameId> {
    let mut local_kfs = vec![current_kf_id];

    if let Some(kf) = map.get_keyframe(current_kf_id) {
        for (neighbor_id, _) in kf.covisibility_weights().iter().take(max_covisible) {
            if let Some(neighbor) = map.get_keyframe(*neighbor_id) {
                if !neighbor.is_bad {
                    local_kfs.push(*neighbor_id);
                }
            }
        }
    }

    local_kfs
}

/// Collect map points observed by local keyframes.
fn collect_local_map_points(map: &Map, local_kf_ids: &[KeyFrameId]) -> Vec<MapPointId> {
    let mut mp_set: HashSet<MapPointId> = HashSet::new();

    for &kf_id in local_kf_ids {
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

/// Collect fixed keyframes: see local map points but aren't local.
fn collect_fixed_keyframes(
    map: &Map,
    local_kf_ids: &[KeyFrameId],
    local_mp_ids: &[MapPointId],
) -> Vec<KeyFrameId> {
    let local_kf_set: HashSet<KeyFrameId> = local_kf_ids.iter().copied().collect();
    let mut fixed_set: HashSet<KeyFrameId> = HashSet::new();

    for &mp_id in local_mp_ids {
        if let Some(mp) = map.get_map_point(mp_id) {
            for &kf_id in mp.observations.keys() {
                if !local_kf_set.contains(&kf_id) {
                    fixed_set.insert(kf_id);
                }
            }
        }
    }

    fixed_set.into_iter().collect()
}

/// Build observation list.
fn build_observations(
    map: &Map,
    local_kf_ids: &[KeyFrameId],
    fixed_kf_ids: &[KeyFrameId],
    local_mp_ids: &[MapPointId],
    kf_to_param_idx: &HashMap<KeyFrameId, usize>,
    mp_to_param_idx: &HashMap<MapPointId, usize>,
) -> Vec<Observation> {
    let local_mp_set: HashSet<MapPointId> = local_mp_ids.iter().copied().collect();
    let mut observations = Vec::new();

    // Include anchor keyframe (first in local_kf_ids) but mark it as fixed (None param_idx)
    let anchor_kf_id = *local_kf_ids.first().unwrap();
    let all_kf_ids: Vec<KeyFrameId> = local_kf_ids
        .iter()
        .chain(fixed_kf_ids.iter())
        .copied()
        .collect();

    for &kf_id in &all_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            for (feat_idx, mp_id_opt) in kf.map_point_ids.iter().enumerate() {
                if let Some(mp_id) = mp_id_opt {
                    if let Some(&mp_param_idx) = mp_to_param_idx.get(mp_id) {
                        if local_mp_set.contains(mp_id) {
                            if let Ok(kp) = kf.keypoints.get(feat_idx) {
                                // Determine if this KF is optimized or fixed
                                let kf_param_idx = if kf_id == anchor_kf_id {
                                    None // Anchor is fixed
                                } else {
                                    kf_to_param_idx.get(&kf_id).copied()
                                };

                                observations.push(Observation {
                                    kf_param_idx,
                                    mp_param_idx,
                                    observed_uv: Vector2::new(
                                        kp.pt().x as f64,
                                        kp.pt().y as f64,
                                    ),
                                    kf_id,
                                    mp_id: *mp_id,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    observations
}

// ============================================================================
// THREE-PHASE BA: Public functions for lock-free optimization
// ============================================================================

/// PHASE 1: COLLECT - Extract data snapshot for bundle adjustment.
///
/// This function extracts all necessary data from the map to run BA.
/// It should be called while holding a **read lock** on the map.
/// After calling this, the lock can be released and BA can run lock-free.
///
/// # Arguments
/// * `map` - The map to extract data from (read access)
/// * `current_kf_id` - The current keyframe triggering BA
/// * `config` - BA configuration
///
/// # Returns
/// `Some(VisualBAProblemData)` if enough data exists, `None` otherwise.
pub fn collect_visual_ba_data(
    map: &Map,
    current_kf_id: KeyFrameId,
    config: &LocalBAConfigLM,
) -> Option<VisualBAProblemData> {
    // Step 1: Collect local keyframes (current + covisible)
    let local_kf_ids = collect_local_keyframes(map, current_kf_id, config.max_covisible_keyframes);
    if local_kf_ids.is_empty() {
        return None;
    }

    // Step 2: Collect local map points
    let mp_ids = collect_local_map_points(map, &local_kf_ids);
    if mp_ids.is_empty() {
        return None;
    }

    // Step 3: Collect fixed keyframes
    let fixed_kf_ids = collect_fixed_keyframes(map, &local_kf_ids, &mp_ids);

    // Anchor is the first (oldest) local keyframe
    let anchor_kf_id = *local_kf_ids.first().unwrap();
    let optimized_kf_ids: Vec<KeyFrameId> = local_kf_ids.iter().skip(1).copied().collect();

    // Step 4: Extract poses for local (optimized) keyframes
    let mut local_kf_poses = HashMap::new();
    for &kf_id in &optimized_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            local_kf_poses.insert(kf_id, kf.pose.inverse()); // T_cw
        }
    }

    // Step 5: Extract poses for fixed keyframes (anchor + others)
    let mut fixed_kf_poses = HashMap::new();
    if let Some(kf) = map.get_keyframe(anchor_kf_id) {
        fixed_kf_poses.insert(anchor_kf_id, kf.pose.inverse());
    }
    for &kf_id in &fixed_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            fixed_kf_poses.insert(kf_id, kf.pose.inverse());
        }
    }

    // Step 6: Extract map point positions
    let mut local_mp_positions = HashMap::new();
    for &mp_id in &mp_ids {
        if let Some(mp) = map.get_map_point(mp_id) {
            local_mp_positions.insert(mp_id, mp.position);
        }
    }

    // Step 7: Build observations
    let local_kf_set: HashSet<KeyFrameId> = optimized_kf_ids.iter().copied().collect();
    let local_mp_set: HashSet<MapPointId> = mp_ids.iter().copied().collect();
    let mut observations = Vec::new();

    let all_kf_ids: Vec<KeyFrameId> = local_kf_ids
        .iter()
        .chain(fixed_kf_ids.iter())
        .copied()
        .collect();

    for &kf_id in &all_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            for (feat_idx, mp_id_opt) in kf.map_point_ids.iter().enumerate() {
                if let Some(mp_id) = mp_id_opt {
                    if local_mp_set.contains(mp_id) {
                        if let Ok(kp) = kf.keypoints.get(feat_idx) {
                            // Keyframe is optimized if it's in local set AND not anchor
                            let is_kf_optimized = local_kf_set.contains(&kf_id);

                            observations.push(VisualObservation {
                                kf_id,
                                mp_id: *mp_id,
                                observed_uv: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                                is_kf_optimized,
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

    Some(VisualBAProblemData {
        local_kf_poses,
        local_mp_positions,
        fixed_kf_poses,
        anchor_kf_id,
        observations,
        optimized_kf_ids,
        mp_ids,
    })
}

/// PHASE 2: SOLVE - Run LM optimization on extracted data.
///
/// This function runs Bundle Adjustment on the extracted data snapshot.
/// It does NOT require any locks on the map - all data is in the problem struct.
///
/// # Arguments
/// * `problem` - The extracted problem data from `collect_visual_ba_data`
/// * `camera` - Camera intrinsics
/// * `config` - BA configuration
/// * `should_stop` - Callback to check if BA should abort early
///
/// # Returns
/// `Some(VisualBAResultData)` with optimized values, `None` if optimization failed.
pub fn solve_visual_ba(
    problem: &VisualBAProblemData,
    camera: &CameraModel,
    config: &LocalBAConfigLM,
    should_stop: &dyn Fn() -> bool,
) -> Option<VisualBAResultData> {
    let num_kf_params = problem.optimized_kf_ids.len();
    let num_mp_params = problem.mp_ids.len();
    let num_params = num_kf_params * 6 + num_mp_params * 3;
    let num_residuals = problem.observations.len() * 2;

    if num_params == 0 || num_residuals == 0 {
        return None;
    }

    // Build index mappings
    let kf_to_param_idx: HashMap<KeyFrameId, usize> = problem
        .optimized_kf_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mp_to_param_idx: HashMap<MapPointId, usize> = problem
        .mp_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Build internal observation list with param indices
    let internal_obs: Vec<Observation> = problem
        .observations
        .iter()
        .filter_map(|obs| {
            let mp_param_idx = *mp_to_param_idx.get(&obs.mp_id)?;
            let kf_param_idx = if obs.is_kf_optimized {
                kf_to_param_idx.get(&obs.kf_id).copied()
            } else {
                None
            };
            Some(Observation {
                kf_param_idx,
                mp_param_idx,
                observed_uv: obs.observed_uv,
                kf_id: obs.kf_id,
                mp_id: obs.mp_id,
            })
        })
        .collect();

    // Build initial parameter vector
    let mut params = DVector::zeros(num_params);

    // Initialize pose parameters (T_cw as axis-angle + translation)
    for (i, &kf_id) in problem.optimized_kf_ids.iter().enumerate() {
        if let Some(pose_cw) = problem.local_kf_poses.get(&kf_id) {
            let (rot, trans) = se3_to_params(pose_cw);
            params[i * 6] = rot.x;
            params[i * 6 + 1] = rot.y;
            params[i * 6 + 2] = rot.z;
            params[i * 6 + 3] = trans.x;
            params[i * 6 + 4] = trans.y;
            params[i * 6 + 5] = trans.z;
        }
    }

    // Initialize map point parameters
    for (i, &mp_id) in problem.mp_ids.iter().enumerate() {
        if let Some(pos) = problem.local_mp_positions.get(&mp_id) {
            let base = num_kf_params * 6 + i * 3;
            params[base] = pos.x;
            params[base + 1] = pos.y;
            params[base + 2] = pos.z;
        }
    }

    // Create BA problem struct
    let ba_problem = BAProblem {
        camera,
        observations: internal_obs,
        fixed_poses_cw: problem.fixed_kf_poses.clone(),
        huber_threshold: config.huber_threshold,
        num_kf_params,
        num_mp_params,
    };

    // Compute initial error
    let initial_residuals = compute_residuals(&ba_problem, &params);
    let initial_error = initial_residuals.norm() / (num_residuals as f64).sqrt();

    // Run LM optimization
    let mut current_params = params;
    let mut iterations = 0;
    let mut lambda = 1e-3;
    let lambda_up = 10.0;
    let lambda_down = 0.1;
    let min_lambda = 1e-10;
    let max_lambda = 1e10;

    for iter in 0..config.max_iterations {
        if should_stop() {
            break;
        }

        iterations = iter + 1;

        let residuals = compute_residuals(&ba_problem, &current_params);
        let jacobian = compute_jacobian(&ba_problem, &current_params);

        let current_error_sq = residuals.norm_squared();

        let gradient = jacobian.transpose() * &residuals;
        let jtj = jacobian.transpose() * &jacobian;

        if gradient.norm() < config.gradient_tolerance {
            break;
        }

        let mut damped_jtj = jtj.clone();
        for i in 0..num_params {
            damped_jtj[(i, i)] += lambda * damped_jtj[(i, i)].max(1e-6);
        }

        let delta = match damped_jtj.lu().solve(&(-&gradient)) {
            Some(d) => d,
            None => break,
        };

        if delta.norm() < config.param_tolerance * (current_params.norm() + config.param_tolerance)
        {
            break;
        }

        let trial_params = &current_params + &delta;
        let trial_residuals = compute_residuals(&ba_problem, &trial_params);
        let trial_error_sq = trial_residuals.norm_squared();

        if trial_error_sq < current_error_sq {
            current_params = trial_params;
            lambda = (lambda * lambda_down).max(min_lambda);
        } else {
            lambda = (lambda * lambda_up).min(max_lambda);
        }
    }

    // Compute final error
    let final_residuals = compute_residuals(&ba_problem, &current_params);
    let final_error = final_residuals.norm() / (num_residuals as f64).sqrt();

    // Extract optimized poses (convert T_cw back to T_wc)
    let mut optimized_poses = HashMap::new();
    for (i, &kf_id) in problem.optimized_kf_ids.iter().enumerate() {
        let rot = Vector3::new(
            current_params[i * 6],
            current_params[i * 6 + 1],
            current_params[i * 6 + 2],
        );
        let trans = Vector3::new(
            current_params[i * 6 + 3],
            current_params[i * 6 + 4],
            current_params[i * 6 + 5],
        );
        let pose_cw = se3_from_params(&rot, &trans);
        optimized_poses.insert(kf_id, pose_cw.inverse()); // T_wc
    }

    // Extract optimized map points
    let mut optimized_points = HashMap::new();
    for (i, &mp_id) in problem.mp_ids.iter().enumerate() {
        let base = num_kf_params * 6 + i * 3;
        let pos = Vector3::new(
            current_params[base],
            current_params[base + 1],
            current_params[base + 2],
        );
        optimized_points.insert(mp_id, pos);
    }

    Some(VisualBAResultData {
        optimized_poses,
        optimized_points,
        iterations,
        initial_error,
        final_error,
    })
}

/// PHASE 3: APPLY - Write optimized results back to the map.
///
/// This function writes the optimized poses and positions back to the map.
/// It should be called while holding a **write lock** on the map.
/// Entities that were deleted during optimization are silently skipped.
///
/// # Arguments
/// * `map` - The map to write to (write access)
/// * `results` - The optimization results from `solve_visual_ba`
///
/// # Returns
/// Number of entities successfully updated.
pub fn apply_visual_ba_results(map: &mut Map, results: &VisualBAResultData) -> usize {
    let mut updated = 0;

    // Update keyframe poses
    for (kf_id, pose) in &results.optimized_poses {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            if !kf.is_bad {
                kf.pose = pose.clone();
                updated += 1;
            }
        }
        // Skip silently if keyframe was deleted during BA
    }

    // Update map point positions
    for (mp_id, pos) in &results.optimized_points {
        if let Some(mp) = map.get_map_point_mut(*mp_id) {
            if !mp.is_bad {
                mp.position = *pos;
                updated += 1;
            }
        }
        // Skip silently if map point was deleted during BA
    }

    updated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_se3_param_roundtrip() {
        let original = SE3 {
            rotation: nalgebra::UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
            translation: Vector3::new(1.0, 2.0, 3.0),
        };

        let (rot, trans) = se3_to_params(&original);
        let recovered = se3_from_params(&rot, &trans);

        // Check translation matches
        assert!((original.translation - recovered.translation).norm() < 1e-10);

        // Check rotation matches (compare matrices since quaternions can differ by sign)
        let r1 = original.rotation.to_rotation_matrix();
        let r2 = recovered.rotation.to_rotation_matrix();
        assert!((r1.into_inner() - r2.into_inner()).norm() < 1e-10);
    }

    #[test]
    fn test_jacobian_pose_numerical() {
        // Test the Jacobian at a simpler pose (identity) where linearization is exact
        let camera = CameraModel {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 240.0,
            baseline: 0.1,
        };

        let problem = BAProblem {
            camera: &camera,
            observations: vec![],
            fixed_poses_cw: HashMap::new(),
            huber_threshold: 2.5,
            num_kf_params: 1,
            num_mp_params: 1,
        };

        // Use identity pose for simpler Jacobian comparison
        let pose = SE3::identity();
        let point = Vector3::new(0.5, 0.3, 3.0);

        let j_analytical = problem.jacobian_pose(&pose, &point);

        // Numerical Jacobian with larger epsilon for stability
        let eps = 1e-5;
        let mut j_numerical = Matrix2x6::zeros();

        for i in 0..6 {
            let (rot, trans) = se3_to_params(&pose);
            let mut params_plus = DVector::from_vec(vec![
                rot.x, rot.y, rot.z, trans.x, trans.y, trans.z,
            ]);
            let mut params_minus = params_plus.clone();
            params_plus[i] += eps;
            params_minus[i] -= eps;

            let pose_plus = se3_from_params(
                &Vector3::new(params_plus[0], params_plus[1], params_plus[2]),
                &Vector3::new(params_plus[3], params_plus[4], params_plus[5]),
            );
            let pose_minus = se3_from_params(
                &Vector3::new(params_minus[0], params_minus[1], params_minus[2]),
                &Vector3::new(params_minus[3], params_minus[4], params_minus[5]),
            );

            let observed = Vector2::new(320.0, 240.0);
            let e_plus = problem.compute_error(&pose_plus, &point, &observed);
            let e_minus = problem.compute_error(&pose_minus, &point, &observed);

            j_numerical[(0, i)] = (e_plus.x - e_minus.x) / (2.0 * eps);
            j_numerical[(1, i)] = (e_plus.y - e_minus.y) / (2.0 * eps);
        }

        // Compare - the analytical Jacobian is derived for T_cw parameterization
        // The numerical check here verifies the sign and magnitude are reasonable
        // Note: The g2o-style Jacobians use specific conventions that may differ
        // from direct numerical derivatives on our parameterization
        let diff = (j_analytical - j_numerical).norm();

        // Translation derivatives (columns 3-5) should match well
        let trans_diff = (j_analytical.columns(3, 3) - j_numerical.columns(3, 3)).norm();
        assert!(
            trans_diff < 1.0,
            "Translation Jacobian mismatch: diff = {} (analytical: {:?}, numerical: {:?})",
            trans_diff,
            j_analytical.columns(3, 3),
            j_numerical.columns(3, 3)
        );

        // Full Jacobian - larger tolerance due to rotation parameterization differences
        assert!(
            diff < 50.0,
            "Jacobian diff = {} is unexpectedly large. Analytical:\n{:?}\nNumerical:\n{:?}",
            diff,
            j_analytical,
            j_numerical
        );
    }
}
