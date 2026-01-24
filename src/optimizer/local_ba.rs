//! Local Bundle Adjustment optimizer.
//!
//! Optimizes keyframe poses and map point positions to minimize reprojection error.
//! Uses Gauss-Newton with Schur complement for efficient solving.
//!
//! # TODO: BA currently DISABLED (see Local Mapper – Step 4)
//!
//! Known issues:
//! - Gradient sign: g2o uses `b = -J^T * Omega * error`; we tried both signs.
//! - Both `b = -J^T * error` and `b = +J^T * error` cause divergence.
//! - Step control rejects all updates (error increases on first iteration).
//!
//! Recommended fixes:
//! - Use an existing Rust optimizer (e.g. `argmin`, `levenberg-marquardt`), or
//! - Port g2o's implementation more directly and verify Jacobians numerically.

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix2x6, Matrix3, Vector2, Vector3};
use opencv::prelude::*;

use crate::atlas::map::{KeyFrameId, Map, MapPointId};
use crate::geometry::SE3;
use crate::tracking::frame::CameraModel;

/// Result of local bundle adjustment.
#[derive(Debug)]
pub struct LocalBAResult {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Initial total reprojection error.
    pub initial_error: f64,
    /// Final total reprojection error.
    pub final_error: f64,
    /// Number of keyframes optimized.
    pub num_keyframes: usize,
    /// Number of map points optimized.
    pub num_map_points: usize,
    /// Number of observations (edges).
    pub num_observations: usize,
}

/// Configuration for local BA.
pub struct LocalBAConfig {
    /// Maximum number of Gauss-Newton iterations.
    pub max_iterations: usize,
    /// Convergence threshold (relative change in error).
    pub convergence_threshold: f64,
    /// Huber kernel threshold for robust estimation (pixels).
    pub huber_threshold: f64,
    /// Chi-square threshold for outlier rejection (pixels squared).
    pub chi2_threshold: f64,
}

impl Default for LocalBAConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            convergence_threshold: 1e-4,
            huber_threshold: 5.991, // 95% chi2 with 2 DOF
            chi2_threshold: 5.991,
        }
    }
}

/// An observation: a map point seen from a keyframe.
struct Observation {
    kf_id: KeyFrameId,
    mp_id: MapPointId,
    observed_uv: Vector2<f64>,
    feature_idx: usize,
}

/// Run local bundle adjustment.
///
/// Optimizes:
/// - Poses of local keyframes (except the first/anchor keyframe)
/// - Positions of map points observed by local keyframes
///
/// Fixed:
/// - Poses of "fixed" keyframes (those that observe local map points but aren't in local set)
pub fn local_bundle_adjustment(
    map: &mut Map,
    current_kf_id: KeyFrameId,
    camera: &CameraModel,
    config: &LocalBAConfig,
    should_stop: &dyn Fn() -> bool,
) -> Option<LocalBAResult> {
    // Step 1: Collect local keyframes (current + covisible)
    let local_kf_ids = collect_local_keyframes(map, current_kf_id);
    if local_kf_ids.is_empty() {
        return None;
    }

    // Step 2: Collect local map points (seen by local keyframes)
    let local_mp_ids = collect_local_map_points(map, &local_kf_ids);
    if local_mp_ids.is_empty() {
        return None;
    }

    // Step 3: Collect fixed keyframes (see local map points but aren't local)
    let fixed_kf_ids = collect_fixed_keyframes(map, &local_kf_ids, &local_mp_ids);

    // Step 4: Build observations
    let observations = build_observations(map, &local_kf_ids, &fixed_kf_ids, &local_mp_ids);
    if observations.is_empty() {
        return None;
    }

    // Step 5: Create index mappings for optimization variables
    // First keyframe is fixed (anchor), rest are optimized
    let anchor_kf_id = *local_kf_ids.first().unwrap();
    let optimized_kf_ids: Vec<KeyFrameId> = local_kf_ids
        .iter()
        .skip(1) // Skip anchor
        .copied()
        .collect();

    let kf_to_idx: HashMap<KeyFrameId, usize> = optimized_kf_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let mp_to_idx: HashMap<MapPointId, usize> = local_mp_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Step 6: Extract initial values
    // IMPORTANT: We store poses as T_wc (camera-to-world) in keyframes,
    // but g2o's Jacobians are for T_cw (world-to-camera).
    // So we work with T_cw during optimization and convert back at the end.
    let mut kf_poses_cw: Vec<SE3> = optimized_kf_ids
        .iter()
        .map(|&id| map.get_keyframe(id).unwrap().pose.inverse())
        .collect();

    let mut mp_positions: Vec<Vector3<f64>> = local_mp_ids
        .iter()
        .map(|&id| map.get_map_point(id).unwrap().position)
        .collect();

    // Get fixed keyframe poses (including anchor) - also as T_cw
    let mut fixed_poses_cw: HashMap<KeyFrameId, SE3> = HashMap::new();
    fixed_poses_cw.insert(
        anchor_kf_id,
        map.get_keyframe(anchor_kf_id).unwrap().pose.inverse(),
    );
    for &kf_id in &fixed_kf_ids {
        if let Some(kf) = map.get_keyframe(kf_id) {
            fixed_poses_cw.insert(kf_id, kf.pose.inverse());
        }
    }

    // Step 7: Gauss-Newton iterations
    // Note: All poses are now in T_cw (world-to-camera) frame
    let initial_error = compute_total_error_cw(
        &observations,
        &kf_poses_cw,
        &mp_positions,
        &kf_to_idx,
        &mp_to_idx,
        &fixed_poses_cw,
        camera,
        config.huber_threshold,
    );

    let mut current_error = initial_error;
    let mut iterations = 0;

    let num_kf_params = kf_poses_cw.len() * 6; // 6 DOF per pose
    let num_mp_params = mp_positions.len() * 3; // 3 DOF per point

    for iter in 0..config.max_iterations {
        if should_stop() {
            break;
        }

        iterations = iter + 1;

        // Build normal equations using Schur complement
        let (h_pp, b_p, h_ll, b_l, h_pl) = build_normal_equations_cw(
            &observations,
            &kf_poses_cw,
            &mp_positions,
            &kf_to_idx,
            &mp_to_idx,
            &fixed_poses_cw,
            camera,
            config.huber_threshold,
            num_kf_params,
            num_mp_params,
        );

        // Solve using Schur complement:
        // H_pp - H_pl * H_ll^-1 * H_pl^T) * delta_p = b_p - H_pl * H_ll^-1 * b_l
        let delta = solve_schur_complement(&h_pp, &b_p, &h_ll, &b_l, &h_pl, num_kf_params);

        if delta.is_none() {
            break;
        }
        let delta = delta.unwrap();

        // Save current state for potential rollback
        let kf_poses_cw_backup = kf_poses_cw.clone();
        let mp_positions_backup = mp_positions.clone();

        // Update poses (T_cw)
        // With g2o ordering: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        // g2o applies updates via left multiplication: T_new = exp(delta) * T_old
        // NOTE: Negating delta to test if direction is inverted
        for (i, pose_cw) in kf_poses_cw.iter_mut().enumerate() {
            let delta_rot = Vector3::new(-delta[i * 6], -delta[i * 6 + 1], -delta[i * 6 + 2]);
            let delta_trans = Vector3::new(-delta[i * 6 + 3], -delta[i * 6 + 4], -delta[i * 6 + 5]);

            // Apply small rotation and translation updates (left multiplication)
            *pose_cw = apply_pose_update_left(pose_cw, &delta_rot, &delta_trans);
        }

        // Update map points (back-substitution)
        // delta_l = H_ll^-1 * (b_l - H_pl^T * delta_p)
        for (i, pos) in mp_positions.iter_mut().enumerate() {
            let h_ll_block = h_ll.fixed_view::<3, 3>(i * 3, i * 3);
            let b_l_block = b_l.fixed_rows::<3>(i * 3);

            // Compute H_pl^T * delta_p contribution
            let mut h_pl_t_delta_p = Vector3::zeros();
            for j in 0..num_kf_params {
                for k in 0..3 {
                    h_pl_t_delta_p[k] += h_pl[(j, i * 3 + k)] * delta[j];
                }
            }

            let rhs = b_l_block - h_pl_t_delta_p;

            // Solve 3x3 system
            if let Some(h_ll_inv) = h_ll_block.try_inverse() {
                let delta_mp = h_ll_inv * rhs;
                *pos += delta_mp;
            }
        }

        // Compute new error
        let new_error = compute_total_error_cw(
            &observations,
            &kf_poses_cw,
            &mp_positions,
            &kf_to_idx,
            &mp_to_idx,
            &fixed_poses_cw,
            camera,
            config.huber_threshold,
        );

        // Step control: reject update if error increased
        if new_error > current_error * 1.001 {
            // Rollback and stop - the algorithm is diverging
            kf_poses_cw.clone_from_slice(&kf_poses_cw_backup);
            mp_positions.clone_from_slice(&mp_positions_backup);
            break;
        }

        // Check convergence
        let relative_change = (current_error - new_error).abs() / current_error.max(1e-10);
        current_error = new_error;

        if relative_change < config.convergence_threshold {
            break;
        }
    }

    // Step 8: Write back optimized values to map
    // Convert T_cw back to T_wc for storage in keyframes
    for (i, &kf_id) in optimized_kf_ids.iter().enumerate() {
        if let Some(kf) = map.get_keyframe_mut(kf_id) {
            kf.pose = kf_poses_cw[i].inverse(); // T_wc = T_cw^-1
        }
    }

    for (i, &mp_id) in local_mp_ids.iter().enumerate() {
        if let Some(mp) = map.get_map_point_mut(mp_id) {
            mp.position = mp_positions[i];
        }
    }

    Some(LocalBAResult {
        iterations,
        initial_error,
        final_error: current_error,
        num_keyframes: local_kf_ids.len(),
        num_map_points: local_mp_ids.len(),
        num_observations: observations.len(),
    })
}

/// Collect local keyframes: current keyframe + covisible neighbors.
fn collect_local_keyframes(map: &Map, current_kf_id: KeyFrameId) -> Vec<KeyFrameId> {
    let mut local_kfs = vec![current_kf_id];

    if let Some(kf) = map.get_keyframe(current_kf_id) {
        // Get covisible keyframes (up to 20)
        for (neighbor_id, _weight) in kf.covisibility_weights().iter().take(20) {
            if let Some(neighbor) = map.get_keyframe(*neighbor_id) {
                if !neighbor.is_bad {
                    local_kfs.push(*neighbor_id);
                }
            }
        }
    }

    local_kfs
}

/// Collect map points seen by local keyframes.
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

/// Collect fixed keyframes: those that see local map points but aren't local.
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
) -> Vec<Observation> {
    let local_mp_set: HashSet<MapPointId> = local_mp_ids.iter().copied().collect();
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
                        // Get the 2D observation from keypoint
                        if let Ok(kp) = kf.keypoints.get(feat_idx) {
                            observations.push(Observation {
                                kf_id,
                                mp_id: *mp_id,
                                observed_uv: Vector2::new(kp.pt().x as f64, kp.pt().y as f64),
                                feature_idx: feat_idx,
                            });
                        }
                    }
                }
            }
        }
    }

    observations
}

/// Compute total reprojection error.
/// All poses are T_cw (world-to-camera).
fn compute_total_error_cw(
    observations: &[Observation],
    kf_poses_cw: &[SE3],
    mp_positions: &[Vector3<f64>],
    kf_to_idx: &HashMap<KeyFrameId, usize>,
    mp_to_idx: &HashMap<MapPointId, usize>,
    fixed_poses_cw: &HashMap<KeyFrameId, SE3>,
    camera: &CameraModel,
    huber_threshold: f64,
) -> f64 {
    let mut total_error = 0.0;

    for obs in observations {
        let mp_idx = match mp_to_idx.get(&obs.mp_id) {
            Some(&idx) => idx,
            None => continue,
        };

        let pose_cw = if let Some(&kf_idx) = kf_to_idx.get(&obs.kf_id) {
            &kf_poses_cw[kf_idx]
        } else if let Some(pose_cw) = fixed_poses_cw.get(&obs.kf_id) {
            pose_cw
        } else {
            continue;
        };

        let error =
            compute_reprojection_error_cw(pose_cw, &mp_positions[mp_idx], &obs.observed_uv, camera);

        let chi2 = error.norm_squared();
        total_error += huber_cost(chi2, huber_threshold);
    }

    total_error
}

/// Compute reprojection error for a single observation.
/// Following g2o convention: error = observed - projected
/// pose_cw is T_cw (world-to-camera), like g2o's SE3Quat.
fn compute_reprojection_error_cw(
    pose_cw: &SE3,
    point_world: &Vector3<f64>,
    observed_uv: &Vector2<f64>,
    camera: &CameraModel,
) -> Vector2<f64> {
    // Transform point to camera frame (T_cw * P_w)
    let p_cam = pose_cw.transform_point(point_world);

    if p_cam.z <= 0.001 {
        return Vector2::new(1000.0, 1000.0); // Large error for invalid projection
    }

    // Project to image
    let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
    let v = camera.fy * p_cam.y / p_cam.z + camera.cy;

    // g2o convention: error = observed - projected
    Vector2::new(observed_uv.x - u, observed_uv.y - v)
}

/// Build the normal equations for Gauss-Newton.
/// All poses are T_cw (world-to-camera).
fn build_normal_equations_cw(
    observations: &[Observation],
    kf_poses_cw: &[SE3],
    mp_positions: &[Vector3<f64>],
    kf_to_idx: &HashMap<KeyFrameId, usize>,
    mp_to_idx: &HashMap<MapPointId, usize>,
    fixed_poses_cw: &HashMap<KeyFrameId, SE3>,
    camera: &CameraModel,
    huber_threshold: f64,
    num_kf_params: usize,
    num_mp_params: usize,
) -> (
    DMatrix<f64>,
    DVector<f64>,
    DMatrix<f64>,
    DVector<f64>,
    DMatrix<f64>,
) {
    let mut h_pp = DMatrix::zeros(num_kf_params, num_kf_params); // Pose-pose block
    let mut b_p = DVector::zeros(num_kf_params);
    let mut h_ll = DMatrix::zeros(num_mp_params, num_mp_params); // Landmark-landmark block
    let mut b_l = DVector::zeros(num_mp_params);
    let mut h_pl = DMatrix::zeros(num_kf_params, num_mp_params); // Pose-landmark block

    for obs in observations {
        let mp_idx = match mp_to_idx.get(&obs.mp_id) {
            Some(&idx) => idx,
            None => continue,
        };

        // Check if this keyframe is optimized or fixed
        let kf_idx_opt = kf_to_idx.get(&obs.kf_id).copied();
        let pose_cw = if let Some(kf_idx) = kf_idx_opt {
            &kf_poses_cw[kf_idx]
        } else if let Some(pose_cw) = fixed_poses_cw.get(&obs.kf_id) {
            pose_cw
        } else {
            continue;
        };

        let point = &mp_positions[mp_idx];
        let error = compute_reprojection_error_cw(pose_cw, point, &obs.observed_uv, camera);

        // Robust weight
        let chi2 = error.norm_squared();
        let weight = huber_weight(chi2, huber_threshold);

        // Compute Jacobians
        let (j_pose, j_point) = compute_jacobians_cw(pose_cw, point, camera);

        // Accumulate normal equations
        let j_point_w = j_point * weight;

        // H_ll block (3x3 for this point)
        let h_ll_contrib = j_point.transpose() * j_point_w;
        for i in 0..3 {
            for j in 0..3 {
                h_ll[(mp_idx * 3 + i, mp_idx * 3 + j)] += h_ll_contrib[(i, j)];
            }
        }

        // b_l block
        // Note: g2o's Jacobians are d(proj)/d(params), not d(error)/d(params)
        // So for error = obs - proj, gradient = -J^T * error
        // and b = -gradient = +J^T * error
        let b_l_contrib = j_point.transpose() * (error * weight);
        for i in 0..3 {
            b_l[mp_idx * 3 + i] += b_l_contrib[i];
        }

        // If keyframe is optimized (not fixed), add pose contributions
        if let Some(kf_idx) = kf_idx_opt {
            let j_pose_w = j_pose * weight;

            // H_pp block (6x6 for this pose)
            let h_pp_contrib = j_pose.transpose() * j_pose_w;
            for i in 0..6 {
                for j in 0..6 {
                    h_pp[(kf_idx * 6 + i, kf_idx * 6 + j)] += h_pp_contrib[(i, j)];
                }
            }

            // b_p block
            let b_p_contrib = j_pose.transpose() * (error * weight);
            for i in 0..6 {
                b_p[kf_idx * 6 + i] += b_p_contrib[i];
            }

            // H_pl block (6x3)
            let h_pl_contrib = j_pose.transpose() * j_point_w;
            for i in 0..6 {
                for j in 0..3 {
                    h_pl[(kf_idx * 6 + i, mp_idx * 3 + j)] += h_pl_contrib[(i, j)];
                }
            }
        }
    }

    // Add damping to diagonal (Levenberg-Marquardt style)
    let damping = 1e-6;
    for i in 0..num_kf_params {
        h_pp[(i, i)] += damping;
    }
    for i in 0..num_mp_params {
        h_ll[(i, i)] += damping;
    }

    (h_pp, b_p, h_ll, b_l, h_pl)
}

/// Compute Jacobians of reprojection error with respect to pose and point.
/// Matches g2o's EdgeSE3ProjectXYZ::linearizeOplus() exactly.
///
/// pose_cw is T_cw (world-to-camera), like g2o's SE3Quat.
/// Pose Jacobian columns: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
/// This matches g2o's convention where rotation comes before translation.
fn compute_jacobians_cw(
    pose_cw: &SE3,
    point_world: &Vector3<f64>,
    camera: &CameraModel,
) -> (Matrix2x6<f64>, Matrix2x3<f64>) {
    // Transform point to camera frame (T_cw * P_w)
    let p_cam = pose_cw.transform_point(point_world);

    let x = p_cam.x;
    let y = p_cam.y;
    let z = p_cam.z;
    let invz = 1.0 / z;
    let invz2 = invz * invz;

    if z.abs() < 1e-6 {
        return (Matrix2x6::zeros(), Matrix2x3::zeros());
    }

    let fx = camera.fx;
    let fy = camera.fy;

    // =========================================================================
    // Jacobian w.r.t. pose (j_pose) - exactly from g2o types_six_dof_expmap.cpp
    // Columns: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
    // =========================================================================
    // Row 0 (u/x error derivative)
    let j00 = x * y * invz2 * fx; // d(u_err)/d(rot_x)
    let j01 = -(1.0 + x * x * invz2) * fx; // d(u_err)/d(rot_y)
    let j02 = y * invz * fx; // d(u_err)/d(rot_z)
    let j03 = -invz * fx; // d(u_err)/d(trans_x)
    let j04 = 0.0; // d(u_err)/d(trans_y)
    let j05 = x * invz2 * fx; // d(u_err)/d(trans_z)

    // Row 1 (v/y error derivative)
    let j10 = (1.0 + y * y * invz2) * fy; // d(v_err)/d(rot_x)
    let j11 = -x * y * invz2 * fy; // d(v_err)/d(rot_y)
    let j12 = -x * invz * fy; // d(v_err)/d(rot_z)
    let j13 = 0.0; // d(v_err)/d(trans_x)
    let j14 = -invz * fy; // d(v_err)/d(trans_y)
    let j15 = y * invz2 * fy; // d(v_err)/d(trans_z)

    // Matrix2x6 in row-major: (row0_col0, row0_col1, ..., row1_col0, ...)
    let j_pose = Matrix2x6::new(j00, j01, j02, j03, j04, j05, j10, j11, j12, j13, j14, j15);

    // =========================================================================
    // Jacobian w.r.t. 3D point (j_point) - from g2o
    // _jacobianOplusXi = -1./z * tmp * T.rotation().toRotationMatrix();
    // where tmp = [fx, 0, -fx*x/z; 0, fy, -fy*y/z]
    // =========================================================================
    let r_cw = pose_cw.rotation.to_rotation_matrix().into_inner();

    // tmp matrix (2x3)
    let tmp = Matrix2x3::new(fx, 0.0, -fx * x * invz, 0.0, fy, -fy * y * invz);

    // j_point = -1/z * tmp * R_cw
    let j_point = (-invz) * tmp * r_cw;

    (j_pose, j_point)
}

/// Solve the Schur complement system.
fn solve_schur_complement(
    h_pp: &DMatrix<f64>,
    b_p: &DVector<f64>,
    h_ll: &DMatrix<f64>,
    b_l: &DVector<f64>,
    h_pl: &DMatrix<f64>,
    num_kf_params: usize,
) -> Option<DVector<f64>> {
    if num_kf_params == 0 {
        return Some(DVector::zeros(0));
    }

    // For efficiency with block-diagonal H_ll, we compute the Schur complement
    // S = H_pp - H_pl * H_ll^-1 * H_pl^T
    // b_s = b_p - H_pl * H_ll^-1 * b_l

    let num_points = h_ll.nrows() / 3;
    let mut h_ll_inv_h_pl_t = DMatrix::zeros(h_ll.nrows(), h_pl.nrows());
    let mut h_ll_inv_b_l = DVector::zeros(h_ll.nrows());

    // H_ll is block diagonal (3x3 blocks), so we can invert block by block
    for i in 0..num_points {
        let block = h_ll.fixed_view::<3, 3>(i * 3, i * 3);
        let block_inv = match block.try_inverse() {
            Some(inv) => inv,
            None => Matrix3::identity() * 1e6, // Fallback for singular blocks
        };

        // H_ll^-1 * H_pl^T
        for j in 0..h_pl.nrows() {
            let h_pl_col =
                Vector3::new(h_pl[(j, i * 3)], h_pl[(j, i * 3 + 1)], h_pl[(j, i * 3 + 2)]);
            let result = block_inv * h_pl_col;
            h_ll_inv_h_pl_t[(i * 3, j)] = result[0];
            h_ll_inv_h_pl_t[(i * 3 + 1, j)] = result[1];
            h_ll_inv_h_pl_t[(i * 3 + 2, j)] = result[2];
        }

        // H_ll^-1 * b_l
        let b_l_block = Vector3::new(b_l[i * 3], b_l[i * 3 + 1], b_l[i * 3 + 2]);
        let result = block_inv * b_l_block;
        h_ll_inv_b_l[i * 3] = result[0];
        h_ll_inv_b_l[i * 3 + 1] = result[1];
        h_ll_inv_b_l[i * 3 + 2] = result[2];
    }

    // S = H_pp - H_pl * H_ll^-1 * H_pl^T
    // h_ll_inv_h_pl_t is already H_ll^-1 * H_pl^T (num_mp_params x num_kf_params)
    // So: H_pl (num_kf_params x num_mp_params) * h_ll_inv_h_pl_t (num_mp_params x num_kf_params)
    let schur = h_pp - h_pl * &h_ll_inv_h_pl_t;

    // b_s = b_p - H_pl * H_ll^-1 * b_l
    let b_schur = b_p - h_pl * &h_ll_inv_b_l;

    // Solve S * delta_p = b_s
    let schur_lu = schur.lu();
    let delta_p = schur_lu.solve(&b_schur)?;

    Some(delta_p)
}

/// Apply pose update using exponential map (left multiplication).
/// This matches g2o's update: T_new = exp(delta) * T_old
/// where delta = [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
fn apply_pose_update_left(pose: &SE3, delta_rot: &Vector3<f64>, delta_trans: &Vector3<f64>) -> SE3 {
    // Create delta SE3 from rotation and translation
    let angle = delta_rot.norm();
    let delta_rotation = if angle > 1e-10 {
        let axis = delta_rot / angle;
        nalgebra::UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
    } else {
        nalgebra::UnitQuaternion::identity()
    };

    // Left multiplication: T_new = exp(delta) * T_old
    // For SE3: (R_delta, t_delta) * (R, t) = (R_delta * R, R_delta * t + t_delta)
    SE3 {
        rotation: delta_rotation * pose.rotation,
        translation: delta_rotation * pose.translation + delta_trans,
    }
}

/// Huber robust cost function.
fn huber_cost(chi2: f64, threshold: f64) -> f64 {
    if chi2 <= threshold {
        0.5 * chi2
    } else {
        let e = chi2.sqrt();
        let th = threshold.sqrt();
        th * (e - 0.5 * th)
    }
}

/// Huber weight for iteratively reweighted least squares.
fn huber_weight(chi2: f64, threshold: f64) -> f64 {
    if chi2 <= threshold {
        1.0
    } else {
        (threshold / chi2).sqrt()
    }
}
