//! Multi-frame triangulation with geometric validation.
//!
//! Implements the CreateNewMapPoints algorithm from ORB-SLAM3's LocalMapping.cc:
//! 1. For each neighbor keyframe pair, find epipolar-constrained matches
//! 2. Triangulate 3D points using DLT (or use stereo depth if available)
//! 3. Validate with parallax, depth, reprojection error, and scale consistency checks

use nalgebra::{Matrix3, Matrix4, Vector3};
use opencv::core::{Mat, KeyPoint, Vector};
use opencv::prelude::*;
use tracing::debug;

use crate::atlas::map::{KeyFrameId, Map, MapPointId};
use crate::geometry::SE3;
use crate::tracking::frame::CameraModel;
use crate::tracking::frame::stereo::{descriptor_distance, TH_LOW};
use crate::vocabulary::FeatureVector;

/// Configuration for multi-frame triangulation.
pub struct TriangulationConfig {
    /// Number of neighbor keyframes to consider.
    pub num_neighbors: usize,
    /// Maximum descriptor distance for matching.
    pub max_descriptor_dist: u32,
    /// Minimum baseline as fraction of scene depth.
    pub min_baseline_ratio: f64,
    /// Minimum parallax angle in radians (for inertial mode).
    pub min_parallax_inertial: f64,
    /// Minimum parallax angle in radians (for visual mode).
    pub min_parallax_visual: f64,
    /// Maximum reprojection error in pixels (chi-squared at 95% confidence).
    pub max_reproj_error_mono: f64,
    /// Maximum reprojection error for stereo (3-DOF chi-squared).
    pub max_reproj_error_stereo: f64,
    /// Scale ratio tolerance factor.
    pub scale_ratio_factor: f64,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        Self {
            num_neighbors: 10,
            max_descriptor_dist: TH_LOW,
            min_baseline_ratio: 0.01,
            min_parallax_inertial: (0.9996_f64).acos(), // ~1.6 degrees
            min_parallax_visual: (0.9998_f64).acos(),   // ~1.1 degrees
            max_reproj_error_mono: 5.991,   // chi-squared 95% with 2 DOF
            max_reproj_error_stereo: 7.8,   // chi-squared 95% with 3 DOF
            scale_ratio_factor: 1.5,
        }
    }
}

/// Result of multi-frame triangulation.
pub struct TriangulationResult {
    pub num_new_points: usize,
    pub num_pairs_checked: usize,
    pub num_matches_found: usize,
    pub num_triangulated: usize,
    pub num_validated: usize,
}

/// Triangulate new map points between the current keyframe and its neighbors.
///
/// This implements the C++ LocalMapping::CreateNewMapPoints() algorithm:
/// 1. Get N best covisible keyframes + temporal neighbors
/// 2. For each pair, check baseline is sufficient
/// 3. Find matches between unmatched features using descriptor matching with epipolar constraint
/// 4. Triangulate and validate each match
/// 5. Create new MapPoints for validated triangulations
pub fn triangulate_from_neighbors(
    map: &mut Map,
    current_kf_id: KeyFrameId,
    camera: &CameraModel,
    config: &TriangulationConfig,
    is_inertial: bool,
) -> TriangulationResult {
    let mut result = TriangulationResult {
        num_new_points: 0,
        num_pairs_checked: 0,
        num_matches_found: 0,
        num_triangulated: 0,
        num_validated: 0,
    };

    // Get neighbor keyframes
    let neighbor_ids = get_neighbor_keyframes(map, current_kf_id, config.num_neighbors, is_inertial);

    if neighbor_ids.is_empty() {
        return result;
    }

    // Get current keyframe data
    let (current_pose, current_center, current_keypoints, current_descriptors, current_map_points, current_points_cam, current_feat_vec) = {
        match map.get_keyframe(current_kf_id) {
            Some(kf) => (
                kf.pose.clone(),
                kf.camera_center(),
                kf.keypoints.clone(),
                kf.descriptors.clone(),
                kf.map_point_ids.clone(),
                kf.points_cam.clone(),
                kf.feature_vector.clone(),
            ),
            None => return result,
        }
    };

    // Minimum parallax based on mode
    let min_parallax_cos = if is_inertial {
        config.min_parallax_inertial.cos()
    } else {
        config.min_parallax_visual.cos()
    };

    // Process each neighbor
    for &neighbor_id in &neighbor_ids {
        result.num_pairs_checked += 1;

        // Get neighbor keyframe data
        let (neighbor_pose, neighbor_center, neighbor_keypoints, neighbor_descriptors, neighbor_map_points, neighbor_points_cam, neighbor_feat_vec) = {
            match map.get_keyframe(neighbor_id) {
                Some(kf) => (
                    kf.pose.clone(),
                    kf.camera_center(),
                    kf.keypoints.clone(),
                    kf.descriptors.clone(),
                    kf.map_point_ids.clone(),
                    kf.points_cam.clone(),
                    kf.feature_vector.clone(),
                ),
                None => continue,
            }
        };

        // Check baseline is sufficient
        let baseline = (neighbor_center - current_center).norm();
        if baseline < camera.baseline {
            // Baseline too short for stereo case
            continue;
        }

        // Find matches between unmatched features
        // Use FeatureVector-based matching if both keyframes have it (faster)
        let matches = match (&current_feat_vec, &neighbor_feat_vec) {
            (Some(fv1), Some(fv2)) => {
                search_for_triangulation_bow(
                    fv1,
                    fv2,
                    &current_keypoints,
                    &current_descriptors,
                    &current_map_points,
                    &current_points_cam,
                    &neighbor_keypoints,
                    &neighbor_descriptors,
                    &neighbor_map_points,
                    &current_pose,
                    &neighbor_pose,
                    camera,
                    config.max_descriptor_dist,
                )
            }
            _ => {
                // Fallback to spatial grid matching
                search_for_triangulation(
                    &current_keypoints,
                    &current_descriptors,
                    &current_map_points,
                    &current_points_cam,
                    &neighbor_keypoints,
                    &neighbor_descriptors,
                    &neighbor_map_points,
                    &current_pose,
                    &neighbor_pose,
                    camera,
                    config.max_descriptor_dist,
                )
            }
        };

        result.num_matches_found += matches.len();

        // Process each match
        for (idx1, idx2) in matches {
            // Check if we can use stereo depth from either keyframe
            let stereo1 = current_points_cam.get(idx1).and_then(|p| p.as_ref());
            let stereo2 = neighbor_points_cam.get(idx2).and_then(|p| p.as_ref());

            // Compute ray directions for parallax check
            let kp1 = &current_keypoints.get(idx1).unwrap();
            let kp2 = &neighbor_keypoints.get(idx2).unwrap();

            // Unproject to normalized camera coordinates
            let xn1 = Vector3::new(
                (kp1.pt().x as f64 - camera.cx) / camera.fx,
                (kp1.pt().y as f64 - camera.cy) / camera.fy,
                1.0,
            );
            let xn2 = Vector3::new(
                (kp2.pt().x as f64 - camera.cx) / camera.fx,
                (kp2.pt().y as f64 - camera.cy) / camera.fy,
                1.0,
            );

            // Transform to world frame
            let ray1 = current_pose.rotation * xn1;
            let ray2 = neighbor_pose.rotation * xn2;
            let cos_parallax = ray1.dot(&ray2) / (ray1.norm() * ray2.norm());

            // Compute stereo parallax if available
            let cos_parallax_stereo1 = stereo1.map(|p| {
                let angle = 2.0 * (camera.baseline / 2.0 / p.z).atan();
                angle.cos()
            });
            let cos_parallax_stereo2 = stereo2.map(|p| {
                let angle = 2.0 * (camera.baseline / 2.0 / p.z).atan();
                angle.cos()
            });

            let cos_parallax_stereo = match (cos_parallax_stereo1, cos_parallax_stereo2) {
                (Some(c1), Some(c2)) => c1.min(c2),
                (Some(c), None) | (None, Some(c)) => c,
                (None, None) => f64::MAX,
            };

            // Decide triangulation method
            let point_3d: Option<Vector3<f64>> = if cos_parallax < cos_parallax_stereo
                && cos_parallax > 0.0
                && (stereo1.is_some() || stereo2.is_some() || cos_parallax < min_parallax_cos)
            {
                // Use DLT triangulation
                triangulate_dlt(
                    &xn1,
                    &xn2,
                    &current_pose,
                    &neighbor_pose,
                )
            } else if let Some(p_cam) = stereo1 {
                if cos_parallax_stereo1.unwrap_or(f64::MAX) < cos_parallax_stereo2.unwrap_or(f64::MAX) {
                    // Use stereo depth from current keyframe
                    Some(current_pose.transform_point(p_cam))
                } else if let Some(p_cam2) = stereo2 {
                    Some(neighbor_pose.transform_point(p_cam2))
                } else {
                    continue; // No good triangulation method
                }
            } else if let Some(p_cam) = stereo2 {
                // Use stereo depth from neighbor keyframe
                Some(neighbor_pose.transform_point(p_cam))
            } else {
                // No stereo and very low parallax - skip
                continue;
            };

            let p_world = match point_3d {
                Some(p) => p,
                None => continue,
            };

            result.num_triangulated += 1;

            // Validate the triangulated point
            if !validate_triangulation(
                &p_world,
                &current_pose,
                &neighbor_pose,
                kp1,
                kp2,
                camera,
                stereo1.is_some(),
                stereo2.is_some(),
                config,
            ) {
                continue;
            }

            result.num_validated += 1;

            // Get descriptor for the new map point (from current keyframe)
            let descriptor = match current_descriptors.row(idx1 as i32) {
                Ok(row) => row.try_clone().unwrap_or_default(),
                Err(_) => Mat::default(),
            };

            // Create new map point
            let mp_id = map.create_map_point(p_world, descriptor, current_kf_id);

            // Associate with both keyframes
            map.associate(current_kf_id, idx1, mp_id);
            map.associate(neighbor_id, idx2, mp_id);

            result.num_new_points += 1;
        }
    }

    if result.num_new_points > 0 {
        debug!(
            "[MultiFrameTriang] Created {} new points (pairs={}, matches={}, triangulated={}, validated={})",
            result.num_new_points,
            result.num_pairs_checked,
            result.num_matches_found,
            result.num_triangulated,
            result.num_validated
        );
    }

    result
}

/// Get neighbor keyframes for triangulation.
///
/// Returns the N best covisible keyframes, plus temporal neighbors if in inertial mode.
fn get_neighbor_keyframes(
    map: &mut Map,
    current_kf_id: KeyFrameId,
    n: usize,
    is_inertial: bool,
) -> Vec<KeyFrameId> {
    let mut neighbors = map.get_local_keyframes(current_kf_id, n);

    // In inertial mode, also add temporal neighbors
    if is_inertial {
        let mut prev_id = map.get_keyframe(current_kf_id).and_then(|kf| kf.prev_kf);
        let mut count = 0;
        while neighbors.len() < n && prev_id.is_some() && count < n {
            let pid = prev_id.unwrap();
            if !neighbors.contains(&pid) {
                neighbors.push(pid);
            }
            prev_id = map.get_keyframe(pid).and_then(|kf| kf.prev_kf);
            count += 1;
        }
    }

    neighbors
}

/// Grid cell size for spatial hashing (pixels).
const GRID_CELL_SIZE: f32 = 32.0;

/// Maximum grid dimension (to limit memory).
const MAX_GRID_DIM: usize = 64;

/// Build a spatial grid for fast feature lookup.
///
/// Returns a 2D grid where each cell contains indices of features in that cell.
/// This reduces matching from O(N²) to O(N·k) where k is features per cell.
fn build_feature_grid(keypoints: &Vector<KeyPoint>, image_width: u32, image_height: u32) -> Vec<Vec<usize>> {
    let grid_cols = ((image_width as f32 / GRID_CELL_SIZE).ceil() as usize).min(MAX_GRID_DIM);
    let grid_rows = ((image_height as f32 / GRID_CELL_SIZE).ceil() as usize).min(MAX_GRID_DIM);
    let num_cells = grid_cols * grid_rows;

    let mut grid = vec![Vec::new(); num_cells];

    for (idx, kp) in keypoints.iter().enumerate() {
        let col = ((kp.pt().x / GRID_CELL_SIZE) as usize).min(grid_cols - 1);
        let row = ((kp.pt().y / GRID_CELL_SIZE) as usize).min(grid_rows - 1);
        let cell_idx = row * grid_cols + col;
        grid[cell_idx].push(idx);
    }

    grid
}

/// Get candidate feature indices from neighboring grid cells.
///
/// Returns indices of features in cells within radius of the query point.
fn get_candidates_in_radius(
    grid: &[Vec<usize>],
    grid_cols: usize,
    grid_rows: usize,
    x: f32,
    y: f32,
    radius: f32,
) -> Vec<usize> {
    let min_col = ((x - radius) / GRID_CELL_SIZE).floor().max(0.0) as usize;
    let max_col = (((x + radius) / GRID_CELL_SIZE).ceil() as usize).min(grid_cols - 1);
    let min_row = ((y - radius) / GRID_CELL_SIZE).floor().max(0.0) as usize;
    let max_row = (((y + radius) / GRID_CELL_SIZE).ceil() as usize).min(grid_rows - 1);

    let mut candidates = Vec::new();
    for row in min_row..=max_row {
        for col in min_col..=max_col {
            let cell_idx = row * grid_cols + col;
            if cell_idx < grid.len() {
                candidates.extend(&grid[cell_idx]);
            }
        }
    }
    candidates
}

/// Search for triangulation matches between two keyframes.
///
/// Finds matches between features that:
/// 1. Are not already associated with map points
/// 2. Have similar descriptors (below threshold)
/// 3. Satisfy epipolar constraint
///
/// Uses spatial grid indexing to reduce complexity from O(N²) to O(N·k).
fn search_for_triangulation(
    kp1: &Vector<KeyPoint>,
    desc1: &Mat,
    mp1: &[Option<MapPointId>],
    points_cam1: &[Option<Vector3<f64>>],
    kp2: &Vector<KeyPoint>,
    desc2: &Mat,
    mp2: &[Option<MapPointId>],
    pose1: &SE3,
    pose2: &SE3,
    camera: &CameraModel,
    max_dist: u32,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();

    // Compute epipole: camera 1 center projected into camera 2
    let c1_world = pose1.translation.clone();
    let pose2_inv = pose2.inverse();
    let c1_in_cam2 = pose2_inv.transform_point(&c1_world);
    let epipole = Vector3::new(
        camera.fx * c1_in_cam2.x / c1_in_cam2.z + camera.cx,
        camera.fy * c1_in_cam2.y / c1_in_cam2.z + camera.cy,
        1.0,
    );

    // Compute relative pose
    let t12 = pose2_inv.translation - pose2_inv.rotation * pose1.translation;
    let r12 = pose2_inv.rotation * pose1.rotation.inverse();

    // Build spatial grid for kf2 features - O(N) construction
    // This reduces matching from O(N²) to approximately O(N·k) where k ~ features per cell
    let image_width = (camera.cx * 2.0) as u32;
    let image_height = (camera.cy * 2.0) as u32;
    let grid = build_feature_grid(kp2, image_width, image_height);
    let grid_cols = ((image_width as f32 / GRID_CELL_SIZE).ceil() as usize).min(MAX_GRID_DIM);
    let grid_rows = ((image_height as f32 / GRID_CELL_SIZE).ceil() as usize).min(MAX_GRID_DIM);

    // Search radius based on epipolar line extent (pixels)
    // Features should be within this distance of the epipolar line
    let search_radius = 100.0_f32;

    // Track which features in kf2 have been matched
    let mut matched2 = vec![false; kp2.len()];

    // For each unmatched feature in kf1
    for (idx1, kp1_item) in kp1.iter().enumerate() {
        // Skip if already has a map point
        if mp1.get(idx1).map_or(false, |mp| mp.is_some()) {
            continue;
        }

        let d1 = match desc1.row(idx1 as i32) {
            Ok(row) => row,
            Err(_) => continue,
        };

        let mut best_dist = max_dist;
        let mut best_idx2: Option<usize> = None;

        // Check if feature has stereo depth
        let has_stereo1 = points_cam1.get(idx1).map_or(false, |p| p.is_some());

        // Get candidate features from spatial grid near the epipolar line
        // Project kp1 to kf2's image plane as approximate search center
        let search_x = kp1_item.pt().x;
        let search_y = kp1_item.pt().y;
        let candidates = get_candidates_in_radius(
            &grid,
            grid_cols,
            grid_rows,
            search_x,
            search_y,
            search_radius,
        );

        // Search only among candidates (not all features)
        // OPTIMIZATION: Check epipolar constraint BEFORE descriptor distance
        // Epipolar check is ~10x cheaper than descriptor distance computation
        for &idx2 in &candidates {
            // Skip if already matched or has map point
            if matched2[idx2] || mp2.get(idx2).map_or(false, |mp| mp.is_some()) {
                continue;
            }

            let kp2_item = match kp2.get(idx2) {
                Ok(kp) => kp,
                Err(_) => continue,
            };

            // FIRST: Check epipolar constraint (cheap geometric check)
            if !has_stereo1 {
                // Check distance from epipole (point near epipole has low parallax)
                let dx = epipole.x - kp2_item.pt().x as f64;
                let dy = epipole.y - kp2_item.pt().y as f64;
                let scale_sq = 1.0; // Could use scale factor from octave
                if dx * dx + dy * dy < 100.0 * scale_sq {
                    continue;
                }
            }

            // Check epipolar line constraint (still cheap)
            if !check_epipolar_constraint(&kp1_item, &kp2_item, &r12, &t12, camera) {
                continue;
            }

            // SECOND: Compute descriptor distance (expensive - only for epipolar-valid candidates)
            let d2 = match desc2.row(idx2 as i32) {
                Ok(row) => row,
                Err(_) => continue,
            };

            let dist = match descriptor_distance(&d1, &d2) {
                Ok(d) => d,
                Err(_) => continue,
            };

            if dist < best_dist && dist <= max_dist {
                best_idx2 = Some(idx2);
                best_dist = dist;
            }
        }

        if let Some(idx2) = best_idx2 {
            matches.push((idx1, idx2));
            matched2[idx2] = true;
        }
    }

    matches
}

/// Search for triangulation matches using FeatureVector grouping.
///
/// This is more efficient than spatial grid when vocabulary is available:
/// - Groups features by vocabulary node (typically ~10 groups at level L-4)
/// - Only compares features that share the same vocabulary node
/// - Reduces false matches since features in the same node are descriptor-similar
///
/// Falls back to spatial grid matching if FeatureVectors are not available.
#[allow(clippy::too_many_arguments)]
fn search_for_triangulation_bow(
    feat_vec1: &FeatureVector,
    feat_vec2: &FeatureVector,
    kp1: &Vector<KeyPoint>,
    desc1: &Mat,
    mp1: &[Option<MapPointId>],
    points_cam1: &[Option<Vector3<f64>>],
    kp2: &Vector<KeyPoint>,
    desc2: &Mat,
    mp2: &[Option<MapPointId>],
    pose1: &SE3,
    pose2: &SE3,
    camera: &CameraModel,
    max_dist: u32,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();

    // Compute epipole: camera 1 center projected into camera 2
    let c1_world = pose1.translation.clone();
    let pose2_inv = pose2.inverse();
    let c1_in_cam2 = pose2_inv.transform_point(&c1_world);
    let epipole = Vector3::new(
        camera.fx * c1_in_cam2.x / c1_in_cam2.z + camera.cx,
        camera.fy * c1_in_cam2.y / c1_in_cam2.z + camera.cy,
        1.0,
    );

    // Compute relative pose for epipolar constraint
    let t12 = pose2_inv.translation - pose2_inv.rotation * pose1.translation;
    let r12 = pose2_inv.rotation * pose1.rotation.inverse();

    // Track which features in kf2 have been matched
    let mut matched2 = vec![false; kp2.len()];

    // Iterate over vocabulary nodes that appear in both keyframes
    for (node_id, indices1) in feat_vec1 {
        // Get corresponding features in kf2 that share this vocabulary node
        let indices2 = match feat_vec2.get(node_id) {
            Some(indices) => indices,
            None => continue, // No features in kf2 share this node
        };

        // For each feature in kf1 at this node
        for &idx1 in indices1 {
            // Skip if already has a map point
            if mp1.get(idx1).map_or(false, |mp| mp.is_some()) {
                continue;
            }

            let kp1_item = match kp1.get(idx1) {
                Ok(kp) => kp,
                Err(_) => continue,
            };

            let d1 = match desc1.row(idx1 as i32) {
                Ok(row) => row,
                Err(_) => continue,
            };

            let has_stereo1 = points_cam1.get(idx1).map_or(false, |p| p.is_some());

            let mut best_dist = max_dist;
            let mut best_idx2: Option<usize> = None;

            // Search among features in kf2 that share the same vocabulary node
            // OPTIMIZATION: Check epipolar constraint BEFORE descriptor distance
            for &idx2 in indices2 {
                // Skip if already matched or has map point
                if matched2[idx2] || mp2.get(idx2).map_or(false, |mp| mp.is_some()) {
                    continue;
                }

                let kp2_item = match kp2.get(idx2) {
                    Ok(kp) => kp,
                    Err(_) => continue,
                };

                // OPTIMIZATION: Check epipolar constraint FIRST (cheaper than descriptor distance)
                if !has_stereo1 {
                    // Check distance from epipole (point near epipole has low parallax)
                    let dx = epipole.x - kp2_item.pt().x as f64;
                    let dy = epipole.y - kp2_item.pt().y as f64;
                    if dx * dx + dy * dy < 100.0 {
                        continue;
                    }
                }

                // Check epipolar line constraint
                if !check_epipolar_constraint(&kp1_item, &kp2_item, &r12, &t12, camera) {
                    continue;
                }

                // SECOND: Compute descriptor distance (only for epipolar-valid candidates)
                let d2 = match desc2.row(idx2 as i32) {
                    Ok(row) => row,
                    Err(_) => continue,
                };

                let dist = match descriptor_distance(&d1, &d2) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                if dist < best_dist && dist <= max_dist {
                    best_dist = dist;
                    best_idx2 = Some(idx2);
                }
            }

            if let Some(idx2) = best_idx2 {
                matches.push((idx1, idx2));
                matched2[idx2] = true;
            }
        }
    }

    matches
}

/// Check if two keypoints satisfy the epipolar constraint.
fn check_epipolar_constraint(
    kp1: &KeyPoint,
    kp2: &KeyPoint,
    r12: &nalgebra::UnitQuaternion<f64>,
    t12: &Vector3<f64>,
    camera: &CameraModel,
) -> bool {
    // Compute essential matrix: E = [t12]_x * R12
    let t_skew = skew_symmetric(t12);
    let r12_mat = r12.to_rotation_matrix();
    let e = t_skew * r12_mat.matrix();

    // Compute fundamental matrix: F = K2^{-T} * E * K1^{-1}
    // For same camera: F = K^{-T} * E * K^{-1}
    let k_inv = Matrix3::new(
        1.0 / camera.fx, 0.0, -camera.cx / camera.fx,
        0.0, 1.0 / camera.fy, -camera.cy / camera.fy,
        0.0, 0.0, 1.0,
    );
    let f = k_inv.transpose() * e * k_inv;

    // Compute epipolar line in image 2: l2 = F * p1
    let p1 = Vector3::new(kp1.pt().x as f64, kp1.pt().y as f64, 1.0);
    let l2 = f * p1;

    // Distance from p2 to epipolar line
    let p2 = Vector3::new(kp2.pt().x as f64, kp2.pt().y as f64, 1.0);
    let num = (l2.dot(&p2)).abs();
    let den = (l2.x * l2.x + l2.y * l2.y).sqrt();

    if den < 1e-10 {
        return false;
    }

    let dist = num / den;

    // Allow some tolerance based on scale (octave)
    let sigma_sq = 1.0; // Could scale by octave level
    dist * dist < 3.84 * sigma_sq // chi-squared 95% with 1 DOF
}

/// Compute skew-symmetric matrix from vector.
fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

/// Triangulate a 3D point using Direct Linear Transform (DLT).
///
/// Given normalized camera coordinates and camera poses, computes the 3D point
/// that minimizes algebraic error.
fn triangulate_dlt(
    xn1: &Vector3<f64>,
    xn2: &Vector3<f64>,
    pose1: &SE3,
    pose2: &SE3,
) -> Option<Vector3<f64>> {
    // Compute world-to-camera transforms
    let t1_cw = pose1.inverse();
    let t2_cw = pose2.inverse();

    // Build projection matrices (3x4)
    let p1 = projection_matrix(&t1_cw);
    let p2 = projection_matrix(&t2_cw);

    // Build the DLT system: A * X = 0
    // Each point gives 2 equations (from x * P[2] - P[0] and y * P[2] - P[1])
    let mut a = Matrix4::<f64>::zeros();

    // Equations from camera 1
    for j in 0..4 {
        a[(0, j)] = xn1.x * p1[(2, j)] - p1[(0, j)];
        a[(1, j)] = xn1.y * p1[(2, j)] - p1[(1, j)];
    }

    // Equations from camera 2
    for j in 0..4 {
        a[(2, j)] = xn2.x * p2[(2, j)] - p2[(0, j)];
        a[(3, j)] = xn2.y * p2[(2, j)] - p2[(1, j)];
    }

    // Solve via SVD - the solution is the right singular vector corresponding to smallest singular value
    let svd = a.svd(true, true);
    let v = svd.v_t?.transpose();
    let x3d_h = v.column(3);

    // Convert from homogeneous
    if x3d_h[3].abs() < 1e-10 {
        return None;
    }

    Some(Vector3::new(
        x3d_h[0] / x3d_h[3],
        x3d_h[1] / x3d_h[3],
        x3d_h[2] / x3d_h[3],
    ))
}

/// Build a 3x4 projection matrix from an SE3 pose (world-to-camera).
fn projection_matrix(pose_cw: &SE3) -> nalgebra::SMatrix<f64, 3, 4> {
    let r = pose_cw.rotation.to_rotation_matrix();
    let t = &pose_cw.translation;

    nalgebra::SMatrix::<f64, 3, 4>::from_columns(&[
        r.matrix().column(0).into(),
        r.matrix().column(1).into(),
        r.matrix().column(2).into(),
        (*t).into(),
    ])
}

/// Validate a triangulated 3D point with geometric checks.
fn validate_triangulation(
    p_world: &Vector3<f64>,
    pose1: &SE3,
    pose2: &SE3,
    kp1: &KeyPoint,
    kp2: &KeyPoint,
    camera: &CameraModel,
    is_stereo1: bool,
    is_stereo2: bool,
    config: &TriangulationConfig,
) -> bool {
    // Transform to camera frames
    let pose1_inv = pose1.inverse();
    let pose2_inv = pose2.inverse();
    let p_cam1 = pose1_inv.transform_point(p_world);
    let p_cam2 = pose2_inv.transform_point(p_world);

    // Check depth is positive (in front of both cameras)
    if p_cam1.z <= 0.0 || p_cam2.z <= 0.0 {
        return false;
    }

    // Check reprojection error in camera 1
    let u1 = camera.fx * p_cam1.x / p_cam1.z + camera.cx;
    let v1 = camera.fy * p_cam1.y / p_cam1.z + camera.cy;
    let err_x1 = u1 - kp1.pt().x as f64;
    let err_y1 = v1 - kp1.pt().y as f64;
    let err_sq1 = err_x1 * err_x1 + err_y1 * err_y1;

    // Scale by octave (inverse sigma squared)
    let sigma_sq1 = 1.0; // Could use scale factor from octave
    let max_err1 = if is_stereo1 { config.max_reproj_error_stereo } else { config.max_reproj_error_mono };
    if err_sq1 / sigma_sq1 > max_err1 {
        return false;
    }

    // Check reprojection error in camera 2
    let u2 = camera.fx * p_cam2.x / p_cam2.z + camera.cx;
    let v2 = camera.fy * p_cam2.y / p_cam2.z + camera.cy;
    let err_x2 = u2 - kp2.pt().x as f64;
    let err_y2 = v2 - kp2.pt().y as f64;
    let err_sq2 = err_x2 * err_x2 + err_y2 * err_y2;

    let sigma_sq2 = 1.0;
    let max_err2 = if is_stereo2 { config.max_reproj_error_stereo } else { config.max_reproj_error_mono };
    if err_sq2 / sigma_sq2 > max_err2 {
        return false;
    }

    // Check scale consistency
    let cam1_center = pose1.translation.clone();
    let cam2_center = pose2.translation.clone();
    let dist1 = (p_world - cam1_center).norm();
    let dist2 = (p_world - cam2_center).norm();

    if dist1 < 1e-6 || dist2 < 1e-6 {
        return false;
    }

    // Ratio of distances should be consistent with ratio of octaves
    // For now, just check the ratio is reasonable
    let ratio_dist = dist2 / dist1;
    let octave1 = kp1.octave() as f64;
    let octave2 = kp2.octave() as f64;
    let scale_factor = 1.2_f64; // ORB scale factor
    let ratio_octave = scale_factor.powf(octave1) / scale_factor.powf(octave2);

    if ratio_dist * config.scale_ratio_factor < ratio_octave
        || ratio_dist > ratio_octave * config.scale_ratio_factor
    {
        return false;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skew_symmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = skew_symmetric(&v);

        // s * v should be zero
        let result = s * v;
        assert!(result.norm() < 1e-10);

        // Check antisymmetry
        assert!((s + s.transpose()).norm() < 1e-10);
    }

    #[test]
    fn test_triangulate_dlt() {
        // Simple test: point at (0, 0, 5) viewed from two cameras
        let pose1 = SE3::identity();
        let mut pose2 = SE3::identity();
        pose2.translation = Vector3::new(1.0, 0.0, 0.0); // 1 meter to the right

        // Point in world
        let p_world = Vector3::new(0.0, 0.0, 5.0);

        // Project to normalized coordinates
        let p_cam1 = pose1.inverse().transform_point(&p_world);
        let p_cam2 = pose2.inverse().transform_point(&p_world);

        let xn1 = Vector3::new(p_cam1.x / p_cam1.z, p_cam1.y / p_cam1.z, 1.0);
        let xn2 = Vector3::new(p_cam2.x / p_cam2.z, p_cam2.y / p_cam2.z, 1.0);

        let result = triangulate_dlt(&xn1, &xn2, &pose1, &pose2);
        assert!(result.is_some());

        let triangulated = result.unwrap();
        assert!((triangulated - p_world).norm() < 0.01);
    }
}
