//! Loop correction: Pose graph correction and map point fusion.
//!
//! This module implements the three-phase correction process:
//! 1. COLLECT (read lock): Gather all data needed for correction
//! 2. SOLVE (no lock): Compute corrected poses via Sim3 propagation
//! 3. APPLY (write lock): Update poses, fuse map points, add loop edges

use std::collections::{HashMap, HashSet, VecDeque};

use nalgebra::Vector3;
use opencv::prelude::*;
use parking_lot::RwLock;

use crate::atlas::atlas::Atlas;
use crate::atlas::map::{KeyFrameId, Map, MapPointId};
use crate::geometry::{SE3, Sim3};
use crate::tracking::frame::CameraModel;

use super::detector::LoopCandidate;
use super::sim3_solver::compute_sim3_from_matches;

/// A verified loop closure ready for correction.
#[derive(Debug, Clone)]
pub struct VerifiedLoop {
    /// Current keyframe that detected the loop.
    pub current_kf_id: KeyFrameId,

    /// Loop keyframe (older keyframe we're closing back to).
    pub loop_kf_id: KeyFrameId,

    /// Computed Sim3 transformation from current to loop keyframe.
    /// This transforms points in current KF frame to loop KF frame.
    pub sim3_current_to_loop: Sim3,

    /// Matched map point pairs: (current_mp_id, loop_mp_id).
    pub matched_map_points: Vec<(MapPointId, MapPointId)>,

    /// Feature matches: (current_feat_idx, loop_feat_idx).
    pub feature_matches: Vec<(usize, usize)>,
}

/// A loop edge in the pose graph.
#[derive(Debug, Clone)]
pub struct LoopEdge {
    /// The other keyframe in the loop.
    pub connected_kf_id: KeyFrameId,

    /// Relative Sim3 transformation to the connected keyframe.
    pub relative_sim3: Sim3,
}

/// Data collected during Phase 1 (read lock).
struct CorrectionData {
    /// Current keyframe pose before correction.
    current_pose: SE3,

    /// Loop keyframe pose.
    loop_pose: SE3,

    /// All keyframes in the current keyframe's connected component.
    /// Maps keyframe ID to (pose, parent_id).
    affected_keyframes: HashMap<KeyFrameId, (SE3, Option<KeyFrameId>)>,

    /// Map points observed by affected keyframes.
    /// Maps map point ID to (position, observing_kf_ids).
    affected_map_points: HashMap<MapPointId, (Vector3<f64>, Vec<KeyFrameId>)>,

    /// Sim3 correction for the current keyframe.
    sim3_correction: Sim3,

    /// Map points to fuse: current -> loop.
    points_to_fuse: Vec<(MapPointId, MapPointId)>,
}

/// Results from Phase 2 (solve).
struct CorrectionResult {
    /// Corrected Sim3 poses for each keyframe.
    corrected_poses: HashMap<KeyFrameId, Sim3>,

    /// Corrected map point positions.
    corrected_points: HashMap<MapPointId, Vector3<f64>>,

    /// Loop edge to add.
    loop_edge: (KeyFrameId, KeyFrameId, Sim3),
}

/// Configuration for loop correction.
#[derive(Debug, Clone)]
pub struct CorrectorConfig {
    /// Minimum covisibility weight to include in essential graph.
    pub min_covisibility_weight: usize,

    /// Maximum depth in spanning tree to propagate correction.
    pub max_propagation_depth: usize,
}

impl Default for CorrectorConfig {
    fn default() -> Self {
        Self {
            min_covisibility_weight: 15,
            max_propagation_depth: 100,
        }
    }
}

/// Verify a loop candidate geometrically using Sim3 RANSAC.
///
/// # Arguments
/// * `candidate` - The loop candidate from detection
/// * `atlas` - The atlas (read-locked internally)
/// * `camera` - Camera model for projection
///
/// # Returns
/// * `Some(VerifiedLoop)` if geometric verification succeeds
/// * `None` if verification fails
pub fn verify_loop_candidate(
    candidate: &LoopCandidate,
    atlas: &RwLock<Atlas>,
    camera: &CameraModel,
) -> Option<VerifiedLoop> {
    let atlas_guard = atlas.read();
    let map = atlas_guard.active_map();

    let current_kf = map.get_keyframe(candidate.current_kf_id)?;
    let loop_kf = map.get_keyframe(candidate.loop_kf_id)?;

    // Get 3D points for both keyframes
    let (current_points, _current_mp_ids, _current_feat_indices) =
        get_keyframe_3d_points(current_kf, map);
    let (loop_points, _loop_mp_ids, _loop_feat_indices) = get_keyframe_3d_points(loop_kf, map);

    if current_points.len() < 20 || loop_points.len() < 20 {
        return None;
    }

    // Match features using BoW-accelerated matching
    let matches = match_features_bow(current_kf, loop_kf, map);

    if matches.len() < 15 {
        return None;
    }

    // Get 3D point pairs for matched features
    let mut pts_current = Vec::new();
    let mut pts_loop = Vec::new();
    let mut matched_map_points = Vec::new();
    let mut feature_matches = Vec::new();

    for &(curr_idx, loop_idx) in &matches {
        // Get 3D points in camera frame
        let curr_pt_cam = match current_kf.points_cam.get(curr_idx) {
            Some(Some(pt)) => pt,
            _ => continue,
        };
        let loop_pt_cam = match loop_kf.points_cam.get(loop_idx) {
            Some(Some(pt)) => pt,
            _ => continue,
        };

        // Transform to world frame
        let curr_pt_world = current_kf.pose.transform_point(curr_pt_cam);
        let loop_pt_world = loop_kf.pose.transform_point(loop_pt_cam);

        pts_current.push(curr_pt_world);
        pts_loop.push(loop_pt_world);

        // Track map point matches if both have associated map points
        if let (Some(curr_mp), Some(loop_mp)) = (
            current_kf.get_map_point(curr_idx),
            loop_kf.get_map_point(loop_idx),
        ) {
            matched_map_points.push((curr_mp, loop_mp));
        }

        feature_matches.push((curr_idx, loop_idx));
    }

    if pts_current.len() < 15 {
        return None;
    }

    // Run Sim3 RANSAC (scale fixed for stereo)
    let sim3_result = compute_sim3_from_matches(&pts_current, &pts_loop, true)?;

    if sim3_result.num_inliers < 15 {
        return None;
    }

    // Verify by reprojection
    let verified_matches =
        verify_by_reprojection(&sim3_result.sim3, current_kf, loop_kf, &feature_matches, camera);

    if verified_matches < 50 {
        return None;
    }

    Some(VerifiedLoop {
        current_kf_id: candidate.current_kf_id,
        loop_kf_id: candidate.loop_kf_id,
        sim3_current_to_loop: sim3_result.sim3,
        matched_map_points,
        feature_matches,
    })
}

/// Get 3D points from a keyframe.
fn get_keyframe_3d_points(
    kf: &crate::atlas::map::KeyFrame,
    _map: &Map,
) -> (Vec<Vector3<f64>>, Vec<Option<MapPointId>>, Vec<usize>) {
    let mut points = Vec::new();
    let mut mp_ids = Vec::new();
    let mut indices = Vec::new();

    for (idx, pt_opt) in kf.points_cam.iter().enumerate() {
        if let Some(pt_cam) = pt_opt {
            // Transform to world frame
            let pt_world = kf.pose.transform_point(pt_cam);
            points.push(pt_world);
            mp_ids.push(kf.get_map_point(idx));
            indices.push(idx);
        }
    }

    (points, mp_ids, indices)
}

/// Match features between two keyframes using BoW-accelerated matching.
fn match_features_bow(
    kf1: &crate::atlas::map::KeyFrame,
    kf2: &crate::atlas::map::KeyFrame,
    _map: &Map,
) -> Vec<(usize, usize)> {
    let mut matches = Vec::new();

    // Use feature vectors if available for accelerated matching
    let fv1 = kf1.feature_vector();
    let fv2 = kf2.feature_vector();

    match (fv1, fv2) {
        (Some(fv1), Some(fv2)) => {
            // Accelerated matching: only compare features in same vocabulary node
            for (node_id, indices1) in fv1 {
                if let Some(indices2) = fv2.get(node_id) {
                    for &idx1 in indices1 {
                        let desc1 = get_descriptor_row(&kf1.descriptors, idx1 as i32);

                        let mut best_dist = u32::MAX;
                        let mut second_dist = u32::MAX;
                        let mut best_idx2 = 0;

                        for &idx2 in indices2 {
                            let desc2 = get_descriptor_row(&kf2.descriptors, idx2 as i32);
                            let dist = hamming_distance(&desc1, &desc2);

                            if dist < best_dist {
                                second_dist = best_dist;
                                best_dist = dist;
                                best_idx2 = idx2;
                            } else if dist < second_dist {
                                second_dist = dist;
                            }
                        }

                        // Lowe's ratio test
                        if best_dist < 50 && (best_dist as f64) < 0.7 * (second_dist as f64) {
                            matches.push((idx1, best_idx2));
                        }
                    }
                }
            }
        }
        _ => {
            // Fallback: brute-force matching (slower)
            let n1 = kf1.descriptors.rows();
            let n2 = kf2.descriptors.rows();

            for i in 0..n1 {
                let desc1 = get_descriptor_row(&kf1.descriptors, i);

                let mut best_dist = u32::MAX;
                let mut second_dist = u32::MAX;
                let mut best_j = 0;

                for j in 0..n2 {
                    let desc2 = get_descriptor_row(&kf2.descriptors, j);
                    let dist = hamming_distance(&desc1, &desc2);

                    if dist < best_dist {
                        second_dist = best_dist;
                        best_dist = dist;
                        best_j = j;
                    } else if dist < second_dist {
                        second_dist = dist;
                    }
                }

                if best_dist < 50 && (best_dist as f64) < 0.7 * (second_dist as f64) {
                    matches.push((i as usize, best_j as usize));
                }
            }
        }
    }

    matches
}

/// Get a descriptor row as bytes.
fn get_descriptor_row(mat: &opencv::core::Mat, row: i32) -> [u8; 32] {
    use opencv::prelude::*;
    let mut desc = [0u8; 32];
    for j in 0..32 {
        if let Ok(val) = mat.at_2d::<u8>(row, j) {
            desc[j as usize] = *val;
        }
    }
    desc
}

/// Compute Hamming distance between two descriptors.
fn hamming_distance(a: &[u8; 32], b: &[u8; 32]) -> u32 {
    let mut dist = 0u32;
    for i in 0..32 {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Verify loop by reprojection.
fn verify_by_reprojection(
    sim3: &Sim3,
    current_kf: &crate::atlas::map::KeyFrame,
    loop_kf: &crate::atlas::map::KeyFrame,
    matches: &[(usize, usize)],
    camera: &CameraModel,
) -> usize {
    let mut good_matches = 0;
    let threshold_sq = 5.991 * 1.0; // Chi-squared threshold for 2 DOF

    for &(curr_idx, loop_idx) in matches {
        // Get 3D point from current KF
        let curr_pt_cam = match current_kf.points_cam.get(curr_idx) {
            Some(Some(pt)) => pt,
            _ => continue,
        };

        // Transform to world, then apply Sim3, then to loop camera
        let pt_world = current_kf.pose.transform_point(curr_pt_cam);
        let pt_corrected = sim3.transform_point(&pt_world);
        let pt_loop_cam = loop_kf.pose.inverse().transform_point(&pt_corrected);

        if pt_loop_cam.z <= 0.0 {
            continue;
        }

        // Project to image
        let u = camera.fx * pt_loop_cam.x / pt_loop_cam.z + camera.cx;
        let v = camera.fy * pt_loop_cam.y / pt_loop_cam.z + camera.cy;

        // Get observed keypoint
        if let Ok(kp) = loop_kf.keypoints.get(loop_idx) {
            let du = u - kp.pt().x as f64;
            let dv = v - kp.pt().y as f64;
            let error_sq = du * du + dv * dv;

            // Get scale-dependent threshold (scale factor = 1.2^octave)
            let octave = kp.octave();
            let scale = 1.2_f64.powi(octave);
            let scaled_threshold = threshold_sq * scale * scale;

            if error_sq < scaled_threshold {
                good_matches += 1;
            }
        }
    }

    good_matches
}

/// Perform loop correction using the three-phase pattern.
///
/// This is the main entry point for loop correction.
pub fn correct_loop(
    verified_loop: &VerifiedLoop,
    atlas: &RwLock<Atlas>,
    _camera: &CameraModel,
    config: &CorrectorConfig,
) {
    // Phase 1: COLLECT (read lock)
    let data = {
        let atlas_guard = atlas.read();
        collect_correction_data(&atlas_guard, verified_loop, config)
    };

    let data = match data {
        Some(d) => d,
        None => return,
    };

    // Phase 2: SOLVE (no lock)
    let result = compute_correction(&data, config);

    // Phase 3: APPLY (write lock)
    {
        let mut atlas_guard = atlas.write();
        apply_correction(&mut atlas_guard, &data, &result, verified_loop);
    }

    tracing::info!(
        "Loop correction complete: {} -> {}, corrected {} keyframes, {} map points",
        verified_loop.current_kf_id,
        verified_loop.loop_kf_id,
        result.corrected_poses.len(),
        result.corrected_points.len()
    );
}

/// Phase 1: Collect all data needed for correction.
fn collect_correction_data(
    atlas: &Atlas,
    verified_loop: &VerifiedLoop,
    config: &CorrectorConfig,
) -> Option<CorrectionData> {
    let map = atlas.active_map();

    let current_kf = map.get_keyframe(verified_loop.current_kf_id)?;
    let loop_kf = map.get_keyframe(verified_loop.loop_kf_id)?;

    // Compute the correction Sim3 for current keyframe
    // The corrected pose should place current KF consistent with loop KF
    let sim3_correction = verified_loop.sim3_current_to_loop.clone();

    // Collect affected keyframes via BFS from current keyframe
    let mut affected_keyframes = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back((verified_loop.current_kf_id, 0));
    visited.insert(verified_loop.current_kf_id);

    while let Some((kf_id, depth)) = queue.pop_front() {
        if depth > config.max_propagation_depth {
            continue;
        }

        if let Some(kf) = map.get_keyframe(kf_id) {
            affected_keyframes.insert(kf_id, (kf.pose.clone(), kf.parent_id));

            // Add spanning tree children
            for &child_id in &kf.children_ids {
                if !visited.contains(&child_id) {
                    visited.insert(child_id);
                    queue.push_back((child_id, depth + 1));
                }
            }

            // Add strong covisibility edges
            for (&cov_id, &weight) in kf.covisibility_weights() {
                if weight >= config.min_covisibility_weight && !visited.contains(&cov_id) {
                    visited.insert(cov_id);
                    queue.push_back((cov_id, depth + 1));
                }
            }
        }
    }

    // Collect affected map points
    let mut affected_map_points = HashMap::new();
    for &kf_id in affected_keyframes.keys() {
        if let Some(kf) = map.get_keyframe(kf_id) {
            for (_, mp_id) in kf.get_map_point_indices() {
                if !affected_map_points.contains_key(&mp_id) {
                    if let Some(mp) = map.get_map_point(mp_id) {
                        let observers: Vec<_> = mp.observations.keys().copied().collect();
                        affected_map_points.insert(mp_id, (mp.position, observers));
                    }
                }
            }
        }
    }

    Some(CorrectionData {
        current_pose: current_kf.pose.clone(),
        loop_pose: loop_kf.pose.clone(),
        affected_keyframes,
        affected_map_points,
        sim3_correction,
        points_to_fuse: verified_loop.matched_map_points.clone(),
    })
}

/// Phase 2: Compute corrected poses and points.
fn compute_correction(data: &CorrectionData, _config: &CorrectorConfig) -> CorrectionResult {
    let mut corrected_poses = HashMap::new();

    // Propagate Sim3 correction through the spanning tree
    // Start with the inverse of the correction (to correct world coordinates)
    let correction_inv = data.sim3_correction.inverse();

    for (&kf_id, (pose, _parent)) in &data.affected_keyframes {
        // Apply correction: new_pose = correction_inv * old_pose
        let corrected_sim3 = Sim3 {
            rotation: correction_inv.rotation * pose.rotation,
            translation: correction_inv.transform_point(&pose.translation),
            scale: correction_inv.scale,
        };
        corrected_poses.insert(kf_id, corrected_sim3);
    }

    // Correct map point positions
    let mut corrected_points = HashMap::new();
    for (&mp_id, (position, _observers)) in &data.affected_map_points {
        let corrected_pos = correction_inv.transform_point(position);
        corrected_points.insert(mp_id, corrected_pos);
    }

    // Create loop edge
    let loop_edge = (
        data.affected_keyframes
            .keys()
            .next()
            .copied()
            .unwrap_or(KeyFrameId::new(0)), // current
        KeyFrameId::new(0),                 // will be set properly
        data.sim3_correction.clone(),
    );

    CorrectionResult {
        corrected_poses,
        corrected_points,
        loop_edge,
    }
}

/// Phase 3: Apply correction to the map.
fn apply_correction(
    atlas: &mut Atlas,
    data: &CorrectionData,
    result: &CorrectionResult,
    verified_loop: &VerifiedLoop,
) {
    let map = atlas.active_map_mut();

    // Update keyframe poses
    for (&kf_id, corrected_sim3) in &result.corrected_poses {
        if let Some(kf) = map.get_keyframe_mut(kf_id) {
            // Convert Sim3 back to SE3 (scale is 1.0 for stereo)
            kf.pose = corrected_sim3.to_se3();
        }
    }

    // Update map point positions
    for (&mp_id, corrected_pos) in &result.corrected_points {
        if let Some(mp) = map.get_map_point_mut(mp_id) {
            mp.position = *corrected_pos;
        }
    }

    // Fuse duplicate map points
    fuse_map_points(map, &data.points_to_fuse);

    // Note: Loop edges would be stored in KeyFrame if we add that field
    // For now, the covisibility graph is updated through map point fusion
    tracing::debug!(
        "Applied loop correction: {} -> {}",
        verified_loop.current_kf_id,
        verified_loop.loop_kf_id
    );
}

/// Fuse duplicate map points (keep the one with more observations).
fn fuse_map_points(map: &mut Map, points_to_fuse: &[(MapPointId, MapPointId)]) {
    for &(current_mp_id, loop_mp_id) in points_to_fuse {
        if current_mp_id == loop_mp_id {
            continue;
        }

        // Get observation counts
        let current_obs = map
            .get_map_point(current_mp_id)
            .map(|mp| mp.num_observations())
            .unwrap_or(0);
        let loop_obs = map
            .get_map_point(loop_mp_id)
            .map(|mp| mp.num_observations())
            .unwrap_or(0);

        // Keep the one with more observations, replace the other
        let (to_keep, to_replace) = if current_obs >= loop_obs {
            (current_mp_id, loop_mp_id)
        } else {
            (loop_mp_id, current_mp_id)
        };

        // Get observations from the point to replace
        let observations_to_move: Vec<_> = map
            .get_map_point(to_replace)
            .map(|mp| mp.observations.iter().map(|(&k, &v)| (k, v)).collect())
            .unwrap_or_default();

        // Move observations to the kept point
        for (kf_id, feat_idx) in observations_to_move {
            // Update keyframe's map point reference
            if let Some(kf) = map.get_keyframe_mut(kf_id) {
                if kf.get_map_point(feat_idx) == Some(to_replace) {
                    kf.set_map_point(feat_idx, to_keep);
                }
            }

            // Add observation to kept point
            if let Some(mp) = map.get_map_point_mut(to_keep) {
                mp.add_observation(kf_id, feat_idx);
            }
        }

        // Remove the replaced point
        map.remove_map_point(to_replace);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        let a = [0u8; 32];
        let b = [0u8; 32];
        assert_eq!(hamming_distance(&a, &b), 0);

        let mut c = [0u8; 32];
        c[0] = 0xFF;
        assert_eq!(hamming_distance(&a, &c), 8);
    }

    #[test]
    fn test_verified_loop_structure() {
        let verified = VerifiedLoop {
            current_kf_id: KeyFrameId::new(100),
            loop_kf_id: KeyFrameId::new(10),
            sim3_current_to_loop: Sim3::identity(),
            matched_map_points: vec![],
            feature_matches: vec![],
        };

        assert_eq!(verified.current_kf_id.0, 100);
        assert_eq!(verified.loop_kf_id.0, 10);
    }

    #[test]
    fn test_corrector_config_default() {
        let config = CorrectorConfig::default();
        assert_eq!(config.min_covisibility_weight, 15);
        assert_eq!(config.max_propagation_depth, 100);
    }
}
