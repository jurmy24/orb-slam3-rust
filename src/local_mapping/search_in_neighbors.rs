//! SearchInNeighbors - Map point fusion between covisible keyframes.
//!
//! This module implements the map point fusion algorithm from ORB-SLAM3's LocalMapping.
//! It fuses duplicate map points between the current keyframe and its neighbors by:
//! 1. Collecting neighbor keyframes (covisible + temporal for inertial)
//! 2. Fusing current KF's map points into neighbors
//! 3. Fusing neighbors' map points into current KF
//! 4. Updating affected points' descriptors and normals

use std::collections::HashSet;

use opencv::prelude::MatTraitConst;
use tracing::debug;

use crate::atlas::map::{KeyFrameId, Map, MapPointId};
use crate::tracking::frame::{descriptor_distance, CameraModel, TH_LOW};

/// Configuration for SearchInNeighbors.
pub struct SearchInNeighborsConfig {
    /// Number of best covisible keyframes to collect.
    pub nn_covisibles: usize,
    /// Minimum number of neighbors for inertial mode (walks temporal chain).
    pub min_neighbors_inertial: usize,
    /// Search radius multiplier based on scale (radius = factor * scale).
    pub radius_factor: f64,
    /// Descriptor distance threshold for matching.
    pub desc_threshold: u32,
}

impl Default for SearchInNeighborsConfig {
    fn default() -> Self {
        Self {
            nn_covisibles: 10,
            min_neighbors_inertial: 20,
            radius_factor: 3.0,
            desc_threshold: TH_LOW,
        }
    }
}

/// Result of SearchInNeighbors operation.
#[derive(Debug, Default)]
pub struct SearchInNeighborsResult {
    /// Number of map points fused.
    pub num_fused: usize,
    /// Number of new observations added.
    pub num_observations_added: usize,
    /// Number of descriptors updated.
    pub num_descriptors_updated: usize,
}

/// Perform map point fusion between current keyframe and its neighbors.
///
/// This is the main entry point for SearchInNeighbors, following ORB-SLAM3's
/// LocalMapping::SearchInNeighbors().
///
/// # Arguments
/// * `map` - Mutable reference to the map
/// * `current_kf_id` - ID of the current (just processed) keyframe
/// * `camera` - Camera model for projection
/// * `is_inertial` - Whether IMU is initialized (affects neighbor collection)
///
/// # Returns
/// Statistics about the fusion operation
pub fn search_in_neighbors(
    map: &mut Map,
    current_kf_id: KeyFrameId,
    camera: &CameraModel,
    is_inertial: bool,
) -> SearchInNeighborsResult {
    let config = SearchInNeighborsConfig::default();
    search_in_neighbors_with_config(map, current_kf_id, camera, is_inertial, &config)
}

/// Perform map point fusion with custom configuration.
pub fn search_in_neighbors_with_config(
    map: &mut Map,
    current_kf_id: KeyFrameId,
    camera: &CameraModel,
    is_inertial: bool,
    config: &SearchInNeighborsConfig,
) -> SearchInNeighborsResult {
    let mut result = SearchInNeighborsResult::default();

    // Phase 1: Collect neighbor keyframes
    let neighbor_ids = collect_neighbors(map, current_kf_id, is_inertial, config);

    if neighbor_ids.is_empty() {
        return result;
    }

    // Collect current KF's map points
    let current_mp_ids: Vec<MapPointId> = {
        match map.get_keyframe(current_kf_id) {
            Some(kf) => kf
                .map_point_ids
                .iter()
                .filter_map(|mp| *mp)
                .collect(),
            None => return result,
        }
    };

    // Phase 2: Fuse current KF's map points into neighbors
    let fused_phase2 = fuse_points_into_keyframes(
        map,
        &current_mp_ids,
        &neighbor_ids,
        camera,
        config,
    );
    result.num_fused += fused_phase2.0;
    result.num_observations_added += fused_phase2.1;

    // Phase 3: Collect neighbors' map points and fuse into current KF
    let neighbor_mp_ids: HashSet<MapPointId> = neighbor_ids
        .iter()
        .filter_map(|&kf_id| map.get_keyframe(kf_id))
        .flat_map(|kf| kf.map_point_ids.iter().filter_map(|mp| *mp))
        .collect();

    // Remove points that are already in current KF
    let current_mp_set: HashSet<MapPointId> = current_mp_ids.iter().copied().collect();
    let neighbor_only_mps: Vec<MapPointId> = neighbor_mp_ids
        .difference(&current_mp_set)
        .copied()
        .collect();

    let fused_phase3 = fuse_points_into_keyframes(
        map,
        &neighbor_only_mps,
        &[current_kf_id],
        camera,
        config,
    );
    result.num_fused += fused_phase3.0;
    result.num_observations_added += fused_phase3.1;

    // Phase 4: Update affected map points' descriptors
    // Collect all affected map points
    let mut affected_mps: HashSet<MapPointId> = HashSet::new();
    affected_mps.extend(current_mp_ids.iter());
    affected_mps.extend(neighbor_mp_ids.iter());

    for mp_id in affected_mps {
        if map.compute_distinctive_descriptors(mp_id) {
            result.num_descriptors_updated += 1;
        }
        map.update_map_point_normal_and_depth(mp_id);
    }

    debug!(
        "[SearchInNeighbors] kf={}: fused={} obs_added={} desc_updated={}",
        current_kf_id.0, result.num_fused, result.num_observations_added, result.num_descriptors_updated
    );

    result
}

/// Collect neighbor keyframes for fusion.
///
/// Returns IDs of neighbor keyframes (best covisibles + secondary covisibles).
/// For inertial mode, also walks the temporal chain to get enough neighbors.
fn collect_neighbors(
    map: &Map,
    current_kf_id: KeyFrameId,
    is_inertial: bool,
    config: &SearchInNeighborsConfig,
) -> Vec<KeyFrameId> {
    let mut neighbors = HashSet::new();

    // Get best covisible keyframes (requires mutable access for caching)
    // Since we only have immutable access here, we'll compute on the fly
    let first_neighbors: Vec<KeyFrameId> = {
        if let Some(kf) = map.get_keyframe(current_kf_id) {
            let mut covisibles: Vec<(KeyFrameId, usize)> = kf
                .covisibility_weights()
                .iter()
                .map(|(&id, &weight)| (id, weight))
                .collect();
            covisibles.sort_by(|a, b| b.1.cmp(&a.1));
            covisibles
                .into_iter()
                .take(config.nn_covisibles)
                .map(|(id, _)| id)
                .collect()
        } else {
            return Vec::new();
        }
    };

    neighbors.extend(first_neighbors.iter().copied());

    // Get secondary covisibles (neighbors of neighbors)
    for &neighbor_id in &first_neighbors {
        if let Some(neighbor_kf) = map.get_keyframe(neighbor_id) {
            let secondary: Vec<KeyFrameId> = neighbor_kf
                .covisibility_weights()
                .iter()
                .filter(|(id, _)| **id != current_kf_id && !neighbors.contains(id))
                .take(5) // Limit secondary neighbors
                .map(|(&id, _)| id)
                .collect();
            neighbors.extend(secondary);
        }
    }

    // For inertial mode, walk temporal chain to ensure enough neighbors
    if is_inertial && neighbors.len() < config.min_neighbors_inertial {
        // Walk backward from current KF
        let mut prev_kf_id = map
            .get_keyframe(current_kf_id)
            .and_then(|kf| kf.prev_kf);

        while neighbors.len() < config.min_neighbors_inertial {
            match prev_kf_id {
                Some(id) => {
                    neighbors.insert(id);
                    prev_kf_id = map.get_keyframe(id).and_then(|kf| kf.prev_kf);
                }
                None => break,
            }
        }
    }

    // Remove current KF from neighbors
    neighbors.remove(&current_kf_id);

    neighbors.into_iter().collect()
}

/// Fuse map points into target keyframes.
///
/// For each map point, project it into each target keyframe and look for
/// matching features. If a match is found:
/// - If the target feature has no map point: add observation
/// - If the target feature has a different map point: merge the points
///
/// Returns (num_fused, num_observations_added)
fn fuse_points_into_keyframes(
    map: &mut Map,
    mp_ids: &[MapPointId],
    target_kf_ids: &[KeyFrameId],
    camera: &CameraModel,
    config: &SearchInNeighborsConfig,
) -> (usize, usize) {
    let mut num_fused = 0;
    let mut num_obs_added = 0;

    // Get ORB scale factor
    let scale_factor = 1.2f64;
    let num_levels = 8;

    for &mp_id in mp_ids {
        // Get map point data
        let mp_data = match map.get_map_point(mp_id) {
            Some(mp) if !mp.is_bad => Some((
                mp.position,
                mp.descriptor.try_clone().unwrap_or_default(),
                mp.observations.keys().copied().collect::<HashSet<_>>(),
            )),
            _ => continue,
        };

        let (position, descriptor, mp_observers) = match mp_data {
            Some(d) => d,
            None => continue,
        };

        for &target_kf_id in target_kf_ids {
            // Skip if map point already observed by this keyframe
            if mp_observers.contains(&target_kf_id) {
                continue;
            }

            // Get target keyframe pose
            let kf_pose_cw = match map.get_keyframe(target_kf_id) {
                Some(kf) => kf.pose.inverse(),
                None => continue,
            };

            // Project point to target keyframe
            let p_cam = kf_pose_cw.transform_point(&position);

            // Check if in front of camera
            if p_cam.z <= 0.0 {
                continue;
            }

            // Project to image
            let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
            let v = camera.fy * p_cam.y / p_cam.z + camera.cy;

            // Check image bounds (with margin)
            let width = camera.cx * 2.0;
            let height = camera.cy * 2.0;
            if u < 0.0 || u >= width || v < 0.0 || v >= height {
                continue;
            }

            // Compute search radius based on depth
            let depth = p_cam.z;
            let radius = config.radius_factor * scale_factor.powi(num_levels as i32 - 1) * depth / camera.fx;
            let search_radius = radius.min(50.0).max(10.0); // Clamp to reasonable range

            // Find candidate features in area
            let candidates = {
                match map.get_keyframe(target_kf_id) {
                    Some(kf) => kf.get_features_in_area(u, v, search_radius, None, None),
                    None => continue,
                }
            };

            if candidates.is_empty() {
                continue;
            }

            // Find best match by descriptor distance
            let mut best_dist = u32::MAX;
            let mut best_idx: Option<usize> = None;

            for &feat_idx in &candidates {
                // Get feature descriptor
                let feat_desc = {
                    match map.get_keyframe(target_kf_id) {
                        Some(kf) => match kf.descriptors.row(feat_idx as i32) {
                            Ok(row) => row.try_clone().ok(),
                            Err(_) => None,
                        },
                        None => None,
                    }
                };

                if let Some(ref feat_desc) = feat_desc {
                    if let Ok(dist) = descriptor_distance(&descriptor, feat_desc) {
                        if dist < best_dist && dist < config.desc_threshold {
                            best_dist = dist;
                            best_idx = Some(feat_idx);
                        }
                    }
                }
            }

            // Process match
            if let Some(feat_idx) = best_idx {
                // Check if feature already has a map point
                let existing_mp = map
                    .get_keyframe(target_kf_id)
                    .and_then(|kf| kf.get_map_point(feat_idx));

                match existing_mp {
                    Some(existing_mp_id) if existing_mp_id != mp_id => {
                        // Feature has different map point - merge them
                        // Keep the one with more observations
                        let (keeper, goner) = {
                            let mp_obs = map
                                .get_map_point(mp_id)
                                .map(|mp| mp.num_observations())
                                .unwrap_or(0);
                            let existing_obs = map
                                .get_map_point(existing_mp_id)
                                .map(|mp| mp.num_observations())
                                .unwrap_or(0);

                            if mp_obs >= existing_obs {
                                (mp_id, existing_mp_id)
                            } else {
                                (existing_mp_id, mp_id)
                            }
                        };

                        if map.merge_map_points(keeper, goner) {
                            num_fused += 1;
                        }
                    }
                    None => {
                        // Feature has no map point - add observation
                        map.associate(target_kf_id, feat_idx, mp_id);
                        num_obs_added += 1;
                    }
                    _ => {
                        // Feature already has this map point - skip
                    }
                }
            }
        }
    }

    (num_fused, num_obs_added)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SearchInNeighborsConfig::default();
        assert_eq!(config.nn_covisibles, 10);
        assert_eq!(config.min_neighbors_inertial, 20);
    }
}
