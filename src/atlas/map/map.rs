//! Map - Container for KeyFrames and MapPoints.
//!
//! The Map is the central data structure that holds the SLAM graph:
//! - KeyFrames with their poses and features
//! - MapPoints (3D landmarks) with their observations
//! - Covisibility relationships between KeyFrames
//!
//! It provides methods for:
//! - Adding/removing KeyFrames and MapPoints
//! - Creating bidirectional associations (KF↔MP)
//! - Querying local neighborhoods
//! - Frustum-based visibility queries
//! - Culling bad MapPoints

use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;
use opencv::core::{KeyPoint, Mat, Vector};

use crate::geometry::SE3;
use crate::tracking::frame::CameraModel;

use super::keyframe::KeyFrame;
use super::map_point::MapPoint;
use super::types::{KeyFrameId, MapPointId};

/// The SLAM map containing KeyFrames and MapPoints.
pub struct Map {
    /// All KeyFrames in the map.
    keyframes: HashMap<KeyFrameId, KeyFrame>,

    /// All MapPoints in the map.
    map_points: HashMap<MapPointId, MapPoint>,

    /// Counter for generating unique KeyFrame IDs.
    next_kf_id: u64,

    /// Counter for generating unique MapPoint IDs.
    next_mp_id: u64,

    /// ORB scale factor (typically 1.2).
    orb_scale_factor: f64,

    /// Number of ORB pyramid levels (typically 8).
    orb_num_levels: u32,

    /// Whether IMU has been initialized for this map.
    imu_initialized: bool,

    /// Whether first visual-inertial BA has been done.
    inertial_ba1_done: bool,

    /// Whether second visual-inertial BA has been done.
    inertial_ba2_done: bool,

    /// Most recent KeyFrame ID (tail of temporal chain).
    /// Used to link new keyframes in temporal order.
    last_keyframe_id: Option<KeyFrameId>,
}

impl Map {
    /// Create a new empty Map.
    // ! Some parameters set here too
    pub fn new() -> Self {
        Self {
            keyframes: HashMap::new(),
            map_points: HashMap::new(),
            next_kf_id: 0,
            next_mp_id: 0,
            orb_scale_factor: 1.2,
            orb_num_levels: 8,
            imu_initialized: false,
            inertial_ba1_done: false,
            inertial_ba2_done: false,
            last_keyframe_id: None,
        }
    }

    /// Create a new Map with custom ORB parameters.
    pub fn with_orb_params(scale_factor: f64, num_levels: u32) -> Self {
        Self {
            orb_scale_factor: scale_factor,
            orb_num_levels: num_levels,
            ..Self::new()
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IMU Initialization State
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if IMU has been initialized for this map.
    pub fn is_imu_initialized(&self) -> bool {
        self.imu_initialized
    }

    /// Set IMU as initialized.
    pub fn set_imu_initialized(&mut self) {
        self.imu_initialized = true;
    }

    /// Check if first visual-inertial BA has been done.
    pub fn is_inertial_ba1_done(&self) -> bool {
        self.inertial_ba1_done
    }

    /// Set first visual-inertial BA as done.
    pub fn set_inertial_ba1_done(&mut self) {
        self.inertial_ba1_done = true;
    }

    /// Check if second visual-inertial BA has been done.
    pub fn is_inertial_ba2_done(&self) -> bool {
        self.inertial_ba2_done
    }

    /// Set second visual-inertial BA as done.
    pub fn set_inertial_ba2_done(&mut self) {
        self.inertial_ba2_done = true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ID Generation
    // ─────────────────────────────────────────────────────────────────────────

    /// Generate a new unique KeyFrame ID.
    pub fn next_keyframe_id(&mut self) -> KeyFrameId {
        let id = KeyFrameId::new(self.next_kf_id);
        self.next_kf_id += 1;
        id
    }

    /// Generate a new unique MapPoint ID.
    pub fn next_map_point_id(&mut self) -> MapPointId {
        let id = MapPointId::new(self.next_mp_id);
        self.next_mp_id += 1;
        id
    }

    // ─────────────────────────────────────────────────────────────────────────
    // KeyFrame Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Create and add a new KeyFrame to the map.
    ///
    /// Automatically links the new keyframe to the previous one in temporal order.
    /// Returns the ID of the created KeyFrame.
    pub fn create_keyframe(
        &mut self,
        timestamp_ns: u64,
        pose: SE3,
        keypoints: Vector<KeyPoint>,
        descriptors: Mat,
        points_cam: Vec<Option<Vector3<f64>>>,
    ) -> KeyFrameId {
        let id = self.next_keyframe_id();
        let mut kf = KeyFrame::new(id, timestamp_ns, pose, keypoints, descriptors, points_cam);

        // Link to previous keyframe in temporal chain
        if let Some(prev_id) = self.last_keyframe_id {
            kf.prev_kf = Some(prev_id);
            // Update the previous keyframe's next_kf pointer
            if let Some(prev_kf) = self.keyframes.get_mut(&prev_id) {
                prev_kf.next_kf = Some(id);
            }
        }

        self.keyframes.insert(id, kf);
        self.last_keyframe_id = Some(id);
        id
    }

    /// Get the most recent KeyFrame ID.
    pub fn last_keyframe_id(&self) -> Option<KeyFrameId> {
        self.last_keyframe_id
    }

    /// Get keyframes in temporal order (oldest to newest).
    ///
    /// Walks the temporal chain from the first keyframe with no prev_kf.
    pub fn keyframes_temporal_order(&self) -> Vec<&KeyFrame> {
        // Find the first keyframe (no prev_kf)
        let first_kf = self.keyframes.values().find(|kf| kf.prev_kf.is_none());

        let mut result = Vec::new();
        let mut current = first_kf;

        while let Some(kf) = current {
            result.push(kf);
            current = kf.next_kf.and_then(|id| self.keyframes.get(&id));
        }

        result
    }

    /// Get the timestamp span of the map in seconds.
    pub fn time_span_seconds(&self) -> f64 {
        let kfs = self.keyframes_temporal_order();
        if kfs.len() < 2 {
            return 0.0;
        }
        let first_ts = kfs.first().unwrap().timestamp_ns as f64 / 1e9;
        let last_ts = kfs.last().unwrap().timestamp_ns as f64 / 1e9;
        last_ts - first_ts
    }

    /// Add an existing KeyFrame to the map.
    pub fn add_keyframe(&mut self, kf: KeyFrame) {
        // Update the ID counter if needed
        if kf.id.0 >= self.next_kf_id {
            self.next_kf_id = kf.id.0 + 1;
        }
        self.keyframes.insert(kf.id, kf);
    }

    /// Get a KeyFrame by ID.
    pub fn get_keyframe(&self, id: KeyFrameId) -> Option<&KeyFrame> {
        self.keyframes.get(&id)
    }

    /// Get a mutable reference to a KeyFrame by ID.
    pub fn get_keyframe_mut(&mut self, id: KeyFrameId) -> Option<&mut KeyFrame> {
        self.keyframes.get_mut(&id)
    }

    /// Get all KeyFrame IDs.
    pub fn keyframe_ids(&self) -> impl Iterator<Item = &KeyFrameId> {
        self.keyframes.keys()
    }

    /// Get all KeyFrames.
    pub fn keyframes(&self) -> impl Iterator<Item = &KeyFrame> {
        self.keyframes.values()
    }

    /// Get the number of KeyFrames.
    pub fn num_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    /// Remove a KeyFrame from the map.
    ///
    /// This does NOT update covisibility or spanning tree relationships.
    /// Call `remove_keyframe_full()` for a complete removal.
    pub fn remove_keyframe(&mut self, id: KeyFrameId) -> Option<KeyFrame> {
        self.keyframes.remove(&id)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MapPoint Operations
    // ─────────────────────────────────────────────────────────────────────────

    /// Create and add a new MapPoint to the map.
    ///
    /// Returns the ID of the created MapPoint.
    pub fn create_map_point(
        &mut self,
        position: Vector3<f64>,
        descriptor: Mat,
        first_kf_id: KeyFrameId,
    ) -> MapPointId {
        let id = self.next_map_point_id();
        let mp = MapPoint::new(id, position, descriptor, first_kf_id);
        self.map_points.insert(id, mp);
        id
    }

    /// Add an existing MapPoint to the map.
    pub fn add_map_point(&mut self, mp: MapPoint) {
        // Update the ID counter if needed
        if mp.id.0 >= self.next_mp_id {
            self.next_mp_id = mp.id.0 + 1;
        }
        self.map_points.insert(mp.id, mp);
    }

    /// Get a MapPoint by ID.
    pub fn get_map_point(&self, id: MapPointId) -> Option<&MapPoint> {
        self.map_points.get(&id)
    }

    /// Get a mutable reference to a MapPoint by ID.
    pub fn get_map_point_mut(&mut self, id: MapPointId) -> Option<&mut MapPoint> {
        self.map_points.get_mut(&id)
    }

    /// Get all MapPoint IDs.
    pub fn map_point_ids(&self) -> impl Iterator<Item = &MapPointId> {
        self.map_points.keys()
    }

    /// Get all MapPoints.
    pub fn map_points(&self) -> impl Iterator<Item = &MapPoint> {
        self.map_points.values()
    }

    /// Get the number of MapPoints.
    pub fn num_map_points(&self) -> usize {
        self.map_points.len()
    }

    /// Remove a MapPoint from the map.
    ///
    /// This does NOT remove observations from KeyFrames.
    /// Call `remove_map_point_full()` for a complete removal.
    pub fn remove_map_point(&mut self, id: MapPointId) -> Option<MapPoint> {
        self.map_points.remove(&id)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Association (KF ↔ MP)
    // ─────────────────────────────────────────────────────────────────────────

    /// Create a bidirectional association between a KeyFrame feature and a MapPoint.
    ///
    /// This:
    /// 1. Links the KeyFrame's feature to the MapPoint
    /// 2. Adds an observation in the MapPoint
    /// 3. Updates covisibility with other KeyFrames observing this MapPoint
    ///
    /// # Arguments
    /// * `kf_id` - KeyFrame ID
    /// * `feature_idx` - Index of the feature in the KeyFrame
    /// * `mp_id` - MapPoint ID
    ///
    /// Returns true if the association was created successfully.
    pub fn associate(&mut self, kf_id: KeyFrameId, feature_idx: usize, mp_id: MapPointId) -> bool {
        // Get existing observers before adding the new one
        let existing_observers: Vec<KeyFrameId> = self
            .map_points
            .get(&mp_id)
            .map(|mp| mp.observations.keys().copied().collect())
            .unwrap_or_default();

        // Update MapPoint
        if let Some(mp) = self.map_points.get_mut(&mp_id) {
            mp.add_observation(kf_id, feature_idx);
        } else {
            return false;
        }

        // Update KeyFrame
        if let Some(kf) = self.keyframes.get_mut(&kf_id) {
            kf.set_map_point(feature_idx, mp_id);
        } else {
            return false;
        }

        // Update covisibility: increment weight with all other observers
        for other_kf_id in existing_observers {
            if other_kf_id == kf_id {
                continue;
            }

            // Get current weights
            let weight_in_other = self
                .keyframes
                .get(&other_kf_id)
                .map(|kf| kf.get_covisibility_weight(kf_id))
                .unwrap_or(0);

            let new_weight = weight_in_other + 1;

            // Update both directions
            if let Some(kf) = self.keyframes.get_mut(&kf_id) {
                kf.add_covisibility(other_kf_id, new_weight);
            }
            if let Some(other_kf) = self.keyframes.get_mut(&other_kf_id) {
                other_kf.add_covisibility(kf_id, new_weight);
            }
        }

        true
    }

    /// Remove the association between a KeyFrame feature and its MapPoint.
    ///
    /// This:
    /// 1. Removes the link from the KeyFrame
    /// 2. Removes the observation from the MapPoint
    /// 3. Updates covisibility with other KeyFrames
    ///
    /// Returns the MapPoint ID that was disassociated, if any.
    pub fn disassociate(&mut self, kf_id: KeyFrameId, feature_idx: usize) -> Option<MapPointId> {
        // Get the MapPoint ID
        let mp_id = self.keyframes.get(&kf_id)?.get_map_point(feature_idx)?;

        // Get other observers for covisibility update
        let other_observers: Vec<KeyFrameId> = self
            .map_points
            .get(&mp_id)
            .map(|mp| {
                mp.observations
                    .keys()
                    .filter(|&&id| id != kf_id)
                    .copied()
                    .collect()
            })
            .unwrap_or_default();

        // Remove from KeyFrame
        if let Some(kf) = self.keyframes.get_mut(&kf_id) {
            kf.erase_map_point(feature_idx);
        }

        // Remove from MapPoint
        if let Some(mp) = self.map_points.get_mut(&mp_id) {
            mp.erase_observation(kf_id);
        }

        // Update covisibility: decrement weight with other observers
        for other_kf_id in other_observers {
            // Get current weight
            let current_weight = self
                .keyframes
                .get(&other_kf_id)
                .map(|kf| kf.get_covisibility_weight(kf_id))
                .unwrap_or(0);

            if current_weight <= 1 {
                // Remove the connection
                if let Some(kf) = self.keyframes.get_mut(&kf_id) {
                    kf.erase_covisibility(other_kf_id);
                }
                if let Some(other_kf) = self.keyframes.get_mut(&other_kf_id) {
                    other_kf.erase_covisibility(kf_id);
                }
            } else {
                // Decrement the weight
                let new_weight = current_weight - 1;
                if let Some(kf) = self.keyframes.get_mut(&kf_id) {
                    kf.update_covisibility(other_kf_id, new_weight);
                }
                if let Some(other_kf) = self.keyframes.get_mut(&other_kf_id) {
                    other_kf.update_covisibility(kf_id, new_weight);
                }
            }
        }

        Some(mp_id)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Queries
    // ─────────────────────────────────────────────────────────────────────────

    /// Get local KeyFrames around a given KeyFrame using covisibility.
    ///
    /// Returns the N KeyFrames with the most shared MapPoints.
    /// Note: This version updates the cached sorted list if dirty.
    pub fn get_local_keyframes(&mut self, kf_id: KeyFrameId, n: usize) -> Vec<KeyFrameId> {
        if let Some(kf) = self.keyframes.get_mut(&kf_id) {
            kf.get_best_covisibles(n)
        } else {
            Vec::new()
        }
    }

    /// Get local KeyFrames around a given KeyFrame using covisibility (immutable version).
    ///
    /// Returns the N KeyFrames with the most shared MapPoints.
    /// This version doesn't use caching - it sorts on each call.
    /// Suitable for read-only access patterns.
    pub fn get_local_keyframes_readonly(&self, kf_id: KeyFrameId, n: usize) -> Vec<KeyFrameId> {
        if let Some(kf) = self.keyframes.get(&kf_id) {
            // Get all covisibility weights and sort by weight descending
            let mut covisibles: Vec<(KeyFrameId, usize)> = kf
                .covisibility_weights()
                .iter()
                .map(|(&id, &weight)| (id, weight))
                .collect();
            covisibles.sort_by(|a, b| b.1.cmp(&a.1));
            covisibles.into_iter().take(n).map(|(id, _)| id).collect()
        } else {
            Vec::new()
        }
    }

    /// Get all MapPoints observed by a set of KeyFrames.
    pub fn get_map_points_from_keyframes(&self, kf_ids: &[KeyFrameId]) -> HashSet<MapPointId> {
        let mut mp_ids = HashSet::new();

        for &kf_id in kf_ids {
            if let Some(kf) = self.keyframes.get(&kf_id) {
                for (_, mp_id) in kf.get_map_point_indices() {
                    mp_ids.insert(mp_id);
                }
            }
        }

        mp_ids
    }

    /// Get MapPoints visible from a camera pose (frustum culling).
    ///
    /// # Arguments
    /// * `pose_wc` - Camera pose (world-to-camera transform)
    /// * `camera` - Camera intrinsics
    /// * `candidate_mp_ids` - Optional set of MapPoints to check (if None, checks all)
    ///
    /// Returns MapPoint IDs that are in the camera frustum.
    pub fn get_map_points_in_frustum(
        &self,
        pose_wc: &SE3,
        camera: &CameraModel,
        candidate_mp_ids: Option<&HashSet<MapPointId>>,
        image_width: u32,
        image_height: u32,
    ) -> Vec<MapPointId> {
        let pose_cw = pose_wc.inverse();
        let cam_pos = pose_wc.translation;

        let check_point = |mp: &MapPoint| -> bool {
            if mp.is_bad {
                return false;
            }

            // Transform to camera frame
            let p_cam = pose_cw.transform_point(&mp.position);

            // Check if in front of camera
            if p_cam.z <= 0.0 {
                return false;
            }

            // Check distance bounds
            let dist = (mp.position - cam_pos).norm();
            if !mp.is_in_distance_range(dist) {
                return false;
            }

            // Project to image
            let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
            let v = camera.fy * p_cam.y / p_cam.z + camera.cy;

            // Check if in image bounds (with small margin)
            let margin = 10.0;
            if u < -margin
                || u >= image_width as f64 + margin
                || v < -margin
                || v >= image_height as f64 + margin
            {
                return false;
            }

            true
        };

        match candidate_mp_ids {
            Some(candidates) => candidates
                .iter()
                .filter(|&&id| self.map_points.get(&id).map_or(false, check_point))
                .copied()
                .collect(),
            None => self
                .map_points
                .iter()
                .filter(|(_, mp)| check_point(mp))
                .map(|(&id, _)| id)
                .collect(),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Culling
    // ─────────────────────────────────────────────────────────────────────────

    /// Cull MapPoints that don't meet quality criteria.
    ///
    /// Marks points as bad and removes them from the map.
    ///
    /// # Arguments
    /// * `min_found_ratio` - Minimum found/visible ratio (typically 0.25)
    /// * `min_observations` - Minimum number of observing KeyFrames (typically 3)
    ///
    /// Returns the number of culled MapPoints.
    pub fn cull_bad_map_points(&mut self, min_found_ratio: f64, min_observations: usize) -> usize {
        // Find MapPoints to cull
        let to_cull: Vec<MapPointId> = self
            .map_points
            .iter()
            .filter(|(_, mp)| mp.should_cull(min_found_ratio, min_observations))
            .map(|(&id, _)| id)
            .collect();

        let count = to_cull.len();

        // Remove each MapPoint and clean up associations
        for mp_id in to_cull {
            self.remove_map_point_full(mp_id);
        }

        count
    }

    /// Fully remove a MapPoint, cleaning up all associations.
    pub fn remove_map_point_full(&mut self, mp_id: MapPointId) {
        // Get observations to clean up
        let observations: Vec<(KeyFrameId, usize)> = self
            .map_points
            .get(&mp_id)
            .map(|mp| {
                mp.observations
                    .iter()
                    .map(|(&kf_id, &feat_idx)| (kf_id, feat_idx))
                    .collect()
            })
            .unwrap_or_default();

        // Remove from each observing KeyFrame
        for (kf_id, feat_idx) in observations {
            if let Some(kf) = self.keyframes.get_mut(&kf_id) {
                kf.erase_map_point(feat_idx);
            }
        }

        // Remove the MapPoint itself
        self.map_points.remove(&mp_id);
    }

    /// Fully remove a KeyFrame, cleaning up all associations.
    pub fn remove_keyframe_full(&mut self, kf_id: KeyFrameId) {
        // Get all MapPoint associations
        let mp_associations: Vec<(usize, MapPointId)> = self
            .keyframes
            .get(&kf_id)
            .map(|kf| kf.get_map_point_indices().collect())
            .unwrap_or_default();

        // Get covisible KeyFrames for cleanup
        let covisibles: Vec<KeyFrameId> = self
            .keyframes
            .get(&kf_id)
            .map(|kf| kf.get_covisibles().copied().collect())
            .unwrap_or_default();

        // Get spanning tree info
        let parent_id = self.keyframes.get(&kf_id).and_then(|kf| kf.parent_id);
        let children: Vec<KeyFrameId> = self
            .keyframes
            .get(&kf_id)
            .map(|kf| kf.children_ids.iter().copied().collect())
            .unwrap_or_default();

        // Remove observations from MapPoints
        for (_, mp_id) in mp_associations {
            if let Some(mp) = self.map_points.get_mut(&mp_id) {
                mp.erase_observation(kf_id);
            }
        }

        // Remove from covisibility graph
        for other_kf_id in covisibles {
            if let Some(other_kf) = self.keyframes.get_mut(&other_kf_id) {
                other_kf.erase_covisibility(kf_id);
            }
        }

        // Update spanning tree
        if let Some(parent) = parent_id {
            // Remove from parent's children
            if let Some(parent_kf) = self.keyframes.get_mut(&parent) {
                parent_kf.erase_child(kf_id);
            }

            // Reparent children to grandparent
            for child_id in &children {
                if let Some(child_kf) = self.keyframes.get_mut(child_id) {
                    child_kf.set_parent(parent);
                }
                if let Some(parent_kf) = self.keyframes.get_mut(&parent) {
                    parent_kf.add_child(*child_id);
                }
            }
        } else {
            // This was a root - first child becomes new root
            if let Some(&new_root) = children.first() {
                if let Some(new_root_kf) = self.keyframes.get_mut(&new_root) {
                    new_root_kf.parent_id = None;
                }
                // Reparent remaining children
                for child_id in children.iter().skip(1) {
                    if let Some(child_kf) = self.keyframes.get_mut(child_id) {
                        child_kf.set_parent(new_root);
                    }
                    if let Some(new_root_kf) = self.keyframes.get_mut(&new_root) {
                        new_root_kf.add_child(*child_id);
                    }
                }
            }
        }

        // Finally remove the KeyFrame
        self.keyframes.remove(&kf_id);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Map Maintenance
    // ─────────────────────────────────────────────────────────────────────────

    /// Update the normal and depth bounds for a MapPoint.
    ///
    /// Should be called after the MapPoint's observations change.
    pub fn update_map_point_normal_and_depth(&mut self, mp_id: MapPointId) {
        // Collect observer positions
        let positions: Vec<(KeyFrameId, Vector3<f64>)> = self
            .map_points
            .get(&mp_id)
            .map(|mp| {
                mp.observations
                    .keys()
                    .filter_map(|&kf_id| {
                        self.keyframes
                            .get(&kf_id)
                            .map(|kf| (kf_id, kf.camera_center()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        // Update the MapPoint
        if let Some(mp) = self.map_points.get_mut(&mp_id) {
            let positions_ref: Vec<_> = positions.iter().map(|(id, pos)| (id, pos)).collect();
            mp.update_normal_and_depth(
                positions_ref.into_iter(),
                self.orb_scale_factor,
                self.orb_num_levels,
            );
        }
    }

    /// Clear the entire map.
    pub fn clear(&mut self) {
        self.keyframes.clear();
        self.map_points.clear();
        self.next_kf_id = 0;
        self.next_mp_id = 0;
    }
}

impl Default for Map {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Map")
            .field("num_keyframes", &self.keyframes.len())
            .field("num_map_points", &self.map_points.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Mat, Vector};

    fn create_test_map() -> Map {
        Map::new()
    }

    #[test]
    fn test_create_keyframe() {
        let mut map = create_test_map();

        let kf_id = map.create_keyframe(
            1000000,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![],
        );

        assert_eq!(kf_id, KeyFrameId::new(0));
        assert_eq!(map.num_keyframes(), 1);

        let kf = map.get_keyframe(kf_id).unwrap();
        assert_eq!(kf.timestamp_ns, 1000000);
    }

    #[test]
    fn test_create_map_point() {
        let mut map = create_test_map();

        let kf_id = map.create_keyframe(0, SE3::identity(), Vector::new(), Mat::default(), vec![]);

        let mp_id = map.create_map_point(Vector3::new(1.0, 2.0, 3.0), Mat::default(), kf_id);

        assert_eq!(mp_id, MapPointId::new(0));
        assert_eq!(map.num_map_points(), 1);

        let mp = map.get_map_point(mp_id).unwrap();
        assert_eq!(mp.position, Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_associate() {
        let mut map = create_test_map();

        // Create two KeyFrames
        let kf1_id = map.create_keyframe(
            0,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );
        let kf2_id = map.create_keyframe(
            1000000,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );

        // Create a MapPoint
        let mp_id = map.create_map_point(Vector3::new(1.0, 0.0, 5.0), Mat::default(), kf1_id);

        // Associate both KeyFrames with the MapPoint
        assert!(map.associate(kf1_id, 0, mp_id));
        assert!(map.associate(kf2_id, 3, mp_id));

        // Check MapPoint has both observations
        let mp = map.get_map_point(mp_id).unwrap();
        assert_eq!(mp.num_observations(), 2);
        assert_eq!(mp.observations.get(&kf1_id), Some(&0));
        assert_eq!(mp.observations.get(&kf2_id), Some(&3));

        // Check KeyFrames point to MapPoint
        assert_eq!(
            map.get_keyframe(kf1_id).unwrap().get_map_point(0),
            Some(mp_id)
        );
        assert_eq!(
            map.get_keyframe(kf2_id).unwrap().get_map_point(3),
            Some(mp_id)
        );

        // Check covisibility was updated
        let kf1 = map.get_keyframe(kf1_id).unwrap();
        assert_eq!(kf1.get_covisibility_weight(kf2_id), 1);

        let kf2 = map.get_keyframe(kf2_id).unwrap();
        assert_eq!(kf2.get_covisibility_weight(kf1_id), 1);
    }

    #[test]
    fn test_disassociate() {
        let mut map = create_test_map();

        // Setup: two KeyFrames observing one MapPoint
        let kf1_id = map.create_keyframe(
            0,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );
        let kf2_id = map.create_keyframe(
            1000000,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );
        let mp_id = map.create_map_point(Vector3::zeros(), Mat::default(), kf1_id);

        map.associate(kf1_id, 0, mp_id);
        map.associate(kf2_id, 0, mp_id);

        // Covisibility should be 1
        assert_eq!(
            map.get_keyframe(kf1_id)
                .unwrap()
                .get_covisibility_weight(kf2_id),
            1
        );

        // Disassociate kf1
        let removed = map.disassociate(kf1_id, 0);
        assert_eq!(removed, Some(mp_id));

        // MapPoint should have only kf2 observation
        let mp = map.get_map_point(mp_id).unwrap();
        assert_eq!(mp.num_observations(), 1);
        assert!(mp.observations.contains_key(&kf2_id));

        // kf1 should have no MapPoint at feature 0
        assert_eq!(map.get_keyframe(kf1_id).unwrap().get_map_point(0), None);

        // Covisibility should be removed (was 1, now 0)
        assert_eq!(
            map.get_keyframe(kf1_id)
                .unwrap()
                .get_covisibility_weight(kf2_id),
            0
        );
    }

    #[test]
    fn test_cull_bad_map_points() {
        let mut map = create_test_map();

        let kf_id = map.create_keyframe(
            0,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );

        // Create a good point (3 observations, good found ratio)
        let good_mp_id = map.create_map_point(Vector3::zeros(), Mat::default(), kf_id);
        if let Some(mp) = map.get_map_point_mut(good_mp_id) {
            mp.add_observation(KeyFrameId::new(0), 0);
            mp.add_observation(KeyFrameId::new(1), 0);
            mp.add_observation(KeyFrameId::new(2), 0);
            mp.visible_count = 10;
            mp.found_count = 8; // 80% found ratio
        }

        // Create a bad point (poor found ratio)
        let bad_mp_id = map.create_map_point(Vector3::new(1.0, 1.0, 1.0), Mat::default(), kf_id);
        if let Some(mp) = map.get_map_point_mut(bad_mp_id) {
            mp.add_observation(KeyFrameId::new(0), 1);
            mp.add_observation(KeyFrameId::new(1), 1);
            mp.add_observation(KeyFrameId::new(2), 1);
            mp.visible_count = 100;
            mp.found_count = 5; // 5% found ratio
        }

        // Cull with 25% minimum found ratio
        let culled = map.cull_bad_map_points(0.25, 3);

        assert_eq!(culled, 1);
        assert!(map.get_map_point(good_mp_id).is_some());
        assert!(map.get_map_point(bad_mp_id).is_none());
    }

    #[test]
    fn test_get_local_keyframes() {
        let mut map = create_test_map();

        // Create KeyFrames with varying covisibility
        let kf0_id = map.create_keyframe(
            0,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );
        let kf1_id = map.create_keyframe(
            1,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );
        let kf2_id = map.create_keyframe(
            2,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );
        let kf3_id = map.create_keyframe(
            3,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![None; 10],
        );

        // Create MapPoints and associations to build covisibility
        // kf0 shares 3 points with kf1, 1 point with kf2, 2 points with kf3
        for i in 0..3 {
            let mp_id = map.create_map_point(Vector3::zeros(), Mat::default(), kf0_id);
            map.associate(kf0_id, i, mp_id);
            map.associate(kf1_id, i, mp_id);
        }
        for i in 3..4 {
            let mp_id = map.create_map_point(Vector3::zeros(), Mat::default(), kf0_id);
            map.associate(kf0_id, i, mp_id);
            map.associate(kf2_id, i, mp_id);
        }
        for i in 4..6 {
            let mp_id = map.create_map_point(Vector3::zeros(), Mat::default(), kf0_id);
            map.associate(kf0_id, i, mp_id);
            map.associate(kf3_id, i, mp_id);
        }

        // Get best 2 covisibles for kf0
        let local = map.get_local_keyframes(kf0_id, 2);
        assert_eq!(local.len(), 2);
        assert_eq!(local[0], kf1_id); // 3 shared points
        assert_eq!(local[1], kf3_id); // 2 shared points
    }
}
