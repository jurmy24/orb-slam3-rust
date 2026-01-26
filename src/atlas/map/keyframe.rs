//! KeyFrame - A selected frame with map structure relationships.
//!
//! KeyFrames are the core nodes of the SLAM graph. They contain:
//! - Sensor data (features, 3D points, IMU preintegration)
//! - Pose estimate (SE3 transform)
//! - Graph relationships (covisibility, spanning tree)
//!
//! The covisibility graph connects KeyFrames that share MapPoint observations,
//! while the spanning tree provides a minimal connected structure for efficient
//! optimization during loop closure.

use std::collections::{HashMap, HashSet};

use nalgebra::Vector3;
use opencv::core::{KeyPoint, Mat, Vector};
use opencv::prelude::KeyPointTraitConst;

use crate::geometry::SE3;
use crate::imu::{ImuBias, PreintegratedState};

use super::types::{KeyFrameId, MapPointId};

/// A KeyFrame in the SLAM map.
///
/// KeyFrames are selected frames that become nodes in the map graph.
/// They store visual features, 3D points, pose estimates, and maintain
/// relationships with other KeyFrames via covisibility and spanning tree edges.
#[derive(Clone)]
pub struct KeyFrame {
    /// Unique identifier for this KeyFrame.
    pub id: KeyFrameId,

    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,

    /// Pose: transform from camera to world (T_wc).
    /// To transform a point from camera to world: p_world = pose.transform_point(p_cam)
    pub pose: SE3,

    /// Velocity in world frame (m/s).
    pub velocity: Vector3<f64>,

    // ─────────────────────────────────────────────────────────────────────────
    // Visual Features
    // ─────────────────────────────────────────────────────────────────────────
    /// Detected keypoints (from left image in stereo).
    pub keypoints: Vector<KeyPoint>,

    /// ORB descriptors for each keypoint.
    pub descriptors: Mat,

    /// 3D points in camera frame (from stereo triangulation).
    /// None if the point couldn't be triangulated (e.g., too far, poor disparity).
    pub points_cam: Vec<Option<Vector3<f64>>>,

    /// Optional Bag-of-Words representation used for place recognition.
    ///
    /// The vocabulary and BoW computation live outside of this type; the
    /// KeyFrame simply stores the resulting sparse histogram.
    pub bow_vector: Option<crate::atlas::keyframe_db::BowVector>,

    /// Optional Feature Vector for accelerated feature matching.
    ///
    /// Groups feature indices by vocabulary node at level L-4 (typically ~10 groups).
    /// This enables O(N·k) matching instead of O(N²) by only comparing features
    /// that share the same vocabulary node.
    pub feature_vector: Option<crate::vocabulary::FeatureVector>,

    // ─────────────────────────────────────────────────────────────────────────
    // Map Associations
    // ─────────────────────────────────────────────────────────────────────────
    /// Feature index → MapPoint association.
    /// If map_point_ids[i] = Some(mp_id), feature i is associated with MapPoint mp_id.
    pub map_point_ids: Vec<Option<MapPointId>>,

    // ─────────────────────────────────────────────────────────────────────────
    // IMU Data
    // ─────────────────────────────────────────────────────────────────────────
    /// Preintegrated IMU measurements from the previous KeyFrame to this one.
    /// None for the first KeyFrame or if IMU data is unavailable.
    pub imu_preintegrated: Option<PreintegratedState>,

    /// IMU bias estimate at this KeyFrame.
    pub imu_bias: ImuBias,

    // ─────────────────────────────────────────────────────────────────────────
    // Temporal Links (for IMU)
    // ─────────────────────────────────────────────────────────────────────────
    /// Previous KeyFrame in temporal order (for IMU preintegration chain).
    /// None for the first KeyFrame in the map.
    pub prev_kf: Option<KeyFrameId>,

    /// Next KeyFrame in temporal order.
    /// None for the most recent KeyFrame.
    pub next_kf: Option<KeyFrameId>,

    // ─────────────────────────────────────────────────────────────────────────
    // Covisibility Graph
    // ─────────────────────────────────────────────────────────────────────────
    /// Covisibility weights: connected KeyFrame → number of shared MapPoints.
    /// This is the adjacency list for the covisibility graph.
    covisibility_weights: HashMap<KeyFrameId, usize>,

    /// Cached ordered list of covisible KeyFrames (best first).
    /// Invalidated when covisibility_weights changes.
    ordered_covisibles: Vec<(KeyFrameId, usize)>,

    /// Flag indicating ordered_covisibles needs rebuilding.
    covisibility_dirty: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Spanning Tree
    // ─────────────────────────────────────────────────────────────────────────
    /// Parent KeyFrame in the spanning tree.
    /// None for the root KeyFrame (typically the first in the map).
    pub parent_id: Option<KeyFrameId>,

    /// Children KeyFrames in the spanning tree.
    pub children_ids: HashSet<KeyFrameId>,

    // ─────────────────────────────────────────────────────────────────────────
    // Status
    // ─────────────────────────────────────────────────────────────────────────
    /// Whether this KeyFrame is marked as bad (to be removed).
    pub is_bad: bool,
}

impl KeyFrame {
    /// Create a new KeyFrame.
    ///
    /// # Arguments
    /// * `id` - Unique identifier
    /// * `timestamp_ns` - Timestamp in nanoseconds
    /// * `pose` - Camera-to-world transform (T_wc)
    /// * `keypoints` - Detected feature keypoints
    /// * `descriptors` - ORB descriptors
    /// * `points_cam` - 3D points in camera frame (from stereo)
    pub fn new(
        id: KeyFrameId,
        timestamp_ns: u64,
        pose: SE3,
        keypoints: Vector<KeyPoint>,
        descriptors: Mat,
        points_cam: Vec<Option<Vector3<f64>>>,
    ) -> Self {
        // Use max of keypoints and points_cam length to handle both
        // (in production they should match, but this handles test cases)
        let num_features = keypoints.len().max(points_cam.len());

        Self {
            id,
            timestamp_ns,
            pose,
            velocity: Vector3::zeros(),
            keypoints,
            descriptors,
            points_cam,
            map_point_ids: vec![None; num_features],
            imu_preintegrated: None,
            imu_bias: ImuBias::zero(),
            prev_kf: None,
            next_kf: None,
            bow_vector: None,
            feature_vector: None,
            covisibility_weights: HashMap::new(),
            ordered_covisibles: Vec::new(),
            covisibility_dirty: false,
            parent_id: None,
            children_ids: HashSet::new(),
            is_bad: false,
        }
    }

    /// Get the camera position in world frame.
    pub fn camera_center(&self) -> Vector3<f64> {
        self.pose.translation.clone()
    }

    /// Get the camera-to-world rotation.
    pub fn rotation_cw(&self) -> nalgebra::UnitQuaternion<f64> {
        self.pose.rotation
    }

    /// Get the world-to-camera transform (inverse of pose).
    pub fn pose_wc_inverse(&self) -> SE3 {
        self.pose.inverse()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bag-of-Words accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// Set the Bag-of-Words vector for this keyframe.
    ///
    /// The caller is responsible for computing the BoW representation using
    /// a vocabulary. This method simply stores the resulting sparse vector.
    pub fn set_bow_vector(&mut self, bow: crate::atlas::keyframe_db::BowVector) {
        self.bow_vector = Some(bow);
    }

    /// Get a reference to the Bag-of-Words vector, if available.
    pub fn bow_vector(&self) -> Option<&crate::atlas::keyframe_db::BowVector> {
        self.bow_vector.as_ref()
    }

    /// Set the Feature Vector for this keyframe.
    ///
    /// The Feature Vector groups feature indices by vocabulary node,
    /// enabling accelerated feature matching during triangulation.
    pub fn set_feature_vector(&mut self, fv: crate::vocabulary::FeatureVector) {
        self.feature_vector = Some(fv);
    }

    /// Get a reference to the Feature Vector, if available.
    pub fn feature_vector(&self) -> Option<&crate::vocabulary::FeatureVector> {
        self.feature_vector.as_ref()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Map Point Associations
    // ─────────────────────────────────────────────────────────────────────────

    /// Get the MapPoint ID for a given feature index.
    pub fn get_map_point(&self, feature_idx: usize) -> Option<MapPointId> {
        self.map_point_ids.get(feature_idx).copied().flatten()
    }

    /// Associate a feature with a MapPoint.
    ///
    /// Returns the previous MapPoint ID if there was one.
    pub fn set_map_point(&mut self, feature_idx: usize, mp_id: MapPointId) -> Option<MapPointId> {
        if feature_idx >= self.map_point_ids.len() {
            return None;
        }
        let prev = self.map_point_ids[feature_idx];
        self.map_point_ids[feature_idx] = Some(mp_id);
        prev
    }

    /// Remove the association for a feature.
    pub fn erase_map_point(&mut self, feature_idx: usize) -> Option<MapPointId> {
        if feature_idx >= self.map_point_ids.len() {
            return None;
        }
        self.map_point_ids[feature_idx].take()
    }

    /// Get all associated MapPoint IDs with their feature indices.
    pub fn get_map_point_indices(&self) -> impl Iterator<Item = (usize, MapPointId)> + '_ {
        self.map_point_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, mp)| mp.map(|id| (idx, id)))
    }

    /// Count the number of associated MapPoints.
    pub fn num_map_points(&self) -> usize {
        self.map_point_ids.iter().filter(|mp| mp.is_some()).count()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Covisibility Graph
    // ─────────────────────────────────────────────────────────────────────────

    /// Add or update a covisibility connection.
    ///
    /// # Arguments
    /// * `kf_id` - The connected KeyFrame
    /// * `weight` - Number of shared MapPoints
    pub fn add_covisibility(&mut self, kf_id: KeyFrameId, weight: usize) {
        if kf_id == self.id {
            return; // Don't connect to self
        }
        self.covisibility_weights.insert(kf_id, weight);
        self.covisibility_dirty = true;
    }

    /// Update the covisibility weight for a connection.
    ///
    /// Does nothing if the connection doesn't exist.
    pub fn update_covisibility(&mut self, kf_id: KeyFrameId, weight: usize) {
        if let Some(w) = self.covisibility_weights.get_mut(&kf_id) {
            *w = weight;
            self.covisibility_dirty = true;
        }
    }

    /// Remove a covisibility connection.
    pub fn erase_covisibility(&mut self, kf_id: KeyFrameId) {
        if self.covisibility_weights.remove(&kf_id).is_some() {
            self.covisibility_dirty = true;
        }
    }

    /// Get the covisibility weight with another KeyFrame.
    pub fn get_covisibility_weight(&self, kf_id: KeyFrameId) -> usize {
        self.covisibility_weights.get(&kf_id).copied().unwrap_or(0)
    }

    /// Get all covisible KeyFrames (unordered).
    pub fn get_covisibles(&self) -> impl Iterator<Item = &KeyFrameId> {
        self.covisibility_weights.keys()
    }

    /// Get read-only access to the covisibility weights map.
    pub fn covisibility_weights(&self) -> &HashMap<KeyFrameId, usize> {
        &self.covisibility_weights
    }

    /// Get the N best covisible KeyFrames (most shared points first).
    ///
    /// This rebuilds the sorted cache if needed.
    pub fn get_best_covisibles(&mut self, n: usize) -> Vec<KeyFrameId> {
        self.ensure_ordered_covisibles();
        self.ordered_covisibles
            .iter()
            .take(n)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get covisible KeyFrames with at least `min_weight` shared points.
    pub fn get_covisibles_above_weight(&mut self, min_weight: usize) -> Vec<KeyFrameId> {
        self.ensure_ordered_covisibles();
        self.ordered_covisibles
            .iter()
            .take_while(|(_, w)| *w >= min_weight)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Rebuild the ordered covisibles list if dirty.
    fn ensure_ordered_covisibles(&mut self) {
        if !self.covisibility_dirty {
            return;
        }

        self.ordered_covisibles = self
            .covisibility_weights
            .iter()
            .map(|(id, w)| (*id, *w))
            .collect();
        self.ordered_covisibles.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by weight
        self.covisibility_dirty = false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Spanning Tree
    // ─────────────────────────────────────────────────────────────────────────

    /// Set the parent KeyFrame in the spanning tree.
    pub fn set_parent(&mut self, parent_id: KeyFrameId) {
        self.parent_id = Some(parent_id);
    }

    /// Add a child KeyFrame in the spanning tree.
    pub fn add_child(&mut self, child_id: KeyFrameId) {
        self.children_ids.insert(child_id);
    }

    /// Remove a child KeyFrame from the spanning tree.
    pub fn erase_child(&mut self, child_id: KeyFrameId) {
        self.children_ids.remove(&child_id);
    }

    /// Check if this KeyFrame has a parent.
    pub fn has_parent(&self) -> bool {
        self.parent_id.is_some()
    }

    /// Check if this is a root KeyFrame (no parent).
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Status
    // ─────────────────────────────────────────────────────────────────────────

    /// Mark this KeyFrame as bad.
    pub fn set_bad(&mut self) {
        self.is_bad = true;
    }

    /// Get number of features in this KeyFrame.
    pub fn num_features(&self) -> usize {
        self.keypoints.len()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Spatial Queries
    // ─────────────────────────────────────────────────────────────────────────

    /// Get features within a radius around a point.
    ///
    /// Returns indices of keypoints within the specified radius of (u, v).
    /// Optionally filters by ORB pyramid level.
    ///
    /// # Arguments
    /// * `u` - X coordinate in image
    /// * `v` - Y coordinate in image
    /// * `radius` - Search radius in pixels
    /// * `min_level` - Minimum ORB pyramid level (inclusive), or None for no minimum
    /// * `max_level` - Maximum ORB pyramid level (inclusive), or None for no maximum
    ///
    /// # Returns
    /// Vector of feature indices within the search area
    pub fn get_features_in_area(
        &self,
        u: f64,
        v: f64,
        radius: f64,
        min_level: Option<i32>,
        max_level: Option<i32>,
    ) -> Vec<usize> {
        let mut indices = Vec::new();
        let radius_sq = radius * radius;

        for i in 0..self.keypoints.len() {
            if let Ok(kp) = self.keypoints.get(i) {
                // Check level bounds
                let octave = kp.octave();
                if let Some(min) = min_level {
                    if octave < min {
                        continue;
                    }
                }
                if let Some(max) = max_level {
                    if octave > max {
                        continue;
                    }
                }

                // Check distance
                let du = kp.pt().x as f64 - u;
                let dv = kp.pt().y as f64 - v;
                let dist_sq = du * du + dv * dv;

                if dist_sq <= radius_sq {
                    indices.push(i);
                }
            }
        }

        indices
    }
}

impl std::fmt::Debug for KeyFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyFrame")
            .field("id", &self.id)
            .field("timestamp_ns", &self.timestamp_ns)
            .field("num_features", &self.num_features())
            .field("num_map_points", &self.num_map_points())
            .field("covisibles", &self.covisibility_weights.len())
            .field("has_parent", &self.parent_id.is_some())
            .field("num_children", &self.children_ids.len())
            .field("is_bad", &self.is_bad)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Mat, Vector};

    fn create_test_keyframe(id: u64) -> KeyFrame {
        KeyFrame::new(
            KeyFrameId::new(id),
            1000000 * id,
            SE3::identity(),
            Vector::new(),
            Mat::default(),
            vec![],
        )
    }

    fn create_keyframe_with_features(id: u64, num_features: usize) -> KeyFrame {
        let mut kf = create_test_keyframe(id);
        kf.map_point_ids = vec![None; num_features];
        kf.points_cam = vec![None; num_features];
        kf
    }

    #[test]
    fn test_map_point_association() {
        let mut kf = create_keyframe_with_features(1, 10);

        // Associate feature 3 with MapPoint 100
        kf.set_map_point(3, MapPointId::new(100));
        assert_eq!(kf.get_map_point(3), Some(MapPointId::new(100)));
        assert_eq!(kf.get_map_point(4), None);
        assert_eq!(kf.num_map_points(), 1);

        // Overwrite association
        let prev = kf.set_map_point(3, MapPointId::new(200));
        assert_eq!(prev, Some(MapPointId::new(100)));
        assert_eq!(kf.get_map_point(3), Some(MapPointId::new(200)));

        // Erase association
        let erased = kf.erase_map_point(3);
        assert_eq!(erased, Some(MapPointId::new(200)));
        assert_eq!(kf.get_map_point(3), None);
    }

    #[test]
    fn test_covisibility_graph() {
        let mut kf = create_test_keyframe(1);

        kf.add_covisibility(KeyFrameId::new(2), 50);
        kf.add_covisibility(KeyFrameId::new(3), 100);
        kf.add_covisibility(KeyFrameId::new(4), 25);

        assert_eq!(kf.get_covisibility_weight(KeyFrameId::new(2)), 50);
        assert_eq!(kf.get_covisibility_weight(KeyFrameId::new(5)), 0); // Not connected

        // Get best covisibles (should be ordered by weight descending)
        let best = kf.get_best_covisibles(2);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0], KeyFrameId::new(3)); // 100 shared
        assert_eq!(best[1], KeyFrameId::new(2)); // 50 shared

        // Get above threshold
        let above_30 = kf.get_covisibles_above_weight(30);
        assert_eq!(above_30.len(), 2); // 100 and 50, not 25
    }

    #[test]
    fn test_covisibility_no_self_connection() {
        let mut kf = create_test_keyframe(1);
        kf.add_covisibility(KeyFrameId::new(1), 100); // Try to connect to self
        assert_eq!(kf.get_covisibility_weight(KeyFrameId::new(1)), 0);
    }

    #[test]
    fn test_spanning_tree() {
        let mut kf1 = create_test_keyframe(1);
        let mut kf2 = create_test_keyframe(2);

        assert!(kf1.is_root());
        assert!(!kf1.has_parent());

        // Set kf1 as parent of kf2
        kf2.set_parent(KeyFrameId::new(1));
        kf1.add_child(KeyFrameId::new(2));

        assert!(kf2.has_parent());
        assert!(!kf2.is_root());
        assert_eq!(kf2.parent_id, Some(KeyFrameId::new(1)));
        assert!(kf1.children_ids.contains(&KeyFrameId::new(2)));
    }
}
