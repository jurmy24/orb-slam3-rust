//! MapPoint - A 3D landmark observed by KeyFrames.
//!
//! MapPoints are the fundamental 3D structure elements in the map.
//! Each MapPoint tracks which KeyFrames observe it, maintains quality
//! metrics for culling decisions, and stores viewing geometry constraints.

use std::collections::HashMap;

use nalgebra::Vector3;
use opencv::core::Mat;

use super::types::{KeyFrameId, MapPointId};

/// A 3D map point (landmark) observed by one or more KeyFrames.
///
/// MapPoints form the sparse 3D reconstruction of the environment.
/// They track observations from KeyFrames and maintain quality metrics
/// used for culling bad points.
#[derive(Clone)]
pub struct MapPoint {
    /// Unique identifier for this MapPoint.
    pub id: MapPointId,

    /// 3D position in world frame.
    pub position: Vector3<f64>,

    /// Representative ORB descriptor (cloned from best observation).
    /// IMPORTANT: This must be a cloned Mat, not a row view, to avoid dangling references.
    pub descriptor: Mat,

    /// KeyFrames observing this point, mapped to feature index in that KeyFrame.
    /// observation[kf_id] = feature_idx means keyframe kf_id sees this point at feature_idx.
    pub observations: HashMap<KeyFrameId, usize>,

    /// Mean viewing direction (unit vector, world frame).
    /// Computed as average of all observation directions.
    pub normal: Vector3<f64>,

    /// Minimum distance at which this point can be reliably observed.
    /// Based on ORB scale pyramid - points have scale invariance limits.
    pub min_distance: f64,

    /// Maximum distance at which this point can be reliably observed.
    pub max_distance: f64,

    /// Number of times this point was visible in a frame (in frustum).
    pub visible_count: u32,

    /// Number of times this point was successfully matched/found.
    pub found_count: u32,

    /// KeyFrame that first created this MapPoint (reference frame).
    pub first_kf_id: KeyFrameId,

    /// Whether this point is marked as bad (should be removed).
    pub is_bad: bool,
}

impl MapPoint {
    /// Create a new MapPoint.
    ///
    /// # Arguments
    /// * `id` - Unique identifier
    /// * `position` - 3D position in world frame
    /// * `descriptor` - ORB descriptor (should be cloned, not a view)
    /// * `first_kf_id` - KeyFrame that created this point
    pub fn new(
        id: MapPointId,
        position: Vector3<f64>,
        descriptor: Mat,
        first_kf_id: KeyFrameId,
    ) -> Self {
        Self {
            id,
            position,
            descriptor,
            observations: HashMap::new(),
            normal: Vector3::zeros(),
            min_distance: 0.0,
            max_distance: f64::INFINITY,
            visible_count: 0,
            found_count: 0,
            first_kf_id,
            is_bad: false,
        }
    }

    /// Add an observation from a KeyFrame.
    ///
    /// # Arguments
    /// * `kf_id` - KeyFrame that observes this point
    /// * `feature_idx` - Index of the feature in the KeyFrame's feature list
    pub fn add_observation(&mut self, kf_id: KeyFrameId, feature_idx: usize) {
        self.observations.insert(kf_id, feature_idx);
    }

    /// Remove an observation.
    ///
    /// Returns true if the observation existed and was removed.
    pub fn erase_observation(&mut self, kf_id: KeyFrameId) -> bool {
        self.observations.remove(&kf_id).is_some()
    }

    /// Get the number of KeyFrames observing this point.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Compute the found ratio: found_count / visible_count.
    ///
    /// Returns 1.0 if visible_count is 0 (avoid division by zero,
    /// and new points shouldn't be penalized).
    pub fn found_ratio(&self) -> f64 {
        if self.visible_count == 0 {
            1.0
        } else {
            self.found_count as f64 / self.visible_count as f64
        }
    }

    /// Increment visible count (point was in camera frustum).
    pub fn increase_visible(&mut self) {
        self.visible_count += 1;
    }

    /// Increment found count (point was successfully matched).
    pub fn increase_found(&mut self) {
        self.found_count += 1;
    }

    /// Check if this MapPoint should be culled based on quality metrics.
    ///
    /// A point is bad if:
    /// - It has too few observations (< min_observations)
    /// - Its found ratio is too low (< min_found_ratio)
    ///
    /// # Arguments
    /// * `min_found_ratio` - Minimum found/visible ratio (typically 0.25)
    /// * `min_observations` - Minimum number of observing KeyFrames (typically 3)
    pub fn should_cull(&self, min_found_ratio: f64, min_observations: usize) -> bool {
        if self.is_bad {
            return true;
        }

        // Too few observations
        if self.num_observations() < min_observations {
            return true;
        }

        // Poor tracking ratio
        if self.found_ratio() < min_found_ratio {
            return true;
        }

        false
    }

    /// Mark this point as bad.
    pub fn set_bad(&mut self) {
        self.is_bad = true;
    }

    /// Update the mean normal vector and distance bounds.
    ///
    /// Should be called after observations change.
    /// Requires access to KeyFrame positions, so this is typically called
    /// from the Map container which has access to both.
    ///
    /// # Arguments
    /// * `kf_positions` - Iterator of (KeyFrameId, position) for observing KeyFrames
    /// * `scale_factor` - ORB scale factor per level (typically 1.2)
    /// * `num_levels` - Number of scale levels in ORB pyramid (typically 8)
    pub fn update_normal_and_depth<'a>(
        &mut self,
        kf_positions: impl Iterator<Item = (&'a KeyFrameId, &'a Vector3<f64>)>,
        scale_factor: f64,
        num_levels: u32,
    ) {
        let mut normal_sum = Vector3::zeros();
        let mut min_dist = f64::INFINITY;
        let mut max_dist = 0.0f64;

        for (_, kf_pos) in kf_positions {
            let dir = self.position - kf_pos;
            let dist = dir.norm();

            if dist > 1e-10 {
                normal_sum += dir / dist;
                min_dist = min_dist.min(dist);
                max_dist = max_dist.max(dist);
            }
        }

        let norm = normal_sum.norm();
        if norm > 1e-10 {
            self.normal = normal_sum / norm;
        }

        // Scale bounds by ORB pyramid range
        let scale_range = scale_factor.powi(num_levels as i32 - 1);
        self.min_distance = min_dist / scale_range;
        self.max_distance = max_dist * scale_range;
    }

    /// Check if a viewing distance is within the valid range.
    pub fn is_in_distance_range(&self, distance: f64) -> bool {
        distance >= self.min_distance && distance <= self.max_distance
    }
}

impl std::fmt::Debug for MapPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapPoint")
            .field("id", &self.id)
            .field("position", &self.position)
            .field("observations", &self.observations.len())
            .field("visible_count", &self.visible_count)
            .field("found_count", &self.found_count)
            .field("is_bad", &self.is_bad)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Mat;

    fn create_test_mappoint() -> MapPoint {
        let descriptor = Mat::default();
        MapPoint::new(
            MapPointId::new(1),
            Vector3::new(1.0, 2.0, 3.0),
            descriptor,
            KeyFrameId::new(0),
        )
    }

    #[test]
    fn test_add_remove_observation() {
        let mut mp = create_test_mappoint();

        mp.add_observation(KeyFrameId::new(1), 5);
        mp.add_observation(KeyFrameId::new(2), 10);

        assert_eq!(mp.num_observations(), 2);
        assert_eq!(mp.observations.get(&KeyFrameId::new(1)), Some(&5));

        assert!(mp.erase_observation(KeyFrameId::new(1)));
        assert_eq!(mp.num_observations(), 1);
        assert!(!mp.erase_observation(KeyFrameId::new(1))); // Already removed
    }

    #[test]
    fn test_found_ratio() {
        let mut mp = create_test_mappoint();

        // No visibility yet - should return 1.0
        assert_eq!(mp.found_ratio(), 1.0);

        // 3 visible, 2 found = 0.667
        mp.visible_count = 3;
        mp.found_count = 2;
        assert!((mp.found_ratio() - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_should_cull() {
        let mut mp = create_test_mappoint();

        // Too few observations
        mp.add_observation(KeyFrameId::new(1), 0);
        mp.add_observation(KeyFrameId::new(2), 0);
        assert!(mp.should_cull(0.25, 3)); // Need 3, have 2

        mp.add_observation(KeyFrameId::new(3), 0);
        assert!(!mp.should_cull(0.25, 3)); // Now have 3

        // Poor found ratio
        mp.visible_count = 100;
        mp.found_count = 10; // 10% found ratio
        assert!(mp.should_cull(0.25, 3)); // 10% < 25%

        mp.found_count = 30; // 30% found ratio
        assert!(!mp.should_cull(0.25, 3)); // 30% > 25%

        // Marked as bad
        mp.set_bad();
        assert!(mp.should_cull(0.0, 0)); // Always cull if bad
    }

    #[test]
    fn test_distance_range() {
        let mut mp = create_test_mappoint();
        mp.min_distance = 0.5;
        mp.max_distance = 10.0;

        assert!(mp.is_in_distance_range(1.0));
        assert!(mp.is_in_distance_range(5.0));
        assert!(!mp.is_in_distance_range(0.3)); // Too close
        assert!(!mp.is_in_distance_range(15.0)); // Too far
    }
}
