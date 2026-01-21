//! Tracking `Frame` representation.
//!
//! This is distinct from [`frame::StereoFrame`], which is the low-level
//! result of stereo processing (images + raw features). A `Frame` here is
//! what the Tracker operates on: it owns a subset of features, optional
//! stereo depth, a Bag-of-Words vector and temporary associations to
//! `MapPoint`s.

use nalgebra::Vector3;
use opencv::core::{KeyPoint, Vector};

use crate::atlas::keyframe_db::BowVector;
use crate::atlas::map::MapPointId;
use crate::tracking::frame::features::FeatureSet;

/// A frame being tracked (not yet a KeyFrame).
pub struct Frame {
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Visual features (typically taken from the left image in stereo).
    pub features: FeatureSet,
    /// 3D points in camera frame (from stereo triangulation).
    /// None if the point couldn't be triangulated.
    pub points_cam: Vec<Option<Vector3<f64>>>,
    /// Optional Bag-of-Words representation for place recognition.
    pub bow_vector: Option<BowVector>,
    /// Temporary MapPoint associations for tracking:
    /// map_point_matches[i] = Some(mp_id) if feature i is associated.
    pub map_point_matches: Vec<Option<MapPointId>>,
}

impl Frame {
    /// Construct a new tracking frame from raw stereo processing output.
    pub fn from_stereo(stereo: crate::tracking::frame::StereoFrame) -> Self {
        let n_feats = stereo.left_features.keypoints.len();
        Self {
            timestamp_ns: stereo.timestamp_ns,
            features: stereo.left_features,
            points_cam: stereo.points_cam,
            bow_vector: None,
            map_point_matches: vec![None; n_feats],
        }
    }

    /// Number of features in this frame.
    pub fn num_features(&self) -> usize {
        self.features.keypoints.len()
    }

    /// Convenience accessor for keypoints.
    pub fn keypoints(&self) -> &Vector<KeyPoint> {
        &self.features.keypoints
    }
}

