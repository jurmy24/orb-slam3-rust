//! Inter-thread message types.
//!
//! These types are sent between threads via channels to coordinate
//! SLAM processing.

use nalgebra::Vector3;
use opencv::core::{KeyPoint, Mat, Vector};

use crate::atlas::map::{KeyFrameId, MapPointId};
use crate::geometry::SE3;
use crate::imu::PreintegratedState;

/// Message sent from Tracking to Local Mapping when a new keyframe is created.
///
/// Contains all the data needed for Local Mapping to:
/// 1. Insert the keyframe into the map
/// 2. Associate existing map point matches
/// 3. Triangulate new map points from unmatched stereo features
pub struct NewKeyFrameMsg {
    /// The ID assigned to this keyframe by Tracking.
    pub keyframe_id: KeyFrameId,

    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,

    /// Pose estimate from Tracking (T_wc: camera to world).
    pub pose: SE3,

    /// Velocity estimate from IMU integration (world frame).
    pub velocity: Vector3<f64>,

    /// Detected keypoints (from left image in stereo).
    pub keypoints: Vector<KeyPoint>,

    /// ORB descriptors for each keypoint.
    pub descriptors: Mat,

    /// 3D points in camera frame from stereo triangulation.
    /// `points_cam[i]` corresponds to `keypoints[i]`.
    /// None if the point couldn't be triangulated (e.g., too far).
    pub points_cam: Vec<Option<Vector3<f64>>>,

    /// Map point associations from tracking.
    /// `matched_map_points[i] = Some(mp_id)` if feature i was matched
    /// to an existing map point during tracking.
    pub matched_map_points: Vec<Option<MapPointId>>,

    /// Preintegrated IMU measurements from the previous keyframe.
    /// None for the first keyframe.
    pub imu_preintegrated: Option<PreintegratedState>,
}
