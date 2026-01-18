//! Map module - Core SLAM map data structures.
//!
//! This module contains:
//! - [`KeyFrame`] - Selected frames with poses and feature observations
//! - [`MapPoint`] - 3D landmarks observed by KeyFrames
//! - [`Map`] - Container managing KeyFrames, MapPoints, and their relationships
//!
//! # Architecture
//!
//! The map forms a bipartite graph structure:
//! - KeyFrames observe MapPoints (KF → MP via `map_point_ids`)
//! - MapPoints track their observers (MP → KF via `observations`)
//!
//! KeyFrames also maintain two graph structures:
//! - **Covisibility Graph**: Edges weighted by shared MapPoint count
//! - **Spanning Tree**: Minimal connected structure for loop closure
//!
//! # Example
//!
//! ```ignore
//! use orb_slam3::atlas::map::{Map, KeyFrameId, MapPointId};
//!
//! let mut map = Map::new();
//!
//! // Create a KeyFrame
//! let kf_id = map.create_keyframe(timestamp, pose, keypoints, descriptors, points_cam);
//!
//! // Create a MapPoint
//! let mp_id = map.create_map_point(position, descriptor, kf_id);
//!
//! // Associate KeyFrame feature with MapPoint (bidirectional)
//! map.associate(kf_id, feature_idx, mp_id);
//!
//! // Query local neighborhood
//! let local_kfs = map.get_local_keyframes(kf_id, 10);
//! ```

pub mod keyframe;
pub mod map;
pub mod map_point;
pub mod types;

pub use keyframe::KeyFrame;
pub use map::Map;
pub use map_point::MapPoint;
pub use types::{KeyFrameId, MapPointId};
