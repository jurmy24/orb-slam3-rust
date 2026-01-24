//! Tracking thread: processes raw Frames + IMU and outputs KeyFrames.
//!
//! This module implements the TRACKING component from ORB-SLAM3 Figure 1:
//! - Frame processing (ORB extraction, stereo matching)
//! - IMU preintegration
//! - Initial pose estimation (PnP-RANSAC + motion model)
//! - Track local map (projection-based search)
//! - New KeyFrame decision

pub mod frame;
pub mod tracking_frame;
pub mod keyframe_decision;
pub mod local_map;
pub mod motion_model;
pub mod pose_estimation;
pub mod state;
pub mod tracker;
pub mod result;

pub use state::TrackingState;
pub use tracker::Tracker;
