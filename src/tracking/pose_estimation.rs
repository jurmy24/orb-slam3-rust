//! Pose estimation utilities for the tracking thread.
//!
//! This module provides helpers for initial pose estimation using:
//! - Motion model (constant velocity)
//! - IMU preintegration
//! - PnP-RANSAC refinement

use nalgebra::Vector3;
use opencv::core::Point2f;

use crate::geometry::{solve_pnp_ransac, SE3};
use crate::tracking::frame::CameraModel;
use crate::tracking::motion_model::MotionModel;

/// Estimate pose using available information.
///
/// Priority:
/// 1. If we have 3D-2D correspondences, use PnP-RANSAC with prior
/// 2. Otherwise, use motion model prediction
/// 3. Fall back to identity if nothing available
pub fn estimate_pose(
    points3d: &[Vector3<f64>],
    points2d: &[Point2f],
    camera: &CameraModel,
    motion_model: &MotionModel,
    imu_prior: Option<&SE3>,
) -> SE3 {
    // Choose the best prior available
    let motion_prediction = motion_model.predict();
    let prior = imu_prior.or(motion_prediction.as_ref());

    if !points3d.is_empty() && points3d.len() >= 4 {
        // Have enough correspondences for PnP
        if let Ok(pose) = solve_pnp_ransac(points3d, points2d, camera, prior) {
            return pose;
        }
    }

    // Fall back to prior or identity
    prior.cloned().unwrap_or_else(SE3::identity)
}
