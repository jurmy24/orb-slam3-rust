//! Track Local Map: projection-based search for map point correspondences.
//!
//! This module implements the "Track Local Map" step from ORB-SLAM3 Figure 1.
//! Given the current pose estimate, it projects visible map points into the
//! current frame and searches for matches.

use nalgebra::Vector3;
use opencv::prelude::*;

use crate::geometry::SE3;
use crate::tracking::frame::{CameraModel, StereoFrame};

/// Tracks the local map by projecting map points into the current frame.
pub struct LocalMapTracker {
    /// Camera model for projection.
    camera: CameraModel,
    /// Search radius in pixels for projection matching.
    search_radius: f64,
}

impl LocalMapTracker {
    pub fn new(camera: CameraModel) -> Self {
        Self {
            camera,
            search_radius: 15.0,
        }
    }

    /// Project a 3D point to 2D image coordinates.
    pub fn project(&self, point_world: &Vector3<f64>, pose: &SE3) -> Option<(f64, f64)> {
        // Transform to camera frame
        let pose_inv = pose.inverse();
        let p_cam = pose_inv.transform_point(point_world);

        // Check if point is in front of camera
        if p_cam.z <= 0.0 {
            return None;
        }

        // Project using pinhole model
        let u = self.camera.fx * p_cam.x / p_cam.z + self.camera.cx;
        let v = self.camera.fy * p_cam.y / p_cam.z + self.camera.cy;

        Some((u, v))
    }

    /// Search for map point matches in the current frame.
    ///
    /// # Arguments
    /// * `map_points` - Visible map points in world coordinates
    /// * `current_frame` - Current stereo frame with detected features
    /// * `pose` - Current camera pose estimate
    ///
    /// # Returns
    /// Vector of (map_point_idx, keypoint_idx) matches
    pub fn search_by_projection(
        &self,
        map_points: &[Vector3<f64>],
        current_frame: &StereoFrame,
        pose: &SE3,
    ) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();

        for (mp_idx, mp) in map_points.iter().enumerate() {
            if let Some((proj_u, proj_v)) = self.project(mp, pose) {
                // Find closest keypoint within search radius
                let mut best_dist = self.search_radius;
                let mut best_kp_idx = None;

                for (kp_idx, kp) in current_frame.left_features.keypoints.iter().enumerate() {
                    let du = kp.pt().x as f64 - proj_u;
                    let dv = kp.pt().y as f64 - proj_v;
                    let dist = (du * du + dv * dv).sqrt();

                    if dist < best_dist {
                        best_dist = dist;
                        best_kp_idx = Some(kp_idx);
                    }
                }

                if let Some(kp_idx) = best_kp_idx {
                    matches.push((mp_idx, kp_idx));
                }
            }
        }

        matches
    }
}
