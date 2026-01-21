//! Rerun-based visualization for stereo SLAM.
//!
//! Entity hierarchy:
//!     camera/
//!         left/
//!             image/              - Rectified left image
//!                 unmatched       - Unmatched features (green, radius 6.0)
//!                 matched         - Matched features (red, radius 10.0)
//!     world/
//!         camera      - Camera pose transform
//!         trajectory  - Camera trajectory line
//!         points      - 3D point cloud (colored by depth)

use nalgebra::Vector3;
use opencv::core::{DMatch, KeyPoint, Mat, Vector};
use opencv::prelude::*;
use rerun::{RecordingStream, external::glam};

use crate::atlas::map::KeyFrame;
use crate::geometry::SE3;
use crate::tracking::TrackingState;
use crate::tracking::frame::StereoFrame;
use crate::tracking::result::{TimingStats, TrackingMetrics};

pub struct RerunVisualizer {
    rec: RecordingStream,
}

impl RerunVisualizer {
    pub fn new(app_name: &str) -> Self {
        let rec = rerun::RecordingStreamBuilder::new(app_name)
            .spawn()
            .expect("Failed to spawn rerun viewer");

        // Set up coordinate system (Right-Down-Forward, typical camera convention) for the world frame
        rec.log_static("world", &rerun::ViewCoordinates::RDF()).ok();

        Self { rec }
    }

    /// Set the current timestamp for all subsequent logs
    pub fn set_time(&self, timestamp_ns: u64) {
        let timestamp_sec = timestamp_ns as f64 / 1e9;
        self.rec.set_duration_secs("timestamp", timestamp_sec);
    }

    /// Log a complete stereo frame with images, features, and 3D points
    pub fn log_stereo_frame(&self, frame: &StereoFrame, left_image: &Mat, right_image: &Mat) {
        self.set_time(frame.timestamp_ns);
        self.log_images(left_image, right_image);
        self.log_features_and_matches(
            &frame.left_features.keypoints,
            &frame.right_features.keypoints,
            &frame.matches_lr,
        );
        self.log_3d_points_from_frame(frame);
    }

    fn log_images(&self, left: &Mat, _right: &Mat) {
        // Convert Mat to image data (only log left image)
        if let Ok((data, width, height)) = mat_to_image_data(left) {
            self.rec
                .log(
                    "camera/left/image",
                    &rerun::Image::from_l8(data, [width, height]),
                )
                .ok();
        }
    }

    fn log_features_and_matches(
        &self,
        left_kps: &Vector<KeyPoint>,
        _right_kps: &Vector<KeyPoint>,
        matches: &Vector<DMatch>,
    ) {
        // Build set of matched indices
        let mut matched_indices = std::collections::HashSet::new();
        for m in matches.iter() {
            matched_indices.insert(m.query_idx as usize);
        }

        let mut unmatched_pts: Vec<[f32; 2]> = Vec::new();
        let mut matched_pts: Vec<[f32; 2]> = Vec::new();

        for (idx, kp) in left_kps.iter().enumerate() {
            let pt = [kp.pt().x, kp.pt().y];
            if matched_indices.contains(&idx) {
                matched_pts.push(pt);
            } else {
                unmatched_pts.push(pt);
            }
        }

        // Log unmatched features (green) - smaller
        if !unmatched_pts.is_empty() {
            self.rec
                .log(
                    "camera/left/image/unmatched",
                    &rerun::Points2D::new(unmatched_pts)
                        .with_colors([[0u8, 255, 0]]) // Green
                        .with_radii([6.0f32]),
                )
                .ok();
        }

        // Log matched features (red) - larger
        if !matched_pts.is_empty() {
            self.rec
                .log(
                    "camera/left/image/matched",
                    &rerun::Points2D::new(matched_pts)
                        .with_colors([[255u8, 0, 0]]) // Red
                        .with_radii([10.0f32]),
                )
                .ok();
        }
    }

    fn log_3d_points_from_frame(&self, frame: &StereoFrame) {
        // Collect valid 3D points
        let valid_points: Vec<Vector3<f64>> = frame
            .points_cam
            .iter()
            .filter_map(|p| *p)
            .filter(|p| p.x.is_finite() && p.y.is_finite() && p.z.is_finite())
            .filter(|p| p.x.abs() < 100.0 && p.y.abs() < 100.0 && p.z > 0.1 && p.z < 100.0)
            .collect();

        if valid_points.is_empty() {
            return;
        }

        // Compute depth range for coloring
        let depths: Vec<f64> = valid_points.iter().map(|p| p.z).collect();
        let mut sorted_depths = depths.clone();
        sorted_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let depth_min = sorted_depths[sorted_depths.len() * 5 / 100]; // 5th percentile
        let depth_max = sorted_depths[sorted_depths.len() * 95 / 100]; // 95th percentile
        let depth_range = (depth_max - depth_min).max(0.1);

        // Convert to rerun format with depth-based colors
        let pts: Vec<[f32; 3]> = valid_points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        // Color by depth: blue (close) -> green -> red (far)
        let colors: Vec<[u8; 3]> = depths
            .iter()
            .map(|&d| {
                let normalized = ((d - depth_min) / depth_range).clamp(0.0, 1.0);
                let r = (normalized * 255.0) as u8;
                let g = ((1.0 - (normalized - 0.5).abs() * 2.0) * 255.0) as u8;
                let b = ((1.0 - normalized) * 255.0) as u8;
                [r, g, b]
            })
            .collect();

        self.rec
            .log(
                "world/points",
                &rerun::Points3D::new(pts)
                    .with_colors(colors)
                    .with_radii([0.02f32]),
            )
            .ok();
    }

    pub fn log_pose(&self, pose: &SE3) {
        let translation = glam::Vec3::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        );
        let rotation = glam::Quat::from_xyzw(
            pose.rotation.coords.x as f32,
            pose.rotation.coords.y as f32,
            pose.rotation.coords.z as f32,
            pose.rotation.w as f32,
        );
        self.rec
            .log(
                "world/camera",
                &rerun::Transform3D::from_translation_rotation(translation, rotation),
            )
            .ok();
    }

    pub fn log_trajectory(&self, positions: &[Vector3<f64>]) {
        if positions.len() < 2 {
            return;
        }
        let pts: Vec<[f32; 3]> = positions
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        // Yellow trajectory line
        self.rec
            .log(
                "world/trajectory",
                &rerun::LineStrips3D::new([pts.clone()])
                    .with_colors([[255u8, 255, 0]])
                    .with_radii([0.01f32]),
            )
            .ok();

        // Current position as cyan point
        if let Some(current) = pts.last() {
            self.rec
                .log(
                    "world/trajectory/current",
                    &rerun::Points3D::new([*current])
                        .with_colors([[0u8, 255, 255]]) // Cyan
                        .with_radii([0.05f32]),
                )
                .ok();
        }
    }

    pub fn log_map_points(&self, points: &[Vector3<f64>]) {
        if points.is_empty() {
            return;
        }
        let pts: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.rec
            .log(
                "world/map",
                &rerun::Points3D::new(pts)
                    .with_colors([[0u8, 200, 255]])
                    .with_radii([0.03f32]),
            )
            .ok();
    }

    /// Log scalar tracking state and emit a textual event.
    pub fn log_tracking_state(&self, state: TrackingState, frame_idx: usize, map_index: usize) {
        let code = match state {
            TrackingState::NotInitialized => 0,
            TrackingState::Ok => 1,
            TrackingState::RecentlyLost => 2,
            TrackingState::Lost => 3,
        };
        self.rec
            .log("metrics/state", &rerun::Scalars::single(code as f64))
            .ok();

        let msg = format!("frame={} map={} state={:?}", frame_idx, map_index, state);
        self.log_event("tracking", &msg);
    }

    /// Log scalar tracking metrics as simple time series.
    pub fn log_tracking_metrics(&self, metrics: &TrackingMetrics) {
        self.rec
            .log(
                "metrics/n_features",
                &rerun::Scalars::single(metrics.n_features as f64),
            )
            .ok();
        self.rec
            .log(
                "metrics/n_matches",
                &rerun::Scalars::single(metrics.n_map_point_matches as f64),
            )
            .ok();
        self.rec
            .log(
                "metrics/n_inliers",
                &rerun::Scalars::single(metrics.n_inliers as f64),
            )
            .ok();
        self.rec
            .log(
                "metrics/inlier_ratio",
                &rerun::Scalars::single(metrics.inlier_ratio),
            )
            .ok();
        self.rec
            .log(
                "metrics/reproj_err_median",
                &rerun::Scalars::single(metrics.reproj_error_median_px),
            )
            .ok();
        self.rec
            .log(
                "metrics/delta_trans",
                &rerun::Scalars::single(metrics.delta_translation_m),
            )
            .ok();
        self.rec
            .log(
                "metrics/delta_rot",
                &rerun::Scalars::single(metrics.delta_rotation_deg),
            )
            .ok();
    }

    /// Log per-stage timing information.
    pub fn log_timing(&self, timing: &TimingStats) {
        self.rec
            .log(
                "metrics/timing/total_ms",
                &rerun::Scalars::single(timing.total_ms),
            )
            .ok();
        self.rec
            .log(
                "metrics/timing/extract_orb_ms",
                &rerun::Scalars::single(timing.extract_orb_ms),
            )
            .ok();
        self.rec
            .log(
                "metrics/timing/match_ms",
                &rerun::Scalars::single(timing.match_ms),
            )
            .ok();
        self.rec
            .log(
                "metrics/timing/solve_pnp_ms",
                &rerun::Scalars::single(timing.solve_pnp_ms),
            )
            .ok();
        self.rec
            .log(
                "metrics/timing/relocal_ms",
                &rerun::Scalars::single(timing.relocal_ms),
            )
            .ok();
    }

    /// Log a simple text event (e.g. LOST / RELOCALIZED).
    pub fn log_event(&self, category: &str, message: &str) {
        let full_msg = format!("[{}] {}", category, message);
        self.rec
            .log("events/log", &rerun::TextLog::new(full_msg))
            .ok();
    }

    /// Log features that were used for pose estimation.
    pub fn log_features_used(&self, points: &[[f32; 2]]) {
        if points.is_empty() {
            return;
        }
        self.rec
            .log(
                "camera/left/features_used",
                &rerun::Points2D::new(points.to_vec())
                    .with_colors([[0u8, 0, 255]]) // Blue
                    .with_radii([4.0f32]),
            )
            .ok();
    }

    /// Log inlier vs outlier feature locations in the left image.
    pub fn log_inliers_outliers(&self, inliers: &[[f32; 2]], outliers: &[[f32; 2]]) {
        if !inliers.is_empty() {
            self.rec
                .log(
                    "camera/left/inliers",
                    &rerun::Points2D::new(inliers.to_vec())
                        .with_colors([[0u8, 255, 0]]) // Green
                        .with_radii([4.0f32]),
                )
                .ok();
        }
        if !outliers.is_empty() {
            self.rec
                .log(
                    "camera/left/outliers",
                    &rerun::Points2D::new(outliers.to_vec())
                        .with_colors([[255u8, 0, 0]]) // Red
                        .with_radii([3.0f32]),
                )
                .ok();
        }
    }

    /// Log reprojection error as radius of points in the left image.
    pub fn log_reproj_errors(&self, points: &[[f32; 2]], errors: &[f64]) {
        if points.is_empty() || points.len() != errors.len() {
            return;
        }
        let radii: Vec<f32> = errors.iter().map(|e| (1.0 + e.min(10.0)) as f32).collect();
        self.rec
            .log(
                "camera/left/reproj_error",
                &rerun::Points2D::new(points.to_vec()).with_radii(radii),
            )
            .ok();
    }

    /// Log keyframe frustums as camera transforms in world space.
    pub fn log_keyframe_frustums(&self, keyframes: &[&KeyFrame]) {
        for kf in keyframes {
            let t = &kf.pose.translation;
            let q = &kf.pose.rotation;
            let translation = glam::Vec3::new(t.x as f32, t.y as f32, t.z as f32);
            let rotation = glam::Quat::from_xyzw(
                q.coords.x as f32,
                q.coords.y as f32,
                q.coords.z as f32,
                q.w as f32,
            );
            let path = format!("world/keyframes/{}", kf.id.0);
            self.rec
                .log(
                    path,
                    &rerun::Transform3D::from_translation_rotation(translation, rotation),
                )
                .ok();
        }
    }

    /// Log local map points for the current frame.
    pub fn log_local_map_points(&self, points: &[Vector3<f64>]) {
        if points.is_empty() {
            return;
        }
        let pts: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.rec
            .log(
                "world/local_points",
                &rerun::Points3D::new(pts)
                    .with_colors([[0u8, 255, 255]])
                    .with_radii([0.03f32]),
            )
            .ok();
    }

    /// Log inlier map points for the current frame.
    pub fn log_inlier_map_points(&self, points: &[Vector3<f64>]) {
        if points.is_empty() {
            return;
        }
        let pts: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.rec
            .log(
                "world/inlier_points",
                &rerun::Points3D::new(pts)
                    .with_colors([[0u8, 255, 0]])
                    .with_radii([0.035f32]),
            )
            .ok();
    }
}

/// Convert OpenCV Mat to image data (bytes, width, height)
fn mat_to_image_data(mat: &Mat) -> Result<(Vec<u8>, u32, u32), opencv::Error> {
    let rows = mat.rows() as u32;
    let cols = mat.cols() as u32;

    // Get raw data
    let data = mat.data_bytes()?;
    let image_data: Vec<u8> = data.to_vec();

    Ok((image_data, cols, rows))
}
