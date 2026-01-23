//! Rerun-based visualization for stereo-inertial SLAM.
//!
//! Entity hierarchy:
//!     status               - Status bar (text document)
//!     camera/
//!         image            - Left camera image
//!         image/inliers    - Inlier features (circles, depth-colored or green)
//!         image/outliers   - Outlier features (triangles, depth-colored or red)
//!         image/unmatched  - Unmatched ORB features (gray dots)
//!         metrics          - Image metrics text
//!     world/
//!         camera           - Current camera frustum (large, bright)
//!         trajectory       - Trajectory line (gray, thin)
//!         keyframes/       - Keyframe boxes (blue)
//!         local_points     - Local map points (white dots)
//!         map_points       - Full map points (dim, LOD filtered)
//!         map_id           - Active map ID text
//!     plots/
//!         inlier_count     - Temporal plot
//!         reproj_error     - Temporal plot
//!         imu_residual     - Temporal plot
//!         bias_magnitude   - Temporal plot

use nalgebra::Vector3;
use opencv::core::Mat;
use opencv::prelude::*;
use rerun::{RecordingStream, external::glam};
use std::collections::HashSet;

use crate::atlas::map::KeyFrame;
use crate::geometry::SE3;
use crate::imu::ImuInitState;
use crate::tracking::TrackingState;
use crate::tracking::frame::StereoFrame;
use crate::tracking::result::{MatchInfo, TimingStats, TrackingMetrics};

pub struct RerunVisualizer {
    rec: RecordingStream,
    depth_coloring_enabled: bool,
    last_fps: f64,
    frame_times: Vec<f64>,
    start_timestamp_ns: Option<u64>,
}

impl RerunVisualizer {
    /// Path to the blueprint file (relative to crate root)
    const BLUEPRINT_PATH: &'static str = "src/viz/rust-orb-slam3-stereo-inertial.rbl";

    pub fn new(app_name: &str) -> Self {
        let rec = rerun::RecordingStreamBuilder::new(app_name)
            .spawn()
            .expect("Failed to spawn rerun viewer");

        // Load saved blueprint layout if it exists
        let blueprint_path = std::path::Path::new(Self::BLUEPRINT_PATH);
        if blueprint_path.exists() {
            if let Err(e) = rec.log_file_from_path(blueprint_path, None, false) {
                eprintln!("Warning: Failed to load blueprint: {}", e);
            }
        }

        // Set up coordinate system (Right-Down-Forward, typical camera convention) for the world frame
        rec.log_static("world", &rerun::ViewCoordinates::RDF()).ok();

        Self {
            rec,
            depth_coloring_enabled: true,
            last_fps: 0.0,
            frame_times: Vec::new(),
            start_timestamp_ns: None,
        }
    }

    /// Toggle depth-based coloring for 2D features.
    pub fn set_depth_coloring(&mut self, enabled: bool) {
        self.depth_coloring_enabled = enabled;
    }

    /// Set the current timestamp for all subsequent logs (uses relative time from first frame)
    pub fn set_time(&mut self, timestamp_ns: u64) {
        // Use relative time from first frame
        let start_ns = *self.start_timestamp_ns.get_or_insert(timestamp_ns);
        let relative_ns = timestamp_ns.saturating_sub(start_ns);
        let relative_sec = relative_ns as f64 / 1e9;
        self.rec.set_duration_secs("time", relative_sec);
    }

    /// Log status bar with tracking state, IMU state, and key metrics.
    pub fn log_status_bar(
        &self,
        tracking_state: TrackingState,
        imu_state: ImuInitState,
        metrics: &TrackingMetrics,
        fps: f64,
    ) {
        let state_indicator = match tracking_state {
            TrackingState::Ok => "**OK**",
            TrackingState::RecentlyLost => "**RECOVERING**",
            TrackingState::Lost => "**LOST**",
            TrackingState::NotInitialized => "**INIT**",
        };

        let imu_indicator = match imu_state {
            ImuInitState::NotInitialized => "NOT_INIT",
            ImuInitState::Initializing => "INITIALIZING",
            ImuInitState::Initialized => "INITIALIZED",
        };

        let inlier_pct = if metrics.n_features > 0 {
            metrics.n_inliers as f64 / metrics.n_features as f64 * 100.0
        } else {
            0.0
        };

        let status_text = format!(
            "{} | IMU: {} | Features: {} | Inliers: {} ({:.0}%) | FPS: {:.1}",
            state_indicator, imu_indicator, metrics.n_features, metrics.n_inliers, inlier_pct, fps
        );

        self.rec
            .log(
                "status",
                &rerun::TextDocument::new(status_text)
                    .with_media_type(rerun::MediaType::markdown()),
            )
            .ok();
    }

    /// Log left camera image feed.
    pub fn log_image_feed(&self, left_image: &Mat) {
        if let Ok((data, width, height)) = mat_to_image_data(left_image) {
            self.rec
                .log(
                    "camera/image",
                    &rerun::Image::from_l8(data, [width, height]),
                )
                .ok();
        }
    }

    /// Log annotated features (inliers, outliers, unmatched) with optional depth coloring.
    pub fn log_annotated_features(
        &self,
        frame: &StereoFrame,
        match_info: &MatchInfo,
        inlier_indices: &[usize],
        outlier_indices: &[usize],
    ) {
        let keypoints = &frame.left_features.keypoints;
        let points_cam = &frame.points_cam;

        // Collect matched feature indices
        let matched_set: HashSet<usize> = match_info
            .matched_map_points
            .iter()
            .map(|(_, idx)| *idx)
            .collect();

        // Separate inliers and outliers with depth info
        let mut inlier_points: Vec<[f32; 2]> = Vec::new();
        let mut inlier_colors: Vec<[u8; 3]> = Vec::new();
        let mut outlier_points: Vec<[f32; 2]> = Vec::new();
        let mut outlier_colors: Vec<[u8; 3]> = Vec::new();
        let mut unmatched_points: Vec<[f32; 2]> = Vec::new();

        // Collect depth range for coloring
        let mut depths: Vec<f64> = Vec::new();
        for idx in inlier_indices.iter().chain(outlier_indices.iter()) {
            if let Some(p_cam) = points_cam.get(*idx).and_then(|p| *p) {
                if p_cam.z > 0.1 && p_cam.z < 100.0 {
                    depths.push(p_cam.z);
                }
            }
        }

        let (depth_min, depth_max) = if !depths.is_empty() {
            let mut sorted = depths.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            (
                sorted[sorted.len() * 5 / 100].max(0.5),
                sorted[sorted.len() * 95 / 100].min(20.0),
            )
        } else {
            (0.5, 20.0)
        };

        // Process inliers
        for &idx in inlier_indices {
            if let Some(kp) = keypoints.get(idx).ok() {
                let pt = [kp.pt().x, kp.pt().y];
                inlier_points.push(pt);

                let color = if self.depth_coloring_enabled {
                    if let Some(Some(p_cam)) = points_cam.get(idx) {
                        depth_to_color(p_cam.z, depth_min, depth_max)
                    } else {
                        [0u8, 255, 0] // Green fallback
                    }
                } else {
                    [0u8, 255, 0] // Green
                };
                inlier_colors.push(color);
            }
        }

        // Process outliers
        for &idx in outlier_indices {
            if let Some(kp) = keypoints.get(idx).ok() {
                let pt = [kp.pt().x, kp.pt().y];
                outlier_points.push(pt);

                let color = if self.depth_coloring_enabled {
                    if let Some(Some(p_cam)) = points_cam.get(idx) {
                        depth_to_color(p_cam.z, depth_min, depth_max)
                    } else {
                        [255u8, 0, 0] // Red fallback
                    }
                } else {
                    [255u8, 0, 0] // Red
                };
                outlier_colors.push(color);
            }
        }

        // Process unmatched features
        for (idx, kp) in keypoints.iter().enumerate() {
            if !matched_set.contains(&idx) {
                unmatched_points.push([kp.pt().x, kp.pt().y]);
            }
        }

        // Log inliers as circles
        if !inlier_points.is_empty() && inlier_points.len() == inlier_colors.len() {
            self.rec
                .log(
                    "camera/image/inliers",
                    &rerun::Points2D::new(inlier_points)
                        .with_colors(inlier_colors)
                        .with_radii([6.0f32]),
                )
                .ok();
        }

        // Log outliers as triangles (using larger radius to distinguish)
        if !outlier_points.is_empty() && outlier_points.len() == outlier_colors.len() {
            self.rec
                .log(
                    "camera/image/outliers",
                    &rerun::Points2D::new(outlier_points)
                        .with_colors(outlier_colors)
                        .with_radii([8.0f32]),
                )
                .ok();
        }

        // Log unmatched as gray dots
        if !unmatched_points.is_empty() {
            self.rec
                .log(
                    "camera/image/unmatched",
                    &rerun::Points2D::new(unmatched_points)
                        .with_colors([[128u8, 128, 128]]) // Gray
                        .with_radii([3.0f32]),
                )
                .ok();
        }
    }

    /// Log key metrics overlay on image.
    pub fn log_image_metrics(&self, metrics: &TrackingMetrics, n_matched: usize) {
        let metrics_text = format!(
            "Features: {} | Matched: {} | Inliers: {} ({:.0}%) | Reproj Error: {:.2} px",
            metrics.n_features,
            n_matched,
            metrics.n_inliers,
            if metrics.n_features > 0 {
                metrics.n_inliers as f64 / metrics.n_features as f64 * 100.0
            } else {
                0.0
            },
            metrics.reproj_error_mean_px
        );

        // Log as annotation (Rerun doesn't have direct text overlay, but we can log it separately)
        self.rec
            .log("camera/metrics", &rerun::TextDocument::new(metrics_text))
            .ok();
    }

    /// Log current camera pose as a bright, large frustum.
    pub fn log_camera_pose(&self, pose: &SE3) {
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

        // Log as transform (Rerun will visualize as camera frustum)
        self.rec
            .log(
                "world/camera",
                &rerun::Transform3D::from_translation_rotation(translation, rotation),
            )
            .ok();
    }

    /// Log trajectory as a thin gray line.
    pub fn log_trajectory(&self, positions: &[Vector3<f64>]) {
        if positions.len() < 2 {
            return;
        }
        let pts: Vec<[f32; 3]> = positions
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        self.rec
            .log(
                "world/trajectory",
                &rerun::LineStrips3D::new([pts])
                    .with_colors([[128u8, 128, 128]]) // Gray
                    .with_radii([0.005f32]), // Thin
            )
            .ok();
    }

    /// Log ground truth trajectory as a green line (grows over time).
    pub fn log_groundtruth_trajectory(&self, positions: &[Vector3<f64>]) {
        if positions.len() < 2 {
            return;
        }
        let pts: Vec<[f32; 3]> = positions
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        self.rec
            .log(
                "world/groundtruth",
                &rerun::LineStrips3D::new([pts])
                    .with_colors([[0u8, 200, 100]]) // Green
                    .with_radii([0.008f32]),        // Slightly thicker than estimated
            )
            .ok();
    }

    /// Log keyframes as small blue boxes.
    pub fn log_keyframes(&self, keyframes: &[&KeyFrame]) {
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

            // Log as transform - Rerun can visualize as boxes
            let path = format!("world/keyframes/{}", kf.id.0);
            self.rec
                .log(
                    path.as_str(),
                    &rerun::Transform3D::from_translation_rotation(translation, rotation),
                )
                .ok();

            // Also log as a small box at the camera center
            let box_path = format!("world/keyframes/{}/box", kf.id.0);
            self.rec
                .log(
                    box_path.as_str(),
                    &rerun::Boxes3D::from_centers_and_sizes(
                        [[translation.x, translation.y, translation.z]],
                        [[0.1f32, 0.1, 0.1]],
                    )
                    .with_colors([[0u8, 100, 255]]), // Blue
                )
                .ok();
        }
    }

    /// Log local map points (white dots).
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
                    .with_colors([[255u8, 255, 255]]) // White
                    .with_radii([0.02f32]),
            )
            .ok();
    }

    /// Log full map points with LOD filtering (dim, semi-transparent).
    pub fn log_map_points_lod(&self, points: &[Vector3<f64>], camera_pos: Vector3<f64>) {
        if points.is_empty() {
            return;
        }

        // Apply LOD filtering
        let filtered_points = filter_points_lod(points, camera_pos, 10.0, 50.0, 10);

        if filtered_points.is_empty() {
            return;
        }

        let pts: Vec<[f32; 3]> = filtered_points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        // Dim, semi-transparent points
        self.rec
            .log(
                "world/map_points",
                &rerun::Points3D::new(pts)
                    .with_colors([[100u8, 100, 100]]) // Dim gray
                    .with_radii([0.01f32]), // Small
            )
            .ok();
    }

    /// Log active map ID.
    pub fn log_map_id(&self, map_id: usize) {
        let text = format!("Active Map: {}", map_id);
        self.rec
            .log("world/map_id", &rerun::TextDocument::new(text))
            .ok();
    }

    /// Log temporal plots for metrics.
    pub fn log_temporal_plots(&mut self, metrics: &TrackingMetrics, _timing: &TimingStats) {
        // Update FPS calculation
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        self.frame_times.push(current_time);
        if self.frame_times.len() > 100 {
            self.frame_times.remove(0);
        }

        let fps = if self.frame_times.len() >= 2 {
            let dt = self.frame_times.last().unwrap() - self.frame_times.first().unwrap();
            if dt > 0.0 {
                (self.frame_times.len() - 1) as f64 / dt
            } else {
                0.0
            }
        } else {
            0.0
        };
        self.last_fps = fps;

        // Inlier count
        self.rec
            .log(
                "plots/inlier_count",
                &rerun::Scalars::new([metrics.n_inliers as f64]),
            )
            .ok();

        // Mean reprojection error
        self.rec
            .log(
                "plots/reproj_error",
                &rerun::Scalars::new([metrics.reproj_error_mean_px]),
            )
            .ok();

        // IMU residual
        self.rec
            .log(
                "plots/imu_residual",
                &rerun::Scalars::new([metrics.imu_preint_residual_m]),
            )
            .ok();
    }

    /// Log IMU bias magnitude.
    pub fn log_bias_magnitude(&self, bias: Option<&crate::imu::ImuBias>) {
        if let Some(b) = bias {
            let bg_mag = b.gyro.norm();
            let ba_mag = b.accel.norm();
            let combined = (bg_mag * bg_mag + ba_mag * ba_mag).sqrt();
            self.rec
                .log("plots/bias_magnitude", &rerun::Scalars::new([combined]))
                .ok();
        } else {
            self.rec
                .log("plots/bias_magnitude", &rerun::Scalars::new([0.0]))
                .ok();
        }
    }

    /// Get current FPS estimate.
    pub fn fps(&self) -> f64 {
        self.last_fps
    }
}

/// Convert depth to color (blue -> green -> red gradient).
fn depth_to_color(depth: f64, min_depth: f64, max_depth: f64) -> [u8; 3] {
    let t = ((depth - min_depth) / (max_depth - min_depth).max(0.1)).clamp(0.0, 1.0);
    // Blue -> Green -> Red gradient
    let r = (t * 2.0).min(1.0);
    let g = (1.0 - (t - 0.5).abs() * 2.0).max(0.0);
    let b = ((1.0 - t) * 2.0).min(1.0);
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Filter points using Level-of-Detail (LOD) based on distance from camera.
///
/// - Points within `near_threshold`: keep all
/// - Points between thresholds: voxel-based sampling
/// - Points beyond `far_threshold`: keep 1 in `far_downsample`
fn filter_points_lod(
    points: &[Vector3<f64>],
    camera_pos: Vector3<f64>,
    near_threshold: f64,
    far_threshold: f64,
    far_downsample: usize,
) -> Vec<Vector3<f64>> {
    let mut result = Vec::new();
    let mut far_points = Vec::new();

    for point in points {
        let dist = (point - camera_pos).norm();

        if dist < near_threshold {
            // Keep all near points
            result.push(*point);
        } else if dist < far_threshold {
            // Medium distance: simple sampling (keep every Nth)
            if (result.len() + far_points.len()) % 3 == 0 {
                result.push(*point);
            }
        } else {
            // Far points: collect for downsampling
            far_points.push(*point);
        }
    }

    // Downsample far points
    for (i, point) in far_points.iter().enumerate() {
        if i % far_downsample == 0 {
            result.push(*point);
        }
    }

    result
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
