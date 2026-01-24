//! Coordinate frame definitions and transformations for visual-inertial SLAM.
//!
//! This module centralizes all coordinate frame conventions, similar to how
//! ORB-SLAM3 uses ImuCalib with T_bc/T_cb transforms.
//!
//! # Overview: How Coordinate Frames Work in This System
//!
//! There are three main coordinate systems to understand:
//!
//! 1. **Camera Frame** - Where SLAM poses live internally
//! 2. **Body/IMU Frame** - Where IMU measurements are expressed
//! 3. **Visualization Frame** - What Rerun displays (Z-up world)
//!
//! ## The Key Insight
//!
//! The SLAM system stores all poses in **camera frame convention** (T_wc).
//! For visualization, we transform these to the **FLU convention** (X-forward, Z-up).
//!
//! Ground truth from EuRoC is in the **Vicon/Leica reference frame**, which is
//! **already Z-up**. This is NOT the same as the body/IMU frame (FRD)!
//! The FRD convention only applies to IMU measurements, not ground truth positions.
//!
//! # Frame Conventions
//!
//! ## Camera Frame (RDF - OpenCV/Computer Vision convention)
//! ```text
//!        +Y (down)
//!         |
//!         |
//!         +------ +X (right)
//!        /
//!       /
//!      +Z (forward, optical axis)
//! ```
//! - X: Right
//! - Y: Down
//! - Z: Forward (optical axis, into the scene)
//!
//! ## Body/IMU Frame (FRD - EuRoC IMU convention)
//! ```text
//!        +Z (down)
//!         |
//!         |
//!         +------ +Y (right)
//!        /
//!       /
//!      +X (forward)
//! ```
//! - X: Forward
//! - Y: Right
//! - Z: Down
//!
//! **Note:** This is the frame for IMU measurements (accelerometer, gyroscope),
//! NOT for ground truth positions!
//!
//! ## Visualization/World Frame (FLU - Rerun convention)
//! ```text
//!        +Z (up)
//!         |
//!         |
//!         +------ +Y (left)
//!        /
//!       /
//!      +X (forward)
//! ```
//! - X: Forward
//! - Y: Left
//! - Z: Up
//!
//! This is what Rerun displays with `ViewCoordinates::FLU()`.
//!
//! ## EuRoC Ground Truth Frame (Vicon/Leica room)
//! The ground truth positions from EuRoC are in the motion capture system's
//! reference frame, which is typically **already Z-up** (similar to FLU).
//! This is why we DON'T apply body-to-viz conversion to ground truth!
//!
//! # Transformation Pipeline
//!
//! ## Camera Trajectory → Visualization
//! ```text
//! SLAM Pose (T_wc, camera convention)
//!     │
//!     ▼ camera_pose_to_viz() / camera_position_to_viz()
//!     │
//! Visualization Pose (FLU, Z-up)
//! ```
//!
//! The camera's +Z (forward) becomes visualization +X (forward).
//! The camera's +Y (down) becomes visualization -Z (up).
//!
//! ## Ground Truth → Visualization
//! ```text
//! EuRoC Ground Truth (Vicon frame, already Z-up)
//!     │
//!     ▼ Translate to origin (first position = 0,0,0)
//!     │
//! Visualization (already in correct frame!)
//! ```
//!
//! **Important:** Ground truth does NOT need rotation conversion because
//! the Vicon reference frame is already Z-up!
//!
//! # Transformation Naming Convention
//!
//! We use the notation `T_target_source` where:
//! - `source` is the frame we're transforming FROM
//! - `target` is the frame we're transforming TO
//!
//! For example, `T_body_cam` transforms a point from camera frame to body frame:
//! ```text
//! p_body = T_body_cam * p_cam
//! ```
//!
//! # Comparison with ORB-SLAM3
//!
//! ORB-SLAM3's approach:
//! - Uses `T_bc` (body-from-camera) from calibration files, not hardcoded
//! - Has ONE world frame with gravity = (0, 0, -9.81), so +Z is up
//! - After IMU initialization, aligns world frame so gravity points down
//!
//! Our approach:
//! - Camera poses are in camera convention internally (for PnP, BA, etc.)
//! - Transform to Z-up FLU only for visualization
//! - Ground truth already Z-up, just center at origin
//!
//! # Usage Examples
//!
//! ## Visualizing a camera pose
//! ```ignore
//! use crate::geometry::frames::camera_pose_to_viz;
//!
//! let slam_pose: SE3 = tracker.get_pose();  // In camera convention
//! let viz_pose = camera_pose_to_viz(&slam_pose);  // For Rerun display
//! ```
//!
//! ## Visualizing map points
//! ```ignore
//! use crate::geometry::frames::camera_position_to_viz;
//!
//! let point_slam: Vector3<f64> = map_point.position;  // In camera world
//! let point_viz = camera_position_to_viz(&point_slam);  // For Rerun display
//! ```
//!
//! ## Ground truth (no conversion needed!)
//! ```ignore
//! // EuRoC ground truth is already Z-up
//! let gt_pos = gt_entry.pose.translation - first_gt_pos;  // Just center it
//! // Use directly in visualization - no frame conversion!
//! ```

use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::SE3;

// ============================================================================
// Rotation Matrices Between Frames
// ============================================================================

/// Fixed rotation matrix to transform from Camera frame (RDF) to Body frame (FRD).
///
/// Camera: X-right, Y-down, Z-forward
/// Body:   X-forward, Y-right, Z-down
///
/// This maps:
/// - Camera +Z (forward) → Body +X (forward)
/// - Camera +X (right)   → Body +Y (right)
/// - Camera +Y (down)    → Body +Z (down)
#[rustfmt::skip]
pub fn rotation_body_cam() -> Matrix3<f64> {
    // Column vectors are where camera axes go in body frame
    // Camera X (right)   → Body Y
    // Camera Y (down)    → Body Z
    // Camera Z (forward) → Body X
    Matrix3::new(
        0.0, 0.0, 1.0,  // Body X = Camera Z
        1.0, 0.0, 0.0,  // Body Y = Camera X
        0.0, 1.0, 0.0,  // Body Z = Camera Y
    )
}

/// Fixed rotation matrix to transform from Body frame (FRD) to Camera frame (RDF).
///
/// This is the inverse of `rotation_body_cam()`.
#[rustfmt::skip]
pub fn rotation_cam_body() -> Matrix3<f64> {
    // Inverse = transpose for rotation matrices
    rotation_body_cam().transpose()
}

/// Fixed rotation matrix to transform from Body frame (FRD) to World/Viz frame (FLU).
///
/// Body: X-forward, Y-right, Z-down
/// Viz:  X-forward, Y-left,  Z-up
///
/// This maps:
/// - Body +X (forward) → Viz +X (forward)
/// - Body +Y (right)   → Viz -Y (left)
/// - Body +Z (down)    → Viz -Z (up)
#[rustfmt::skip]
pub fn rotation_viz_body() -> Matrix3<f64> {
    Matrix3::new(
        1.0,  0.0,  0.0,  // Viz X = Body X (forward)
        0.0, -1.0,  0.0,  // Viz Y = -Body Y (right → left)
        0.0,  0.0, -1.0,  // Viz Z = -Body Z (down → up)
    )
}

/// Fixed rotation matrix to transform from World/Viz frame (FLU) to Body frame (FRD).
#[rustfmt::skip]
pub fn rotation_body_viz() -> Matrix3<f64> {
    rotation_viz_body().transpose()
}

/// Fixed rotation matrix to transform from Camera frame (RDF) to World/Viz frame (FLU).
///
/// Camera: X-right, Y-down, Z-forward
/// Viz:    X-forward, Y-left, Z-up
///
/// This is the composition: R_viz_cam = R_viz_body * R_body_cam
///
/// Maps:
/// - Camera +Z (forward) → Viz +X (forward)
/// - Camera +X (right)   → Viz -Y (left)
/// - Camera +Y (down)    → Viz -Z (up)
#[rustfmt::skip]
pub fn rotation_viz_cam() -> Matrix3<f64> {
    Matrix3::new(
        0.0,  0.0, 1.0,  // Viz X = Camera Z
       -1.0,  0.0, 0.0,  // Viz Y = -Camera X
        0.0, -1.0, 0.0,  // Viz Z = -Camera Y
    )
}

/// Fixed rotation matrix to transform from World/Viz frame (FLU) to Camera frame (RDF).
#[rustfmt::skip]
pub fn rotation_cam_viz() -> Matrix3<f64> {
    rotation_viz_cam().transpose()
}

// ============================================================================
// Frame Converter (for calibration-based transforms)
// ============================================================================

/// Coordinate frame converter for visual-inertial SLAM.
///
/// Stores the camera-to-body transform (from calibration) and provides
/// methods to convert poses between camera, body, and visualization frames.
///
/// In ORB-SLAM3, this corresponds to the T_bc/T_cb transforms in ImuCalib.
#[derive(Debug, Clone)]
pub struct FrameConverter {
    /// Transform from camera to body frame (from EuRoC calibration T_BS).
    /// This includes both rotation AND translation (camera offset from IMU).
    pub t_body_cam: SE3,

    /// Inverse: transform from body to camera frame.
    pub t_cam_body: SE3,
}

impl FrameConverter {
    /// Create a converter from the camera-to-body transform (T_BS from EuRoC).
    pub fn new(t_body_cam: SE3) -> Self {
        let t_cam_body = t_body_cam.inverse();
        Self {
            t_body_cam,
            t_cam_body,
        }
    }

    /// Create a converter assuming camera and body frames are co-located
    /// (only rotation difference, no translation).
    pub fn aligned() -> Self {
        let rotation = UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(rotation_body_cam()),
        );
        let t_body_cam = SE3 {
            rotation,
            translation: Vector3::zeros(),
        };
        Self::new(t_body_cam)
    }

    /// Convert a camera-frame pose (T_wc) to body-frame pose (T_wb).
    ///
    /// Given: T_wc (camera pose in SLAM world frame)
    /// Returns: T_wb (body pose in SLAM world frame)
    ///
    /// Formula: T_wb = T_wc * T_cb = T_wc * T_body_cam^(-1)
    pub fn camera_pose_to_body(&self, t_wc: &SE3) -> SE3 {
        t_wc.compose(&self.t_cam_body)
    }

    /// Convert a body-frame pose to camera-frame pose.
    ///
    /// Given: T_wb (body pose in world frame)
    /// Returns: T_wc (camera pose in world frame)
    ///
    /// Formula: T_wc = T_wb * T_bc = T_wb * T_body_cam
    pub fn body_pose_to_camera(&self, t_wb: &SE3) -> SE3 {
        t_wb.compose(&self.t_body_cam)
    }

    /// Transform a camera-frame pose for visualization in FLU world frame.
    ///
    /// This applies the rotation to convert from camera convention (RDF)
    /// to visualization convention (FLU).
    pub fn camera_pose_to_viz(&self, t_wc: &SE3) -> SE3 {
        // First convert to body frame, then to viz frame
        let t_wb = self.camera_pose_to_body(t_wc);
        body_pose_to_viz(&t_wb)
    }
}

// ============================================================================
// Position Transformations
// ============================================================================

/// Transform a body-frame position to visualization frame (FLU).
///
/// Body (FRD): X-forward, Y-right, Z-down
/// Viz (FLU):  X-forward, Y-left, Z-up
pub fn body_position_to_viz(p_body: &Vector3<f64>) -> Vector3<f64> {
    let r = rotation_viz_body();
    r * p_body
}

/// Transform a visualization-frame position to body frame.
pub fn viz_position_to_body(p_viz: &Vector3<f64>) -> Vector3<f64> {
    let r = rotation_body_viz();
    r * p_viz
}

/// Transform a camera-frame position to visualization frame (FLU).
///
/// This is the main transformation used for SLAM trajectory and map points.
///
/// Camera (RDF): X-right, Y-down, Z-forward
/// Viz (FLU):    X-forward, Y-left, Z-up
pub fn camera_position_to_viz(p_cam: &Vector3<f64>) -> Vector3<f64> {
    let r = rotation_viz_cam();
    r * p_cam
}

// ============================================================================
// Pose Transformations
// ============================================================================

/// Transform a body-frame pose (T_wb) to visualization frame.
///
/// The pose rotation is adjusted so that the body's forward direction (+X in FRD)
/// appears as +X in the FLU visualization frame, and body's down (+Z in FRD)
/// becomes -Z (up) in FLU.
pub fn body_pose_to_viz(t_wb: &SE3) -> SE3 {
    // Position: transform the translation vector
    let pos_viz = body_position_to_viz(&t_wb.translation);

    // Rotation: R_viz = R_viz_body * R_wb * R_body_viz
    // This sandwiches the world-to-body rotation between frame conversions
    let r_viz_body = UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(rotation_viz_body()),
    );
    let r_body_viz = r_viz_body.inverse();

    let rot_viz = r_viz_body * t_wb.rotation * r_body_viz;

    SE3 {
        rotation: rot_viz,
        translation: pos_viz,
    }
}

/// Transform a camera-frame pose (T_wc) to visualization frame.
///
/// This is the main transformation used for visualizing camera poses.
///
/// The pose rotation is adjusted so that the camera's forward direction (+Z in RDF)
/// appears as +X in the FLU visualization frame.
pub fn camera_pose_to_viz(t_wc: &SE3) -> SE3 {
    // Position: transform the translation vector
    let pos_viz = camera_position_to_viz(&t_wc.translation);

    // Rotation: R_viz = R_viz_cam * R_wc * R_cam_viz
    let r_viz_cam = UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(rotation_viz_cam()),
    );
    let r_cam_viz = r_viz_cam.inverse();

    let rot_viz = r_viz_cam * t_wc.rotation * r_cam_viz;

    SE3 {
        rotation: rot_viz,
        translation: pos_viz,
    }
}
