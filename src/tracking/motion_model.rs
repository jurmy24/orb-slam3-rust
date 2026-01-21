//! Velocity-based motion model for pose prediction.

use nalgebra::{UnitQuaternion, Vector3};

use crate::geometry::SE3;

/// Constant velocity motion model.
///
/// Predicts the next pose based on the velocity observed between
/// the previous two frames.
pub struct MotionModel {
    /// Previous pose.
    prev_pose: Option<SE3>,
    /// Velocity in world frame (translation per frame).
    velocity: Vector3<f64>,
    /// Angular velocity (rotation per frame).
    angular_velocity: UnitQuaternion<f64>,
}

impl MotionModel {
    pub fn new() -> Self {
        Self {
            prev_pose: None,
            velocity: Vector3::zeros(),
            angular_velocity: UnitQuaternion::identity(),
        }
    }

    /// Update the model with a new pose observation.
    pub fn update(&mut self, pose: &SE3) {
        if let Some(ref prev) = self.prev_pose {
            // Compute velocity as delta between poses
            self.velocity = pose.translation - prev.translation;
            self.angular_velocity = prev.rotation.inverse() * pose.rotation;
        }
        self.prev_pose = Some(pose.clone());
    }

    /// Predict the next pose based on constant velocity assumption.
    pub fn predict(&self) -> Option<SE3> {
        self.prev_pose.as_ref().map(|prev| SE3 {
            rotation: prev.rotation * self.angular_velocity,
            translation: prev.translation + self.velocity,
        })
    }

    /// Reset the motion model.
    pub fn reset(&mut self) {
        self.prev_pose = None;
        self.velocity = Vector3::zeros();
        self.angular_velocity = UnitQuaternion::identity();
    }
}

impl Default for MotionModel {
    fn default() -> Self {
        Self::new()
    }
}
