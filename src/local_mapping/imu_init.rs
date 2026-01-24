//! IMU Initialization for Visual-Inertial SLAM.
//!
//! Implements the gravity direction estimation and IMU initialization
//! as described in ORB-SLAM3. For stereo-inertial, this estimates:
//! - Gravity direction (Rwg - rotation from gravity frame to world)
//! - Initial velocities for each keyframe
//! - IMU biases (initial estimate, refined later)
//!
//! The initialization requires:
//! - At least 10 keyframes with IMU preintegration
//! - At least 1 second of data for stereo

use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};
use tracing::info;

use crate::atlas::map::KeyFrameId;
use crate::imu::{ImuBias, ImuInitState};
use crate::system::shared_state::SharedState;

/// Minimum number of keyframes required for IMU initialization.
const MIN_KEYFRAMES_FOR_INIT: usize = 10;

/// Minimum time span (seconds) for stereo-inertial initialization.
const MIN_TIME_SPAN_STEREO: f64 = 1.0;

/// Result of IMU initialization.
#[derive(Debug, Clone)]
pub struct ImuInitResult {
    /// Rotation from gravity-aligned frame to world frame.
    /// After initialization, gravity in world frame is: g_w = Rwg * [0, 0, -9.81]
    pub rwg: UnitQuaternion<f64>,

    /// Scale factor (1.0 for stereo since we have metric scale).
    pub scale: f64,

    /// Estimated IMU bias.
    pub bias: ImuBias,

    /// Estimated velocities for each keyframe.
    pub velocities: Vec<(KeyFrameId, Vector3<f64>)>,
}

/// Attempt to initialize IMU from the current map state.
///
/// Returns `Some(ImuInitResult)` if initialization succeeds, `None` otherwise.
///
/// For stereo-inertial, we have metric scale from stereo, so we only need to estimate:
/// 1. Gravity direction (Rwg)
/// 2. Keyframe velocities
/// 3. Initial bias estimate (can be zero initially)
pub fn initialize_imu(shared: &Arc<SharedState>) -> Option<ImuInitResult> {
    let mut atlas = shared.atlas.write();
    let map = atlas.active_map_mut();

    // Check if already initialized
    if map.is_imu_initialized() {
        return None;
    }

    // Set state to INITIALIZING when we start the attempt
    if matches!(map.imu_init_state(), ImuInitState::NotInitialized) {
        map.set_imu_init_state(ImuInitState::Initializing);
    }

    // Check minimum keyframes
    if map.num_keyframes() < MIN_KEYFRAMES_FOR_INIT {
        return None;
    }

    // Check minimum time span
    let time_span = map.time_span_seconds();
    if time_span < MIN_TIME_SPAN_STEREO {
        return None;
    }

    // Get keyframes in temporal order (need to clone IDs since we'll drop the lock)
    let keyframe_ids: Vec<KeyFrameId> = map
        .keyframes_temporal_order()
        .iter()
        .map(|kf| kf.id)
        .collect();
    if keyframe_ids.len() < MIN_KEYFRAMES_FOR_INIT {
        return None;
    }

    // Drop write lock and get read lock to access keyframe data
    drop(atlas);
    let atlas = shared.atlas.read();
    let map = atlas.active_map();

    // Get keyframes by ID
    let keyframes: Vec<_> = keyframe_ids
        .iter()
        .filter_map(|&id| map.get_keyframe(id))
        .collect();

    if keyframes.len() < MIN_KEYFRAMES_FOR_INIT {
        return None;
    }

    // Estimate gravity direction from preintegrated velocities
    // The idea: Sum of (R_prev * delta_v) should point opposite to gravity
    // because delta_v includes gravity integration
    let mut dir_g = Vector3::zeros();
    let mut valid_preint_count = 0;

    for kf in &keyframes {
        if let Some(ref preint) = kf.imu_preintegrated {
            if let Some(prev_kf_id) = kf.prev_kf {
                if let Some(prev_kf) = map.get_keyframe(prev_kf_id) {
                    // delta_v was integrated assuming zero gravity, so the "missing"
                    // gravity appears as: actual_dv = preint_dv + R_prev * g * dt
                    // We accumulate -R_prev * delta_v to estimate gravity direction
                    let r_prev = prev_kf.pose.rotation;
                    dir_g -= r_prev * preint.delta_vel;
                    valid_preint_count += 1;
                }
            }
        }
    }

    if valid_preint_count < 2 {
        return None;
    }

    // Normalize to get gravity direction
    let dir_g_norm = dir_g.norm();
    if dir_g_norm < 1e-6 {
        return None;
    }
    let dir_g = dir_g / dir_g_norm;

    // Compute Rwg: rotation that aligns [0, 0, -1] (gravity in inertial frame) with dir_g
    let g_inertial = Vector3::new(0.0, 0.0, -1.0);
    let rwg = rotation_between_vectors(&g_inertial, &dir_g);

    // Estimate velocities for each keyframe
    // Simple approach: use position differences / time
    let mut velocities = Vec::new();
    for window in keyframes.windows(2) {
        let kf_prev = window[0];
        let kf_curr = window[1];

        if let Some(ref preint) = kf_curr.imu_preintegrated {
            let dt = preint.dt;
            if dt > 1e-6 {
                let dp = kf_curr.pose.translation - kf_prev.pose.translation;
                let vel = dp / dt;
                velocities.push((kf_prev.id, vel));
            }
        }
    }
    // Add velocity for last keyframe (use same as previous)
    if let Some(&(_, last_vel)) = velocities.last() {
        if let Some(last_kf) = keyframes.last() {
            velocities.push((last_kf.id, last_vel));
        }
    }

    Some(ImuInitResult {
        rwg,
        scale: 1.0,            // Stereo has metric scale
        bias: ImuBias::zero(), // Initial bias estimate
        velocities,
    })
}

/// Compute rotation that transforms vector `from` to vector `to`.
fn rotation_between_vectors(from: &Vector3<f64>, to: &Vector3<f64>) -> UnitQuaternion<f64> {
    let from_normalized = from.normalize();
    let to_normalized = to.normalize();

    let cross = from_normalized.cross(&to_normalized);
    let dot = from_normalized.dot(&to_normalized);

    // Handle parallel/anti-parallel cases
    if cross.norm() < 1e-10 {
        if dot > 0.0 {
            // Vectors are parallel
            return UnitQuaternion::identity();
        } else {
            // Vectors are anti-parallel - rotate 180° around any perpendicular axis
            let perp = if from_normalized.x.abs() < 0.9 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            let axis = from_normalized.cross(&perp).normalize();
            return UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(axis),
                std::f64::consts::PI,
            );
        }
    }

    let angle = cross.norm().atan2(dot);
    let axis = nalgebra::Unit::new_normalize(cross);
    UnitQuaternion::from_axis_angle(&axis, angle)
}

/// Apply IMU initialization result to the map.
///
/// This updates:
/// - Keyframe velocities
/// - Sets the IMU initialized flag
pub fn apply_imu_init(shared: &Arc<SharedState>, result: &ImuInitResult) {
    let mut atlas = shared.atlas.write();
    let map = atlas.active_map_mut();

    // Update velocities for each keyframe
    for (kf_id, vel) in &result.velocities {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            kf.velocity = *vel;
        }
    }

    // Mark IMU as initialized
    map.set_imu_init_state(ImuInitState::Initialized);

    info!(
        "IMU initialized! Gravity direction estimated from {} keyframes over {:.2}s",
        map.num_keyframes(),
        map.time_span_seconds()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_between_vectors() {
        // Test identity case
        let v1 = Vector3::new(0.0, 0.0, 1.0);
        let rot = rotation_between_vectors(&v1, &v1);
        assert!((rot.angle() - 0.0).abs() < 1e-10);

        // Test 90 degree rotation
        let v2 = Vector3::new(1.0, 0.0, 0.0);
        let rot = rotation_between_vectors(&v1, &v2);
        let v1_rotated = rot * v1;
        assert!((v1_rotated - v2).norm() < 1e-10);

        // Test 180 degree rotation
        let v3 = Vector3::new(0.0, 0.0, -1.0);
        let rot = rotation_between_vectors(&v1, &v3);
        let v1_rotated = rot * v1;
        assert!((v1_rotated - v3).norm() < 1e-10);
    }
}
