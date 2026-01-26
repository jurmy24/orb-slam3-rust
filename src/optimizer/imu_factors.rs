//! IMU Factors for Visual-Inertial Bundle Adjustment.
//!
//! This module provides IMU residuals and Jacobians for joint optimization
//! of poses, velocities, and biases following the "On-Manifold Preintegration"
//! framework by Forster et al.
//!
//! # IMU Residual
//!
//! The preintegrated IMU factor constrains consecutive keyframes i and j:
//!
//! ```text
//! r_imu = [ Log(ΔR_ij^T · R_i^T · R_j) ]    // Rotation residual (3)
//!         [ R_i^T · (v_j - v_i - g·Δt) - Δv_ij ]    // Velocity residual (3)
//!         [ R_i^T · (p_j - p_i - v_i·Δt - 0.5·g·Δt²) - Δp_ij ]  // Position residual (3)
//! ```
//!
//! where ΔR_ij, Δv_ij, Δp_ij are the preintegrated measurements.

use nalgebra::{Matrix3, Vector3};

use crate::geometry::SE3;
use crate::imu::sample::GRAVITY;
use crate::imu::PreintegratedState;

/// IMU residual between two states connected by preintegration.
///
/// The residual is 9-dimensional: [δθ (3), δv (3), δp (3)]
#[derive(Debug, Clone)]
pub struct ImuResidual {
    /// Rotation residual in tangent space (axis-angle).
    pub rotation: Vector3<f64>,
    /// Velocity residual.
    pub velocity: Vector3<f64>,
    /// Position residual.
    pub position: Vector3<f64>,
}

impl ImuResidual {
    /// Compute the 9-dimensional residual vector.
    pub fn as_vector(&self) -> nalgebra::SVector<f64, 9> {
        nalgebra::SVector::<f64, 9>::from_iterator(
            self.rotation
                .iter()
                .chain(self.velocity.iter())
                .chain(self.position.iter())
                .copied(),
        )
    }

    /// Compute squared Mahalanobis norm using information matrix.
    pub fn mahalanobis_norm_squared(&self, info: &nalgebra::SMatrix<f64, 9, 9>) -> f64 {
        let r = self.as_vector();
        (r.transpose() * info * r)[(0, 0)]
    }
}

/// Compute IMU residual between two keyframe states.
///
/// # Arguments
/// * `pose_i` - Pose of keyframe i (T_w_ci, camera-to-world)
/// * `vel_i` - Velocity of keyframe i in world frame
/// * `pose_j` - Pose of keyframe j (T_w_cj, camera-to-world)
/// * `vel_j` - Velocity of keyframe j in world frame
/// * `preint` - Preintegrated IMU measurements from i to j
///
/// # Returns
/// IMU residual (9D)
pub fn compute_imu_residual(
    pose_i: &SE3,
    vel_i: &Vector3<f64>,
    pose_j: &SE3,
    vel_j: &Vector3<f64>,
    preint: &PreintegratedState,
) -> ImuResidual {
    let dt = preint.dt;
    let r_i = pose_i.rotation;
    let p_i = pose_i.translation;
    let r_j = pose_j.rotation;
    let p_j = pose_j.translation;

    // Rotation residual: Log(ΔR_ij^T · R_i^T · R_j)
    // ΔR_ij is the preintegrated rotation
    let delta_r = preint.delta_rot;
    let rotation_error = delta_r.inverse() * r_i.inverse() * r_j;
    let rotation_residual = rotation_error.scaled_axis();

    // Velocity residual: R_i^T · (v_j - v_i - g·Δt) - Δv_ij
    let expected_delta_v = r_i.inverse() * (vel_j - vel_i - GRAVITY * dt);
    let velocity_residual = expected_delta_v - preint.delta_vel;

    // Position residual: R_i^T · (p_j - p_i - v_i·Δt - 0.5·g·Δt²) - Δp_ij
    let expected_delta_p =
        r_i.inverse() * (p_j - p_i - vel_i * dt - 0.5 * GRAVITY * dt * dt);
    let position_residual = expected_delta_p - preint.delta_pos;

    ImuResidual {
        rotation: rotation_residual,
        velocity: velocity_residual,
        position: position_residual,
    }
}

/// Jacobian of IMU residual with respect to state i (15D: pose 6, vel 3, bias 6).
///
/// Returns a 9×15 Jacobian matrix.
#[allow(dead_code)]
pub fn jacobian_wrt_state_i(
    pose_i: &SE3,
    _vel_i: &Vector3<f64>,
    preint: &PreintegratedState,
) -> nalgebra::SMatrix<f64, 9, 15> {
    let dt = preint.dt;
    let r_i = pose_i.rotation.to_rotation_matrix().into_inner();

    let mut jacobian = nalgebra::SMatrix::<f64, 9, 15>::zeros();

    // d(r_rotation) / d(pose_i.rotation) - using right Jacobian approximation
    // Simplified: ≈ -I for small rotations
    jacobian
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-Matrix3::identity()));

    // d(r_velocity) / d(pose_i.rotation)
    // = -R_i^T · [v_j - v_i - g·Δt]×
    // Approximated as zeros for now (second-order effect)

    // d(r_velocity) / d(vel_i) = -R_i^T
    jacobian
        .fixed_view_mut::<3, 3>(3, 6)
        .copy_from(&(-r_i.transpose()));

    // d(r_position) / d(pose_i.rotation) - similar approximation
    // Approximated as zeros

    // d(r_position) / d(pose_i.translation) = -R_i^T
    jacobian
        .fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(-r_i.transpose()));

    // d(r_position) / d(vel_i) = -R_i^T · Δt
    jacobian
        .fixed_view_mut::<3, 3>(6, 6)
        .copy_from(&(-r_i.transpose() * dt));

    // Bias Jacobians (from preintegration covariance if available)
    if let Some(ref cov) = preint.covariance {
        // d(r_rotation) / d(bg) = -J_r_bg
        jacobian
            .fixed_view_mut::<3, 3>(0, 9)
            .copy_from(&(-cov.j_r_bg));

        // d(r_velocity) / d(bg) = -J_v_bg
        jacobian
            .fixed_view_mut::<3, 3>(3, 9)
            .copy_from(&(-cov.j_v_bg));

        // d(r_velocity) / d(ba) = -J_v_ba
        jacobian
            .fixed_view_mut::<3, 3>(3, 12)
            .copy_from(&(-cov.j_v_ba));

        // d(r_position) / d(bg) = -J_p_bg
        jacobian
            .fixed_view_mut::<3, 3>(6, 9)
            .copy_from(&(-cov.j_p_bg));

        // d(r_position) / d(ba) = -J_p_ba
        jacobian
            .fixed_view_mut::<3, 3>(6, 12)
            .copy_from(&(-cov.j_p_ba));
    }

    jacobian
}

/// Jacobian of IMU residual with respect to state j (9D: pose 6, vel 3).
///
/// Returns a 9×9 Jacobian matrix.
#[allow(dead_code)]
pub fn jacobian_wrt_state_j(
    pose_i: &SE3,
    _pose_j: &SE3,
    preint: &PreintegratedState,
) -> nalgebra::SMatrix<f64, 9, 9> {
    let r_i = pose_i.rotation.to_rotation_matrix().into_inner();
    let dt = preint.dt;
    let _ = dt; // Used in position computation

    let mut jacobian = nalgebra::SMatrix::<f64, 9, 9>::zeros();

    // d(r_rotation) / d(pose_j.rotation) ≈ I (right Jacobian)
    jacobian
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&Matrix3::identity());

    // d(r_velocity) / d(vel_j) = R_i^T
    jacobian
        .fixed_view_mut::<3, 3>(3, 6)
        .copy_from(&r_i.transpose());

    // d(r_position) / d(pose_j.translation) = R_i^T
    jacobian
        .fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&r_i.transpose());

    jacobian
}

/// State vector layout for Visual-Inertial BA.
///
/// For each keyframe we optimize:
/// - Pose (6): [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
/// - Velocity (3): [vx, vy, vz]
///
/// Biases are typically optimized as a separate block or with a prior.
#[derive(Debug, Clone, Copy)]
pub struct VIStateLayout {
    /// Number of keyframes.
    pub num_keyframes: usize,
    /// Number of map points.
    pub num_map_points: usize,
}

impl VIStateLayout {
    /// Parameters per keyframe (pose 6 + velocity 3).
    pub const KF_PARAMS: usize = 9;
    /// Parameters per map point.
    pub const MP_PARAMS: usize = 3;

    /// Create a new layout.
    pub fn new(num_keyframes: usize, num_map_points: usize) -> Self {
        Self {
            num_keyframes,
            num_map_points,
        }
    }

    /// Total number of parameters.
    pub fn total_params(&self) -> usize {
        self.num_keyframes * Self::KF_PARAMS + self.num_map_points * Self::MP_PARAMS
    }

    /// Starting index for keyframe i's pose.
    pub fn pose_start(&self, kf_idx: usize) -> usize {
        kf_idx * Self::KF_PARAMS
    }

    /// Starting index for keyframe i's velocity.
    pub fn vel_start(&self, kf_idx: usize) -> usize {
        kf_idx * Self::KF_PARAMS + 6
    }

    /// Starting index for map point i.
    pub fn mp_start(&self, mp_idx: usize) -> usize {
        self.num_keyframes * Self::KF_PARAMS + mp_idx * Self::MP_PARAMS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::PreintegratedState;

    #[test]
    fn test_imu_residual_zero_motion() {
        // Two identical states with zero preintegration should give zero residual
        let pose = SE3::identity();
        let vel = Vector3::zeros();
        let preint = PreintegratedState::identity();

        let residual = compute_imu_residual(&pose, &vel, &pose, &vel, &preint);

        assert!(residual.rotation.norm() < 1e-10);
        assert!(residual.velocity.norm() < 1e-10);
        assert!(residual.position.norm() < 1e-10);
    }

    #[test]
    fn test_imu_residual_pure_gravity() {
        // Test that the residual formula is consistent:
        // If we have matching states and preintegration, residual should be zero.
        let pose = SE3::identity();
        let vel_i = Vector3::zeros();
        let dt = 0.1;

        // Simulate: camera in free fall for 0.1s
        // vel_j = vel_i + g * dt = 0 + (-9.81) * 0.1 = -0.981 in z
        let vel_j = vel_i + GRAVITY * dt;

        // Preintegration in body frame:
        // For identity rotation, body frame = world frame
        // delta_vel = R_i^T * (g * dt) = g * dt (since R_i = I)
        let mut preint = PreintegratedState::identity();
        preint.dt = dt;
        preint.delta_vel = GRAVITY * dt; // This is what preintegration would compute

        let _residual = compute_imu_residual(&pose, &vel_i, &pose, &vel_j, &preint);

        // Velocity residual: R_i^T * (v_j - v_i - g*dt) - delta_vel
        // = I * (g*dt - 0 - g*dt) - g*dt = -g*dt
        // Actually, let me trace through:
        // expected_delta_v = R_i^-1 * (v_j - v_i - g*dt) = I * (g*dt - g*dt) = 0
        // residual = expected_delta_v - delta_vel = 0 - g*dt = -g*dt
        // So this test is checking if preint.delta_vel matches the expected

        // For the residual to be zero, we need:
        // delta_vel = R_i^T * (v_j - v_i - g*dt) = 0
        // So let's set delta_vel = 0 (no preintegrated velocity change when
        // the actual velocity change equals gravity)
        let mut preint_correct = PreintegratedState::identity();
        preint_correct.dt = dt;
        preint_correct.delta_vel = Vector3::zeros(); // No change in body frame velocity

        let residual_correct = compute_imu_residual(&pose, &vel_i, &pose, &vel_j, &preint_correct);

        assert!(
            residual_correct.velocity.norm() < 0.01,
            "Velocity residual: {}",
            residual_correct.velocity.norm()
        );
    }

    #[test]
    fn test_vi_state_layout() {
        let layout = VIStateLayout::new(5, 100);

        assert_eq!(layout.total_params(), 5 * 9 + 100 * 3);
        assert_eq!(layout.pose_start(0), 0);
        assert_eq!(layout.vel_start(0), 6);
        assert_eq!(layout.pose_start(1), 9);
        assert_eq!(layout.mp_start(0), 45);
        assert_eq!(layout.mp_start(1), 48);
    }
}
