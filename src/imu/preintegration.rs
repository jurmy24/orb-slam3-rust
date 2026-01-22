use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::sample::{ImuBias, ImuNoise, ImuSample, GRAVITY};
use super::types::{Matrix9, Matrix9x6};
use crate::geometry::{right_jacobian_so3, skew};

/// Covariance and bias Jacobians for preintegrated measurements.
///
/// Tracks uncertainty in the 9-dimensional state [δθ, δv, δp] and stores
/// Jacobians for first-order bias correction following Forster et al.
#[derive(Debug, Clone)]
pub struct PreintegratedCovariance {
    /// 9×9 covariance matrix for [δθ, δv, δp].
    pub cov: Matrix9,
    /// Jacobian ∂(ΔR)/∂(bg) - rotation w.r.t. gyro bias.
    pub j_r_bg: Matrix3<f64>,
    /// Jacobian ∂(Δv)/∂(bg) - velocity w.r.t. gyro bias.
    pub j_v_bg: Matrix3<f64>,
    /// Jacobian ∂(Δv)/∂(ba) - velocity w.r.t. accel bias.
    pub j_v_ba: Matrix3<f64>,
    /// Jacobian ∂(Δp)/∂(bg) - position w.r.t. gyro bias.
    pub j_p_bg: Matrix3<f64>,
    /// Jacobian ∂(Δp)/∂(ba) - position w.r.t. accel bias.
    pub j_p_ba: Matrix3<f64>,
}

impl PreintegratedCovariance {
    /// Create a new covariance structure with zero covariance and identity-like Jacobians.
    pub fn new() -> Self {
        Self {
            cov: Matrix9::zeros(),
            j_r_bg: Matrix3::zeros(),
            j_v_bg: Matrix3::zeros(),
            j_v_ba: Matrix3::zeros(),
            j_p_bg: Matrix3::zeros(),
            j_p_ba: Matrix3::zeros(),
        }
    }

    /// Returns the information matrix (inverse covariance) if covariance is invertible.
    ///
    /// The information matrix is used in factor graph optimization as the weight
    /// for IMU residuals.
    pub fn information_matrix(&self) -> Option<Matrix9> {
        // Add small regularization to ensure invertibility
        let regularized = self.cov + Matrix9::identity() * 1e-10;
        regularized.try_inverse()
    }
}

impl Default for PreintegratedCovariance {
    fn default() -> Self {
        Self::new()
    }
}

/// Preintegrated motion between two frames.
///
/// Note: This struct uses `Clone` rather than `Copy` because the covariance
/// matrix (9×9 = 648 bytes) is too large for efficient copying.
#[derive(Debug, Clone)]
pub struct PreintegratedState {
    pub delta_rot: UnitQuaternion<f64>,
    pub delta_vel: Vector3<f64>,
    pub delta_pos: Vector3<f64>,
    pub dt: f64,
    /// Optional covariance tracking. When enabled, tracks uncertainty
    /// and stores bias Jacobians for first-order correction.
    pub covariance: Option<PreintegratedCovariance>,
}

impl PreintegratedState {
    pub fn identity() -> Self {
        Self {
            delta_rot: UnitQuaternion::identity(),
            delta_vel: Vector3::zeros(),
            delta_pos: Vector3::zeros(),
            dt: 0.0,
            covariance: None,
        }
    }

    /// Create an identity state with covariance tracking enabled.
    pub fn identity_with_covariance() -> Self {
        Self {
            delta_rot: UnitQuaternion::identity(),
            delta_vel: Vector3::zeros(),
            delta_pos: Vector3::zeros(),
            dt: 0.0,
            covariance: Some(PreintegratedCovariance::new()),
        }
    }

    /// Apply first-order bias correction given a change in bias estimates.
    ///
    /// This allows updating the preintegrated measurements when the bias
    /// estimate changes without re-integrating all IMU samples.
    ///
    /// # Arguments
    /// * `delta_bg` - Change in gyroscope bias estimate
    /// * `delta_ba` - Change in accelerometer bias estimate
    ///
    /// # Returns
    /// A new `PreintegratedState` with corrected measurements, or `None` if
    /// covariance tracking was not enabled.
    pub fn correct_for_bias_change(
        &self,
        delta_bg: &Vector3<f64>,
        delta_ba: &Vector3<f64>,
    ) -> Option<PreintegratedState> {
        let cov = self.covariance.as_ref()?;

        // First-order correction for rotation: ΔR_corrected = ΔR * Exp(J_R_bg * δbg)
        let delta_theta = cov.j_r_bg * delta_bg;
        let delta_rot_correction = UnitQuaternion::from_scaled_axis(delta_theta);
        let corrected_rot = self.delta_rot * delta_rot_correction;

        // First-order correction for velocity: Δv_corrected = Δv + J_v_bg * δbg + J_v_ba * δba
        let corrected_vel = self.delta_vel + cov.j_v_bg * delta_bg + cov.j_v_ba * delta_ba;

        // First-order correction for position: Δp_corrected = Δp + J_p_bg * δbg + J_p_ba * δba
        let corrected_pos = self.delta_pos + cov.j_p_bg * delta_bg + cov.j_p_ba * delta_ba;

        Some(PreintegratedState {
            delta_rot: corrected_rot,
            delta_vel: corrected_vel,
            delta_pos: corrected_pos,
            dt: self.dt,
            covariance: self.covariance.clone(),
        })
    }
}

/// IMU preintegrator: integrates high-rate IMU into a relative motion prior.
pub struct Preintegrator {
    pub bias: ImuBias,
    pub noise: ImuNoise,
    pub state: PreintegratedState,
    /// Whether to track covariance during integration.
    track_covariance: bool,
}

impl Preintegrator {
    pub fn new(bias: ImuBias, noise: ImuNoise) -> Self {
        Self {
            bias,
            noise,
            state: PreintegratedState::identity(),
            track_covariance: false,
        }
    }

    /// Create a new preintegrator with covariance tracking enabled.
    pub fn new_with_covariance(bias: ImuBias, noise: ImuNoise) -> Self {
        Self {
            bias,
            noise,
            state: PreintegratedState::identity_with_covariance(),
            track_covariance: true,
        }
    }

    /// Enable or disable covariance tracking.
    pub fn set_covariance_tracking(&mut self, enabled: bool) {
        self.track_covariance = enabled;
        if enabled && self.state.covariance.is_none() {
            self.state.covariance = Some(PreintegratedCovariance::new());
        } else if !enabled {
            self.state.covariance = None;
        }
    }

    pub fn reset(&mut self) {
        if self.track_covariance {
            self.state = PreintegratedState::identity_with_covariance();
        } else {
            self.state = PreintegratedState::identity();
        }
    }

    /// Integrate a single time step using midpoint integration.
    ///
    /// When covariance tracking is enabled, this also propagates the 9×9
    /// covariance matrix and updates the bias Jacobians.
    pub fn integrate(&mut self, prev: ImuSample, curr: ImuSample) {
        let dt = curr.timestamp_s - prev.timestamp_s;
        if dt <= 0.0 {
            return;
        }

        // Bias-corrected measurements (midpoint)
        let gyro_prev = prev.gyro - self.bias.gyro;
        let gyro_curr = curr.gyro - self.bias.gyro;
        let omega = 0.5 * (gyro_prev + gyro_curr);

        let accel_prev = prev.accel - self.bias.accel;
        let accel_curr = curr.accel - self.bias.accel;
        let accel_body = 0.5 * (accel_prev + accel_curr);

        // Incremental rotation
        let angle_axis = omega * dt;
        let delta_q_inc = UnitQuaternion::from_scaled_axis(angle_axis);
        let delta_r_inc = delta_q_inc.to_rotation_matrix().into_inner();

        // Current rotation matrix (before update)
        let delta_r = self.state.delta_rot.to_rotation_matrix().into_inner();

        // Propagate covariance and Jacobians if enabled
        if let Some(ref mut cov) = self.state.covariance {
            // Compute the right Jacobian for the incremental rotation
            let jr = right_jacobian_so3(&angle_axis);

            // Skew-symmetric matrix of acceleration
            let skew_accel = skew(&accel_body);

            // Build the state transition matrix A (9×9)
            // A = | ΔR_inc^T           0        0   |
            //     | -ΔR·[a]×·dt        I        0   |
            //     | -0.5·ΔR·[a]×·dt²   I·dt     I   |
            let mut a_mat = Matrix9::identity();

            // Top-left 3×3: ΔR_inc^T
            a_mat
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&delta_r_inc.transpose());

            // Middle-left 3×3: -ΔR·[a]×·dt
            let neg_dr_skew_dt = -delta_r * skew_accel * dt;
            a_mat.fixed_view_mut::<3, 3>(3, 0).copy_from(&neg_dr_skew_dt);

            // Bottom-left 3×3: -0.5·ΔR·[a]×·dt²
            let neg_half_dr_skew_dt2 = -0.5 * delta_r * skew_accel * dt * dt;
            a_mat
                .fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&neg_half_dr_skew_dt2);

            // Bottom-middle 3×3: I·dt
            a_mat
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(Matrix3::identity() * dt));

            // Build the noise input matrix B (9×6)
            // B = | Jr·dt      0           |
            //     | 0          ΔR·dt       |
            //     | 0          0.5·ΔR·dt²  |
            let mut b_mat = Matrix9x6::zeros();

            // Top-left 3×3: Jr·dt
            b_mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&(jr * dt));

            // Middle-right 3×3: ΔR·dt
            b_mat
                .fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(delta_r * dt));

            // Bottom-right 3×3: 0.5·ΔR·dt²
            b_mat
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(0.5 * delta_r * dt * dt));

            // Get measurement noise covariance Q (6×6)
            let q_mat = self.noise.measurement_covariance(dt);

            // Propagate covariance: Σ_{k+1} = A · Σ_k · A^T + B · Q · B^T
            cov.cov = a_mat * cov.cov * a_mat.transpose() + b_mat * q_mat * b_mat.transpose();

            // Update bias Jacobians
            // J_R_bg ← ΔR_inc^T · J_R_bg - Jr · dt
            cov.j_r_bg = delta_r_inc.transpose() * cov.j_r_bg - jr * dt;

            // J_v_bg ← J_v_bg - ΔR · [a]× · J_R_bg · dt
            cov.j_v_bg = cov.j_v_bg - delta_r * skew_accel * cov.j_r_bg * dt;

            // J_v_ba ← J_v_ba - ΔR · dt
            cov.j_v_ba = cov.j_v_ba - delta_r * dt;

            // J_p_bg ← J_p_bg + J_v_bg · dt - 0.5 · ΔR · [a]× · J_R_bg · dt²
            cov.j_p_bg =
                cov.j_p_bg + cov.j_v_bg * dt - 0.5 * delta_r * skew_accel * cov.j_r_bg * dt * dt;

            // J_p_ba ← J_p_ba + J_v_ba · dt - 0.5 · ΔR · dt²
            cov.j_p_ba = cov.j_p_ba + cov.j_v_ba * dt - 0.5 * delta_r * dt * dt;
        }

        // Update mean state (same as before)
        self.state.delta_rot = self.state.delta_rot * delta_q_inc;

        // Rotate acceleration into world frame (approx using updated rotation)
        let accel_world = self.state.delta_rot * accel_body + GRAVITY;

        // Update velocity and position
        let vel_prev = self.state.delta_vel;
        self.state.delta_vel += accel_world * dt;
        self.state.delta_pos += vel_prev * dt + 0.5 * accel_world * dt * dt;
        self.state.dt += dt;
    }

    /// Predict pose/velocity from a previous state.
    pub fn propagate(
        &self,
        prev_rot: UnitQuaternion<f64>,
        prev_pos: Vector3<f64>,
        prev_vel: Vector3<f64>,
    ) -> (UnitQuaternion<f64>, Vector3<f64>, Vector3<f64>) {
        let rot = prev_rot * self.state.delta_rot;
        let vel = prev_vel + self.state.delta_vel;
        let pos = prev_pos + self.state.delta_pos + prev_vel * self.state.dt;
        (rot, pos, vel)
    }

    /// Convenience: build rotation matrix of current delta.
    pub fn delta_rotation_matrix(&self) -> Matrix3<f64> {
        self.state.delta_rot.to_rotation_matrix().into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_preintegrator(with_covariance: bool) -> Preintegrator {
        let bias = ImuBias::zero();
        let noise = ImuNoise::default();
        if with_covariance {
            Preintegrator::new_with_covariance(bias, noise)
        } else {
            Preintegrator::new(bias, noise)
        }
    }

    #[test]
    fn test_covariance_grows_with_time() {
        let mut preint = create_test_preintegrator(true);

        // Integrate several samples
        let samples: Vec<ImuSample> = (0..10)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.01,
                accel: Vector3::new(0.0, 0.0, 9.81), // Counteract gravity
                gyro: Vector3::new(0.01, 0.0, 0.0),  // Small rotation
            })
            .collect();

        for i in 0..samples.len() - 1 {
            preint.integrate(samples[i], samples[i + 1]);
        }

        let cov = preint.state.covariance.as_ref().unwrap();

        // Covariance diagonal elements should be positive
        for i in 0..9 {
            assert!(cov.cov[(i, i)] > 0.0, "Diagonal element {} should be positive", i);
        }
    }

    #[test]
    fn test_covariance_symmetry() {
        let mut preint = create_test_preintegrator(true);

        let samples: Vec<ImuSample> = (0..5)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.01,
                accel: Vector3::new(0.1, 0.2, 9.81),
                gyro: Vector3::new(0.01, 0.02, 0.03),
            })
            .collect();

        for i in 0..samples.len() - 1 {
            preint.integrate(samples[i], samples[i + 1]);
        }

        let cov = preint.state.covariance.as_ref().unwrap();

        // Covariance should be symmetric
        for i in 0..9 {
            for j in 0..9 {
                assert_relative_eq!(cov.cov[(i, j)], cov.cov[(j, i)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_bias_correction_small_change() {
        let mut preint = create_test_preintegrator(true);

        let samples: Vec<ImuSample> = (0..5)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.01,
                accel: Vector3::new(0.0, 0.0, 9.81),
                gyro: Vector3::new(0.1, 0.0, 0.0),
            })
            .collect();

        for i in 0..samples.len() - 1 {
            preint.integrate(samples[i], samples[i + 1]);
        }

        let original_pos = preint.state.delta_pos;
        let original_vel = preint.state.delta_vel;

        // Apply small bias correction
        let delta_bg = Vector3::new(1e-5, 0.0, 0.0);
        let delta_ba = Vector3::new(0.0, 0.0, 1e-4);

        let corrected = preint
            .state
            .correct_for_bias_change(&delta_bg, &delta_ba)
            .unwrap();

        // Correction should be small for small bias change
        let pos_diff = (corrected.delta_pos - original_pos).norm();
        let vel_diff = (corrected.delta_vel - original_vel).norm();

        assert!(pos_diff < 1e-3, "Position correction too large: {}", pos_diff);
        assert!(vel_diff < 1e-3, "Velocity correction too large: {}", vel_diff);
    }

    #[test]
    fn test_information_matrix() {
        let mut preint = create_test_preintegrator(true);

        // Integrate for longer to build up meaningful covariance
        let samples: Vec<ImuSample> = (0..100)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.01,
                accel: Vector3::new(0.5, 0.3, 9.81),
                gyro: Vector3::new(0.1, 0.05, 0.02),
            })
            .collect();

        for i in 0..samples.len() - 1 {
            preint.integrate(samples[i], samples[i + 1]);
        }

        let cov = preint.state.covariance.as_ref().unwrap();
        let info_mat = cov.information_matrix();

        assert!(info_mat.is_some(), "Information matrix should be computable");

        // Verify (Σ + εI)⁻¹ is reasonable by checking it produces finite values
        let info = info_mat.unwrap();
        for i in 0..9 {
            for j in 0..9 {
                assert!(info[(i, j)].is_finite(), "Information matrix should be finite");
            }
        }

        // The regularized inverse won't give exact identity, but diagonal should be positive
        // and the product should be approximately identity for large covariance
        let product = (cov.cov + Matrix9::identity() * 1e-10) * info;
        for i in 0..9 {
            // Diagonal should be close to 1
            assert!(product[(i, i)] > 0.5, "Diagonal should be positive: {}", product[(i, i)]);
            assert!(product[(i, i)] < 2.0, "Diagonal should be bounded: {}", product[(i, i)]);
        }
    }

    #[test]
    fn test_backward_compatibility() {
        // Without covariance tracking, behavior should be unchanged
        let mut preint = create_test_preintegrator(false);

        let samples: Vec<ImuSample> = (0..5)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.01,
                accel: Vector3::new(0.0, 0.0, 9.81),
                gyro: Vector3::zeros(),
            })
            .collect();

        for i in 0..samples.len() - 1 {
            preint.integrate(samples[i], samples[i + 1]);
        }

        assert!(preint.state.covariance.is_none());
        assert!(preint.state.dt > 0.0);
    }

    #[test]
    fn test_reset_preserves_covariance_setting() {
        let mut preint = create_test_preintegrator(true);

        let sample1 = ImuSample {
            timestamp_s: 0.0,
            accel: Vector3::new(0.0, 0.0, 9.81),
            gyro: Vector3::zeros(),
        };
        let sample2 = ImuSample {
            timestamp_s: 0.01,
            accel: Vector3::new(0.0, 0.0, 9.81),
            gyro: Vector3::zeros(),
        };

        preint.integrate(sample1, sample2);
        assert!(preint.state.covariance.is_some());

        preint.reset();
        assert!(preint.state.covariance.is_some());
        assert_eq!(preint.state.dt, 0.0);
    }
}
