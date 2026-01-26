use nalgebra::{Matrix3, UnitQuaternion, Vector3};

use super::sample::{ImuBias, ImuNoise, ImuSample, GRAVITY};
use super::types::{Matrix9, Matrix9x6, Matrix15};
use crate::geometry::{right_jacobian_so3, skew};

/// Covariance and bias Jacobians for preintegrated measurements.
///
/// Tracks uncertainty in the 15-dimensional state [δθ, δv, δp, δbg, δba] and stores
/// Jacobians for first-order bias correction following Forster et al.
///
/// The 15x15 covariance matrix structure:
/// - [0:3, 0:3]: rotation covariance
/// - [3:6, 3:6]: velocity covariance
/// - [6:9, 6:9]: position covariance
/// - [9:12, 9:12]: gyro bias covariance
/// - [12:15, 12:15]: accel bias covariance
#[derive(Debug, Clone)]
pub struct PreintegratedCovariance {
    /// 15×15 covariance matrix for [δθ, δv, δp, δbg, δba].
    pub cov: Matrix15,
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
            cov: Matrix15::zeros(),
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
    /// for IMU residuals. Only returns the 9x9 state portion (excluding bias).
    pub fn information_matrix(&self) -> Option<Matrix9> {
        // Extract the 9x9 state covariance portion
        let state_cov = self.cov.fixed_view::<9, 9>(0, 0).clone_owned();
        // Add small regularization to ensure invertibility
        let regularized = state_cov + Matrix9::identity() * 1e-10;
        regularized.try_inverse()
    }

    /// Returns the full 15x15 information matrix if invertible.
    pub fn full_information_matrix(&self) -> Option<Matrix15> {
        let regularized = self.cov + Matrix15::identity() * 1e-10;
        regularized.try_inverse()
    }
}

impl Default for PreintegratedCovariance {
    fn default() -> Self {
        Self::new()
    }
}

/// Stored IMU measurement for reintegration.
#[derive(Debug, Clone, Copy)]
pub struct IntegrableMeasurement {
    pub accel: Vector3<f64>,
    pub gyro: Vector3<f64>,
    pub dt: f64,
}

/// Preintegrated motion between two frames.
///
/// Note: This struct uses `Clone` rather than `Copy` because the covariance
/// matrix (15×15 = 1800 bytes) and measurement storage are too large for efficient copying.
#[derive(Debug, Clone)]
pub struct PreintegratedState {
    pub delta_rot: UnitQuaternion<f64>,
    pub delta_vel: Vector3<f64>,
    pub delta_pos: Vector3<f64>,
    pub dt: f64,
    /// Optional covariance tracking. When enabled, tracks uncertainty
    /// and stores bias Jacobians for first-order correction.
    pub covariance: Option<PreintegratedCovariance>,
    /// Stored measurements for reintegration when bias changes significantly.
    /// Only populated when covariance tracking is enabled.
    pub measurements: Vec<IntegrableMeasurement>,
    /// Original bias used during integration.
    pub original_bias: ImuBias,
}

impl PreintegratedState {
    pub fn identity() -> Self {
        Self {
            delta_rot: UnitQuaternion::identity(),
            delta_vel: Vector3::zeros(),
            delta_pos: Vector3::zeros(),
            dt: 0.0,
            covariance: None,
            measurements: Vec::new(),
            original_bias: ImuBias::zero(),
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
            measurements: Vec::new(),
            original_bias: ImuBias::zero(),
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
            measurements: self.measurements.clone(),
            original_bias: self.original_bias,
        })
    }

    /// Get corrected rotation given current and original bias.
    pub fn get_delta_rotation(&self, current_bias: &ImuBias) -> UnitQuaternion<f64> {
        if let Some(ref cov) = self.covariance {
            let delta_bg = current_bias.gyro - self.original_bias.gyro;
            let delta_theta = cov.j_r_bg * delta_bg;
            self.delta_rot * UnitQuaternion::from_scaled_axis(delta_theta)
        } else {
            self.delta_rot
        }
    }

    /// Get corrected velocity given current and original bias.
    pub fn get_delta_velocity(&self, current_bias: &ImuBias) -> Vector3<f64> {
        if let Some(ref cov) = self.covariance {
            let delta_bg = current_bias.gyro - self.original_bias.gyro;
            let delta_ba = current_bias.accel - self.original_bias.accel;
            self.delta_vel + cov.j_v_bg * delta_bg + cov.j_v_ba * delta_ba
        } else {
            self.delta_vel
        }
    }

    /// Get corrected position given current and original bias.
    pub fn get_delta_position(&self, current_bias: &ImuBias) -> Vector3<f64> {
        if let Some(ref cov) = self.covariance {
            let delta_bg = current_bias.gyro - self.original_bias.gyro;
            let delta_ba = current_bias.accel - self.original_bias.accel;
            self.delta_pos + cov.j_p_bg * delta_bg + cov.j_p_ba * delta_ba
        } else {
            self.delta_pos
        }
    }

    /// Merge another preintegrated state into this one.
    ///
    /// Used when merging two temporal segments (e.g., during keyframe culling).
    /// The other state should come BEFORE this one in time.
    pub fn merge_previous(&mut self, previous: &PreintegratedState) {
        // Merge measurements
        let mut new_measurements = previous.measurements.clone();
        new_measurements.extend(self.measurements.iter().cloned());
        self.measurements = new_measurements;

        // Merge preintegrated values:
        // Combined rotation: ΔR_merged = ΔR_prev * ΔR_curr
        let prev_rot_mat = previous.delta_rot.to_rotation_matrix().into_inner();
        let new_rot = previous.delta_rot * self.delta_rot;

        // Combined velocity: Δv_merged = Δv_prev + ΔR_prev * Δv_curr
        let new_vel = previous.delta_vel + prev_rot_mat * self.delta_vel;

        // Combined position: Δp_merged = Δp_prev + Δv_prev * dt_curr + ΔR_prev * Δp_curr
        let new_pos = previous.delta_pos
            + previous.delta_vel * self.dt
            + prev_rot_mat * self.delta_pos;

        // Combined time
        let new_dt = previous.dt + self.dt;

        // Update covariance and Jacobians if both have them
        if let (Some(prev_cov), Some(curr_cov)) = (&previous.covariance, &mut self.covariance) {
            // Merge Jacobians:
            // J_R_bg_merged = ΔR_curr^T * J_R_bg_prev + J_R_bg_curr
            curr_cov.j_r_bg = self.delta_rot.inverse().to_rotation_matrix().into_inner()
                * prev_cov.j_r_bg + curr_cov.j_r_bg;

            // J_v_bg_merged = J_v_bg_prev + ΔR_prev * J_v_bg_curr - ΔR_prev * [Δv_curr]× * J_R_bg_prev
            let skew_vel = skew(&self.delta_vel);
            curr_cov.j_v_bg = prev_cov.j_v_bg
                + prev_rot_mat * curr_cov.j_v_bg
                - prev_rot_mat * skew_vel * prev_cov.j_r_bg;

            // J_v_ba_merged = J_v_ba_prev + ΔR_prev * J_v_ba_curr
            curr_cov.j_v_ba = prev_cov.j_v_ba + prev_rot_mat * curr_cov.j_v_ba;

            // J_p_bg_merged = J_p_bg_prev + J_v_bg_prev * dt_curr + ΔR_prev * J_p_bg_curr
            //                 - ΔR_prev * [Δp_curr]× * J_R_bg_prev
            let skew_pos = skew(&self.delta_pos);
            curr_cov.j_p_bg = prev_cov.j_p_bg
                + prev_cov.j_v_bg * self.dt
                + prev_rot_mat * curr_cov.j_p_bg
                - prev_rot_mat * skew_pos * prev_cov.j_r_bg;

            // J_p_ba_merged = J_p_ba_prev + J_v_ba_prev * dt_curr + ΔR_prev * J_p_ba_curr
            curr_cov.j_p_ba = prev_cov.j_p_ba
                + prev_cov.j_v_ba * self.dt
                + prev_rot_mat * curr_cov.j_p_ba;

            // Merge covariance matrices (simplified: just sum them)
            // A more accurate approach would involve the full state transition matrix
            curr_cov.cov = prev_cov.cov + curr_cov.cov;
        }

        self.delta_rot = new_rot;
        self.delta_vel = new_vel;
        self.delta_pos = new_pos;
        self.dt = new_dt;
        self.original_bias = previous.original_bias;
    }
}

/// IMU preintegrator: integrates high-rate IMU into a relative motion prior.
pub struct Preintegrator {
    pub bias: ImuBias,
    pub noise: ImuNoise,
    pub state: PreintegratedState,
    /// Whether to track covariance during integration.
    track_covariance: bool,
    /// Whether to store measurements for reintegration.
    store_measurements: bool,
}

impl Preintegrator {
    pub fn new(bias: ImuBias, noise: ImuNoise) -> Self {
        Self {
            bias,
            noise,
            state: PreintegratedState::identity(),
            track_covariance: false,
            store_measurements: false,
        }
    }

    /// Create a new preintegrator with covariance tracking enabled.
    pub fn new_with_covariance(bias: ImuBias, noise: ImuNoise) -> Self {
        let mut state = PreintegratedState::identity_with_covariance();
        state.original_bias = bias;
        Self {
            bias,
            noise,
            state,
            track_covariance: true,
            store_measurements: true,
        }
    }

    /// Enable or disable covariance tracking.
    pub fn set_covariance_tracking(&mut self, enabled: bool) {
        self.track_covariance = enabled;
        self.store_measurements = enabled;
        if enabled && self.state.covariance.is_none() {
            self.state.covariance = Some(PreintegratedCovariance::new());
        } else if !enabled {
            self.state.covariance = None;
            self.state.measurements.clear();
        }
    }

    pub fn reset(&mut self) {
        if self.track_covariance {
            self.state = PreintegratedState::identity_with_covariance();
            self.state.original_bias = self.bias;
        } else {
            self.state = PreintegratedState::identity();
        }
    }

    /// Reset with a new bias.
    pub fn reset_with_bias(&mut self, new_bias: ImuBias) {
        self.bias = new_bias;
        self.reset();
    }

    /// Reintegrate all stored measurements with the current bias.
    ///
    /// This is called when the bias estimate changes significantly and
    /// first-order correction is no longer accurate.
    pub fn reintegrate(&mut self) {
        if self.state.measurements.is_empty() {
            return;
        }

        let measurements = self.state.measurements.clone();

        // Reset state but keep covariance tracking setting
        if self.track_covariance {
            self.state = PreintegratedState::identity_with_covariance();
            self.state.original_bias = self.bias;
        } else {
            self.state = PreintegratedState::identity();
        }

        // Reintegrate all measurements with current bias
        for m in measurements {
            self.integrate_measurement(m.accel, m.gyro, m.dt);
        }
    }

    /// Integrate a single measurement (internal, doesn't use midpoint).
    fn integrate_measurement(&mut self, accel: Vector3<f64>, gyro: Vector3<f64>, dt: f64) {
        if dt <= 0.0 {
            return;
        }

        // Bias-corrected measurements
        let omega = gyro - self.bias.gyro;
        let accel_body = accel - self.bias.accel;

        // Store measurement if enabled
        if self.store_measurements {
            self.state.measurements.push(IntegrableMeasurement {
                accel,
                gyro,
                dt,
            });
        }

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

            // Build the state transition matrix A (9×9 portion)
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

            // Propagate the 9x9 state covariance: Σ_{k+1} = A · Σ_k · A^T + B · Q · B^T
            let state_cov = cov.cov.fixed_view::<9, 9>(0, 0).clone_owned();
            let new_state_cov = a_mat * state_cov * a_mat.transpose() + b_mat * q_mat * b_mat.transpose();
            cov.cov.fixed_view_mut::<9, 9>(0, 0).copy_from(&new_state_cov);

            // Add bias random walk to bias covariance (bottom-right 6x6)
            let bias_walk_cov = self.noise.bias_walk_covariance(dt);
            let bias_cov = cov.cov.fixed_view::<6, 6>(9, 9).clone_owned();
            cov.cov.fixed_view_mut::<6, 6>(9, 9).copy_from(&(bias_cov + bias_walk_cov));

            // Update bias Jacobians (following C++ order: update BEFORE rotation update)
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

        // Update mean state
        self.state.delta_rot = self.state.delta_rot * delta_q_inc;

        // Rotate acceleration into world frame (using updated rotation)
        let accel_world = self.state.delta_rot * accel_body + GRAVITY;

        // Update velocity and position
        let vel_prev = self.state.delta_vel;
        self.state.delta_vel += accel_world * dt;
        self.state.delta_pos += vel_prev * dt + 0.5 * accel_world * dt * dt;
        self.state.dt += dt;
    }

    /// Integrate a single time step using midpoint integration.
    ///
    /// When covariance tracking is enabled, this also propagates the 15×15
    /// covariance matrix and updates the bias Jacobians.
    pub fn integrate(&mut self, prev: ImuSample, curr: ImuSample) {
        let dt = curr.timestamp_s - prev.timestamp_s;
        if dt <= 0.0 {
            return;
        }

        // Midpoint integration for better accuracy
        let gyro = 0.5 * (prev.gyro + curr.gyro);
        let accel = 0.5 * (prev.accel + curr.accel);

        self.integrate_measurement(accel, gyro, dt);
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

        // Covariance diagonal elements should be positive (state portion)
        for i in 0..9 {
            assert!(cov.cov[(i, i)] > 0.0, "Diagonal element {} should be positive", i);
        }

        // Bias covariance should also grow
        for i in 9..15 {
            assert!(cov.cov[(i, i)] > 0.0, "Bias diagonal element {} should be positive", i);
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
        for i in 0..15 {
            for j in 0..15 {
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
    fn test_measurement_storage() {
        let mut preint = create_test_preintegrator(true);

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

        // Should have stored 4 measurements (one per integration step)
        assert_eq!(preint.state.measurements.len(), 4);
    }

    #[test]
    fn test_reintegration() {
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

        let original_rot = preint.state.delta_rot;
        let original_dt = preint.state.dt;

        // Reintegrate with same bias should give same result
        preint.reintegrate();

        assert_relative_eq!(preint.state.dt, original_dt, epsilon = 1e-10);
        assert_relative_eq!(preint.state.delta_rot.w, original_rot.w, epsilon = 1e-10);
    }

    #[test]
    fn test_merge_previous() {
        let mut preint1 = create_test_preintegrator(true);
        let mut preint2 = create_test_preintegrator(true);

        // First integration
        let samples1: Vec<ImuSample> = (0..3)
            .map(|i| ImuSample {
                timestamp_s: i as f64 * 0.01,
                accel: Vector3::new(0.0, 0.0, 9.81),
                gyro: Vector3::new(0.1, 0.0, 0.0),
            })
            .collect();

        for i in 0..samples1.len() - 1 {
            preint1.integrate(samples1[i], samples1[i + 1]);
        }

        // Second integration
        let samples2: Vec<ImuSample> = (0..3)
            .map(|i| ImuSample {
                timestamp_s: (i + 2) as f64 * 0.01,
                accel: Vector3::new(0.0, 0.0, 9.81),
                gyro: Vector3::new(0.1, 0.0, 0.0),
            })
            .collect();

        for i in 0..samples2.len() - 1 {
            preint2.integrate(samples2[i], samples2[i + 1]);
        }

        let total_dt_before = preint1.state.dt + preint2.state.dt;

        // Merge preint1 into preint2
        preint2.state.merge_previous(&preint1.state);

        // Total time should be sum
        assert_relative_eq!(preint2.state.dt, total_dt_before, epsilon = 1e-10);

        // Should have all measurements
        assert_eq!(preint2.state.measurements.len(), 4);
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
        assert!(preint.state.measurements.is_empty());
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
        assert!(preint.state.measurements.is_empty());
    }
}
