use nalgebra::Vector3;

use super::types::{Matrix6, Matrix12};

/// Gravity vector in world frame (m/s^2).
pub const GRAVITY: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);

/// IMU noise parameters (1-sigma).
///
/// Includes both measurement noise and bias random walk parameters.
#[derive(Debug, Clone, Copy)]
pub struct ImuNoise {
    /// Gyroscope measurement noise (rad/s/√Hz).
    pub sigma_gyro: f64,
    /// Accelerometer measurement noise (m/s²/√Hz).
    pub sigma_accel: f64,
    /// Gyroscope bias random walk (rad/s²/√Hz).
    pub sigma_gyro_walk: f64,
    /// Accelerometer bias random walk (m/s³/√Hz).
    pub sigma_accel_walk: f64,
}

impl ImuNoise {
    pub fn default() -> Self {
        Self {
            // Approximate EuRoC noise values
            sigma_gyro: 1.7e-4,
            sigma_accel: 2.0e-3,
            // Bias random walk (typical values for MEMS IMUs)
            sigma_gyro_walk: 1.9e-5,
            sigma_accel_walk: 3.0e-3,
        }
    }

    /// Create noise parameters with custom values.
    pub fn new(sigma_gyro: f64, sigma_accel: f64, sigma_gyro_walk: f64, sigma_accel_walk: f64) -> Self {
        Self {
            sigma_gyro,
            sigma_accel,
            sigma_gyro_walk,
            sigma_accel_walk,
        }
    }

    /// Constructs the 6×6 measurement noise covariance matrix Q for a time step dt.
    ///
    /// Q = diag(σ_g², σ_g², σ_g², σ_a², σ_a², σ_a²) * dt
    ///
    /// The noise is scaled by dt because discrete-time noise variance accumulates
    /// proportionally with integration time.
    pub fn measurement_covariance(&self, dt: f64) -> Matrix6 {
        let sigma_gyro_sq = self.sigma_gyro * self.sigma_gyro * dt;
        let sigma_accel_sq = self.sigma_accel * self.sigma_accel * dt;

        Matrix6::from_diagonal(&nalgebra::Vector6::new(
            sigma_gyro_sq,
            sigma_gyro_sq,
            sigma_gyro_sq,
            sigma_accel_sq,
            sigma_accel_sq,
            sigma_accel_sq,
        ))
    }

    /// Constructs the 12×12 full noise covariance matrix for a time step dt.
    ///
    /// Includes measurement noise and bias random walk:
    /// Q = diag(σ_g², σ_g², σ_g², σ_a², σ_a², σ_a², σ_bg², σ_bg², σ_bg², σ_ba², σ_ba², σ_ba²) * dt
    pub fn full_covariance(&self, dt: f64) -> Matrix12 {
        let sigma_gyro_sq = self.sigma_gyro * self.sigma_gyro * dt;
        let sigma_accel_sq = self.sigma_accel * self.sigma_accel * dt;
        let sigma_bg_sq = self.sigma_gyro_walk * self.sigma_gyro_walk * dt;
        let sigma_ba_sq = self.sigma_accel_walk * self.sigma_accel_walk * dt;

        Matrix12::from_diagonal(&nalgebra::SVector::<f64, 12>::from_row_slice(&[
            sigma_gyro_sq, sigma_gyro_sq, sigma_gyro_sq,
            sigma_accel_sq, sigma_accel_sq, sigma_accel_sq,
            sigma_bg_sq, sigma_bg_sq, sigma_bg_sq,
            sigma_ba_sq, sigma_ba_sq, sigma_ba_sq,
        ]))
    }

    /// Get bias random walk covariance (6×6) for bias states [δbg, δba].
    ///
    /// This is added to the bias portion of the covariance at each timestep.
    pub fn bias_walk_covariance(&self, dt: f64) -> Matrix6 {
        let sigma_bg_sq = self.sigma_gyro_walk * self.sigma_gyro_walk * dt;
        let sigma_ba_sq = self.sigma_accel_walk * self.sigma_accel_walk * dt;

        Matrix6::from_diagonal(&nalgebra::Vector6::new(
            sigma_bg_sq, sigma_bg_sq, sigma_bg_sq,
            sigma_ba_sq, sigma_ba_sq, sigma_ba_sq,
        ))
    }
}

/// IMU biases.
#[derive(Debug, Clone, Copy)]
pub struct ImuBias {
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
}

impl ImuBias {
    pub fn zero() -> Self {
        Self {
            gyro: Vector3::zeros(),
            accel: Vector3::zeros(),
        }
    }
}

/// Single IMU measurement.
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    pub timestamp_s: f64,
    pub accel: Vector3<f64>,
    pub gyro: Vector3<f64>,
}
