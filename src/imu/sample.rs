use nalgebra::Vector3;

use super::types::Matrix6;

/// Gravity vector in world frame (m/s^2).
pub const GRAVITY: Vector3<f64> = Vector3::new(0.0, 0.0, -9.81);

/// IMU noise parameters (1-sigma).
#[derive(Debug, Clone, Copy)]
pub struct ImuNoise {
    pub sigma_gyro: f64,
    pub sigma_accel: f64,
}

impl ImuNoise {
    pub fn default() -> Self {
        Self {
            // Approximate EuRoC noise values
            sigma_gyro: 1.7e-4,
            sigma_accel: 2.0e-3,
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
