use nalgebra::Vector3;

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
