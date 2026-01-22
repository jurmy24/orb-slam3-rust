//! Type aliases for IMU covariance matrices.
//!
//! These types support the 9-dimensional state space [δθ, δv, δp]
//! used in preintegration covariance propagation.

use nalgebra::{SMatrix, SVector};

/// 9×9 covariance matrix for preintegrated state [δθ, δv, δp].
pub type Matrix9 = SMatrix<f64, 9, 9>;

/// 9-dimensional state vector [δθ, δv, δp].
pub type Vector9 = SVector<f64, 9>;

/// 9×6 Jacobian matrix (state w.r.t. noise).
pub type Matrix9x6 = SMatrix<f64, 9, 6>;

/// 6×6 measurement noise covariance matrix.
pub type Matrix6 = SMatrix<f64, 6, 6>;
