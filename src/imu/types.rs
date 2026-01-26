//! Type aliases for IMU covariance matrices.
//!
//! These types support the 15-dimensional state space [δθ, δv, δp, δbg, δba]
//! used in preintegration covariance propagation.
//!
//! The 15x15 covariance matrix structure:
//! - [0:3, 0:3]: rotation covariance
//! - [3:6, 3:6]: velocity covariance
//! - [6:9, 6:9]: position covariance
//! - [9:12, 9:12]: gyro bias covariance
//! - [12:15, 12:15]: accel bias covariance

use nalgebra::{SMatrix, SVector};

/// 9×9 covariance matrix for preintegrated state [δθ, δv, δp] (legacy).
pub type Matrix9 = SMatrix<f64, 9, 9>;

/// 15×15 covariance matrix for full state [δθ, δv, δp, δbg, δba].
pub type Matrix15 = SMatrix<f64, 15, 15>;

/// 9-dimensional state vector [δθ, δv, δp].
pub type Vector9 = SVector<f64, 9>;

/// 15-dimensional state vector [δθ, δv, δp, δbg, δba].
pub type Vector15 = SVector<f64, 15>;

/// 9×6 Jacobian matrix (state w.r.t. noise).
pub type Matrix9x6 = SMatrix<f64, 9, 6>;

/// 15×12 Jacobian matrix (full state w.r.t. full noise [gyro, accel, bg_walk, ba_walk]).
pub type Matrix15x12 = SMatrix<f64, 15, 12>;

/// 6×6 measurement noise covariance matrix.
pub type Matrix6 = SMatrix<f64, 6, 6>;

/// 12×12 full noise covariance matrix [gyro, accel, bg_walk, ba_walk].
pub type Matrix12 = SMatrix<f64, 12, 12>;
