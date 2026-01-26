//! Optimization module for Bundle Adjustment.
//!
//! Implements optimization for:
//! - Local Bundle Adjustment (keyframe poses + map point positions)
//! - Local Inertial BA (poses + velocities + biases + IMU constraints)
//! - Pose-only optimization with IMU (for tracking)
//!
//! The LM-based implementation (`local_ba_lm`) is recommended as it provides
//! stable convergence. The original Gauss-Newton implementation (`local_ba`)
//! is kept for reference but has known issues with Jacobian signs.

pub mod imu_factors;
pub mod local_ba;
pub mod local_ba_lm;
pub mod local_inertial_ba;
pub mod pose_inertial_optim;

pub use imu_factors::{compute_imu_residual, ImuResidual, VIStateLayout};
pub use local_ba::{local_bundle_adjustment, LocalBAConfig, LocalBAResult};
pub use local_ba_lm::{
    local_bundle_adjustment_lm, LocalBAConfigLM, LocalBAResultLM,
    // Three-phase visual BA exports
    collect_visual_ba_data, solve_visual_ba, apply_visual_ba_results,
    VisualBAProblemData, VisualObservation, VisualBAResultData,
};
pub use local_inertial_ba::{
    local_inertial_ba, LocalInertialBAConfig, LocalInertialBAResult,
    // Three-phase inertial BA exports
    collect_inertial_ba_data, solve_inertial_ba, apply_inertial_ba_results,
    InertialBAProblemData, InertialVisualObs, ImuEdgeData, InertialBAResultData,
};
pub use pose_inertial_optim::{
    pose_inertial_optimization, PoseInertialConfig, PoseInertialResult, PoseObservation,
};
