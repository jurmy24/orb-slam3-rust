//! Optimization module for Bundle Adjustment.
//!
//! Implements optimization for:
//! - Local Bundle Adjustment (keyframe poses + map point positions)
//! - Local Inertial BA (poses + velocities + biases + IMU constraints)
//! - Pose-only optimization with IMU (for tracking)
//! - Pose Graph optimization (for loop closing)
//! - Global BA (full map optimization after loop closing)
//!
//! The LM-based implementation (`local_ba_lm`) is recommended as it provides
//! stable convergence. The original Gauss-Newton implementation (`local_ba`)
//! is kept for reference but has known issues with Jacobian signs.

pub mod global_ba;
pub mod imu_factors;
pub mod inertial_init_optim;
pub mod local_ba;
pub mod local_ba_lm;
pub mod local_inertial_ba;
pub mod pose_graph;
pub mod pose_inertial_optim;

pub use global_ba::{
    collect_global_ba_data, solve_global_ba, apply_global_ba_results, run_global_ba,
    GlobalBAConfig, GlobalBAProblemData, GlobalBAObservation, GlobalBAResult,
};
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
pub use pose_graph::{
    collect_pose_graph_data, solve_pose_graph, apply_pose_graph_results,
    PoseGraphConfig, PoseGraphProblemData, PoseGraphEdge, EdgeType, PoseGraphResult,
};
pub use pose_inertial_optim::{
    pose_inertial_optimization, PoseInertialConfig, PoseInertialResult, PoseObservation,
};
pub use inertial_init_optim::{
    optimize_inertial_init_full, optimize_inertial_init_bias_only,
    optimize_inertial_init_scale_refinement, apply_inertial_init_result,
    InertialInitConfig, InertialInitOptimResult, InertialInitProblem,
};
