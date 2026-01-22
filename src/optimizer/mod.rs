//! Optimization module for Bundle Adjustment.
//!
//! Implements Gauss-Newton optimization for:
//! - Local Bundle Adjustment (keyframe poses + map point positions)
//! - Pose-only optimization (for tracking)

pub mod local_ba;

pub use local_ba::{local_bundle_adjustment, LocalBAConfig, LocalBAResult};
