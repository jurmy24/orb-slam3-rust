//! Loop Closing module for ORB-SLAM3.
//!
//! This module implements loop detection, geometric verification, and pose graph correction
//! to eliminate accumulated drift over long trajectories.
//!
//! # Architecture
//!
//! The loop closing pipeline consists of:
//! 1. **Detection** (`detector.rs`): BoW-based loop candidate detection with consistency checking
//! 2. **Verification** (`sim3_solver.rs`): Geometric verification via Sim3 RANSAC
//! 3. **Correction** (`corrector.rs`): Pose graph optimization and map point fusion
//! 4. **Thread** (`loop_closer.rs`): Main loop closing thread receiving keyframes
//!
//! # Threading Model
//!
//! The LoopCloser runs in its own thread, receiving keyframes from LocalMapping via a channel.
//! It uses the three-phase locking pattern for lock efficiency:
//! - Phase 1 (COLLECT): Read lock to gather data
//! - Phase 2 (SOLVE): No lock, pure computation
//! - Phase 3 (APPLY): Write lock to update the map

pub mod corrector;
pub mod detector;
pub mod loop_closer;
pub mod sim3_solver;

pub use corrector::{correct_loop, VerifiedLoop};
pub use detector::{detect_loop_candidates, ConsistencyChecker, LoopCandidate};
pub use loop_closer::{LoopCloser, LoopCloserConfig};
pub use sim3_solver::{compute_sim3_from_matches, compute_sim3_ransac, Sim3Result, Sim3SolverConfig};
