//! SLAM system orchestration and thread management.
//!
//! This module contains the top-level `SlamSystem` that spawns and coordinates
//! the Tracking and Local Mapping threads, along with shared state and
//! inter-thread messaging types.

pub mod messages;
pub mod shared_state;
mod slam_system;

pub use messages::NewKeyFrameMsg;
pub use shared_state::SharedState;
pub use slam_system::SlamSystem;
