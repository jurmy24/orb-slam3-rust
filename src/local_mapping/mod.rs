//! Local Mapping thread for ORB-SLAM3.
//!
//! The Local Mapping thread is responsible for:
//! - Inserting new keyframes into the map
//! - Triangulating new map points from stereo observations
//! - Multi-frame triangulation with geometric validation
//! - Maintaining the covisibility graph
//! - Running local bundle adjustment (deferred)
//! - Culling redundant keyframes and map points (deferred)
//! - IMU initialization (for visual-inertial mode)

pub mod imu_init;
mod local_mapper;
pub mod triangulation;

pub use imu_init::{apply_imu_init, initialize_imu, ImuInitResult};
pub use local_mapper::LocalMapper;
pub use triangulation::{triangulate_from_neighbors, TriangulationConfig, TriangulationResult};
