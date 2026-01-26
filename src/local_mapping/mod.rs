//! Local Mapping thread for ORB-SLAM3.
//!
//! The Local Mapping thread is responsible for:
//! - Inserting new keyframes into the map
//! - Triangulating new map points from stereo observations
//! - Multi-frame triangulation with geometric validation
//! - SearchInNeighbors: map point fusion between covisible keyframes
//! - Maintaining the covisibility graph
//! - Running local bundle adjustment (deferred)
//! - Culling redundant keyframes and map points (deferred)
//! - IMU initialization (for visual-inertial mode)

pub mod imu_init;
mod local_mapper;
pub mod search_in_neighbors;
pub mod triangulation;

pub use imu_init::{apply_imu_init, check_sufficient_motion, initialize_imu, ImuInitResult};
pub use local_mapper::LocalMapper;
pub use search_in_neighbors::{
    search_in_neighbors, search_in_neighbors_with_config, SearchInNeighborsConfig,
    SearchInNeighborsResult,
};
pub use triangulation::{triangulate_from_neighbors, TriangulationConfig, TriangulationResult};
