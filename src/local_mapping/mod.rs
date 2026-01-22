//! Local Mapping thread for ORB-SLAM3.
//!
//! The Local Mapping thread is responsible for:
//! - Inserting new keyframes into the map
//! - Triangulating new map points from stereo observations
//! - Maintaining the covisibility graph
//! - Running local bundle adjustment (deferred)
//! - Culling redundant keyframes and map points (deferred)

mod local_mapper;

pub use local_mapper::LocalMapper;
