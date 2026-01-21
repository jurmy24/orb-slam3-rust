//! Atlas module - Multi-map SLAM data structures.
//!
//! The Atlas is the top-level data structure in ORB-SLAM3 that manages:
//! - Active map (current working map)
//! - Non-active maps (for multi-session and map merging)
//! - DBoW2 KeyFrame database (for place recognition)
//!
//! # Current Implementation Status
//!
//! This module currently implements the **Map** component:
//! - [`map::KeyFrame`] - KeyFrames with covisibility and spanning tree
//! - [`map::MapPoint`] - 3D landmarks with observation tracking
//! - [`map::Map`] - Container with association and culling operations
//!
//! # Future Components
//!
//! - `keyframe_db` - DBoW2 visual vocabulary and recognition database
//! - `multi_map` - Non-active map management
//! - `atlas` - Full Atlas container with active/non-active map switching

pub mod map;
pub mod atlas;
pub mod keyframe_db;

// Re-export commonly used types
pub use map::{KeyFrame, KeyFrameId, Map, MapPoint, MapPointId};
