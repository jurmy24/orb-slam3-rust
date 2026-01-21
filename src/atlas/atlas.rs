//! Atlas - multi-map container for ORB-SLAM3.
//!
//! The Atlas owns one or more `Map` instances:
//! - One **active** map where Tracking, Local Mapping and Loop Closing operate.
//! - Zero or more **non‑active** maps which may be re‑activated via
//!   relocalization or map merging.
//!
//! It also owns the shared `KeyFrameDatabase` used for place recognition
//! (global relocalization and map merging).

use crate::atlas::map::Map;

use super::keyframe_db::KeyFrameDatabase;

/// Index of a map inside the Atlas.
pub type MapIndex = usize;

/// Top‑level multi‑map container, mirroring ORB‑SLAM3's Atlas.
pub struct Atlas {
    /// All maps managed by the Atlas.
    maps: Vec<Map>,
    /// Index of the active map inside `maps`.
    active_map_idx: MapIndex,
    /// Shared keyframe database for place recognition.
    pub keyframe_db: KeyFrameDatabase,
}

impl Atlas {
    /// Create a new Atlas with a single empty active map.
    pub fn new() -> Self {
        let mut maps = Vec::new();
        maps.push(Map::new());

        Self {
            maps,
            active_map_idx: 0,
            keyframe_db: KeyFrameDatabase::new(),
        }
    }

    /// Number of maps in the Atlas.
    pub fn num_maps(&self) -> usize {
        self.maps.len()
    }

    /// Index of the active map.
    pub fn active_map_index(&self) -> MapIndex {
        self.active_map_idx
    }

    /// Immutable reference to the active map.
    pub fn active_map(&self) -> &Map {
        &self.maps[self.active_map_idx]
    }

    /// Mutable reference to the active map.
    pub fn active_map_mut(&mut self) -> &mut Map {
        &mut self.maps[self.active_map_idx]
    }

    /// Immutable slice of all maps.
    pub fn all_maps(&self) -> &[Map] {
        &self.maps
    }

    /// Mutable slice of all maps.
    pub fn all_maps_mut(&mut self) -> &mut [Map] {
        &mut self.maps
    }

    /// Create a new empty map and make it the active map.
    ///
    /// Returns the index of the newly created map.
    pub fn create_new_map(&mut self) -> MapIndex {
        self.maps.push(Map::new());
        let idx = self.maps.len() - 1;
        self.active_map_idx = idx;
        idx
    }

    /// Set the active map by index.
    ///
    /// Panics if `idx` is out of bounds.
    pub fn set_active_map(&mut self, idx: MapIndex) {
        assert!(idx < self.maps.len(), "active map index out of range");
        self.active_map_idx = idx;
    }
}

impl Default for Atlas {
    fn default() -> Self {
        Self::new()
    }
}
