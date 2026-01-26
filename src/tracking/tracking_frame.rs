//! Tracking `Frame` representation.
//!
//! This is distinct from [`frame::StereoFrame`], which is the low-level
//! result of stereo processing (images + raw features). A `Frame` here is
//! what the Tracker operates on: it owns a subset of features, optional
//! stereo depth, a Bag-of-Words vector and temporary associations to
//! `MapPoint`s.

use nalgebra::Vector3;
use opencv::core::{KeyPoint, Vector};
use opencv::prelude::KeyPointTraitConst;

use crate::atlas::keyframe_db::BowVector;
use crate::atlas::map::MapPointId;
use crate::tracking::frame::FeatureSet;

/// Spatial grid for O(1) feature lookup, matching C++ ORB-SLAM3.
///
/// The grid divides the image into GRID_COLS × GRID_ROWS cells.
/// Each cell stores indices of features whose keypoints fall within that cell.
/// This enables O(k) feature search in a radius, where k << N (total features).
#[derive(Clone)]
pub struct FeatureGrid {
    /// Grid cells, each containing indices of features in that cell.
    /// Stored as row-major: cell[row * GRID_COLS + col]
    cells: Vec<Vec<usize>>,
    /// Inverse of cell width (1.0 / cell_width)
    grid_element_width_inv: f64,
    /// Inverse of cell height (1.0 / cell_height)
    grid_element_height_inv: f64,
    /// Minimum x coordinate (image left edge, typically 0)
    min_x: f64,
    /// Minimum y coordinate (image top edge, typically 0)
    min_y: f64,
    /// Number of columns in the grid
    grid_cols: usize,
    /// Number of rows in the grid
    grid_rows: usize,
}

impl FeatureGrid {
    /// Default grid dimensions matching C++ ORB-SLAM3
    pub const GRID_COLS: usize = 64;
    pub const GRID_ROWS: usize = 48;

    /// Create a new feature grid from keypoints.
    ///
    /// # Arguments
    /// * `keypoints` - The keypoints to index
    /// * `img_width` - Image width in pixels
    /// * `img_height` - Image height in pixels
    pub fn new(keypoints: &Vector<KeyPoint>, img_width: f64, img_height: f64) -> Self {
        let min_x = 0.0;
        let min_y = 0.0;
        let max_x = img_width;
        let max_y = img_height;

        let grid_element_width_inv = Self::GRID_COLS as f64 / (max_x - min_x);
        let grid_element_height_inv = Self::GRID_ROWS as f64 / (max_y - min_y);

        let num_cells = Self::GRID_COLS * Self::GRID_ROWS;
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); num_cells];

        // Distribute features into grid cells
        for (idx, kp) in keypoints.iter().enumerate() {
            let x = kp.pt().x as f64;
            let y = kp.pt().y as f64;

            let cell_x = ((x - min_x) * grid_element_width_inv) as usize;
            let cell_y = ((y - min_y) * grid_element_height_inv) as usize;

            // Clamp to valid range
            let cell_x = cell_x.min(Self::GRID_COLS - 1);
            let cell_y = cell_y.min(Self::GRID_ROWS - 1);

            let cell_idx = cell_y * Self::GRID_COLS + cell_x;
            cells[cell_idx].push(idx);
        }

        Self {
            cells,
            grid_element_width_inv,
            grid_element_height_inv,
            min_x,
            min_y,
            grid_cols: Self::GRID_COLS,
            grid_rows: Self::GRID_ROWS,
        }
    }

    /// Get feature indices within a circular search region.
    ///
    /// This is the key optimization: instead of O(N) linear scan,
    /// we only check features in nearby grid cells - O(k) where k << N.
    ///
    /// # Arguments
    /// * `x` - Query x coordinate
    /// * `y` - Query y coordinate
    /// * `r` - Search radius in pixels
    ///
    /// # Returns
    /// Vector of feature indices within the search radius (spatial candidates only,
    /// actual distance check should be done by caller for exact radius filtering)
    pub fn get_features_in_area(&self, x: f64, y: f64, r: f64) -> Vec<usize> {
        let mut candidates = Vec::new();

        // Compute cell range to search
        let min_cell_x = ((x - self.min_x - r) * self.grid_element_width_inv).floor() as i32;
        let max_cell_x = ((x - self.min_x + r) * self.grid_element_width_inv).ceil() as i32;
        let min_cell_y = ((y - self.min_y - r) * self.grid_element_height_inv).floor() as i32;
        let max_cell_y = ((y - self.min_y + r) * self.grid_element_height_inv).ceil() as i32;

        // Clamp to valid grid range
        let min_cell_x = min_cell_x.max(0) as usize;
        let max_cell_x = (max_cell_x as usize).min(self.grid_cols - 1);
        let min_cell_y = min_cell_y.max(0) as usize;
        let max_cell_y = (max_cell_y as usize).min(self.grid_rows - 1);

        // Collect features from all cells in range
        for cell_y in min_cell_y..=max_cell_y {
            for cell_x in min_cell_x..=max_cell_x {
                let cell_idx = cell_y * self.grid_cols + cell_x;
                candidates.extend(&self.cells[cell_idx]);
            }
        }

        candidates
    }

    /// Get features in area with octave level filtering.
    ///
    /// Filters candidates by pyramid level (octave) to ensure scale consistency.
    ///
    /// # Arguments
    /// * `x` - Query x coordinate
    /// * `y` - Query y coordinate
    /// * `r` - Base search radius in pixels
    /// * `min_level` - Minimum octave level to accept (-1 for any)
    /// * `max_level` - Maximum octave level to accept (-1 for any)
    /// * `keypoints` - Reference to keypoints for octave checking
    pub fn get_features_in_area_with_level(
        &self,
        x: f64,
        y: f64,
        r: f64,
        min_level: i32,
        max_level: i32,
        keypoints: &Vector<KeyPoint>,
    ) -> Vec<usize> {
        let candidates = self.get_features_in_area(x, y, r);

        if min_level < 0 && max_level < 0 {
            return candidates;
        }

        candidates
            .into_iter()
            .filter(|&idx| {
                if let Ok(kp) = keypoints.get(idx) {
                    let octave = kp.octave();
                    (min_level < 0 || octave >= min_level)
                        && (max_level < 0 || octave <= max_level)
                } else {
                    false
                }
            })
            .collect()
    }
}

/// A frame being tracked (not yet a KeyFrame).
pub struct Frame {
    /// Timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Visual features (typically taken from the left image in stereo).
    pub features: FeatureSet,
    /// 3D points in camera frame (from stereo triangulation).
    /// None if the point couldn't be triangulated.
    pub points_cam: Vec<Option<Vector3<f64>>>,
    /// Optional Bag-of-Words representation for place recognition.
    pub bow_vector: Option<BowVector>,
    /// Temporary MapPoint associations for tracking:
    /// map_point_matches[i] = Some(mp_id) if feature i is associated.
    pub map_point_matches: Vec<Option<MapPointId>>,
    /// Spatial grid for O(1) feature lookup.
    pub grid: FeatureGrid,
}

impl Frame {
    /// Construct a new tracking frame from raw stereo processing output.
    ///
    /// Uses default EuRoC image dimensions (752x480). For other datasets,
    /// use `from_stereo_with_size`.
    pub fn from_stereo(stereo: crate::tracking::frame::StereoFrame) -> Self {
        // EuRoC MAV dataset image dimensions
        Self::from_stereo_with_size(stereo, 752.0, 480.0)
    }

    /// Construct a new tracking frame with explicit image dimensions.
    pub fn from_stereo_with_size(
        stereo: crate::tracking::frame::StereoFrame,
        img_width: f64,
        img_height: f64,
    ) -> Self {
        let n_feats = stereo.left_features.keypoints.len();
        let grid = FeatureGrid::new(&stereo.left_features.keypoints, img_width, img_height);
        Self {
            timestamp_ns: stereo.timestamp_ns,
            features: stereo.left_features,
            points_cam: stereo.points_cam,
            bow_vector: None,
            map_point_matches: vec![None; n_feats],
            grid,
        }
    }

    /// Number of features in this frame.
    pub fn num_features(&self) -> usize {
        self.features.keypoints.len()
    }

    /// Convenience accessor for keypoints.
    pub fn keypoints(&self) -> &Vector<KeyPoint> {
        &self.features.keypoints
    }
}
