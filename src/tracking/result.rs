//! Tracking results and diagnostics structures.
//!
//! These types describe what happened during processing of a single frame:
//! - high level tracking state (OK / LOST / etc.)
//! - pose and velocity estimates
//! - correspondence counts and reprojection statistics
//! - timing information for profiling

use nalgebra::Vector3;

use crate::atlas::map::{KeyFrameId, MapPointId};
use crate::geometry::SE3;
use crate::tracking::TrackingState;

/// Summary of tracking for a single frame.
pub struct TrackingResult {
    pub state: TrackingState,
    pub pose: SE3,
    pub velocity: Vector3<f64>,
    pub reference_kf_id: Option<KeyFrameId>,
    pub metrics: TrackingMetrics,
    pub timing: TimingStats,
    pub matches: MatchInfo,
}

/// Scalar metrics useful for debugging tracking quality.
pub struct TrackingMetrics {
    pub n_features: usize,
    pub n_map_point_matches: usize,
    pub n_inliers: usize,
    pub inlier_ratio: f64,
    pub reproj_error_median_px: f64,
    pub delta_translation_m: f64,
    pub delta_rotation_deg: f64,
}

/// Timing breakdown for a frame.
pub struct TimingStats {
    pub total_ms: f64,
    pub extract_orb_ms: f64,
    pub match_ms: f64,
    pub solve_pnp_ms: f64,
    pub relocal_ms: f64,
}

impl TimingStats {
    pub fn zero() -> Self {
        Self {
            total_ms: 0.0,
            extract_orb_ms: 0.0,
            match_ms: 0.0,
            solve_pnp_ms: 0.0,
            relocal_ms: 0.0,
        }
    }
}

/// Detailed correspondence information for visualization.
pub struct MatchInfo {
    /// All map point matches: (MapPointId, feature_idx).
    pub matched_map_points: Vec<(MapPointId, usize)>,
    /// Indices of inlier matches within `matched_map_points`.
    pub inlier_indices: Vec<usize>,
    /// Indices of outlier matches within `matched_map_points`.
    pub outlier_indices: Vec<usize>,
    /// Reprojection error for each match (same order as `matched_map_points`).
    pub reproj_errors: Vec<f64>,
    /// IDs of MapPoints considered in the local map for this frame.
    pub local_map_point_ids: Vec<MapPointId>,
}

