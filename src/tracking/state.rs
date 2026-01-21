//! Tracking state machine as described in ORB-SLAM3.

/// State of the tracking thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingState {
    /// System not yet initialized (waiting for initial pose).
    NotInitialized,
    /// Tracking successfully.
    Ok,
    /// Lost tracking recently, attempting recovery.
    RecentlyLost,
    /// Completely lost, need relocalization.
    Lost,
}

impl Default for TrackingState {
    fn default() -> Self {
        Self::NotInitialized
    }
}
