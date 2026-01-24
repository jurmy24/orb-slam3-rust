//! IMU initialization state tracking.

/// State of IMU initialization for a map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuInitState {
    /// IMU not yet initialized (waiting for enough keyframes/data).
    NotInitialized,
    /// IMU initialization in progress (attempting to estimate gravity, biases, velocities).
    Initializing,
    /// IMU successfully initialized.
    Initialized,
}

impl Default for ImuInitState {
    fn default() -> Self {
        Self::NotInitialized
    }
}
