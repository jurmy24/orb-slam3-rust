//! KeyFrame decision criteria as described in ORB-SLAM3.

use crate::tracking::frame::StereoFrame;

/// Time threshold (seconds) for creating keyframes before IMU initialization.
/// For stereo-inertial, we create keyframes every 0.25 seconds to gather
/// enough data for IMU initialization.
const TIME_THRESHOLD_BEFORE_IMU_INIT: f64 = 0.25;

/// Criteria for deciding whether to create a new KeyFrame.
///
/// Based on ORB-SLAM3 paper Section V-A:
/// - Time since last KeyFrame
/// - Number of tracked map points
/// - Parallax with respect to last KeyFrame
pub struct KeyFrameDecision {
    /// Minimum frames between KeyFrames.
    min_frames: usize,
    /// Maximum frames between KeyFrames.
    max_frames: usize,
    /// Minimum ratio of tracked points to trigger new KF.
    min_tracked_ratio: f64,
    /// Frame counter since last KeyFrame.
    frames_since_kf: usize,
}

impl KeyFrameDecision {
    pub fn new() -> Self {
        Self {
            min_frames: 0,
            max_frames: 15,
            min_tracked_ratio: 0.9,
            frames_since_kf: 0,
        }
    }

    /// Decide whether to create a new KeyFrame (original method for compatibility).
    ///
    /// # Arguments
    /// * `current_frame` - The current stereo frame
    /// * `tracked_points` - Number of successfully tracked map points
    /// * `reference_points` - Number of points in the reference KeyFrame
    pub fn should_create_keyframe(
        &mut self,
        _current_frame: &StereoFrame,
        tracked_points: usize,
        reference_points: usize,
    ) -> bool {
        self.frames_since_kf += 1;

        // Always create KF if max frames exceeded
        if self.frames_since_kf >= self.max_frames {
            self.frames_since_kf = 0;
            return true;
        }

        // Don't create KF too soon
        if self.frames_since_kf < self.min_frames {
            return false;
        }

        // Create KF if tracking quality dropped
        if reference_points > 0 {
            let ratio = tracked_points as f64 / reference_points as f64;
            if ratio < self.min_tracked_ratio {
                self.frames_since_kf = 0;
                return true;
            }
        }

        false
    }

    /// Decide whether to create a new KeyFrame for stereo-inertial mode.
    ///
    /// Before IMU initialization, uses time-based decision (every 0.25s).
    /// After IMU initialization, uses the standard tracking-quality-based decision.
    ///
    /// # Arguments
    /// * `current_frame` - The current stereo frame
    /// * `tracked_points` - Number of successfully tracked map points
    /// * `reference_points` - Number of points in the reference KeyFrame
    /// * `time_since_last_kf` - Time in seconds since last keyframe was created
    /// * `imu_initialized` - Whether IMU has been initialized
    pub fn should_create_keyframe_stereo_inertial(
        &mut self,
        _current_frame: &StereoFrame,
        tracked_points: usize,
        reference_points: usize,
        time_since_last_kf: f64,
        imu_initialized: bool,
    ) -> bool {
        self.frames_since_kf += 1;

        // Before IMU initialization: use time-based decision
        // This ensures we gather enough keyframes with IMU preintegration for initialization
        if !imu_initialized {
            if time_since_last_kf >= TIME_THRESHOLD_BEFORE_IMU_INIT {
                self.frames_since_kf = 0;
                return true;
            }
            return false;
        }

        // After IMU initialization: use standard decision logic

        // Always create KF if max frames exceeded
        if self.frames_since_kf >= self.max_frames {
            self.frames_since_kf = 0;
            return true;
        }

        // Don't create KF too soon
        if self.frames_since_kf < self.min_frames {
            return false;
        }

        // Create KF if tracking quality dropped
        if reference_points > 0 {
            let ratio = tracked_points as f64 / reference_points as f64;
            if ratio < self.min_tracked_ratio {
                self.frames_since_kf = 0;
                return true;
            }
        }

        false
    }

    /// Reset after KeyFrame creation.
    pub fn reset(&mut self) {
        self.frames_since_kf = 0;
    }
}

impl Default for KeyFrameDecision {
    fn default() -> Self {
        Self::new()
    }
}
