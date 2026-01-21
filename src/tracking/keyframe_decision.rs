//! KeyFrame decision criteria as described in ORB-SLAM3.

use crate::tracking::frame::StereoFrame;

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
            max_frames: 30,
            min_tracked_ratio: 0.9,
            frames_since_kf: 0,
        }
    }

    /// Decide whether to create a new KeyFrame.
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
