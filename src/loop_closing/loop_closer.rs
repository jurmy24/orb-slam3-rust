//! LoopCloser - Main loop closing thread.
//!
//! This module implements the main loop closing thread that:
//! 1. Receives keyframes from LocalMapping
//! 2. Detects loop candidates using BoW
//! 3. Verifies candidates with Sim3 RANSAC
//! 4. Corrects the map via pose graph optimization
//! 5. Launches Global BA in the background

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError};

use crate::atlas::map::KeyFrameId;
use crate::optimizer::{GlobalBAConfig, PoseGraphConfig};
use crate::system::SharedState;
use crate::tracking::frame::CameraModel;

use super::corrector::{correct_loop, verify_loop_candidate, CorrectorConfig};
use super::detector::{detect_loop_candidates, ConsistencyChecker, LoopDetectorConfig};

/// Timeout for receiving keyframes from the channel.
const RECV_TIMEOUT: Duration = Duration::from_millis(100);

/// Configuration for the LoopCloser.
#[derive(Debug, Clone)]
pub struct LoopCloserConfig {
    /// Loop detection configuration.
    pub detector: LoopDetectorConfig,

    /// Loop correction configuration.
    pub corrector: CorrectorConfig,

    /// Pose graph optimization configuration.
    pub pose_graph: PoseGraphConfig,

    /// Global BA configuration.
    pub global_ba: GlobalBAConfig,

    /// Whether to run Global BA after loop correction.
    pub run_global_ba: bool,
}

impl Default for LoopCloserConfig {
    fn default() -> Self {
        Self {
            detector: LoopDetectorConfig::default(),
            corrector: CorrectorConfig::default(),
            pose_graph: PoseGraphConfig::default(),
            global_ba: GlobalBAConfig::default(),
            run_global_ba: true,
        }
    }
}

/// Statistics for the loop closer.
#[derive(Debug, Default, Clone)]
pub struct LoopCloserStats {
    /// Number of keyframes processed.
    pub keyframes_processed: usize,

    /// Number of loop candidates detected.
    pub candidates_detected: usize,

    /// Number of loops verified.
    pub loops_verified: usize,

    /// Number of loops corrected.
    pub loops_corrected: usize,
}

/// The LoopCloser thread handler.
pub struct LoopCloser {
    /// Shared state with other threads.
    shared: Arc<SharedState>,

    /// Camera model.
    camera: CameraModel,

    /// Configuration.
    config: LoopCloserConfig,

    /// Consistency checker for temporal validation.
    consistency_checker: ConsistencyChecker,

    /// Statistics.
    stats: LoopCloserStats,

    /// Flag indicating Global BA is running.
    global_ba_running: Arc<AtomicBool>,
}

impl LoopCloser {
    /// Create a new LoopCloser.
    pub fn new(shared: Arc<SharedState>, camera: CameraModel, config: LoopCloserConfig) -> Self {
        let consistency_checker = ConsistencyChecker::new(config.detector.clone());

        Self {
            shared,
            camera,
            config,
            consistency_checker,
            stats: LoopCloserStats::default(),
            global_ba_running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run the main loop closing thread.
    ///
    /// This function blocks and processes keyframes until the channel is closed
    /// or shutdown is requested.
    pub fn run(&mut self, kf_receiver: Receiver<KeyFrameId>) {
        tracing::info!("LoopCloser thread started");

        loop {
            // Check for shutdown
            if self.shared.is_shutdown_requested() {
                break;
            }

            // Wait for pause to complete if local mapping is paused
            if self.shared.pause_local_mapping.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Receive keyframe with timeout
            match kf_receiver.recv_timeout(RECV_TIMEOUT) {
                Ok(kf_id) => {
                    self.process_keyframe(kf_id);
                }
                Err(RecvTimeoutError::Timeout) => {
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    tracing::info!("LoopCloser channel disconnected");
                    break;
                }
            }
        }

        tracing::info!(
            "LoopCloser thread exiting. Stats: processed={}, detected={}, verified={}, corrected={}",
            self.stats.keyframes_processed,
            self.stats.candidates_detected,
            self.stats.loops_verified,
            self.stats.loops_corrected
        );
    }

    /// Process a single keyframe for loop detection.
    fn process_keyframe(&mut self, kf_id: KeyFrameId) {
        self.stats.keyframes_processed += 1;

        // Skip if Global BA is running
        if self.global_ba_running.load(Ordering::SeqCst) {
            return;
        }

        // Step 1: Detect loop candidates
        let vocabulary = self.shared.vocabulary();
        let candidates = detect_loop_candidates(
            kf_id,
            &self.shared.atlas,
            vocabulary.map(|v| v.as_ref()),
            &self.config.detector,
        );

        if candidates.is_empty() {
            return;
        }

        self.stats.candidates_detected += candidates.len();

        // Step 2: Check temporal consistency
        let consistent_candidate = self.consistency_checker.add_and_check(kf_id, &candidates);

        let candidate = match consistent_candidate {
            Some(c) => c,
            None => return,
        };

        tracing::info!(
            "Loop candidate detected: {} -> {} (score: {:.3})",
            candidate.current_kf_id,
            candidate.loop_kf_id,
            candidate.bow_score
        );

        // Step 3: Verify with Sim3
        let verified_loop = verify_loop_candidate(&candidate, &self.shared.atlas, &self.camera);

        let verified = match verified_loop {
            Some(v) => {
                self.stats.loops_verified += 1;
                tracing::info!(
                    "Loop verified: {} -> {} ({} matched points)",
                    v.current_kf_id,
                    v.loop_kf_id,
                    v.matched_map_points.len()
                );
                v
            }
            None => {
                tracing::debug!("Loop candidate failed verification");
                return;
            }
        };

        // Step 4: Pause local mapping
        self.shared.pause_local_mapping.store(true, Ordering::SeqCst);
        self.wait_for_local_mapping_pause();

        // Step 5: Correct the loop
        correct_loop(
            &verified,
            &self.shared.atlas,
            &self.camera,
            &self.config.corrector,
        );

        self.stats.loops_corrected += 1;
        self.shared.loop_corrected.store(true, Ordering::SeqCst);

        tracing::info!(
            "Loop closed: {} -> {}",
            verified.current_kf_id,
            verified.loop_kf_id
        );

        // Step 6: Resume local mapping
        self.shared.pause_local_mapping.store(false, Ordering::SeqCst);

        // Step 7: Launch Global BA
        if self.config.run_global_ba {
            self.launch_global_ba();
        }
    }

    /// Wait for local mapping to pause.
    fn wait_for_local_mapping_pause(&self) {
        let max_wait = Duration::from_millis(5000);
        let start = std::time::Instant::now();

        while !self.shared.local_mapping_paused.load(Ordering::SeqCst) {
            if start.elapsed() > max_wait {
                tracing::warn!("Timeout waiting for local mapping to pause");
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// Launch Global BA.
    ///
    /// Note: Currently runs synchronously due to thread safety constraints
    /// with OpenCV types. In production, consider extracting all required
    /// data before spawning a thread.
    fn launch_global_ba(&mut self) {
        tracing::info!("Running Global BA");

        // Run Global BA synchronously using three-phase pattern
        let _ = crate::optimizer::run_global_ba(
            &self.shared.atlas,
            &self.camera,
            &self.config.global_ba,
            &self.global_ba_running,
        );
    }

    /// Get current statistics.
    pub fn stats(&self) -> &LoopCloserStats {
        &self.stats
    }

    /// Check if Global BA is currently running.
    pub fn is_global_ba_running(&self) -> bool {
        self.global_ba_running.load(Ordering::SeqCst)
    }

    /// Stop Global BA if running.
    pub fn stop_global_ba(&self) {
        self.global_ba_running.store(false, Ordering::SeqCst);
    }
}

/// Spawn the loop closer thread.
///
/// Returns a handle to the spawned thread.
pub fn spawn_loop_closer(
    shared: Arc<SharedState>,
    kf_receiver: Receiver<KeyFrameId>,
    camera: CameraModel,
    config: LoopCloserConfig,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut loop_closer = LoopCloser::new(shared, camera, config);
        loop_closer.run(kf_receiver);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_closer_config_default() {
        let config = LoopCloserConfig::default();
        assert!(config.run_global_ba);
    }

    #[test]
    fn test_loop_closer_stats_default() {
        let stats = LoopCloserStats::default();
        assert_eq!(stats.keyframes_processed, 0);
        assert_eq!(stats.loops_corrected, 0);
    }
}
