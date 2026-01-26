//! Shared state between Tracking, Local Mapping, and Loop Closing threads.
//!
//! The `SharedState` struct holds all data that needs to be accessed by
//! multiple threads, protected by appropriate synchronization primitives.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::atlas::atlas::Atlas;
use crate::vocabulary::OrbVocabulary;

/// Shared state accessible by Tracking, Local Mapping, and Loop Closing threads.
pub struct SharedState {
    /// The Atlas containing all maps, keyframes, and map points.
    /// Protected by RwLock: Tracking reads, Local Mapping writes.
    pub atlas: RwLock<Atlas>,

    /// ORB vocabulary for Bag-of-Words computation.
    /// None if vocabulary hasn't been loaded.
    pub vocabulary: Option<Arc<OrbVocabulary>>,

    /// Flow control: when true, Tracking should not create new keyframes.
    /// Set by Local Mapping when the keyframe queue is too long.
    pub stop_keyframe_creation: AtomicBool,

    /// Signal to abort Local BA early when a new keyframe arrives.
    /// Set by Tracking when sending a new keyframe to Local Mapping.
    pub abort_ba: AtomicBool,

    /// Request Local Mapping to finish processing and exit.
    pub shutdown_requested: AtomicBool,

    // ─────────────────────────────────────────────────────────────────────────
    // Loop Closing coordination flags
    // ─────────────────────────────────────────────────────────────────────────

    /// Request Local Mapping to pause (set by Loop Closing before correction).
    pub pause_local_mapping: AtomicBool,

    /// Acknowledgment that Local Mapping has paused.
    pub local_mapping_paused: AtomicBool,

    /// Flag indicating Global BA is currently running.
    pub global_ba_running: AtomicBool,

    /// Flag set when a loop has been corrected (for tracking to reset).
    pub loop_corrected: AtomicBool,

    // ─────────────────────────────────────────────────────────────────────────
    // IMU initialization flags
    // ─────────────────────────────────────────────────────────────────────────

    /// Flag indicating IMU initialization failed due to insufficient motion.
    /// Set by Local Mapping when motion < 2cm over 10s. Tracker should reset map.
    pub bad_imu: AtomicBool,
}

impl SharedState {
    /// Create a new SharedState with an empty Atlas and no vocabulary.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            atlas: RwLock::new(Atlas::new()),
            vocabulary: None,
            stop_keyframe_creation: AtomicBool::new(false),
            abort_ba: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            pause_local_mapping: AtomicBool::new(false),
            local_mapping_paused: AtomicBool::new(false),
            global_ba_running: AtomicBool::new(false),
            loop_corrected: AtomicBool::new(false),
            bad_imu: AtomicBool::new(false),
        })
    }

    /// Create a new SharedState with a vocabulary.
    pub fn with_vocabulary(vocabulary: OrbVocabulary) -> Arc<Self> {
        Arc::new(Self {
            atlas: RwLock::new(Atlas::new()),
            vocabulary: Some(Arc::new(vocabulary)),
            stop_keyframe_creation: AtomicBool::new(false),
            abort_ba: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            pause_local_mapping: AtomicBool::new(false),
            local_mapping_paused: AtomicBool::new(false),
            global_ba_running: AtomicBool::new(false),
            loop_corrected: AtomicBool::new(false),
            bad_imu: AtomicBool::new(false),
        })
    }

    /// Get a reference to the vocabulary, if loaded.
    pub fn vocabulary(&self) -> Option<&Arc<OrbVocabulary>> {
        self.vocabulary.as_ref()
    }

    /// Check if keyframe creation should be stopped (flow control).
    pub fn should_stop_keyframe_creation(&self) -> bool {
        self.stop_keyframe_creation.load(Ordering::SeqCst)
    }

    /// Set the stop_keyframe_creation flag.
    pub fn set_stop_keyframe_creation(&self, value: bool) {
        self.stop_keyframe_creation.store(value, Ordering::SeqCst);
    }

    /// Check if BA should be aborted.
    pub fn should_abort_ba(&self) -> bool {
        self.abort_ba.load(Ordering::SeqCst)
    }

    /// Signal that BA should be aborted (new keyframe arriving).
    pub fn request_abort_ba(&self) {
        self.abort_ba.store(true, Ordering::SeqCst);
    }

    /// Clear the abort BA flag (after BA completes or is aborted).
    pub fn clear_abort_ba(&self) {
        self.abort_ba.store(false, Ordering::SeqCst);
    }

    /// Request shutdown of the Local Mapping thread.
    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::SeqCst);
    }

    /// Check if shutdown was requested.
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Loop Closing coordination methods
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if Local Mapping should pause.
    pub fn should_pause_local_mapping(&self) -> bool {
        self.pause_local_mapping.load(Ordering::SeqCst)
    }

    /// Signal that Local Mapping has paused.
    pub fn set_local_mapping_paused(&self, paused: bool) {
        self.local_mapping_paused.store(paused, Ordering::SeqCst);
    }

    /// Check if Global BA is running.
    pub fn is_global_ba_running(&self) -> bool {
        self.global_ba_running.load(Ordering::SeqCst)
    }

    /// Check if a loop has been corrected (and tracking should reset).
    pub fn check_and_clear_loop_corrected(&self) -> bool {
        self.loop_corrected.swap(false, Ordering::SeqCst)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IMU initialization methods
    // ─────────────────────────────────────────────────────────────────────────

    /// Check if IMU initialization failed due to insufficient motion.
    pub fn is_bad_imu(&self) -> bool {
        self.bad_imu.load(Ordering::SeqCst)
    }

    /// Set the bad_imu flag (called by Local Mapping when insufficient motion detected).
    pub fn set_bad_imu(&self, value: bool) {
        self.bad_imu.store(value, Ordering::SeqCst);
    }

    /// Check and clear the bad_imu flag (returns true if it was set).
    pub fn check_and_clear_bad_imu(&self) -> bool {
        self.bad_imu.swap(false, Ordering::SeqCst)
    }
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            atlas: RwLock::new(Atlas::new()),
            vocabulary: None,
            stop_keyframe_creation: AtomicBool::new(false),
            abort_ba: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
            pause_local_mapping: AtomicBool::new(false),
            local_mapping_paused: AtomicBool::new(false),
            global_ba_running: AtomicBool::new(false),
            loop_corrected: AtomicBool::new(false),
            bad_imu: AtomicBool::new(false),
        }
    }
}

// SAFETY: SharedState is safe to share between threads because:
// 1. All access to Atlas is synchronized through RwLock
// 2. The OpenCV KeyPoint's *mut c_void is an artifact of the Rust bindings -
//    the actual data is POD (point coordinates, size, angle, etc.) and safe to share
// 3. We never modify KeyPoint data after creation, only read it
unsafe impl Send for SharedState {}
unsafe impl Sync for SharedState {}
