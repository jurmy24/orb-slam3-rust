//! Shared state between Tracking and Local Mapping threads.
//!
//! The `SharedState` struct holds all data that needs to be accessed by
//! multiple threads, protected by appropriate synchronization primitives.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;

use crate::atlas::atlas::Atlas;

/// Shared state accessible by both Tracking and Local Mapping threads.
pub struct SharedState {
    /// The Atlas containing all maps, keyframes, and map points.
    /// Protected by RwLock: Tracking reads, Local Mapping writes.
    pub atlas: RwLock<Atlas>,

    /// Flow control: when true, Tracking should not create new keyframes.
    /// Set by Local Mapping when the keyframe queue is too long.
    pub stop_keyframe_creation: AtomicBool,

    /// Signal to abort Local BA early when a new keyframe arrives.
    /// Set by Tracking when sending a new keyframe to Local Mapping.
    pub abort_ba: AtomicBool,

    /// Request Local Mapping to finish processing and exit.
    pub shutdown_requested: AtomicBool,
}

impl SharedState {
    /// Create a new SharedState with an empty Atlas.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            atlas: RwLock::new(Atlas::new()),
            stop_keyframe_creation: AtomicBool::new(false),
            abort_ba: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
        })
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
}

impl Default for SharedState {
    fn default() -> Self {
        Self {
            atlas: RwLock::new(Atlas::new()),
            stop_keyframe_creation: AtomicBool::new(false),
            abort_ba: AtomicBool::new(false),
            shutdown_requested: AtomicBool::new(false),
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
