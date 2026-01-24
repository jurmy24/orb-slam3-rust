//! SLAM System - Main entry point and thread orchestration.
//!
//! The `SlamSystem` is the top-level struct that users interact with.
//! It owns the shared state and spawns the Tracking and Local Mapping threads.

use std::sync::Arc;
use std::thread::{self, JoinHandle};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};

use crate::io::euroc::ImuEntry;
use crate::local_mapping::LocalMapper;
use crate::tracking::Tracker;
use crate::tracking::frame::{CameraModel, StereoFrame};
use crate::tracking::result::TrackingResult;

use super::messages::NewKeyFrameMsg;
use super::shared_state::SharedState;

/// Capacity of the keyframe channel between Tracking and Local Mapping.
/// When the channel is full, Tracking will block briefly.
const KEYFRAME_CHANNEL_CAPACITY: usize = 5;

/// Main SLAM system orchestrating Tracking and Local Mapping.
pub struct SlamSystem {
    /// Shared state (Atlas, flags) accessible by all threads.
    shared: Arc<SharedState>,

    /// The tracker runs in the main thread (called by user).
    tracker: Tracker,

    /// Channel sender for keyframes (Tracking -> Local Mapping).
    kf_sender: Sender<NewKeyFrameMsg>,

    /// Handle to the Local Mapping thread.
    local_mapping_handle: Option<JoinHandle<()>>,
}

impl SlamSystem {
    /// Create a new SLAM system with the given camera model.
    ///
    /// This creates the shared state and spawns the Local Mapping thread.
    pub fn new(camera: CameraModel) -> Result<Self> {
        let shared = SharedState::new();

        // Create bounded channel for keyframe communication
        let (kf_sender, kf_receiver) = bounded::<NewKeyFrameMsg>(KEYFRAME_CHANNEL_CAPACITY);

        // Create the tracker
        let tracker = Tracker::new(camera, shared.clone(), kf_sender.clone())?;

        // Spawn Local Mapping thread
        let local_mapping_handle = Self::spawn_local_mapping(shared.clone(), kf_receiver, camera);

        Ok(Self {
            shared,
            tracker,
            kf_sender,
            local_mapping_handle: Some(local_mapping_handle),
        })
    }

    /// Spawn the Local Mapping thread.
    fn spawn_local_mapping(
        shared: Arc<SharedState>,
        kf_receiver: Receiver<NewKeyFrameMsg>,
        camera: CameraModel,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut local_mapper = LocalMapper::new(camera);
            local_mapper.run(kf_receiver, shared);
        })
    }

    /// Process a stereo frame with IMU measurements.
    ///
    /// This runs in the calling thread (typically the main thread).
    /// Returns the tracking result including pose and metrics.
    pub fn process_frame(
        &mut self,
        stereo_frame: StereoFrame,
        imu_measurements: &[ImuEntry],
    ) -> Result<TrackingResult> {
        self.tracker.process_frame(stereo_frame, imu_measurements)
    }

    /// Get a reference to the shared state for visualization.
    pub fn shared_state(&self) -> &Arc<SharedState> {
        &self.shared
    }

    /// Get the current trajectory (poses of all processed frames).
    pub fn trajectory(&self) -> &[crate::geometry::SE3] {
        &self.tracker.trajectory
    }

    /// Shutdown the system gracefully.
    ///
    /// Signals the Local Mapping thread to finish and waits for it.
    pub fn shutdown(&mut self) {
        // Signal shutdown
        self.shared.request_shutdown();

        // Drop sender to unblock receiver (Local Mapping will exit its loop)
        // Note: We can't drop self.kf_sender here since we don't own it exclusively.
        // The thread will exit when it checks the shutdown flag.

        // Wait for Local Mapping to finish
        if let Some(handle) = self.local_mapping_handle.take() {
            // Send a dummy message or let channel close naturally
            // Actually, we need to signal somehow - the thread might be blocking on recv
            // We'll handle this by having LocalMapper check shutdown periodically
            let _ = handle.join();
        }
    }
}

impl Drop for SlamSystem {
    fn drop(&mut self) {
        self.shutdown();
    }
}
