//! SLAM System - Main entry point and thread orchestration.
//!
//! The `SlamSystem` is the top-level struct that users interact with.
//! It owns the shared state and spawns the Tracking and Local Mapping threads.

use std::path::Path;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};

use crate::atlas::map::KeyFrameId;
use crate::io::euroc::ImuEntry;
use crate::local_mapping::LocalMapper;
use crate::loop_closing::loop_closer::{spawn_loop_closer, LoopCloserConfig};
use crate::tracking::Tracker;
use crate::tracking::frame::{CameraModel, StereoFrame};
use crate::tracking::result::TrackingResult;
use crate::vocabulary::OrbVocabulary;

use super::messages::NewKeyFrameMsg;
use super::shared_state::SharedState;

/// Capacity of the keyframe channel between Tracking and Local Mapping.
/// When the channel is full, Tracking will block briefly.
const KEYFRAME_CHANNEL_CAPACITY: usize = 5;

/// Main SLAM system orchestrating Tracking, Local Mapping, and Loop Closing.
pub struct SlamSystem {
    /// Shared state (Atlas, flags) accessible by all threads.
    shared: Arc<SharedState>,

    /// The tracker runs in the main thread (called by user).
    tracker: Tracker,

    /// Channel sender for keyframes (Tracking -> Local Mapping).
    kf_sender: Sender<NewKeyFrameMsg>,

    /// Handle to the Local Mapping thread.
    local_mapping_handle: Option<JoinHandle<()>>,

    /// Handle to the Loop Closing thread.
    loop_closing_handle: Option<JoinHandle<()>>,
}

impl SlamSystem {
    /// Create a new SLAM system with the given camera model.
    ///
    /// This creates the shared state and spawns the Local Mapping thread.
    /// Use `with_vocabulary` to also load an ORB vocabulary for place recognition.
    pub fn new(camera: CameraModel) -> Result<Self> {
        Self::with_shared_state(camera, SharedState::new())
    }

    /// Create a new SLAM system with a vocabulary loaded from a file.
    ///
    /// The vocabulary is used for Bag-of-Words computation to accelerate
    /// feature matching and enable place recognition.
    ///
    /// # Arguments
    ///
    /// * `camera` - Camera model with intrinsics and baseline
    /// * `vocabulary_path` - Path to ORBvoc.txt (DBoW2 format)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let system = SlamSystem::with_vocabulary(camera, "data/ORBvoc.txt")?;
    /// ```
    pub fn with_vocabulary<P: AsRef<Path>>(camera: CameraModel, vocabulary_path: P) -> Result<Self> {
        let vocabulary = OrbVocabulary::load_from_text(vocabulary_path)?;
        let shared = SharedState::with_vocabulary(vocabulary);
        Self::with_shared_state(camera, shared)
    }

    /// Create a SLAM system with a pre-created shared state.
    fn with_shared_state(camera: CameraModel, shared: Arc<SharedState>) -> Result<Self> {
        // Channel: Tracking -> LocalMapping
        let (kf_sender, kf_receiver) = bounded::<NewKeyFrameMsg>(KEYFRAME_CHANNEL_CAPACITY);

        // Channel: LocalMapping -> LoopClosing
        let (lc_sender, lc_receiver) = bounded::<KeyFrameId>(KEYFRAME_CHANNEL_CAPACITY);

        // Create the tracker
        let tracker = Tracker::new(camera, shared.clone(), kf_sender.clone())?;

        // Spawn Local Mapping thread (now with loop closer sender)
        let local_mapping_handle =
            Self::spawn_local_mapping(shared.clone(), kf_receiver, lc_sender, camera);

        // Spawn Loop Closing thread
        let loop_closing_handle = spawn_loop_closer(
            shared.clone(),
            lc_receiver,
            camera,
            LoopCloserConfig::default(),
        );

        Ok(Self {
            shared,
            tracker,
            kf_sender,
            local_mapping_handle: Some(local_mapping_handle),
            loop_closing_handle: Some(loop_closing_handle),
        })
    }

    /// Spawn the Local Mapping thread.
    fn spawn_local_mapping(
        shared: Arc<SharedState>,
        kf_receiver: Receiver<NewKeyFrameMsg>,
        lc_sender: Sender<KeyFrameId>,
        camera: CameraModel,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut local_mapper = LocalMapper::new(camera, Some(lc_sender));
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
    /// Signals all threads to finish and waits for them.
    pub fn shutdown(&mut self) {
        // Signal shutdown
        self.shared.request_shutdown();

        // Wait for Local Mapping to finish
        if let Some(handle) = self.local_mapping_handle.take() {
            let _ = handle.join();
        }

        // Wait for Loop Closing to finish
        if let Some(handle) = self.loop_closing_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SlamSystem {
    fn drop(&mut self) {
        self.shutdown();
    }
}
