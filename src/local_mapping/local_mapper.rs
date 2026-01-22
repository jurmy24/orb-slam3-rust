//! Local Mapper - Main Local Mapping thread implementation.
//!
//! Processes keyframes received from Tracking:
//! 1. Inserts keyframe into the map
//! 2. Associates matched map points
//! 3. Triangulates new map points from unmatched stereo features
//! 4. Updates covisibility graph (automatic via associate)
//! 5. Local BA (DISABLED - needs debugging)
//! 6. IMU initialization (for visual-inertial mode)

use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError};
use opencv::core::Mat;
use opencv::prelude::*;

use crate::atlas::keyframe_db::BowVector;
use crate::atlas::map::KeyFrameId;
use crate::optimizer::{local_bundle_adjustment as run_local_ba, LocalBAConfig};
use crate::system::messages::NewKeyFrameMsg;
use crate::system::shared_state::SharedState;
use crate::tracking::frame::CameraModel;

use super::imu_init::{apply_imu_init, initialize_imu};

/// Flow control threshold: if queue has more than this many keyframes,
/// signal Tracking to stop creating new ones.
const MAX_QUEUE_SIZE: usize = 3;

/// Timeout for receiving keyframes. Allows periodic shutdown checks.
const RECV_TIMEOUT: Duration = Duration::from_millis(100);

/// Local Mapping thread state.
pub struct LocalMapper {
    /// Camera model for triangulation.
    camera: CameraModel,
}

impl LocalMapper {
    /// Create a new LocalMapper.
    pub fn new(camera: CameraModel) -> Self {
        Self { camera }
    }

    /// Main thread loop: receive keyframes and process them.
    ///
    /// This runs until shutdown is requested or the channel is closed.
    pub fn run(&mut self, kf_receiver: Receiver<NewKeyFrameMsg>, shared: Arc<SharedState>) {
        let shared = Arc::clone(&shared);
        loop {
            // Check for shutdown
            if shared.is_shutdown_requested() {
                break;
            }

            // Update flow control based on queue size
            let queue_len = kf_receiver.len();
            shared.set_stop_keyframe_creation(queue_len > MAX_QUEUE_SIZE);

            // Try to receive a keyframe with timeout (allows shutdown checks)
            match kf_receiver.recv_timeout(RECV_TIMEOUT) {
                Ok(msg) => {
                    self.process_keyframe(msg, &shared);
                }
                Err(RecvTimeoutError::Timeout) => {
                    // No keyframe available, continue to check shutdown
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Channel closed, exit
                    break;
                }
            }
        }
    }

    /// Process a single keyframe message.
    fn process_keyframe(&mut self, msg: NewKeyFrameMsg, shared: &Arc<SharedState>) {
        // Clear abort flag (we're starting fresh)
        shared.clear_abort_ba();

        // Step 1: Insert keyframe into the map
        let kf_id = self.insert_keyframe(&msg, shared);

        // Step 2: Associate existing map point matches
        self.associate_matched_points(&msg, kf_id, shared);

        // Step 3: Triangulate new map points from unmatched stereo features
        self.triangulate_new_points(&msg, kf_id, shared);

        // Step 4: Local BA
        // TODO: BA is currently DISABLED - the Jacobian/gradient computation has bugs
        // that cause divergence. Tracking works well without BA for now.
        // See issue: Gauss-Newton BA diverges on first iteration (error increases)
        // Needs: Deep comparison with g2o's exact implementation
        // self.local_bundle_adjustment(kf_id, shared);

        // Step 5: Cull map points (stub - does nothing for now)
        self.cull_map_points(shared);

        // Step 6: Try to initialize IMU if not yet done
        self.try_imu_initialization(shared);
    }

    /// Attempt IMU initialization if conditions are met.
    fn try_imu_initialization(&self, shared: &Arc<SharedState>) {
        // Check if already initialized
        {
            let atlas = shared.atlas.read();
            if atlas.active_map().is_imu_initialized() {
                return;
            }
        }

        // Try to initialize
        if let Some(result) = initialize_imu(shared) {
            apply_imu_init(shared, &result);
        }
    }

    /// Insert the keyframe into the map.
    fn insert_keyframe(&self, msg: &NewKeyFrameMsg, shared: &Arc<SharedState>) -> KeyFrameId {
        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();

        // Create the keyframe in the map
        let kf_id = map.create_keyframe(
            msg.timestamp_ns,
            msg.pose.clone(),
            msg.keypoints.clone(),
            msg.descriptors.clone(),
            msg.points_cam.clone(),
        );

        // Set velocity and IMU preintegration on the keyframe
        if let Some(kf) = map.get_keyframe_mut(kf_id) {
            kf.velocity = msg.velocity;
            kf.imu_preintegrated = msg.imu_preintegrated.clone();

            // Compute simple BoW vector and add to database
            let bow = compute_bow_stub(&msg.descriptors);
            kf.set_bow_vector(bow.clone());
            let map_idx = atlas.active_map_index();
            atlas.keyframe_db.add(kf_id, bow, map_idx);
        }

        kf_id
    }

    /// Associate map points that were matched during tracking.
    fn associate_matched_points(
        &self,
        msg: &NewKeyFrameMsg,
        kf_id: KeyFrameId,
        shared: &Arc<SharedState>,
    ) {
        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();

        for (feat_idx, mp_id_opt) in msg.matched_map_points.iter().enumerate() {
            if let Some(mp_id) = mp_id_opt {
                // Create bidirectional association (also updates covisibility)
                map.associate(kf_id, feat_idx, *mp_id);
            }
        }
    }

    /// Triangulate new map points from unmatched stereo features.
    ///
    /// For each feature that:
    /// 1. Has a valid stereo depth (points_cam[i] is Some)
    /// 2. Was NOT matched to an existing map point during tracking
    ///
    /// We create a new MapPoint at the triangulated world position.
    fn triangulate_new_points(
        &self,
        msg: &NewKeyFrameMsg,
        kf_id: KeyFrameId,
        shared: &Arc<SharedState>,
    ) {
        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();

        for (feat_idx, p_cam_opt) in msg.points_cam.iter().enumerate() {
            // Skip if no valid stereo depth
            let p_cam = match p_cam_opt {
                Some(p) => p,
                None => continue,
            };

            // Skip if already matched to an existing map point
            if msg.matched_map_points[feat_idx].is_some() {
                continue;
            }

            // Transform to world coordinates
            let p_world = msg.pose.transform_point(p_cam);

            // Get descriptor for this feature
            let descriptor = match msg.descriptors.row(feat_idx as i32) {
                Ok(row) => row.try_clone().unwrap_or_default(),
                Err(_) => Mat::default(),
            };

            // Create new map point
            let mp_id = map.create_map_point(p_world, descriptor, kf_id);

            // Associate with the keyframe (also updates covisibility)
            map.associate(kf_id, feat_idx, mp_id);
        }
    }

    /// Local Bundle Adjustment.
    ///
    /// Optimizes:
    /// - Poses of recent keyframes (covisible with current)
    /// - Positions of map points observed by those keyframes
    ///
    /// The optimization checks `shared.should_abort_ba()` periodically
    /// and exits early if a new keyframe arrived.
    fn local_bundle_adjustment(&self, kf_id: KeyFrameId, shared: &Arc<SharedState>) {
        let config = LocalBAConfig::default();
        let should_stop = || shared.should_abort_ba();
        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();
        if let Some(result) = run_local_ba(map, kf_id, &self.camera, &config, &should_stop) {
            if result.iterations > 0 {
                eprintln!(
                    "[LocalBA] kf={} iters={} error: {:.2} -> {:.2} (kfs={}, mps={}, obs={})",
                    kf_id.0,
                    result.iterations,
                    result.initial_error,
                    result.final_error,
                    result.num_keyframes,
                    result.num_map_points,
                    result.num_observations
                );
            }
        }
    }

    /// Cull bad map points (stub - does nothing for now).
    ///
    /// In a full implementation, this would remove map points that:
    /// - Have too few observations
    /// - Have high reprojection error
    /// - Are occluded in most recent frames
    fn cull_map_points(&self, _shared: &Arc<SharedState>) {
        // TODO: Implement map point culling
    }
}

/// Compute a simple placeholder Bag-of-Words vector.
///
/// This just assigns a unit weight to each descriptor row index,
/// which is enough to exercise the KeyFrameDatabase.
fn compute_bow_stub(descriptors: &Mat) -> BowVector {
    let mut bow = BowVector::new();
    let rows = descriptors.rows();
    for i in 0..rows {
        bow.insert(i as u32, 1.0);
    }
    bow
}
