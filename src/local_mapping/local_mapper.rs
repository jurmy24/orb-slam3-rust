//! Local Mapper - Main Local Mapping thread implementation.
//!
//! Processes keyframes received from Tracking:
//! 1. Inserts keyframe into the map
//! 2. Associates matched map points
//! 3. Triangulates new map points from unmatched stereo features
//! 4. Updates covisibility graph (automatic via associate)
//! 5. Local BA (using Levenberg-Marquardt)
//! 6. Map point culling (removes low-quality points)
//! 7. Keyframe culling (removes redundant keyframes)
//! 8. IMU initialization (for visual-inertial mode)

use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use opencv::core::Mat;
use opencv::prelude::*;
use tracing::{debug, info};

use crate::atlas::keyframe_db::BowVector;
use crate::atlas::map::KeyFrameId;
use crate::optimizer::{
    collect_visual_ba_data, solve_visual_ba, apply_visual_ba_results,
    collect_inertial_ba_data, solve_inertial_ba, apply_inertial_ba_results,
    LocalBAConfigLM, LocalInertialBAConfig,
};
use crate::system::messages::NewKeyFrameMsg;
use crate::system::shared_state::SharedState;
use crate::tracking::frame::CameraModel;

use super::imu_init::{apply_imu_init, check_sufficient_motion, initialize_imu};
use super::search_in_neighbors::search_in_neighbors;
use super::triangulation::{triangulate_from_neighbors as do_multiframe_triangulation, TriangulationConfig};

/// Flow control threshold: if queue has more than this many keyframes,
/// signal Tracking to stop creating new ones.
const MAX_QUEUE_SIZE: usize = 3;

/// Timeout for receiving keyframes. Allows periodic shutdown checks.
const RECV_TIMEOUT: Duration = Duration::from_millis(100);

/// Local Mapping thread state.
pub struct LocalMapper {
    /// Camera model for triangulation.
    camera: CameraModel,

    /// Channel sender to Loop Closing thread.
    lc_sender: Option<Sender<KeyFrameId>>,
}

impl LocalMapper {
    /// Create a new LocalMapper.
    ///
    /// # Arguments
    /// * `camera` - Camera model for triangulation
    /// * `lc_sender` - Optional sender to forward keyframe IDs to Loop Closing
    pub fn new(camera: CameraModel, lc_sender: Option<Sender<KeyFrameId>>) -> Self {
        Self { camera, lc_sender }
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

            // Handle pause request from LoopCloser
            if shared.should_pause_local_mapping() {
                shared.set_local_mapping_paused(true);
                while shared.should_pause_local_mapping() && !shared.is_shutdown_requested() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                shared.set_local_mapping_paused(false);
                continue;
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

        // Step 3b: Multi-frame triangulation with geometric validation
        // This creates additional map points by matching between neighbor keyframes
        self.triangulate_from_neighbors(kf_id, shared);

        // Step 3c: Search in neighbors - fuse duplicate map points
        // This merges map points between the current KF and its covisible neighbors
        self.search_in_neighbors(kf_id, shared);

        // Step 4: Local Bundle Adjustment (LM-based)
        self.local_bundle_adjustment(kf_id, shared);

        // Step 5: Cull bad map points
        self.cull_map_points(kf_id, shared);

        // Step 6: Cull redundant keyframes
        self.cull_keyframes(kf_id, shared);

        // Step 7: Try to initialize IMU if not yet done
        self.try_imu_initialization(shared);

        // Step 8: Send keyframe to Loop Closing (non-blocking)
        if let Some(ref sender) = self.lc_sender {
            let _ = sender.try_send(kf_id);
        }
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

        // Check for sufficient motion (sets bad_imu flag if insufficient)
        if !check_sufficient_motion(shared) {
            // Insufficient motion detected - Tracker will handle map reset
            return;
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

            // Compute BoW vector and FeatureVector using vocabulary if available
            let (bow, feat_opt) = if let Some(vocab) = shared.vocabulary() {
                // Use real vocabulary - levels_up=4 for typical L=5 vocabulary (groups at level 1)
                let (bow, feat) = vocab.transform(&msg.descriptors, 4);
                (bow, Some(feat))
            } else {
                // Fallback to stub if no vocabulary loaded
                (compute_bow_stub(&msg.descriptors), None)
            };

            kf.set_bow_vector(bow.clone());
            if let Some(feat) = feat_opt {
                kf.set_feature_vector(feat);
            }

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

    /// Multi-frame triangulation with geometric validation.
    ///
    /// Creates new map points by matching features between the current keyframe
    /// and its neighbors, then triangulating with DLT and validating with:
    /// - Parallax angle check
    /// - Depth positivity check
    /// - Reprojection error check
    /// - Scale consistency check
    fn triangulate_from_neighbors(&self, kf_id: KeyFrameId, shared: &Arc<SharedState>) {
        let config = TriangulationConfig::default();

        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();

        // Check if IMU is initialized (affects parallax threshold)
        let is_inertial = map.is_imu_initialized();

        let _result = do_multiframe_triangulation(map, kf_id, &self.camera, &config, is_inertial);
    }

    /// Search in neighbors: fuse duplicate map points between covisible keyframes.
    ///
    /// This function:
    /// 1. Collects neighbor keyframes (covisible + temporal for inertial)
    /// 2. Fuses current KF's map points into neighbors (adds observations or merges)
    /// 3. Fuses neighbors' map points into current KF
    /// 4. Updates affected points' descriptors and normals
    ///
    /// This reduces map point count and improves map consistency.
    fn search_in_neighbors(&self, kf_id: KeyFrameId, shared: &Arc<SharedState>) {
        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();

        let is_inertial = map.is_imu_initialized();
        let result = search_in_neighbors(map, kf_id, &self.camera, is_inertial);

        if result.num_fused > 0 || result.num_observations_added > 0 {
            debug!(
                "[SearchInNeighbors] kf={}: fused {} points, added {} observations",
                kf_id.0, result.num_fused, result.num_observations_added
            );
        }
    }

    /// Local Bundle Adjustment using three-phase locking.
    ///
    /// This implementation minimizes lock contention by separating BA into three phases:
    /// 1. **COLLECT** (~100ms): Read lock to extract data snapshot
    /// 2. **SOLVE** (6-20s): NO lock - run LM optimization on extracted data
    /// 3. **APPLY** (~10ms): Write lock to write results back
    ///
    /// This reduces lock hold time from 6-20+ seconds to ~110ms (99% reduction).
    ///
    /// When IMU is initialized, uses Local Inertial BA which optimizes:
    /// - Poses, velocities, and biases of recent keyframes (temporal window)
    /// - Positions of map points observed by those keyframes
    /// - IMU preintegration constraints between consecutive keyframes
    ///
    /// Otherwise, uses visual-only Local BA which optimizes:
    /// - Poses of recent keyframes (covisible with current)
    /// - Positions of map points observed by those keyframes
    ///
    /// The optimization checks `shared.should_abort_ba()` periodically
    /// and exits early if a new keyframe arrived.
    fn local_bundle_adjustment(&self, kf_id: KeyFrameId, shared: &Arc<SharedState>) {
        let should_stop = || shared.should_abort_ba();

        // Check if IMU is initialized (quick read lock)
        let is_inertial = {
            let atlas = shared.atlas.read();
            atlas.active_map().is_imu_initialized()
        };

        if is_inertial {
            // ========== INERTIAL BA (THREE-PHASE) ==========
            let config = LocalInertialBAConfig::default();

            // PHASE 1: COLLECT (read lock)
            let problem = {
                let atlas = shared.atlas.read();
                match collect_inertial_ba_data(atlas.active_map(), kf_id, &config) {
                    Some(p) => p,
                    None => return,
                }
            }; // Read lock released here

            // PHASE 2: SOLVE (no lock held!)
            let result = match solve_inertial_ba(&problem, &self.camera, &config, &should_stop) {
                Some(r) => r,
                None => return,
            };

            // PHASE 3: APPLY (write lock)
            if result.iterations > 0 {
                let mut atlas = shared.atlas.write();
                let updated = apply_inertial_ba_results(atlas.active_map_mut(), &result);

                debug!(
                    "[LocalInertialBA] kf={} iters={} error: {:.2} -> {:.2} (updated={})",
                    kf_id.0,
                    result.iterations,
                    result.initial_error,
                    result.final_error,
                    updated
                );
            } // Write lock released here
        } else {
            // ========== VISUAL BA (THREE-PHASE) ==========
            let config = LocalBAConfigLM::default();

            // PHASE 1: COLLECT (read lock)
            let problem = {
                let atlas = shared.atlas.read();
                match collect_visual_ba_data(atlas.active_map(), kf_id, &config) {
                    Some(p) => p,
                    None => return,
                }
            }; // Read lock released here

            // PHASE 2: SOLVE (no lock held!)
            let result = match solve_visual_ba(&problem, &self.camera, &config, &should_stop) {
                Some(r) => r,
                None => return,
            };

            // PHASE 3: APPLY (write lock)
            if result.iterations > 0 {
                let mut atlas = shared.atlas.write();
                let updated = apply_visual_ba_results(atlas.active_map_mut(), &result);

                debug!(
                    "[LocalBA-LM] kf={} iters={} error: {:.3} -> {:.3} px (updated={})",
                    kf_id.0,
                    result.iterations,
                    result.initial_error,
                    result.final_error,
                    updated
                );
            } // Write lock released here
        }
    }

    /// Cull bad map points.
    ///
    /// Removes map points that:
    /// - Have found_count/visible_count < 0.25 (poor tracking ratio)
    /// - Have < 3 observations after grace period (3 keyframes since creation)
    ///
    /// # Arguments
    /// * `current_kf_id` - Current keyframe ID (used for grace period)
    /// * `shared` - Shared state containing the atlas
    fn cull_map_points(&self, current_kf_id: KeyFrameId, shared: &Arc<SharedState>) {
        const MIN_FOUND_RATIO: f64 = 0.25;
        const MIN_OBSERVATIONS: usize = 3;
        const GRACE_KEYFRAMES: u64 = 3;

        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();

        // Collect points to cull
        let to_cull: Vec<_> = map
            .map_points()
            .filter(|mp| {
                // Skip if already bad
                if mp.is_bad {
                    return true;
                }

                // Grace period: don't cull points created in the last few keyframes
                let age_keyframes = current_kf_id.0.saturating_sub(mp.first_kf_id.0);
                if age_keyframes < GRACE_KEYFRAMES {
                    // During grace period, only cull if very few observations
                    // and the point should have been seen more
                    if mp.num_observations() < 2 && mp.visible_count > 2 {
                        return true;
                    }
                    return false;
                }

                // After grace period, apply full culling criteria
                mp.should_cull(MIN_FOUND_RATIO, MIN_OBSERVATIONS)
            })
            .map(|mp| mp.id)
            .collect();

        let num_culled = to_cull.len();

        // Remove culled points
        for mp_id in to_cull {
            map.remove_map_point_full(mp_id);
        }

        if num_culled > 0 {
            debug!(
                "[MapCulling] Removed {} bad map points (remaining: {})",
                num_culled,
                map.num_map_points()
            );
        }
    }

    /// Cull redundant keyframes.
    ///
    /// Removes keyframes where >90% (visual) or >50% (inertial) of their MapPoints
    /// are observed by at least 3 other keyframes. This keeps the map compact
    /// while maintaining coverage.
    ///
    /// Does NOT cull:
    /// - The anchor (first) keyframe
    /// - The most recent keyframe
    ///
    /// When IMU is initialized, this function properly merges the IMU preintegration
    /// from the culled keyframe into its successor to maintain the temporal chain.
    ///
    /// # Arguments
    /// * `current_kf_id` - Current keyframe ID (will not be culled)
    /// * `shared` - Shared state containing the atlas
    fn cull_keyframes(&self, current_kf_id: KeyFrameId, shared: &Arc<SharedState>) {
        const MIN_OBSERVATIONS_PER_POINT: usize = 3;

        let mut atlas = shared.atlas.write();
        let map = atlas.active_map_mut();

        // Check if IMU is initialized (affects culling rules and threshold)
        let imu_initialized = map.is_imu_initialized();

        // Use different redundancy threshold for inertial vs visual
        // Inertial mode is more aggressive (0.5) to keep computational cost down
        let redundancy_threshold = if imu_initialized { 0.5 } else { 0.9 };

        // Get local keyframes to check (covisible with current)
        let local_kf_ids: Vec<KeyFrameId> = {
            if let Some(kf) = map.get_keyframe(current_kf_id) {
                kf.covisibility_weights()
                    .keys()
                    .copied()
                    .collect()
            } else {
                return;
            }
        };

        let mut to_cull = Vec::new();

        for &kf_id in &local_kf_ids {
            // Never cull the current keyframe
            if kf_id == current_kf_id {
                continue;
            }

            let kf = match map.get_keyframe(kf_id) {
                Some(kf) => kf,
                None => continue,
            };

            // Don't cull if already bad
            if kf.is_bad {
                continue;
            }

            // Don't cull root keyframe (no parent)
            if kf.is_root() {
                continue;
            }

            // Count how many of this KF's map points are well-observed elsewhere
            let mut redundant_count = 0;
            let mut total_points = 0;

            for mp_id_opt in &kf.map_point_ids {
                if let Some(mp_id) = mp_id_opt {
                    if let Some(mp) = map.get_map_point(*mp_id) {
                        if mp.is_bad {
                            continue;
                        }

                        total_points += 1;

                        // Count observations in other keyframes
                        let obs_in_others = mp.observations.keys()
                            .filter(|&&obs_kf_id| obs_kf_id != kf_id)
                            .count();

                        if obs_in_others >= MIN_OBSERVATIONS_PER_POINT {
                            redundant_count += 1;
                        }
                    }
                }
            }

            // Check if keyframe is redundant
            if total_points > 0 {
                let redundancy_ratio = redundant_count as f64 / total_points as f64;
                if redundancy_ratio > redundancy_threshold {
                    to_cull.push(kf_id);
                }
            }
        }

        let num_culled = to_cull.len();

        // Remove culled keyframes with proper IMU handling
        for kf_id in to_cull {
            self.remove_keyframe_with_imu_merge(map, kf_id, imu_initialized);
        }

        if num_culled > 0 {
            info!(
                "[KFCulling] Removed {} redundant keyframes (remaining: {})",
                num_culled,
                map.num_keyframes()
            );
        }
    }

    /// Remove a keyframe while properly merging IMU preintegration if needed.
    ///
    /// When IMU is initialized, this function:
    /// 1. Merges the culled KF's preintegration into the next KF
    /// 2. Updates the temporal chain (prev_kf, next_kf links)
    /// 3. Removes the keyframe from the map
    fn remove_keyframe_with_imu_merge(
        &self,
        map: &mut crate::atlas::map::Map,
        kf_id: KeyFrameId,
        imu_initialized: bool,
    ) {
        // Get temporal chain info before removal
        let (prev_kf_id, next_kf_id, preint_to_merge) = {
            match map.get_keyframe(kf_id) {
                Some(kf) => (kf.prev_kf, kf.next_kf, kf.imu_preintegrated.clone()),
                None => return,
            }
        };

        // If IMU is initialized and there's a next keyframe, merge preintegration
        if imu_initialized {
            if let (Some(next_id), Some(preint)) = (next_kf_id, preint_to_merge) {
                // Merge preintegration: next_kf's preintegration should now include
                // the culled kf's preintegration
                if let Some(next_kf) = map.get_keyframe_mut(next_id) {
                    if let Some(ref mut next_preint) = next_kf.imu_preintegrated {
                        // Merge the culled KF's preintegration into the next KF's
                        next_preint.merge_previous(&preint);
                        debug!(
                            "[KFCulling] Merged IMU preintegration from KF {} into KF {}",
                            kf_id.0, next_id.0
                        );
                    }
                }
            }

            // Update temporal chain: connect prev and next
            if let Some(prev_id) = prev_kf_id {
                if let Some(prev_kf) = map.get_keyframe_mut(prev_id) {
                    prev_kf.next_kf = next_kf_id;
                }
            }
            if let Some(next_id) = next_kf_id {
                if let Some(next_kf) = map.get_keyframe_mut(next_id) {
                    next_kf.prev_kf = prev_kf_id;
                }
            }
        }

        // Now remove the keyframe (handles covisibility, spanning tree, etc.)
        map.remove_keyframe_full(kf_id);
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
