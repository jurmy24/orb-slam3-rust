//! Main tracker: orchestrates frame processing and pose estimation.
//!
//! This implementation follows the ORB-SLAM3 design:
//! - Uses shared `Atlas` via RwLock for thread-safe access
//! - Tracks 3D `MapPoint`s by projection into the current frame
//! - Uses IMU preintegration to obtain a motion prior
//! - Estimates pose with PnP given 3D–2D correspondences
//! - Sends new keyframes to Local Mapping via channel

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use crossbeam_channel::Sender;
use nalgebra::Vector3;
use opencv::core::{Mat, Point2f, Vector};
use opencv::features2d::BFMatcher;
use opencv::prelude::*;
use tracing::{debug, error};

use crate::atlas::keyframe_db::BowVector;
use crate::atlas::map::{KeyFrameId, MapPointId};
use crate::geometry::{SE3, solve_pnp_ransac_detailed};
use crate::imu::{ImuBias, ImuNoise, Preintegrator};
use crate::io::euroc::ImuEntry;
use crate::system::messages::NewKeyFrameMsg;
use crate::system::shared_state::SharedState;
use crate::tracking::TrackingState;
use crate::tracking::frame::{CameraModel, StereoFrame};
use crate::tracking::keyframe_decision::KeyFrameDecision;
use crate::tracking::matching::{NN_RATIO, TH_HIGH, descriptor_distance};
use crate::tracking::result::{MatchInfo, TimingStats, TrackingMetrics, TrackingResult};
use crate::tracking::tracking_frame::Frame;

/// Time threshold for transitioning from RECENTLY_LOST to LOST (seconds).
const TIME_RECENTLY_LOST_THRESHOLD: f64 = 5.0;

/// Minimum number of keyframes to keep a map (below this, reset instead of create new).
const MIN_KEYFRAMES_FOR_NEW_MAP: usize = 10;

/// Minimum inliers for tracking to be considered OK.
const MIN_INLIERS_OK: usize = 20;

/// Minimum inliers to consider tracking "weak but usable" (for RECENTLY_LOST KF creation).
const MIN_INLIERS_WEAK: usize = 10;

/// Main tracking structure.
pub struct Tracker {
    camera: CameraModel,
    preintegrator: Preintegrator,

    /// Current pose (T_wc) and velocity (world frame).
    pub pose: SE3,
    pub velocity: Vector3<f64>,
    pub trajectory: Vec<SE3>,

    /// Reference keyframe used for tracking.
    reference_kf: Option<KeyFrameId>,

    /// Running frame counter.
    frame_count: usize,

    /// Tracking state machine (NotInitialized, Ok, RecentlyLost, Lost).
    pub state: TrackingState,

    /// Number of consecutive frames with poor tracking.
    lost_frames: usize,

    /// Shared state (Atlas, flags) - thread-safe access.
    shared: Arc<SharedState>,

    /// Channel to send keyframes to Local Mapping.
    kf_sender: Sender<NewKeyFrameMsg>,

    /// Keyframe creation decision logic.
    kf_decision: KeyFrameDecision,

    /// Separate preintegrator for accumulating IMU since last keyframe.
    /// Reset when a new keyframe is created.
    kf_preintegrator: Preintegrator,

    /// Number of map points in the reference keyframe (for KF decision).
    reference_kf_num_points: usize,

    /// Timestamp when tracking was lost (for RECENTLY_LOST timeout).
    time_stamp_lost: Option<f64>,

    /// Last keyframe timestamp (for time-based keyframe decision).
    last_kf_timestamp: f64,

    // ─────────────────────────────────────────────────────────────────────────
    // Motion Model Tracking State
    // ─────────────────────────────────────────────────────────────────────────
    /// Previous frame pose for motion model tracking.
    last_frame_pose: Option<SE3>,

    /// Motion model velocity: relative transform T_curr_prev = T_curr * T_prev^-1
    motion_model_velocity: Option<SE3>,

    /// Previous frame's matched map points (feature index -> MapPointId).
    last_frame_map_points: Vec<Option<MapPointId>>,

    /// Previous frame's 3D points in camera frame (for projection).
    last_frame_points_cam: Vec<Option<Vector3<f64>>>,

    /// Previous frame's descriptors for matching.
    last_frame_descriptors: Mat,
}

impl Tracker {
    pub fn new(
        camera: CameraModel,
        shared: Arc<SharedState>,
        kf_sender: Sender<NewKeyFrameMsg>,
    ) -> Result<Self> {
        Ok(Self {
            camera,
            preintegrator: Preintegrator::new_with_covariance(ImuBias::zero(), ImuNoise::default()),
            pose: SE3::identity(),
            velocity: Vector3::zeros(),
            trajectory: vec![SE3::identity()],
            reference_kf: None,
            frame_count: 0,
            state: TrackingState::NotInitialized,
            lost_frames: 0,
            shared,
            kf_sender,
            kf_decision: KeyFrameDecision::new(),
            kf_preintegrator: Preintegrator::new(ImuBias::zero(), ImuNoise::default()),
            reference_kf_num_points: 0,
            time_stamp_lost: None,
            last_kf_timestamp: 0.0,
            // Motion model state
            last_frame_pose: None,
            motion_model_velocity: None,
            last_frame_map_points: Vec::new(),
            last_frame_points_cam: Vec::new(),
            last_frame_descriptors: Mat::default(),
        })
    }

    /// Process a stereo frame with accompanying IMU interval and return a
    /// detailed `TrackingResult` suitable for visualization and debugging.
    pub fn process_frame(
        &mut self,
        stereo_frame: StereoFrame,
        imu_measurements: &[ImuEntry],
    ) -> Result<TrackingResult> {
        let t_start = Instant::now();
        let prev_pose = self.pose.clone();
        let current_timestamp = stereo_frame.timestamp_ns as f64 / 1e9; // Convert to seconds

        self.frame_count += 1;

        // Build tracking frame from stereo output.
        let frame = Frame::from_stereo(stereo_frame.clone());

        // Integrate IMU for motion prior
        self.preintegrator.reset();
        for pair in imu_measurements.windows(2) {
            let prev = pair[0].sample;
            let curr = pair[1].sample;
            self.preintegrator.integrate(prev, curr);

            // Also accumulate for keyframe preintegration
            self.kf_preintegrator.integrate(prev, curr);
        }

        // Propagate the preintegrated state to get the motion prior
        let (pred_rot, pred_pos, pred_vel) =
            self.preintegrator
                .propagate(self.pose.rotation, self.pose.translation, self.velocity);
        let imu_prior = SE3 {
            rotation: pred_rot,
            translation: pred_pos,
        };

        // --- Map initialization or tracking ---
        let n_inliers: usize;
        let matched_map_points: Vec<Option<MapPointId>>;
        let mut reproj_errors: Vec<f64> = Vec::new();
        let mut inlier_indices: Vec<usize> = Vec::new();
        let mut outlier_indices: Vec<usize> = Vec::new();
        let tracking_ok;

        // Check if the active map needs initialization
        let needs_init = {
            let atlas = self.shared.atlas.read();
            atlas.active_map().num_keyframes() == 0
        };

        if needs_init {
            // Initialize map from first frame - this creates the first keyframe directly
            self.initialize_map(&frame, &stereo_frame, &imu_prior)?;
            n_inliers = frame.points_cam.iter().filter(|p| p.is_some()).count();
            matched_map_points = vec![None; frame.num_features()];
            tracking_ok = n_inliers >= MIN_INLIERS_OK;
            self.last_kf_timestamp = current_timestamp;
        } else {
            // Handle different tracking states
            match self.state {
                TrackingState::Ok | TrackingState::NotInitialized => {
                    // Normal tracking: reference KF + local map
                    let (inliers, matches, reproj_errs, inlier_idx, outlier_idx) =
                        self.track_normal(&frame, &imu_prior)?;
                    n_inliers = inliers;
                    matched_map_points = matches;
                    reproj_errors = reproj_errs;
                    inlier_indices = inlier_idx;
                    outlier_indices = outlier_idx;
                    tracking_ok = n_inliers >= MIN_INLIERS_OK;

                    // Try to create keyframe even with weak tracking (helps map growth)
                    if n_inliers >= MIN_INLIERS_WEAK {
                        self.maybe_create_keyframe(
                            &frame,
                            &stereo_frame,
                            n_inliers,
                            &matched_map_points,
                            current_timestamp,
                        );
                    }
                }

                TrackingState::RecentlyLost => {
                    // IMU-only prediction while recently lost
                    // Try to recover with visual tracking
                    let (inliers, matches, reproj_errs, inlier_idx, outlier_idx) =
                        self.track_normal(&frame, &imu_prior)?;
                    n_inliers = inliers;
                    matched_map_points = matches;
                    reproj_errors = reproj_errs;
                    inlier_indices = inlier_idx;
                    outlier_indices = outlier_idx;
                    tracking_ok = n_inliers >= MIN_INLIERS_OK;

                    // For stereo-inertial: create keyframes even when RECENTLY_LOST
                    // This helps gather IMU data and allows the map to grow
                    // even during brief tracking failures (ORB-SLAM3's mInsertKFsLost)
                    if n_inliers >= MIN_INLIERS_WEAK {
                        self.maybe_create_keyframe(
                            &frame,
                            &stereo_frame,
                            n_inliers,
                            &matched_map_points,
                            current_timestamp,
                        );
                    }

                    if !tracking_ok {
                        // Use IMU prediction only
                        self.pose = imu_prior.clone();

                        // Check timeout for RECENTLY_LOST -> LOST transition
                        if let Some(lost_time) = self.time_stamp_lost {
                            if current_timestamp - lost_time > TIME_RECENTLY_LOST_THRESHOLD {
                                // Transition to LOST
                                self.state = TrackingState::Lost;
                            }
                        }
                    }
                }

                TrackingState::Lost => {
                    // When LOST: create new map or reset current map
                    self.handle_lost_state()?;

                    // After creating new map, return early - next frame will initialize
                    n_inliers = 0;
                    matched_map_points = vec![None; frame.num_features()];
                    tracking_ok = false;
                }
            }
        }

        // Update tracking state based on results
        self.update_state_with_result(tracking_ok, current_timestamp);

        // Update motion model if tracking was successful
        if tracking_ok {
            self.update_motion_model(&frame, &matched_map_points);
        } else if self.state == TrackingState::Lost {
            // Clear motion model on tracking loss
            self.clear_motion_model();
        }

        // Update velocity from IMU prediction
        self.velocity = pred_vel;
        self.trajectory.push(self.pose.clone());

        // Compute IMU preintegration residual
        let imu_residual = (imu_prior.translation - self.pose.translation).norm();

        // Build metrics
        let n_features = frame.features.keypoints.len();
        let n_map_point_matches = n_inliers;
        let inlier_ratio = if n_map_point_matches > 0 {
            n_inliers as f64 / n_map_point_matches as f64
        } else {
            0.0
        };

        // Compute reprojection error statistics
        let reproj_error_median_px = if !reproj_errors.is_empty() {
            let mut sorted_errors = reproj_errors.clone();
            sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted_errors[sorted_errors.len() / 2]
        } else {
            0.0
        };

        let reproj_error_mean_px = if !reproj_errors.is_empty() {
            reproj_errors.iter().sum::<f64>() / reproj_errors.len() as f64
        } else {
            0.0
        };

        let delta_t = (self.pose.translation - prev_pose.translation).norm();
        let dq = prev_pose.rotation.inverse() * self.pose.rotation;
        let delta_rot_rad = dq.angle();
        let delta_rot_deg = delta_rot_rad.to_degrees();

        // Get IMU bias from preintegrator
        let imu_bias = Some(self.preintegrator.bias);

        // Get local map point IDs for visualization
        let local_map_point_ids = {
            let atlas = self.shared.atlas.read();
            let map = atlas.active_map();
            if let Some(ref_kf) = self.reference_kf {
                let local_kfs = map.get_local_keyframes_readonly(ref_kf, 10);
                map.get_map_points_from_keyframes(&local_kfs)
                    .into_iter()
                    .collect()
            } else {
                Vec::new()
            }
        };

        let metrics = TrackingMetrics {
            n_features,
            n_map_point_matches,
            n_inliers,
            inlier_ratio,
            reproj_error_median_px,
            reproj_error_mean_px,
            imu_preint_residual_m: imu_residual,
            delta_translation_m: delta_t,
            delta_rotation_deg: delta_rot_deg,
        };

        let mut timing = TimingStats::zero();
        timing.total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        // Build matched map points list for MatchInfo
        let matched_map_points_list: Vec<(MapPointId, usize)> = matched_map_points
            .iter()
            .enumerate()
            .filter_map(|(idx, mp_id_opt)| mp_id_opt.map(|mp_id| (mp_id, idx)))
            .collect();

        let matches = MatchInfo {
            matched_map_points: matched_map_points_list,
            inlier_indices,
            outlier_indices,
            reproj_errors,
            local_map_point_ids,
        };

        Ok(TrackingResult {
            state: self.state,
            pose: self.pose.clone(),
            velocity: self.velocity,
            reference_kf_id: self.reference_kf,
            metrics,
            timing,
            matches,
            imu_bias,
        })
    }

    /// Normal tracking: motion model or reference KF, followed by local map tracking.
    ///
    /// ORB-SLAM3 order:
    /// 1. Try TrackWithMotionModel (constant velocity assumption)
    /// 2. If that fails, try TrackReferenceKeyFrame
    /// 3. Refine with TrackLocalMap
    fn track_normal(
        &mut self,
        frame: &Frame,
        imu_prior: &SE3,
    ) -> Result<(
        usize,
        Vec<Option<MapPointId>>,
        Vec<f64>,
        Vec<usize>,
        Vec<usize>,
    )> {
        // Stage 1: Initial pose estimation
        // Try motion model first (if we have velocity), then fall back to reference KF
        let initial_pose = {
            let atlas = self.shared.atlas.read();

            // Try motion model first (requires previous frame data)
            let motion_model_result = if self.motion_model_velocity.is_some() {
                self.track_with_motion_model(frame, &atlas)
            } else {
                None
            };

            if let Some(pose_mm) = motion_model_result {
                pose_mm
            } else {
                // Fall back to reference keyframe tracking
                if let Some(pose_ref) = self.track_with_reference_kf(frame, &atlas) {
                    pose_ref
                } else {
                    // Last resort: use IMU prior
                    imu_prior.clone()
                }
            }
        };
        self.pose = initial_pose.clone();

        // Stage 2: Refine pose by tracking the local map
        let (estimated_pose, inliers, matches, reproj_errors, inlier_indices, outlier_indices) = {
            let atlas = self.shared.atlas.read();
            self.track_local_map(&atlas, frame, &initial_pose)?
        };
        self.pose = estimated_pose;

        Ok((
            inliers,
            matches,
            reproj_errors,
            inlier_indices,
            outlier_indices,
        ))
    }

    /// Handle LOST state: create new map or reset current map.
    fn handle_lost_state(&mut self) -> Result<()> {
        let num_keyframes = {
            let atlas = self.shared.atlas.read();
            atlas.active_map().num_keyframes()
        };

        if num_keyframes < MIN_KEYFRAMES_FOR_NEW_MAP {
            // Too few keyframes - reset current map
            error!(
                "Tracking LOST with {} keyframes - resetting active map",
                num_keyframes
            );
            let mut atlas = self.shared.atlas.write();
            atlas.reset_active_map();
        } else {
            // Enough keyframes - create new map (old map preserved for potential merging)
            error!("Tracking LOST - creating new map in Atlas");
            let mut atlas = self.shared.atlas.write();
            atlas.create_new_map();
        }

        // Reset tracker state for new map
        self.reference_kf = None;
        self.reference_kf_num_points = 0;
        self.state = TrackingState::NotInitialized;
        self.lost_frames = 0;
        self.time_stamp_lost = None;
        self.kf_decision.reset();
        self.kf_preintegrator.reset();
        self.clear_motion_model();

        Ok(())
    }

    /// Update tracking state based on tracking result.
    fn update_state_with_result(&mut self, tracking_ok: bool, current_timestamp: f64) {
        match self.state {
            TrackingState::NotInitialized => {
                if tracking_ok {
                    self.state = TrackingState::Ok;
                    self.lost_frames = 0;
                    self.time_stamp_lost = None;
                }
            }
            TrackingState::Ok => {
                if tracking_ok {
                    self.lost_frames = 0;
                    self.time_stamp_lost = None;
                } else {
                    // Transition to RECENTLY_LOST
                    self.state = TrackingState::RecentlyLost;
                    self.time_stamp_lost = Some(current_timestamp);
                    self.lost_frames = 1;
                }
            }
            TrackingState::RecentlyLost => {
                if tracking_ok {
                    // Recovered!
                    self.state = TrackingState::Ok;
                    self.lost_frames = 0;
                    self.time_stamp_lost = None;
                }
                // If not OK, state transition to LOST is handled in track_normal via timeout
            }
            TrackingState::Lost => {
                // State change happens in handle_lost_state
            }
        }
    }

    /// Decide whether to create a new keyframe, and if so, send it to Local Mapping.
    fn maybe_create_keyframe(
        &mut self,
        frame: &Frame,
        stereo_frame: &StereoFrame,
        n_inliers: usize,
        matched_map_points: &[Option<MapPointId>],
        current_timestamp: f64,
    ) {
        // Don't create keyframes if flow control says to stop
        if self.shared.should_stop_keyframe_creation() {
            return;
        }

        // Don't create keyframes if tracking is completely lost
        // For stereo-inertial, allow KF creation during RECENTLY_LOST (mInsertKFsLost in ORB-SLAM3)
        if self.state == TrackingState::Lost {
            return;
        }

        // Check if IMU is initialized (for time-based KF decision)
        let imu_initialized = {
            let atlas = self.shared.atlas.read();
            atlas.active_map().is_imu_initialized()
        };

        // Time since last keyframe
        let time_since_last_kf = current_timestamp - self.last_kf_timestamp;

        // Use KeyFrameDecision to check if we should create a keyframe
        let should_create = self.kf_decision.should_create_keyframe_stereo_inertial(
            stereo_frame,
            n_inliers,
            self.reference_kf_num_points,
            time_since_last_kf,
            imu_initialized,
        );

        if !should_create {
            return;
        }

        // Create and send keyframe message to Local Mapping
        let msg = NewKeyFrameMsg {
            keyframe_id: KeyFrameId::new(0), // Local Mapping will assign the real ID
            timestamp_ns: frame.timestamp_ns,
            pose: self.pose.clone(),
            velocity: self.velocity,
            keypoints: frame.features.keypoints.clone(),
            descriptors: frame.features.descriptors.clone(),
            points_cam: frame.points_cam.clone(),
            matched_map_points: matched_map_points.to_vec(),
            imu_preintegrated: Some(self.kf_preintegrator.state.clone()),
        };

        // Signal Local Mapping that a new keyframe is coming (abort current BA)
        self.shared.request_abort_ba();

        // Send to Local Mapping (may block briefly if channel is full)
        if self.kf_sender.send(msg).is_ok() {
            // Reset preintegration accumulator
            self.kf_preintegrator.reset();

            // Update last keyframe timestamp
            self.last_kf_timestamp = current_timestamp;

            // Update reference keyframe info (we don't have the ID yet, but we know
            // the number of points for the next decision)
            self.reference_kf_num_points = n_inliers;

            // Note: reference_kf will be updated when we see the keyframe in the map
            // For now, we'll update it based on the most recent keyframe we can find
            self.update_reference_kf_from_map();
        }
    }

    /// Update reference keyframe to the most recent one in the map.
    fn update_reference_kf_from_map(&mut self) {
        let atlas = self.shared.atlas.read();
        let map = atlas.active_map();

        // Find the keyframe with the highest ID (most recent)
        let mut max_id: Option<KeyFrameId> = None;
        for kf_id in map.keyframe_ids() {
            match max_id {
                None => max_id = Some(*kf_id),
                Some(current_max) if kf_id.0 > current_max.0 => max_id = Some(*kf_id),
                _ => {}
            }
        }

        if let Some(kf_id) = max_id {
            self.reference_kf = Some(kf_id);
        }
    }

    /// Initialize the map from the first frame.
    ///
    /// For the first keyframe, we create it directly in the map (not via Local Mapping)
    /// because we need it immediately for tracking.
    fn initialize_map(
        &mut self,
        frame: &Frame,
        _stereo_frame: &StereoFrame,
        imu_prior: &SE3,
    ) -> Result<()> {
        self.pose = imu_prior.clone();

        let mut atlas = self.shared.atlas.write();
        let map = atlas.active_map_mut();

        // Create the first keyframe
        let kf_id = map.create_keyframe(
            frame.timestamp_ns,
            self.pose.clone(),
            frame.features.keypoints.clone(),
            frame.features.descriptors.clone(),
            frame.points_cam.clone(),
        );

        // Create MapPoints for each valid 3D point
        let mut num_points = 0;
        for (feat_idx, p_cam_opt) in frame.points_cam.iter().enumerate() {
            if let Some(p_cam) = p_cam_opt {
                let p_world = self.pose.transform_point(p_cam);

                // Get descriptor for this feature
                let descriptor = match frame.features.descriptors.row(feat_idx as i32) {
                    Ok(row) => row.try_clone().unwrap_or_default(),
                    Err(_) => Mat::default(),
                };

                let mp_id = map.create_map_point(p_world, descriptor, kf_id);
                map.associate(kf_id, feat_idx, mp_id);
                num_points += 1;
            }
        }

        self.reference_kf = Some(kf_id);
        self.reference_kf_num_points = num_points;

        // Compute BoW and add to database
        if let Some(kf) = map.get_keyframe_mut(kf_id) {
            let bow = compute_bow_stub(&kf.descriptors);
            kf.set_bow_vector(bow.clone());
            let map_idx = atlas.active_map_index();
            atlas.keyframe_db.add(kf_id, bow, map_idx);
        }

        // Reset keyframe decision counter
        self.kf_decision.reset();

        Ok(())
    }

    /// Track the local map by projecting MapPoints into the current frame and solving PnP.
    ///
    /// Returns (pose, n_inliers, matched_map_points, reproj_errors, inlier_indices, outlier_indices).
    fn track_local_map(
        &self,
        atlas: &crate::atlas::atlas::Atlas,
        frame: &Frame,
        imu_prior: &SE3,
    ) -> Result<(
        SE3,
        usize,
        Vec<Option<MapPointId>>,
        Vec<f64>,
        Vec<usize>,
        Vec<usize>,
    )> {
        let map = atlas.active_map();
        let mut matched_map_points: Vec<Option<MapPointId>> = vec![None; frame.num_features()];

        // Build local keyframe set K1 ∪ K2 using covisibility graph
        let mut k1: HashSet<KeyFrameId> = HashSet::new();
        if let Some(ref_kf) = self.reference_kf {
            k1.insert(ref_kf);
        } else {
            for id in map.keyframe_ids() {
                k1.insert(*id);
            }
        }

        let mut k2: HashSet<KeyFrameId> = HashSet::new();
        for &kf_id in &k1 {
            for nid in map.get_local_keyframes_readonly(kf_id, 10) {
                k2.insert(nid);
            }
        }

        let local_kfs: Vec<KeyFrameId> = k1.union(&k2).copied().collect();
        if local_kfs.is_empty() {
            return Ok((
                imu_prior.clone(),
                0,
                matched_map_points,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ));
        }

        // Collect local MapPoints
        let local_mp_ids = map.get_map_points_from_keyframes(&local_kfs);

        // Build 3D-2D correspondences
        let mut pts3d = Vec::new();
        let mut pts2d = Vec::new();
        let mut mp_indices: Vec<(MapPointId, usize)> = Vec::new(); // (mp_id, feature_idx)

        for mp_id in &local_mp_ids {
            let mp = match map.get_map_point(*mp_id) {
                Some(mp) => mp,
                None => continue,
            };

            // Transform to camera frame
            let pose_cw = self.pose.inverse();
            let p_cam = pose_cw.transform_point(&mp.position);
            if p_cam.z <= 0.0 {
                continue;
            }

            // Project to image plane
            let u = self.camera.fx * p_cam.x / p_cam.z + self.camera.cx;
            let v = self.camera.fy * p_cam.y / p_cam.z + self.camera.cy;

            // Find candidates within search radius
            const SEARCH_RADIUS: f64 = 15.0;
            let mut candidates: Vec<usize> = Vec::new();
            for (kp_idx, kp) in frame.keypoints().iter().enumerate() {
                let du = kp.pt().x as f64 - u;
                let dv = kp.pt().y as f64 - v;
                let spatial_dist_sq = du * du + dv * dv;
                if spatial_dist_sq < SEARCH_RADIUS * SEARCH_RADIUS {
                    candidates.push(kp_idx);
                }
            }

            if candidates.is_empty() {
                continue;
            }

            // Find best descriptor match
            let mut best_dist = u32::MAX;
            let mut second_best_dist = u32::MAX;
            let mut best_idx: Option<usize> = None;

            for &kp_idx in &candidates {
                let frame_desc_row = frame.features.descriptors.row(kp_idx as i32)?;
                let frame_desc = frame_desc_row.try_clone()?;
                let dist = descriptor_distance(&mp.descriptor, &frame_desc)?;

                if dist < best_dist {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = Some(kp_idx);
                } else if dist < second_best_dist {
                    second_best_dist = dist;
                }
            }

            if best_dist > TH_HIGH {
                continue;
            }

            if candidates.len() > 1 {
                if (best_dist as f32) > NN_RATIO * (second_best_dist as f32) {
                    continue;
                }
            }

            if let Some(kp_idx) = best_idx {
                let kp = frame.keypoints().get(kp_idx)?;
                pts3d.push(mp.position);
                pts2d.push(Point2f::new(kp.pt().x, kp.pt().y));
                mp_indices.push((*mp_id, kp_idx));
            }
        }

        let n_corr = pts3d.len();

        if self.frame_count <= 5 || self.frame_count % 100 == 0 {
            debug!(
                "[track_local_map] frame={} local_kfs={} local_mps={} correspondences={}",
                self.frame_count,
                local_kfs.len(),
                local_mp_ids.len(),
                n_corr
            );
        }

        if n_corr < 4 {
            return Ok((
                imu_prior.clone(),
                n_corr,
                matched_map_points,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ));
        }

        let pnp = solve_pnp_ransac_detailed(&pts3d, &pts2d, &self.camera, Some(imu_prior))
            .unwrap_or_else(|_| crate::geometry::PnPResult {
                pose: imu_prior.clone(),
                inlier_mask: Vec::new(),
                reproj_errors: Vec::new(),
            });

        // Record which map points were matched (inliers only) and collect inlier/outlier indices
        let mut inlier_indices = Vec::new();
        let mut outlier_indices = Vec::new();
        let mut reproj_errors_full = vec![f64::INFINITY; mp_indices.len()];

        for (i, &is_inlier) in pnp.inlier_mask.iter().enumerate() {
            if i < mp_indices.len() {
                let (mp_id, feat_idx) = mp_indices[i];
                if i < pnp.reproj_errors.len() {
                    reproj_errors_full[i] = pnp.reproj_errors[i];
                }
                if is_inlier {
                    matched_map_points[feat_idx] = Some(mp_id);
                    inlier_indices.push(i);
                } else {
                    outlier_indices.push(i);
                }
            }
        }

        let n_inliers = pnp.inlier_mask.iter().filter(|&&b| b).count();

        if self.frame_count <= 5 || self.frame_count % 100 == 0 {
            debug!("[track_local_map] pnp_inliers={}", n_inliers);
        }

        Ok((
            pnp.pose,
            n_inliers,
            matched_map_points,
            reproj_errors_full,
            inlier_indices,
            outlier_indices,
        ))
    }

    /// Initial pose estimation via reference keyframe tracking.
    fn track_with_reference_kf(
        &self,
        frame: &Frame,
        atlas: &crate::atlas::atlas::Atlas,
    ) -> Option<SE3> {
        let ref_kf_id = self.reference_kf?;
        let map = atlas.active_map();
        let kf = map.get_keyframe(ref_kf_id)?;

        let matcher = BFMatcher::new(opencv::core::NORM_HAMMING, true).ok()?;
        let mut matches = Vector::<opencv::core::DMatch>::new();
        matcher
            .train_match(
                &kf.descriptors,
                &frame.features.descriptors,
                &mut matches,
                &Mat::default(),
            )
            .ok()?;

        let mut pts3d = Vec::new();
        let mut pts2d = Vec::new();

        let total_matches = matches.len();
        let mut matches_with_mp = 0;

        for m in matches.iter() {
            let feat_idx = m.query_idx as usize;

            // Skip matches where the keyframe feature doesn't have an associated map point
            let mp_id = match kf.get_map_point(feat_idx) {
                Some(id) => id,
                None => continue,
            };

            // Skip invalid map points
            let mp = match map.get_map_point(mp_id) {
                Some(mp) => mp,
                None => continue,
            };

            matches_with_mp += 1;

            if let Ok(kp) = frame.keypoints().get(m.train_idx as usize) {
                pts3d.push(mp.position);
                pts2d.push(Point2f::new(kp.pt().x, kp.pt().y));
            }
        }

        if self.frame_count <= 5 || self.frame_count % 100 == 0 {
            debug!(
                "[track_ref_kf] frame={} total_matches={} matches_with_mp={} pts3d={}",
                self.frame_count,
                total_matches,
                matches_with_mp,
                pts3d.len()
            );
        }

        if pts3d.len() < 4 {
            return None;
        }

        let result = solve_pnp_ransac_detailed(&pts3d, &pts2d, &self.camera, Some(&self.pose));
        if let Ok(ref res) = result {
            let n_inliers = res.inlier_mask.iter().filter(|&&b| b).count();
            if self.frame_count <= 5 || self.frame_count % 100 == 0 {
                debug!("[track_ref_kf] pnp_inliers={}", n_inliers);
            }
        }

        result.ok().map(|res| res.pose)
    }

    /// Motion model tracking: predict pose using constant velocity and match previous frame's map points.
    ///
    /// This is more robust than reference keyframe tracking when the camera is moving
    /// because the previous frame is temporally adjacent.
    fn track_with_motion_model(
        &self,
        frame: &Frame,
        atlas: &crate::atlas::atlas::Atlas,
    ) -> Option<SE3> {
        // Need motion model velocity and previous frame data
        let velocity = self.motion_model_velocity.as_ref()?;
        let last_pose = self.last_frame_pose.as_ref()?;

        // Predict current pose: T_curr = velocity * T_prev
        // velocity is T_curr_prev, so T_curr_w = T_curr_prev * T_prev_w
        let predicted_pose = velocity.compose(last_pose);

        // Get the active map
        let map = atlas.active_map();

        // Project previous frame's map points into current frame
        let mut pts3d = Vec::new();
        let mut pts2d = Vec::new();

        // Search radius for stereo (in pixels)
        let search_radius: f64 = 15.0;

        for (_feat_idx, mp_id_opt) in self.last_frame_map_points.iter().enumerate() {
            let mp_id = match mp_id_opt {
                Some(id) => *id,
                None => continue,
            };

            let mp = match map.get_map_point(mp_id) {
                Some(mp) => mp,
                None => continue,
            };

            // Project map point to current frame using predicted pose
            let pose_cw = predicted_pose.inverse();
            let p_cam = pose_cw.transform_point(&mp.position);

            // Check if point is in front of camera
            if p_cam.z <= 0.0 {
                continue;
            }

            // Project to image plane
            let u = self.camera.fx * p_cam.x / p_cam.z + self.camera.cx;
            let v = self.camera.fy * p_cam.y / p_cam.z + self.camera.cy;

            // Simple bounds check using principal point as approximate center
            // (image is roughly 2*cx by 2*cy)
            let approx_width = 2.0 * self.camera.cx;
            let approx_height = 2.0 * self.camera.cy;
            if u < 0.0 || u >= approx_width || v < 0.0 || v >= approx_height {
                continue;
            }

            // Find candidate features within search radius
            let mut best_dist = u32::MAX;
            let mut best_idx: Option<usize> = None;

            for (kp_idx, kp) in frame.keypoints().iter().enumerate() {
                let du = kp.pt().x as f64 - u;
                let dv = kp.pt().y as f64 - v;
                let dist_sq = du * du + dv * dv;

                if dist_sq > search_radius * search_radius {
                    continue;
                }

                // Compare descriptors
                let frame_desc_row = match frame.features.descriptors.row(kp_idx as i32) {
                    Ok(row) => row,
                    Err(_) => continue,
                };
                let frame_desc = match frame_desc_row.try_clone() {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let desc_dist = match descriptor_distance(&mp.descriptor, &frame_desc) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                if desc_dist < best_dist && desc_dist < TH_HIGH {
                    best_dist = desc_dist;
                    best_idx = Some(kp_idx);
                }
            }

            if let Some(kp_idx) = best_idx {
                if let Ok(kp) = frame.keypoints().get(kp_idx) {
                    pts3d.push(mp.position);
                    pts2d.push(Point2f::new(kp.pt().x, kp.pt().y));
                }
            }
        }

        if self.frame_count <= 5 || self.frame_count % 100 == 0 {
            debug!(
                "[track_motion] frame={} prev_mps={} correspondences={}",
                self.frame_count,
                self.last_frame_map_points
                    .iter()
                    .filter(|x| x.is_some())
                    .count(),
                pts3d.len()
            );
        }

        // Need at least 10 correspondences for motion model (ORB-SLAM3 uses 20)
        if pts3d.len() < 10 {
            return None;
        }

        // Solve PnP with predicted pose as initial guess
        let result = solve_pnp_ransac_detailed(&pts3d, &pts2d, &self.camera, Some(&predicted_pose));

        if let Ok(ref res) = result {
            let n_inliers = res.inlier_mask.iter().filter(|&&b| b).count();
            if self.frame_count <= 5 || self.frame_count % 100 == 0 {
                debug!("[track_motion] pnp_inliers={}", n_inliers);
            }
            // Only accept if we have enough inliers
            if n_inliers < 10 {
                return None;
            }
        }

        result.ok().map(|res| res.pose)
    }

    /// Update motion model state at the end of a successful tracking frame.
    ///
    /// Call this after pose has been estimated and before moving to next frame.
    fn update_motion_model(&mut self, frame: &Frame, matched_map_points: &[Option<MapPointId>]) {
        // Compute velocity: T_curr_prev = T_curr * T_prev^-1
        if let Some(ref last_pose) = self.last_frame_pose {
            // velocity = current_pose * last_pose.inverse()
            let velocity = self.pose.compose(&last_pose.inverse());
            self.motion_model_velocity = Some(velocity);
        }

        // Store current frame data for next iteration
        self.last_frame_pose = Some(self.pose.clone());
        self.last_frame_map_points = matched_map_points.to_vec();
        self.last_frame_points_cam = frame.points_cam.clone();
        self.last_frame_descriptors = frame.features.descriptors.try_clone().unwrap_or_default();
    }

    /// Clear motion model state (e.g., after tracking loss).
    fn clear_motion_model(&mut self) {
        self.last_frame_pose = None;
        self.motion_model_velocity = None;
        self.last_frame_map_points.clear();
        self.last_frame_points_cam.clear();
        self.last_frame_descriptors = Mat::default();
    }
}

/// Simple placeholder Bag-of-Words computation.
fn compute_bow_stub(descriptors: &Mat) -> BowVector {
    let mut bow = BowVector::new();
    let rows = descriptors.rows();
    for i in 0..rows {
        bow.insert(i as u32, 1.0);
    }
    bow
}
