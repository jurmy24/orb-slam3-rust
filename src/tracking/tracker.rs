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

        // Check if the active map needs initialization
        let needs_init = {
            let atlas = self.shared.atlas.read();
            atlas.active_map().num_keyframes() == 0
        };

        if needs_init {
            // Initialize map from first frame - this creates the first keyframe directly
            self.initialize_map(&frame, &stereo_frame, &imu_prior)?;
            n_inliers = frame.points_cam.iter().filter(|p| p.is_some()).count();
            matched_map_points = vec![None; frame.num_features()]; // Not used in init path
        } else {
            // Stage 1: Initial pose estimation via reference keyframe tracking
            let initial_pose = {
                let atlas = self.shared.atlas.read();
                if let Some(pose_ref) = self.track_with_reference_kf(&frame, &atlas) {
                    pose_ref
                } else {
                    imu_prior.clone()
                }
            };
            self.pose = initial_pose.clone();

            // Stage 2: Refine pose by tracking the local map
            let (estimated_pose, inliers, matches) = {
                let atlas = self.shared.atlas.read();
                self.track_local_map(&atlas, &frame, &initial_pose)?
            };
            self.pose = estimated_pose;
            n_inliers = inliers;
            matched_map_points = matches;

            // Decide whether to create a new keyframe
            self.maybe_create_keyframe(&frame, &stereo_frame, n_inliers, &matched_map_points);
        }

        // Update tracking state from inlier count.
        self.update_state(n_inliers);

        // Update state and pose history.
        self.velocity = pred_vel;
        self.trajectory.push(self.pose.clone());

        // Build metrics.
        let n_features = frame.features.keypoints.len();
        let n_map_point_matches = n_inliers;
        let inlier_ratio = if n_map_point_matches > 0 {
            n_inliers as f64 / n_map_point_matches as f64
        } else {
            0.0
        };

        let delta_t = (self.pose.translation - prev_pose.translation).norm();
        let dq = prev_pose.rotation.inverse() * self.pose.rotation;
        let delta_rot_rad = dq.angle();
        let delta_rot_deg = delta_rot_rad.to_degrees();

        let metrics = TrackingMetrics {
            n_features,
            n_map_point_matches,
            n_inliers,
            inlier_ratio,
            reproj_error_median_px: 0.0,
            delta_translation_m: delta_t,
            delta_rotation_deg: delta_rot_deg,
        };

        let mut timing = TimingStats::zero();
        timing.total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        let matches = MatchInfo {
            matched_map_points: Vec::new(),
            inlier_indices: Vec::new(),
            outlier_indices: Vec::new(),
            reproj_errors: Vec::new(),
            local_map_point_ids: Vec::new(),
        };

        Ok(TrackingResult {
            state: self.state,
            pose: self.pose.clone(),
            velocity: self.velocity,
            reference_kf_id: self.reference_kf,
            metrics,
            timing,
            matches,
        })
    }

    /// Decide whether to create a new keyframe, and if so, send it to Local Mapping.
    fn maybe_create_keyframe(
        &mut self,
        frame: &Frame,
        stereo_frame: &StereoFrame,
        n_inliers: usize,
        matched_map_points: &[Option<MapPointId>],
    ) {
        // Don't create keyframes if flow control says to stop
        if self.shared.should_stop_keyframe_creation() {
            return;
        }

        // Don't create keyframes if tracking is not OK
        if self.state != TrackingState::Ok && self.state != TrackingState::NotInitialized {
            return;
        }

        // Use KeyFrameDecision to check if we should create a keyframe
        let should_create = self.kf_decision.should_create_keyframe(
            stereo_frame,
            n_inliers,
            self.reference_kf_num_points,
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

    /// Update the high-level tracking state based on the number of inliers.
    fn update_state(&mut self, n_inliers: usize) {
        const MIN_INLIERS_OK: usize = 30;
        const MIN_INLIERS_RECENTLY_LOST: usize = 15;
        const MAX_LOST_FRAMES: usize = 5;

        let tracking_good = n_inliers >= MIN_INLIERS_OK;
        let tracking_weak = n_inliers >= MIN_INLIERS_RECENTLY_LOST;

        self.state = match self.state {
            TrackingState::NotInitialized => {
                if tracking_good {
                    TrackingState::Ok
                } else {
                    TrackingState::NotInitialized
                }
            }
            TrackingState::Ok => {
                if tracking_good || tracking_weak {
                    self.lost_frames = 0;
                    TrackingState::Ok
                } else {
                    self.lost_frames = 1;
                    TrackingState::RecentlyLost
                }
            }
            TrackingState::RecentlyLost => {
                if tracking_good || tracking_weak {
                    self.lost_frames = 0;
                    TrackingState::Ok
                } else {
                    self.lost_frames += 1;
                    if self.lost_frames > MAX_LOST_FRAMES {
                        TrackingState::Lost
                    } else {
                        TrackingState::RecentlyLost
                    }
                }
            }
            TrackingState::Lost => {
                if tracking_good {
                    self.lost_frames = 0;
                    TrackingState::Ok
                } else {
                    TrackingState::Lost
                }
            }
        };
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
    /// Returns (pose, n_inliers, matched_map_points).
    fn track_local_map(
        &self,
        atlas: &crate::atlas::atlas::Atlas,
        frame: &Frame,
        imu_prior: &SE3,
    ) -> Result<(SE3, usize, Vec<Option<MapPointId>>)> {
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
            return Ok((imu_prior.clone(), 0, matched_map_points));
        }

        // Collect local MapPoints
        let local_mp_ids = map.get_map_points_from_keyframes(&local_kfs);

        // Build 3D-2D correspondences
        let mut pts3d = Vec::new();
        let mut pts2d = Vec::new();
        let mut mp_indices: Vec<(MapPointId, usize)> = Vec::new(); // (mp_id, feature_idx)

        for mp_id in local_mp_ids {
            let mp = match map.get_map_point(mp_id) {
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
                mp_indices.push((mp_id, kp_idx));
            }
        }

        let n_corr = pts3d.len();

        if n_corr < 4 {
            return Ok((imu_prior.clone(), n_corr, matched_map_points));
        }

        let pnp = solve_pnp_ransac_detailed(&pts3d, &pts2d, &self.camera, Some(imu_prior))
            .unwrap_or_else(|_| crate::geometry::PnPResult {
                pose: imu_prior.clone(),
                inlier_mask: Vec::new(),
                reproj_errors: Vec::new(),
            });

        // Record which map points were matched (inliers only)
        for (i, &is_inlier) in pnp.inlier_mask.iter().enumerate() {
            if is_inlier && i < mp_indices.len() {
                let (mp_id, feat_idx) = mp_indices[i];
                matched_map_points[feat_idx] = Some(mp_id);
            }
        }

        let n_inliers = pnp.inlier_mask.iter().filter(|&&b| b).count();
        Ok((pnp.pose, n_inliers, matched_map_points))
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

        for m in matches.iter() {
            let feat_idx = m.query_idx as usize;
            let mp_id = kf.get_map_point(feat_idx)?;
            let mp = map.get_map_point(mp_id)?;

            if let Ok(kp) = frame.keypoints().get(m.train_idx as usize) {
                pts3d.push(mp.position);
                pts2d.push(Point2f::new(kp.pt().x, kp.pt().y));
            }
        }

        if pts3d.len() < 4 {
            return None;
        }

        solve_pnp_ransac_detailed(&pts3d, &pts2d, &self.camera, Some(&self.pose))
            .ok()
            .map(|res| res.pose)
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
