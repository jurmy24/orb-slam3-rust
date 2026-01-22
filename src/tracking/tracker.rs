//! Main tracker: orchestrates frame processing and pose estimation.
//!
//! This implementation follows the ORB-SLAM3 design more closely than a
//! simple frame-to-frame visual odometry:
//! - maintains an `Atlas` with an active `Map`
//! - tracks 3D `MapPoint`s by projection into the current frame
//! - uses IMU preintegration to obtain a motion prior
//! - estimates pose with PnP given 3D–2D correspondences

use std::collections::HashSet;
use std::time::Instant;

use anyhow::Result;
use nalgebra::Vector3;
use opencv::core::{Mat, Point2f, Vector};
use opencv::features2d::BFMatcher;
use opencv::prelude::*;

use crate::atlas::atlas::Atlas;
use crate::atlas::keyframe_db::BowVector;
use crate::atlas::map::{KeyFrameId, MapPointId};
use crate::geometry::{SE3, solve_pnp_ransac_detailed};
use crate::imu::{ImuBias, ImuNoise, Preintegrator};
use crate::io::euroc::ImuEntry;
use crate::tracking::TrackingState;
use crate::tracking::frame::{CameraModel, StereoFrame};
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
}

impl Tracker {
    pub fn new(camera: CameraModel) -> Result<Self> {
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
        })
    }

    /// Process a stereo frame with accompanying IMU interval and return a
    /// detailed `TrackingResult` suitable for visualization and debugging.
    pub fn process_frame(
        &mut self,
        stereo_frame: StereoFrame,
        imu_measurements: &[ImuEntry],
        atlas: &mut Atlas,
    ) -> Result<TrackingResult> {
        let t_start = Instant::now();
        let prev_pose = self.pose.clone(); // pose before updating this tracker

        self.frame_count += 1;

        // Build tracking frame from stereo output.
        let frame = Frame::from_stereo(stereo_frame);

        // Integrate IMU for motion prior by iterating over IMU pairs
        self.preintegrator.reset();
        for pair in imu_measurements.windows(2) {
            let prev = pair[0].sample; // previous IMU sample
            let curr = pair[1].sample; // current IMU sample
            self.preintegrator.integrate(prev, curr);
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
        // Get the active map
        let active_map = atlas.active_map_mut();

        let mut n_inliers: usize;

        // If the active map is empty, initialize it
        if active_map.num_keyframes() == 0 {
            // Initialize map from first frame.
            self.initialize_map(atlas, &frame, &imu_prior)?;
            // Consider initialization as a good tracking step.
            n_inliers = frame.points_cam.iter().filter(|p| p.is_some()).count();
        } else {
            // Track using projection of existing map points.
            let (estimated_pose, inliers) = self.track_local_map(active_map, &frame, &imu_prior)?;
            self.pose = estimated_pose;
            n_inliers = inliers;

            // If tracking is weak, try reference keyframe based tracking.
            if inliers < 15 {
                if let Some(pose_ref) = self.track_with_reference_kf(&frame, atlas) {
                    self.pose = pose_ref;
                    // Boost inlier count heuristically to mark tracking as OK.
                    n_inliers = 30;
                }
            }
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
            // For now we don't compute reprojection errors here; this will be
            // populated once the detailed PnP path is fully wired.
            reproj_error_median_px: 0.0,
            delta_translation_m: delta_t,
            delta_rotation_deg: delta_rot_deg,
        };

        // Timing: we only measure total_ms for now; the per-stage breakdown is
        // filled in future refinements.
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

    /// Initialize the map from the first frame by creating a KeyFrame and
    /// MapPoints from all valid stereo points.
    fn initialize_map(&mut self, atlas: &mut Atlas, frame: &Frame, imu_prior: &SE3) -> Result<()> {
        let map = atlas.active_map_mut();
        // Use IMU prior as first pose.
        self.pose = imu_prior.clone();

        // Create a keyframe.
        let kf_id = map.create_keyframe(
            frame.timestamp_ns,
            self.pose.clone(),
            frame.features.keypoints.clone(),
            frame.features.descriptors.clone(),
            frame.points_cam.clone(),
        );

        // Create MapPoints for each valid 3D point and associate.
        for (feat_idx, p_cam_opt) in frame.points_cam.iter().enumerate() {
            if let Some(p_cam) = p_cam_opt {
                let p_world = self.pose.transform_point(p_cam);
                // For now store an empty descriptor; descriptor-based matching
                // can be added later without changing the API.
                let descriptor = opencv::core::Mat::default();
                let mp_id = map.create_map_point(p_world, descriptor, kf_id);
                map.associate(kf_id, feat_idx, mp_id);
            }
        }

        self.reference_kf = Some(kf_id);

        // Compute a very simple BoW vector for this keyframe and add it to
        // the shared KeyFrame database.
        if let Some(kf) = map.get_keyframe_mut(kf_id) {
            let bow = compute_bow_stub(&kf.descriptors);
            kf.set_bow_vector(bow.clone());
            let map_idx = atlas.active_map_index();
            atlas.keyframe_db.add(kf_id, bow, map_idx);
        }
        Ok(())
    }

    /// Track the local map by projecting MapPoints into the current frame and
    /// solving PnP.
    fn track_local_map(
        &mut self,
        map: &mut crate::atlas::map::Map,
        frame: &Frame,
        imu_prior: &SE3,
    ) -> Result<(SE3, usize)> {
        // --- Build local keyframe set K1 ∪ K2 using covisibility graph ---
        // K1 is the reference keyframe (typically the last keyframe)
        let mut k1: HashSet<KeyFrameId> = HashSet::new();
        if let Some(ref_kf) = self.reference_kf {
            // Add reference keyframe to K1
            k1.insert(ref_kf);
        } else {
            // If no reference keyframe, use all keyframes from map
            // On first frame this is the only keyframe
            for id in map.keyframe_ids() {
                k1.insert(*id);
            }
        }

        // Set of keyframes that are covisible with K1 (the local keyframes)
        // Two frames are covisible if they share many of the same points
        // We're looking for keyframes that look similar to the reference keyframe
        let mut k2: HashSet<KeyFrameId> = HashSet::new();
        for &kf_id in &k1 {
            for nid in map.get_local_keyframes(kf_id, 10) {
                k2.insert(nid);
            }
        }

        // Union of K1 and K2
        let local_kfs: Vec<KeyFrameId> = k1.union(&k2).copied().collect();
        if local_kfs.is_empty() {
            // No keyframes yet – fall back to prior.
            return Ok((imu_prior.clone(), 0));
        }

        // Collect local MapPoints observed by K1 ∪ K2.
        // This is all 3D map points that were triangulated from observations in any of the local keyframes
        let local_mp_ids = map.get_map_points_from_keyframes(&local_kfs);

        // Build 3D-2D correspondences from projected local map points.
        let mut pts3d = Vec::new();
        let mut pts2d = Vec::new();

        // For each local map point, project it into the current frame and find the closest keypoint
        // Note that a map point is a 3D point that was triangulated from observations in one or more keyframes
        for mp_id in local_mp_ids {
            let mp = if let Some(mp) = map.get_map_point(mp_id) {
                mp
            } else {
                continue;
            };

            // Transform to camera frame (world to camera)
            let pose_cw = self.pose.inverse();
            let p_cam = pose_cw.transform_point(&mp.position);
            // If the point is behind the camera, skip it
            if p_cam.z <= 0.0 {
                continue;
            }

            // Project to image plane using pinhole model
            let u = self.camera.fx * p_cam.x / p_cam.z + self.camera.cx;
            let v = self.camera.fy * p_cam.y / p_cam.z + self.camera.cy;

            // ORB-SLAM3-style descriptor matching: collect candidates within radius
            const SEARCH_RADIUS: f64 = 15.0; // pixels
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

            // Find best and second-best descriptor matches
            let mut best_dist = u32::MAX;
            let mut second_best_dist = u32::MAX;
            let mut best_idx: Option<usize> = None;

            for &kp_idx in &candidates {
                let frame_desc_row = frame.features.descriptors.row(kp_idx as i32)?;
                // Clone the row to convert BoxedRef<Mat> to Mat
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

            // Apply threshold check
            if best_dist > TH_HIGH {
                continue;
            }

            // Apply ratio test (Lowe's ratio) - skip if only one candidate
            if candidates.len() > 1 {
                if (best_dist as f32) > NN_RATIO * (second_best_dist as f32) {
                    continue; // Ambiguous match, reject
                }
            }

            // Accept match
            if let Some(kp_idx) = best_idx {
                let kp = frame.keypoints().get(kp_idx)?;
                pts3d.push(mp.position);
                pts2d.push(Point2f::new(kp.pt().x, kp.pt().y));
            }
        }

        let n_corr = pts3d.len();

        if n_corr < 4 {
            // Not enough correspondences, fall back to IMU prior.
            return Ok((imu_prior.clone(), n_corr));
        }

        let pnp = solve_pnp_ransac_detailed(&pts3d, &pts2d, &self.camera, Some(imu_prior))
            .unwrap_or_else(|_| crate::geometry::PnPResult {
                pose: imu_prior.clone(),
                inlier_mask: Vec::new(),
                reproj_errors: Vec::new(),
            });
        let n_inliers = pnp.inlier_mask.iter().filter(|&&b| b).count();
        Ok((pnp.pose, n_inliers))
    }

    /// Placeholder for reference keyframe tracking, to be filled in a later
    /// phase. This uses descriptor matching between the reference keyframe
    /// and the current frame to obtain an initial pose estimate.
    fn track_with_reference_kf(&self, frame: &Frame, atlas: &Atlas) -> Option<SE3> {
        let ref_kf_id = self.reference_kf?;
        let map = atlas.active_map();
        let kf = map.get_keyframe(ref_kf_id)?;

        // Match descriptors between reference keyframe and current frame.
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
            // Get associated MapPoint for this keyframe feature.
            let mp_id: MapPointId = if let Some(id) = kf.get_map_point(feat_idx) {
                id
            } else {
                continue;
            };
            let mp = if let Some(mp) = map.get_map_point(mp_id) {
                mp
            } else {
                continue;
            };

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

/// Very simple, placeholder Bag-of-Words computation.
///
/// This just assigns a unit weight to each descriptor row index, which is
/// enough to exercise the KeyFrameDatabase and relocalization plumbing.
fn compute_bow_stub(descriptors: &Mat) -> BowVector {
    let mut bow = BowVector::new();
    let rows = descriptors.rows();
    for i in 0..rows {
        bow.insert(i as u32, 1.0);
    }
    bow
}
