use anyhow::Result;
use nalgebra::Vector3;
use opencv::core::{DMatch, Mat, Vector};
use opencv::features2d::BFMatcher;
use opencv::prelude::*;
use opencv::boxed_ref::BoxedRef;

use crate::tracking::frame::camera::CameraModel;
use crate::tracking::frame::features::{FeatureDetector, FeatureSet, keypoints_to_points};

#[derive(Clone)]
pub struct StereoFrame {
    pub left_features: FeatureSet,
    pub right_features: FeatureSet,
    pub matches_lr: Vector<DMatch>,
    /// 3D points per left keypoint index (None if no valid depth).
    pub points_cam: Vec<Option<Vector3<f64>>>,
    pub timestamp_ns: u64,
}

pub struct StereoProcessor {
    detector: FeatureDetector,
    matcher: BFMatcher,
    camera: CameraModel,
}

impl StereoProcessor {
    pub fn new(camera: CameraModel, n_features: i32) -> Result<Self> {
        let detector = FeatureDetector::new(n_features)?;
        // TODO: This might not be what ORB-SLAM3 uses/recommends, plus crossCheck=true might not be what we want
        // Perhaps replace BFMather with candidate selection and BFMatcher... (gives ORB-SLAM3 efficiency)
        let matcher = BFMatcher::new(opencv::core::NORM_HAMMING, false)?;
        Ok(Self {
            detector,
            matcher,
            camera,
        })
    }

    pub fn process(&mut self, left: &Mat, right: &Mat, timestamp_ns: u64) -> Result<StereoFrame> {
        let left_features = self.detector.detect(left)?;
        let right_features = self.detector.detect(right)?;

        let matches_lr = self.match_features(&left_features, &right_features)?;
        let points_cam = triangulate(&left_features, &right_features, &matches_lr, self.camera);

        Ok(StereoFrame {
            left_features,
            right_features,
            matches_lr,
            points_cam,
            timestamp_ns,
        })
    }

    fn match_features(&self, left: &FeatureSet, right: &FeatureSet) -> Result<Vector<DMatch>> {
        // ORB-SLAM3 approach: For each left feature, search along epipolar line (horizontal in rectified stereo)
        // constrained by disparity min/max from baseline and depth range

        const MIN_DEPTH: f64 = 0.1; // meters
        const MAX_DEPTH: f64 = 40.0; // meters
        const TH_HIGH: f32 = 100.0; // ORB descriptor distance threshold
        const VERTICAL_MARGIN: f32 = 2.0; // pixels tolerance for y-coordinate

        // Calculate disparity bounds from depth range
        let max_disparity = (self.camera.fx * self.camera.baseline / MIN_DEPTH) as f32;
        let min_disparity = (self.camera.fx * self.camera.baseline / MAX_DEPTH) as f32;

        let mut matches = Vector::<DMatch>::new();

        // For each left keypoint, search for best match in right image
        for (left_idx, left_kp) in left.keypoints.iter().enumerate() {
            let ul = left_kp.pt().x;
            let vl = left_kp.pt().y;

            // Define horizontal search range based on disparity constraints
            let min_u = (ul - max_disparity).max(0.0);
            let max_u = (ul - min_disparity).min((right.keypoints.len() as f32) * ul / left.keypoints.len() as f32);

            let mut best_dist = TH_HIGH;
            let mut best_right_idx: Option<usize> = None;
            let mut second_best_dist = TH_HIGH;

            // Get left descriptor
            let left_desc = left.descriptors.row(left_idx as i32)?;

            // Search for matches in right image along epipolar line
            for (right_idx, right_kp) in right.keypoints.iter().enumerate() {
                let ur = right_kp.pt().x;
                let vr = right_kp.pt().y;

                // Check epipolar constraint (y coordinates should match in rectified stereo)
                if (vl - vr).abs() > VERTICAL_MARGIN {
                    continue;
                }

                // Check horizontal disparity range
                if ur < min_u || ur > max_u {
                    continue;
                }

                // Ensure positive disparity (left feature should be to the right of right feature)
                if ul <= ur {
                    continue;
                }

                // Compute ORB descriptor distance
                let right_desc = right.descriptors.row(right_idx as i32)?;
                let dist = descriptor_distance(&left_desc, &right_desc)?;

                if dist < best_dist {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_right_idx = Some(right_idx);
                } else if dist < second_best_dist {
                    second_best_dist = dist;
                }
            }

            // Apply ratio test (Lowe's ratio)
            if let Some(right_idx) = best_right_idx {
                if best_dist < 0.9 * second_best_dist || second_best_dist == TH_HIGH {
                    let dmatch = DMatch {
                        query_idx: left_idx as i32,
                        train_idx: right_idx as i32,
                        img_idx: 0,
                        distance: best_dist,
                    };
                    matches.push(dmatch);
                }
            }
        }

        Ok(matches)
    }
}

/// Compute Hamming distance between two ORB descriptors (32-byte binary)
fn descriptor_distance(desc1: &BoxedRef<Mat>, desc2: &BoxedRef<Mat>) -> Result<f32> {
    // Compute Hamming distance manually by XOR-ing bytes and counting bits
    let mut hamming_dist = 0u32;

    // ORB descriptors are typically 1 row x 32 cols (256 bits)
    for j in 0..desc1.cols() {
        let val1 = *desc1.at_2d::<u8>(0, j)?;
        let val2 = *desc2.at_2d::<u8>(0, j)?;
        hamming_dist += (val1 ^ val2).count_ones();
    }

    Ok(hamming_dist as f32)
}

/// Triangulate 3D points from stereo matches using pinhole model.
fn triangulate(
    left: &FeatureSet,
    right: &FeatureSet,
    matches: &Vector<DMatch>,
    cam: CameraModel,
) -> Vec<Option<Vector3<f64>>> {
    // Convert keypoints to pixel coordinates in image frame
    let left_pts = keypoints_to_points(&left.keypoints);
    let right_pts = keypoints_to_points(&right.keypoints);

    // Initialize 3D points as None for each left keypoint
    let mut points = vec![None; left_pts.len()];

    for m in matches {
        if let (Some(l), Some(r)) = (
            left_pts.get(m.query_idx as usize),
            right_pts.get(m.train_idx as usize),
        ) {
            let disparity = l.x - r.x;
            if disparity.abs() < 0.5 {
                continue;
            }
            let z = cam.fx * cam.baseline / disparity;
            let x = (l.x - cam.cx) * z / cam.fx;
            let y = (l.y - cam.cy) * z / cam.fy;
            points[m.query_idx as usize] = Some(Vector3::new(x, y, z));
        }
    }

    points
}
