use anyhow::Result;
use nalgebra::{Point2, Vector3};
use opencv::core::{DMatch, KeyPoint, Mat, Ptr, Vector};
use opencv::features2d;
use opencv::prelude::*;

use crate::tracking::frame::camera::CameraModel;

/// ORB-SLAM3 matching thresholds
pub const TH_HIGH: u32 = 100; // Max descriptor distance for acceptance
pub const TH_LOW: u32 = 50; // Stricter threshold
pub const NN_RATIO: f32 = 0.75; // Ratio test threshold (best/second_best)

/// A set of ORB features extracted from an image.
#[derive(Clone)]
pub struct FeatureSet {
    pub keypoints: Vector<KeyPoint>,
    pub descriptors: Mat,
}

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
    orb: Ptr<features2d::ORB>,
    camera: CameraModel,
}

impl StereoProcessor {
    pub fn new(camera: CameraModel, n_features: i32) -> Result<Self> {
        let orb = features2d::ORB::create(
            n_features,
            1.2,
            8,
            31,
            0,
            2,
            features2d::ORB_ScoreType::HARRIS_SCORE,
            31,
            20,
        )?;
        Ok(Self { orb, camera })
    }

    pub fn process(&mut self, left: &Mat, right: &Mat, timestamp_ns: u64) -> Result<StereoFrame> {
        let left_features = self.detect_features(left)?;
        let right_features = self.detect_features(right)?;

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

    fn detect_features(&mut self, image: &Mat) -> Result<FeatureSet> {
        let mut keypoints = Vector::<KeyPoint>::new();
        let mut descriptors = Mat::default();
        let mask = Mat::default();
        self.orb
            .detect_and_compute(image, &mask, &mut keypoints, &mut descriptors, false)?;
        Ok(FeatureSet {
            keypoints,
            descriptors,
        })
    }

    fn match_features(&self, left: &FeatureSet, right: &FeatureSet) -> Result<Vector<DMatch>> {
        // ORB-SLAM3 approach: For each left feature, search along epipolar line (horizontal in rectified stereo)
        // constrained by disparity min/max from baseline and depth range

        const MIN_DEPTH: f64 = 0.1; // meters
        const MAX_DEPTH: f64 = 40.0; // meters
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
            let max_u = (ul - min_disparity)
                .min((right.keypoints.len() as f32) * ul / left.keypoints.len() as f32);

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
                if (best_dist as f32) < 0.9 * (second_best_dist as f32)
                    || second_best_dist == TH_HIGH
                {
                    let dmatch = DMatch {
                        query_idx: left_idx as i32,
                        train_idx: right_idx as i32,
                        img_idx: 0,
                        distance: best_dist as f32,
                    };
                    matches.push(dmatch);
                }
            }
        }

        Ok(matches)
    }
}

/// Compute Hamming distance between two ORB descriptors (32-byte binary).
/// Returns the number of differing bits.
pub fn descriptor_distance(desc1: &impl MatTraitConst, desc2: &impl MatTraitConst) -> Result<u32> {
    let mut hamming_dist = 0u32;
    let cols = desc1.cols().min(desc2.cols());
    for j in 0..cols {
        let val1 = *desc1.at_2d::<u8>(0, j)?;
        let val2 = *desc2.at_2d::<u8>(0, j)?;
        hamming_dist += (val1 ^ val2).count_ones();
    }
    Ok(hamming_dist)
}

/// Convert OpenCV keypoints to (x, y) points in pixel coordinates (image frame)
fn keypoints_to_points(keypoints: &Vector<KeyPoint>) -> Vec<Point2<f64>> {
    keypoints
        .iter()
        .map(|kp| Point2::new(kp.pt().x as f64, kp.pt().y as f64))
        .collect()
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
