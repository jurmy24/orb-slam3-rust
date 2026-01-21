use anyhow::Result;
use nalgebra::Point2;
use opencv::core::{KeyPoint, Ptr, Vector};
use opencv::features2d;
use opencv::prelude::*;

#[derive(Clone)]
pub struct FeatureSet {
    pub keypoints: Vector<KeyPoint>,
    pub descriptors: Mat,
}

pub struct FeatureDetector {
    orb: Ptr<features2d::ORB>,
}

impl FeatureDetector {
    // ! Parameters are set here
    pub fn new(n_features: i32) -> Result<Self> {
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
        Ok(Self { orb })
    }

    pub fn detect(&mut self, image: &Mat) -> Result<FeatureSet> {
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
}

/// Convert OpenCV keypoints to (x, y) points in pixel coordinates (image frame)
pub fn keypoints_to_points(keypoints: &Vector<KeyPoint>) -> Vec<Point2<f64>> {
    keypoints
        .iter()
        .map(|kp| Point2::new(kp.pt().x as f64, kp.pt().y as f64))
        .collect()
}
