//! Temporal feature matching between consecutive frames.

use anyhow::Result;
use nalgebra::Vector3;
use opencv::core::{DMatch, Mat, Vector};
use opencv::features2d::BFMatcher;
use opencv::prelude::*;

use crate::tracking::frame::features::FeatureSet;

/// ORB-SLAM3 matching thresholds
pub const TH_HIGH: u32 = 100;  // Max descriptor distance for acceptance
pub const TH_LOW: u32 = 50;    // Stricter threshold
pub const NN_RATIO: f32 = 0.75; // Ratio test threshold (best/second_best)

/// Compute Hamming distance between two ORB descriptors (32-byte binary).
/// Returns the number of differing bits.
///
/// # Arguments
/// * `desc1` - First ORB descriptor (1 row x 32 cols)
/// * `desc2` - Second ORB descriptor (1 row x 32 cols)
///
/// # Returns
/// Hamming distance as u32 (number of differing bits)
pub fn descriptor_distance(desc1: &Mat, desc2: &Mat) -> Result<u32> {
    let mut hamming_dist = 0u32;
    // ORB descriptors are 1 row x 32 cols (256 bits)
    let cols = desc1.cols().min(desc2.cols());
    for j in 0..cols {
        let val1 = *desc1.at_2d::<u8>(0, j)?;
        let val2 = *desc2.at_2d::<u8>(0, j)?;
        hamming_dist += (val1 ^ val2).count_ones();
    }
    Ok(hamming_dist)
}

pub struct TemporalMatcher {
    matcher: BFMatcher,
}

#[derive(Clone)]
pub struct TemporalMatchResult {
    pub matches: Vector<DMatch>,
}

impl TemporalMatcher {
    pub fn new() -> Result<Self> {
        let matcher = BFMatcher::new(opencv::core::NORM_HAMMING, true)?;
        Ok(Self { matcher })
    }

    pub fn match_features(
        &self,
        prev: &FeatureSet,
        curr: &FeatureSet,
    ) -> Result<TemporalMatchResult> {
        let mut matches = Vector::<DMatch>::new();
        self.matcher.train_match(
            &prev.descriptors,
            &curr.descriptors,
            &mut matches,
            &Mat::default(),
        )?;
        Ok(TemporalMatchResult { matches })
    }
}

/// Convert stereo 3D point vector into nalgebra Vec.
pub fn points3d_from_vectors(points: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
    points.to_vec()
}
