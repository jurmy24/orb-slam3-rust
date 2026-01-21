//! Temporal feature matching between consecutive frames.

use anyhow::Result;
use nalgebra::Vector3;
use opencv::core::{DMatch, Mat, Vector};
use opencv::features2d::BFMatcher;
use opencv::prelude::*;

use crate::tracking::frame::features::FeatureSet;

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
