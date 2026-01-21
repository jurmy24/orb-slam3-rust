//! KeyFrameDatabase - lightweight place recognition database.
//!
//! In the original ORB‑SLAM3 implementation this is backed by a DBoW2
//! visual vocabulary and an inverted index. Here we implement a very small
//! subset of the functionality needed by the Tracker:
//! - add / erase keyframes with a (stubbed) BoW vector
//! - retrieve candidate keyframes given a query BoW vector
//!
//! The actual scoring is intentionally simple and can be upgraded later
//! without touching the rest of the tracking code.

use std::collections::HashMap;

use crate::atlas::map::KeyFrameId;

/// Bag‑of‑Words vector: word_id -> weight.
///
/// For now this is a simple alias that can be produced by a future
/// vocabulary module or a lightweight in‑house implementation.
pub type BowVector = HashMap<u32, f64>;

/// Candidate keyframe with similarity score.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub keyframe_id: KeyFrameId,
    pub map_index: usize,
    pub score: f64,
}

/// Very small KeyFrame database suitable for relocalization experiments.
pub struct KeyFrameDatabase {
    /// For each keyframe, store its BoW vector and owning map index.
    entries: HashMap<KeyFrameId, (BowVector, usize)>,
}

impl KeyFrameDatabase {
    /// Create an empty database.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add or update a keyframe entry.
    pub fn add(&mut self, kf_id: KeyFrameId, bow: BowVector, map_idx: usize) {
        self.entries.insert(kf_id, (bow, map_idx));
    }

    /// Remove a keyframe from the database.
    pub fn erase(&mut self, kf_id: &KeyFrameId) {
        self.entries.remove(kf_id);
    }

    /// Detect candidate keyframes similar to the provided BoW vector.
    ///
    /// The scoring here is a simple dot‑product between BoW vectors.
    /// Results are returned sorted by decreasing score.
    pub fn detect_candidates(
        &self,
        query: &BowVector,
        exclude_map: Option<usize>,
        max_results: usize,
    ) -> Vec<Candidate> {
        let mut cands = Vec::new();

        for (kf_id, (bow, map_idx)) in &self.entries {
            if let Some(excluded) = exclude_map {
                if *map_idx == excluded {
                    continue;
                }
            }

            let mut score = 0.0;
            // Very cheap dot product between sparse histograms
            for (word_id, weight) in query {
                if let Some(other_w) = bow.get(word_id) {
                    score += weight * other_w;
                }
            }

            if score > 0.0 {
                cands.push(Candidate {
                    keyframe_id: *kf_id,
                    map_index: *map_idx,
                    score,
                });
            }
        }

        // Sort by score descending and truncate.
        cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(max_results);
        cands
    }
}

impl Default for KeyFrameDatabase {
    fn default() -> Self {
        Self::new()
    }
}

