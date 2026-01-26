//! Loop detection using Bag-of-Words with temporal consistency checking.
//!
//! This module implements the first stage of loop closing: detecting potential
//! loop candidates based on visual similarity (BoW) and temporal consistency.

use std::collections::{HashMap, HashSet, VecDeque};

use parking_lot::RwLock;

use crate::atlas::atlas::Atlas;
use crate::atlas::keyframe_db::BowVector;
use crate::atlas::map::{KeyFrameId, Map};
use crate::vocabulary::OrbVocabulary;

/// Configuration for loop detection.
#[derive(Debug, Clone)]
pub struct LoopDetectorConfig {
    /// Minimum BoW score ratio compared to best covisible keyframe.
    /// Candidates must score at least this fraction of the best covisible score.
    pub min_score_ratio: f64,

    /// Number of consecutive frames required for temporal consistency.
    pub consistency_threshold: usize,

    /// Minimum number of covisible keyframes to consider for threshold computation.
    pub min_covisibles_for_threshold: usize,

    /// Maximum number of keyframes to check in the covisibility group.
    pub max_covisibles_to_check: usize,

    /// Minimum time gap (in number of keyframes) between current and loop candidate.
    /// Prevents detecting recent keyframes as loops.
    pub min_temporal_gap: usize,
}

impl Default for LoopDetectorConfig {
    fn default() -> Self {
        Self {
            min_score_ratio: 0.75,
            consistency_threshold: 3,
            min_covisibles_for_threshold: 5,
            max_covisibles_to_check: 10,
            min_temporal_gap: 30,
        }
    }
}

/// A potential loop closure candidate.
#[derive(Debug, Clone)]
pub struct LoopCandidate {
    /// Current keyframe ID that detected the loop.
    pub current_kf_id: KeyFrameId,

    /// Loop keyframe ID (the older keyframe we're closing back to).
    pub loop_kf_id: KeyFrameId,

    /// BoW similarity score between current and loop keyframe.
    pub bow_score: f64,

    /// Covisible keyframes of the loop keyframe (for extended matching).
    pub loop_covisibles: Vec<KeyFrameId>,
}

/// Tracks temporal consistency of loop detections.
///
/// A loop is only considered valid if we detect similar candidates
/// for `consistency_threshold` consecutive keyframes.
pub struct ConsistencyChecker {
    /// Configuration.
    config: LoopDetectorConfig,

    /// History of candidate groups for recent keyframes.
    /// Each entry is (keyframe_id, set of candidate keyframe IDs).
    history: VecDeque<(KeyFrameId, HashSet<KeyFrameId>)>,

    /// Current consistent candidates with their consistency count.
    /// Maps candidate region (represented by a keyframe) to count.
    consistent_counts: HashMap<KeyFrameId, usize>,
}

impl ConsistencyChecker {
    /// Create a new consistency checker.
    pub fn new(config: LoopDetectorConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            consistent_counts: HashMap::new(),
        }
    }

    /// Add candidates for a new keyframe and check for consistency.
    ///
    /// Returns the first candidate that has been consistently detected.
    pub fn add_and_check(&mut self, kf_id: KeyFrameId, candidates: &[LoopCandidate]) -> Option<LoopCandidate> {
        // Build set of candidate keyframe IDs (including their covisibles)
        let mut candidate_set: HashSet<KeyFrameId> = HashSet::new();
        for c in candidates {
            candidate_set.insert(c.loop_kf_id);
            for &cov_id in &c.loop_covisibles {
                candidate_set.insert(cov_id);
            }
        }

        // Update consistency counts
        let mut new_counts: HashMap<KeyFrameId, usize> = HashMap::new();

        for &cand_id in &candidate_set {
            // Check if this candidate (or its region) was seen in recent history
            let prev_count = self.get_region_count(&cand_id);
            new_counts.insert(cand_id, prev_count + 1);
        }

        // Check if any candidate has reached the consistency threshold
        let mut best_consistent: Option<&LoopCandidate> = None;

        for candidate in candidates {
            if let Some(&count) = new_counts.get(&candidate.loop_kf_id) {
                if count >= self.config.consistency_threshold {
                    if best_consistent.is_none()
                        || candidate.bow_score > best_consistent.unwrap().bow_score
                    {
                        best_consistent = Some(candidate);
                    }
                }
            }
        }

        // Update history
        self.history.push_back((kf_id, candidate_set));
        if self.history.len() > self.config.consistency_threshold + 2 {
            self.history.pop_front();
        }

        // Update counts
        self.consistent_counts = new_counts;

        // If we found a consistent candidate, clear the history to avoid
        // repeated detections of the same loop
        if let Some(candidate) = best_consistent {
            let result = candidate.clone();
            self.clear();
            return Some(result);
        }

        None
    }

    /// Get the consistency count for a candidate region.
    fn get_region_count(&self, candidate_id: &KeyFrameId) -> usize {
        // Check if this candidate or any of its neighbors appeared in recent frames
        let mut count = 0;

        for (_kf_id, candidate_set) in &self.history {
            if candidate_set.contains(candidate_id) {
                count += 1;
            }
        }

        count
    }

    /// Clear the consistency history.
    pub fn clear(&mut self) {
        self.history.clear();
        self.consistent_counts.clear();
    }
}

/// Detect loop closure candidates for a given keyframe.
///
/// This function:
/// 1. Computes a minimum BoW score threshold from covisible keyframes
/// 2. Queries the KeyFrameDatabase for candidates above threshold
/// 3. Filters out recently connected keyframes
/// 4. Returns candidates sorted by score
///
/// # Arguments
/// * `kf_id` - The current keyframe to find loops for
/// * `atlas` - The atlas containing all maps and the keyframe database
/// * `vocabulary` - The ORB vocabulary for BoW scoring
/// * `config` - Detection configuration
///
/// # Returns
/// Vector of loop candidates (may be empty if no good candidates found)
pub fn detect_loop_candidates(
    kf_id: KeyFrameId,
    atlas: &RwLock<Atlas>,
    vocabulary: Option<&OrbVocabulary>,
    config: &LoopDetectorConfig,
) -> Vec<LoopCandidate> {
    let atlas_guard = atlas.read();
    let map = atlas_guard.active_map();

    // Get the current keyframe
    let current_kf = match map.get_keyframe(kf_id) {
        Some(kf) => kf,
        None => return vec![],
    };

    // Must have BoW vector computed
    let current_bow = match current_kf.bow_vector() {
        Some(bow) => bow,
        None => return vec![],
    };

    // Get connected keyframes (covisibles + spanning tree neighbors)
    let connected_kfs = get_connected_keyframes(kf_id, map);

    // Compute minimum score threshold from covisibles
    let min_score = compute_min_score(current_bow, &connected_kfs, map, vocabulary, config);

    if min_score < 0.01 {
        // No meaningful threshold could be computed
        return vec![];
    }

    // Query all keyframes for candidates
    let candidates = find_candidates_above_threshold(
        kf_id,
        current_bow,
        min_score,
        &connected_kfs,
        map,
        vocabulary,
        config,
    );

    candidates
}

/// Get all keyframes connected to the given keyframe (covisibles + tree neighbors).
fn get_connected_keyframes(kf_id: KeyFrameId, map: &Map) -> HashSet<KeyFrameId> {
    let mut connected = HashSet::new();
    connected.insert(kf_id);

    if let Some(kf) = map.get_keyframe(kf_id) {
        // Add covisibles
        for cov_id in kf.get_covisibles() {
            connected.insert(*cov_id);
        }

        // Add parent
        if let Some(parent_id) = kf.parent_id {
            connected.insert(parent_id);
        }

        // Add children
        for &child_id in &kf.children_ids {
            connected.insert(child_id);
        }

        // Add temporal neighbors
        if let Some(prev_id) = kf.prev_kf {
            connected.insert(prev_id);
        }
        if let Some(next_id) = kf.next_kf {
            connected.insert(next_id);
        }
    }

    connected
}

/// Compute minimum BoW score threshold based on covisible keyframes.
fn compute_min_score(
    current_bow: &BowVector,
    connected_kfs: &HashSet<KeyFrameId>,
    map: &Map,
    vocabulary: Option<&OrbVocabulary>,
    config: &LoopDetectorConfig,
) -> f64 {
    let mut best_covisible_score = 0.0;
    let mut num_checked = 0;

    // Score current against covisible keyframes
    for &cov_id in connected_kfs {
        if num_checked >= config.max_covisibles_to_check {
            break;
        }

        if let Some(cov_kf) = map.get_keyframe(cov_id) {
            if let Some(cov_bow) = cov_kf.bow_vector() {
                let score = compute_bow_score(current_bow, cov_bow, vocabulary);
                if score > best_covisible_score {
                    best_covisible_score = score;
                }
                num_checked += 1;
            }
        }
    }

    if num_checked < config.min_covisibles_for_threshold {
        return 0.0;
    }

    // Threshold is a fraction of the best covisible score
    best_covisible_score * config.min_score_ratio
}

/// Find all candidate keyframes with BoW score above threshold.
fn find_candidates_above_threshold(
    current_kf_id: KeyFrameId,
    current_bow: &BowVector,
    min_score: f64,
    connected_kfs: &HashSet<KeyFrameId>,
    map: &Map,
    vocabulary: Option<&OrbVocabulary>,
    config: &LoopDetectorConfig,
) -> Vec<LoopCandidate> {
    let mut candidates = Vec::new();

    // Get the numeric ID for temporal gap checking
    let current_id_num = current_kf_id.0;

    for other_kf in map.keyframes() {
        let other_kf_id = &other_kf.id;
        // Skip connected keyframes
        if connected_kfs.contains(other_kf_id) {
            continue;
        }

        // Skip keyframes that are too recent (temporal gap check)
        let gap = if current_id_num > other_kf_id.0 {
            current_id_num - other_kf_id.0
        } else {
            other_kf_id.0 - current_id_num
        };

        if gap < config.min_temporal_gap as u64 {
            continue;
        }

        // Skip bad keyframes
        if other_kf.is_bad {
            continue;
        }

        // Must have BoW vector
        let other_bow = match other_kf.bow_vector() {
            Some(bow) => bow,
            None => continue,
        };

        // Compute similarity score
        let score = compute_bow_score(current_bow, other_bow, vocabulary);

        if score >= min_score {
            // Get covisibles of the candidate for extended verification
            let loop_covisibles: Vec<_> = other_kf.get_covisibles().copied().collect();

            candidates.push(LoopCandidate {
                current_kf_id,
                loop_kf_id: *other_kf_id,
                bow_score: score,
                loop_covisibles,
            });
        }
    }

    // Sort by score descending
    candidates.sort_by(|a, b| {
        b.bow_score
            .partial_cmp(&a.bow_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates
}

/// Compute BoW similarity score between two vectors.
fn compute_bow_score(
    bow1: &BowVector,
    bow2: &BowVector,
    vocabulary: Option<&OrbVocabulary>,
) -> f64 {
    if let Some(_vocab) = vocabulary {
        OrbVocabulary::score(bow1, bow2)
    } else {
        // Fallback: simple dot product
        let mut score = 0.0;
        for (word_id, w1) in bow1 {
            if let Some(w2) = bow2.get(word_id) {
                score += w1 * w2;
            }
        }
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_checker() {
        let config = LoopDetectorConfig {
            consistency_threshold: 3,
            ..Default::default()
        };
        let mut checker = ConsistencyChecker::new(config);

        // Create candidates
        let candidate = LoopCandidate {
            current_kf_id: KeyFrameId::new(10),
            loop_kf_id: KeyFrameId::new(1),
            bow_score: 0.8,
            loop_covisibles: vec![KeyFrameId::new(2), KeyFrameId::new(3)],
        };

        // First detection - not consistent yet
        let result = checker.add_and_check(KeyFrameId::new(10), &[candidate.clone()]);
        assert!(result.is_none());

        // Second detection - still not consistent
        let candidate2 = LoopCandidate {
            current_kf_id: KeyFrameId::new(11),
            loop_kf_id: KeyFrameId::new(1),
            bow_score: 0.85,
            loop_covisibles: vec![KeyFrameId::new(2)],
        };
        let result = checker.add_and_check(KeyFrameId::new(11), &[candidate2]);
        assert!(result.is_none());

        // Third detection - should be consistent now
        let candidate3 = LoopCandidate {
            current_kf_id: KeyFrameId::new(12),
            loop_kf_id: KeyFrameId::new(1),
            bow_score: 0.9,
            loop_covisibles: vec![],
        };
        let result = checker.add_and_check(KeyFrameId::new(12), &[candidate3]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().loop_kf_id, KeyFrameId::new(1));
    }

    #[test]
    fn test_consistency_checker_no_match() {
        let config = LoopDetectorConfig {
            consistency_threshold: 3,
            ..Default::default()
        };
        let mut checker = ConsistencyChecker::new(config);

        // Different candidates each time
        for i in 10..15 {
            let candidate = LoopCandidate {
                current_kf_id: KeyFrameId::new(i),
                loop_kf_id: KeyFrameId::new(i - 9), // Different each time
                bow_score: 0.8,
                loop_covisibles: vec![],
            };
            let result = checker.add_and_check(KeyFrameId::new(i), &[candidate]);
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_bow_score_identical() {
        let mut bow = BowVector::new();
        bow.insert(1, 0.5);
        bow.insert(2, 0.3);
        bow.insert(3, 0.2);

        let score = compute_bow_score(&bow, &bow, None);
        assert!(score > 0.3); // Should be sum of squares
    }

    #[test]
    fn test_bow_score_different() {
        let mut bow1 = BowVector::new();
        bow1.insert(1, 0.5);
        bow1.insert(2, 0.5);

        let mut bow2 = BowVector::new();
        bow2.insert(3, 0.5);
        bow2.insert(4, 0.5);

        let score = compute_bow_score(&bow1, &bow2, None);
        assert_eq!(score, 0.0); // No common words
    }
}
