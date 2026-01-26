//! ORB Vocabulary for Bag-of-Words place recognition.
//!
//! This module implements loading and querying of a DBoW2-format vocabulary tree
//! for accelerating feature matching and enabling place recognition.
//!
//! # Vocabulary Structure
//!
//! The vocabulary is a hierarchical k-means tree trained on ORB descriptors:
//! - Branching factor k (typically 10)
//! - Depth L levels (typically 5-6)
//! - ~100,000 leaf nodes ("visual words")
//!
//! # Key Types
//!
//! - [`BowVector`]: Histogram of word occurrences with TF-IDF weights (for place recognition)
//! - [`FeatureVector`]: Groups feature indices by vocabulary node (for accelerated matching)
//! - [`OrbVocabulary`]: The vocabulary tree structure with quantization methods

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use opencv::core::Mat;
use opencv::prelude::*;

/// Bag-of-Words vector: word_id -> TF-IDF weight.
///
/// Used for place recognition and loop closure detection.
/// The weights are L1-normalized after accumulation.
pub type BowVector = HashMap<u32, f64>;

/// Feature vector: node_id -> list of feature indices.
///
/// Groups features by which vocabulary node they fall into at a specific level.
/// Used to accelerate feature matching by only comparing features in the same group.
pub type FeatureVector = HashMap<u32, Vec<usize>>;

/// A node in the vocabulary tree.
#[derive(Debug, Clone)]
pub struct VocabNode {
    /// Node ID (0 is root, children numbered sequentially)
    pub id: u32,
    /// Parent node ID (0 for root's direct children, u32::MAX for root)
    pub parent: u32,
    /// Child node IDs (empty for leaf nodes)
    pub children: Vec<u32>,
    /// ORB binary descriptor (32 bytes) for this node
    pub descriptor: [u8; 32],
    /// IDF weight (non-zero for leaf nodes)
    pub weight: f64,
    /// Word ID if this is a leaf node
    pub word_id: Option<u32>,
}

impl VocabNode {
    /// Create a new vocabulary node.
    fn new(id: u32, parent: u32) -> Self {
        Self {
            id,
            parent,
            children: Vec::new(),
            descriptor: [0u8; 32],
            weight: 0.0,
            word_id: None,
        }
    }

    /// Check if this node is a leaf (visual word).
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// ORB Vocabulary tree (DBoW2 format).
///
/// Provides methods for:
/// - Loading vocabulary from DBoW2 text format
/// - Transforming descriptors to BowVector and FeatureVector
/// - Quantizing individual descriptors to word IDs
#[derive(Debug)]
pub struct OrbVocabulary {
    /// All nodes in the tree (nodes[0] is root)
    nodes: Vec<VocabNode>,
    /// Word ID to node ID mapping (for leaf nodes)
    words: Vec<u32>,
    /// Branching factor (typically 10)
    k: usize,
    /// Depth levels (typically 5-6)
    l: usize,
}

impl OrbVocabulary {
    /// Load vocabulary from DBoW2 text format.
    ///
    /// # File Format
    ///
    /// ```text
    /// k L scoring weighting
    /// parent_id is_leaf desc[0] desc[1] ... desc[31] weight
    /// ...
    /// ```
    ///
    /// - Line 1: k=branching factor, L=depth, scoring type, weighting type
    /// - Lines 2+: One line per node (excluding root)
    ///   - parent_id: Parent node index (0-based, 0 means child of root)
    ///   - is_leaf: 1 if leaf node, 0 otherwise
    ///   - desc[0..31]: 32 descriptor bytes
    ///   - weight: IDF weight (non-zero for leaves)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let vocab = OrbVocabulary::load_from_text("data/ORBvoc.txt")?;
    /// ```
    pub fn load_from_text<P: AsRef<Path>>(path: P) -> Result<Self, VocabularyError> {
        let file = File::open(path.as_ref())
            .map_err(|e| VocabularyError::Io(format!("Failed to open vocabulary file: {}", e)))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse header: k L scoring weighting
        let header = lines
            .next()
            .ok_or_else(|| VocabularyError::Parse("Empty vocabulary file".to_string()))?
            .map_err(|e| VocabularyError::Io(e.to_string()))?;

        let header_parts: Vec<&str> = header.split_whitespace().collect();
        if header_parts.len() < 2 {
            return Err(VocabularyError::Parse(
                "Invalid header format, expected: k L [scoring weighting]".to_string(),
            ));
        }

        let k: usize = header_parts[0]
            .parse()
            .map_err(|_| VocabularyError::Parse("Invalid k value".to_string()))?;
        let l: usize = header_parts[1]
            .parse()
            .map_err(|_| VocabularyError::Parse("Invalid L value".to_string()))?;

        // Create root node
        let mut nodes = vec![VocabNode::new(0, u32::MAX)];
        let mut words = Vec::new();
        let mut word_count = 0u32;

        // Parse node lines
        for (line_num, line_result) in lines.enumerate() {
            let line = line_result.map_err(|e| VocabularyError::Io(e.to_string()))?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            // Expected format: parent_id is_leaf desc[32] weight
            // That's 1 + 1 + 32 + 1 = 35 parts
            if parts.len() < 35 {
                continue; // Skip malformed lines
            }

            let parent_id: u32 = parts[0]
                .parse()
                .map_err(|_| VocabularyError::Parse(format!("Invalid parent_id at line {}", line_num + 2)))?;
            let is_leaf: bool = parts[1] == "1";

            // Parse descriptor (32 bytes)
            let mut descriptor = [0u8; 32];
            for (i, byte_str) in parts[2..34].iter().enumerate() {
                descriptor[i] = byte_str
                    .parse()
                    .map_err(|_| VocabularyError::Parse(format!("Invalid descriptor byte at line {}", line_num + 2)))?;
            }

            // Parse weight
            let weight: f64 = parts[34]
                .parse()
                .map_err(|_| VocabularyError::Parse(format!("Invalid weight at line {}", line_num + 2)))?;

            // Create node with sequential ID
            let node_id = nodes.len() as u32;
            let mut node = VocabNode::new(node_id, parent_id);
            node.descriptor = descriptor;
            node.weight = weight;

            // If leaf, assign word ID
            if is_leaf {
                node.word_id = Some(word_count);
                words.push(node_id);
                word_count += 1;
            }

            // Link to parent
            if (parent_id as usize) < nodes.len() {
                nodes[parent_id as usize].children.push(node_id);
            }

            nodes.push(node);
        }

        tracing::info!(
            "Loaded vocabulary: k={}, L={}, {} nodes, {} words",
            k,
            l,
            nodes.len(),
            words.len()
        );

        Ok(Self { nodes, words, k, l })
    }

    /// Get vocabulary parameters.
    pub fn params(&self) -> (usize, usize) {
        (self.k, self.l)
    }

    /// Get number of visual words (leaf nodes).
    pub fn num_words(&self) -> usize {
        self.words.len()
    }

    /// Get number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Quantize a single descriptor to its leaf word.
    ///
    /// Traverses the tree from root to leaf, following the closest child at each level.
    ///
    /// # Returns
    ///
    /// Tuple of (word_id, leaf_node_id)
    fn transform_descriptor(&self, desc: &[u8]) -> (u32, u32) {
        let mut node_id = 0usize; // Start at root

        // Descend tree by following closest child at each level
        while !self.nodes[node_id].children.is_empty() {
            let mut best_child = self.nodes[node_id].children[0];
            let mut best_dist = hamming_distance(desc, &self.nodes[best_child as usize].descriptor);

            for &child in &self.nodes[node_id].children[1..] {
                let dist = hamming_distance(desc, &self.nodes[child as usize].descriptor);
                if dist < best_dist {
                    best_dist = dist;
                    best_child = child;
                }
            }

            node_id = best_child as usize;
        }

        let word_id = self.nodes[node_id].word_id.unwrap_or(0);
        (word_id, node_id as u32)
    }

    /// Get the ancestor node at a specific number of levels up from a leaf.
    ///
    /// # Arguments
    ///
    /// * `leaf_id` - The leaf node ID
    /// * `levels_up` - Number of levels to go up (0 = leaf itself)
    ///
    /// # Returns
    ///
    /// The node ID at the requested level, or the root if levels_up exceeds depth.
    fn get_parent_at_level(&self, leaf_id: u32, levels_up: usize) -> u32 {
        let mut node_id = leaf_id;

        for _ in 0..levels_up {
            let parent = self.nodes[node_id as usize].parent;
            if parent == u32::MAX {
                // Reached root
                break;
            }
            node_id = parent;
        }

        node_id
    }

    /// Transform descriptors to BowVector and FeatureVector.
    ///
    /// # Arguments
    ///
    /// * `descriptors` - OpenCV Mat with one descriptor per row (CV_8U, 32 columns)
    /// * `levels_up` - Levels to go up from leaf for FeatureVector grouping (typically 4 for L=5)
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `BowVector`: word_id -> L1-normalized TF-IDF weight
    /// - `FeatureVector`: node_id (at level L-levels_up) -> list of feature indices
    pub fn transform(&self, descriptors: &Mat, levels_up: usize) -> (BowVector, FeatureVector) {
        let mut bow = BowVector::new();
        let mut feat = FeatureVector::new();

        let rows = descriptors.rows();
        for i in 0..rows {
            // Get descriptor row as bytes
            let desc = match get_descriptor_row(descriptors, i) {
                Some(d) => d,
                None => continue,
            };

            let (word_id, leaf_id) = self.transform_descriptor(&desc);

            // BowVector: accumulate word weights
            let weight = self.nodes[leaf_id as usize].weight;
            *bow.entry(word_id).or_insert(0.0) += weight;

            // FeatureVector: group by node at level L - levels_up
            let node_at_level = self.get_parent_at_level(leaf_id, levels_up);
            feat.entry(node_at_level).or_default().push(i as usize);
        }

        // Normalize BowVector (L1 norm)
        let sum: f64 = bow.values().sum();
        if sum > 0.0 {
            for v in bow.values_mut() {
                *v /= sum;
            }
        }

        (bow, feat)
    }

    /// Transform descriptors to BowVector only (faster, no FeatureVector).
    ///
    /// Use this when you only need place recognition and not feature matching.
    pub fn transform_bow_only(&self, descriptors: &Mat) -> BowVector {
        let mut bow = BowVector::new();

        let rows = descriptors.rows();
        for i in 0..rows {
            let desc = match get_descriptor_row(descriptors, i) {
                Some(d) => d,
                None => continue,
            };

            let (word_id, leaf_id) = self.transform_descriptor(&desc);
            let weight = self.nodes[leaf_id as usize].weight;
            *bow.entry(word_id).or_insert(0.0) += weight;
        }

        // Normalize
        let sum: f64 = bow.values().sum();
        if sum > 0.0 {
            for v in bow.values_mut() {
                *v /= sum;
            }
        }

        bow
    }

    /// Compute similarity score between two BowVectors.
    ///
    /// Uses L1 scoring: 1 - 0.5 * ||v1 - v2||_1
    /// Returns a score in [0, 1] where 1 means identical.
    pub fn score(v1: &BowVector, v2: &BowVector) -> f64 {
        let mut diff_sum = 0.0;

        // Sum |v1[i] - v2[i]| for all words
        for (word_id, w1) in v1 {
            let w2 = v2.get(word_id).copied().unwrap_or(0.0);
            diff_sum += (w1 - w2).abs();
        }

        // Add v2 entries not in v1
        for (word_id, w2) in v2 {
            if !v1.contains_key(word_id) {
                diff_sum += w2.abs();
            }
        }

        1.0 - 0.5 * diff_sum
    }
}

/// Errors that can occur when loading or using vocabulary.
#[derive(Debug)]
pub enum VocabularyError {
    /// I/O error reading vocabulary file
    Io(String),
    /// Parse error in vocabulary format
    Parse(String),
}

impl std::fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VocabularyError::Io(msg) => write!(f, "Vocabulary I/O error: {}", msg),
            VocabularyError::Parse(msg) => write!(f, "Vocabulary parse error: {}", msg),
        }
    }
}

impl std::error::Error for VocabularyError {}

/// Compute Hamming distance between two 32-byte descriptors.
#[inline]
fn hamming_distance(a: &[u8], b: &[u8; 32]) -> u32 {
    let mut dist = 0u32;
    for i in 0..32 {
        dist += (a.get(i).copied().unwrap_or(0) ^ b[i]).count_ones();
    }
    dist
}

/// Extract a descriptor row from an OpenCV Mat as a byte slice.
fn get_descriptor_row(mat: &Mat, row: i32) -> Option<[u8; 32]> {
    if row < 0 || row >= mat.rows() {
        return None;
    }

    let mut desc = [0u8; 32];
    let cols = mat.cols().min(32);

    for j in 0..cols {
        if let Ok(val) = mat.at_2d::<u8>(row, j) {
            desc[j as usize] = *val;
        }
    }

    Some(desc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance() {
        let a = [0u8; 32];
        let b = [0u8; 32];
        assert_eq!(hamming_distance(&a, &b), 0);

        let mut c = [0u8; 32];
        c[0] = 0xFF; // 8 bits different
        assert_eq!(hamming_distance(&a, &c), 8);

        c[1] = 0x0F; // 4 more bits
        assert_eq!(hamming_distance(&a, &c), 12);
    }

    #[test]
    fn test_bow_score() {
        let mut v1 = BowVector::new();
        v1.insert(0, 0.5);
        v1.insert(1, 0.5);

        let mut v2 = BowVector::new();
        v2.insert(0, 0.5);
        v2.insert(1, 0.5);

        // Identical vectors should score 1.0
        let score = OrbVocabulary::score(&v1, &v2);
        assert!((score - 1.0).abs() < 1e-10);

        // Completely different vectors
        let mut v3 = BowVector::new();
        v3.insert(2, 0.5);
        v3.insert(3, 0.5);

        let score2 = OrbVocabulary::score(&v1, &v3);
        assert!(score2 < 0.01);
    }

    #[test]
    fn test_vocab_node_creation() {
        let node = VocabNode::new(1, 0);
        assert_eq!(node.id, 1);
        assert_eq!(node.parent, 0);
        assert!(node.is_leaf());
        assert!(node.word_id.is_none());
    }

    #[test]
    #[ignore] // Takes ~8 seconds to load the vocabulary
    fn test_load_dbow2_vocabulary() {
        // Uses the vocabulary file in data/ORBvoc.txt
        let vocab_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("data/ORBvoc.txt");

        if !vocab_path.exists() {
            eprintln!("Skipping test: {} not found", vocab_path.display());
            return;
        }

        let vocab = OrbVocabulary::load_from_text(&vocab_path)
            .expect("Failed to load vocabulary");

        let (k, l) = vocab.params();
        assert_eq!(k, 10, "Expected branching factor k=10");
        assert_eq!(l, 6, "Expected depth L=6");

        // Expected nodes: (k^L - 1) / (k - 1) = (10^6 - 1) / 9 = 111,111
        // Plus root = 111,112
        assert!(vocab.num_nodes() > 100_000, "Expected >100k nodes");

        // Expected words (leaves): k^(L-1) = 10^5 = 100,000
        assert!(vocab.num_words() > 90_000, "Expected ~100k words");

        println!("Vocabulary loaded: k={}, L={}, {} nodes, {} words",
            k, l, vocab.num_nodes(), vocab.num_words());
    }
}
