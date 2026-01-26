//! Pose Graph Optimization for Loop Closing.
//!
//! Optimizes the Essential Graph (spanning tree + loop edges + strong covisibility)
//! using Sim3 constraints. For stereo mode, scale is fixed to 1.0.
//!
//! This module uses the three-phase pattern:
//! 1. COLLECT: Extract poses and edges from the map
//! 2. SOLVE: Run LM optimization without locks
//! 3. APPLY: Write back optimized poses

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector};

use crate::atlas::map::{KeyFrameId, Map};
use crate::geometry::{SE3, Sim3};

/// Configuration for pose graph optimization.
#[derive(Debug, Clone)]
pub struct PoseGraphConfig {
    /// Maximum number of LM iterations.
    pub max_iterations: usize,

    /// Convergence threshold on parameter change.
    pub param_tolerance: f64,

    /// Convergence threshold on gradient norm.
    pub gradient_tolerance: f64,

    /// Minimum covisibility weight to include as edge.
    pub min_covisibility_weight: usize,

    /// Whether to fix scale (true for stereo).
    pub fix_scale: bool,
}

impl Default for PoseGraphConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            param_tolerance: 1e-6,
            gradient_tolerance: 1e-6,
            min_covisibility_weight: 100,
            fix_scale: true,
        }
    }
}

/// A pose graph edge (constraint between two keyframes).
#[derive(Debug, Clone)]
pub struct PoseGraphEdge {
    /// First keyframe ID.
    pub kf_id_i: KeyFrameId,

    /// Second keyframe ID.
    pub kf_id_j: KeyFrameId,

    /// Relative Sim3 measurement from i to j.
    pub measurement: Sim3,

    /// Information weight (inverse covariance).
    pub information: f64,

    /// Edge type for debugging.
    pub edge_type: EdgeType,
}

/// Type of pose graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// Spanning tree edge.
    SpanningTree,
    /// Loop closure edge.
    LoopClosure,
    /// Strong covisibility edge.
    Covisibility,
}

/// Data extracted for pose graph optimization.
pub struct PoseGraphProblemData {
    /// Keyframe IDs in optimization order.
    pub kf_ids: Vec<KeyFrameId>,

    /// Initial Sim3 poses for each keyframe.
    pub initial_poses: HashMap<KeyFrameId, Sim3>,

    /// Pose graph edges.
    pub edges: Vec<PoseGraphEdge>,

    /// ID of fixed keyframe (anchor).
    pub fixed_kf_id: KeyFrameId,

    /// Whether to fix scale.
    pub fix_scale: bool,
}

/// Result of pose graph optimization.
pub struct PoseGraphResult {
    /// Optimized Sim3 poses.
    pub optimized_poses: HashMap<KeyFrameId, Sim3>,

    /// Number of iterations.
    pub iterations: usize,

    /// Initial error.
    pub initial_error: f64,

    /// Final error.
    pub final_error: f64,
}

/// PHASE 1: Collect pose graph data from the map.
pub fn collect_pose_graph_data(
    map: &Map,
    loop_kf_id: KeyFrameId,
    current_kf_id: KeyFrameId,
    loop_sim3: &Sim3,
    config: &PoseGraphConfig,
) -> Option<PoseGraphProblemData> {
    let mut kf_ids = Vec::new();
    let mut initial_poses = HashMap::new();
    let mut edges = Vec::new();
    let mut visited = HashSet::new();

    // Find the oldest keyframe as anchor
    let first_kf = map.keyframes().min_by_key(|kf| kf.id.0)?;
    let fixed_kf_id = first_kf.id;

    // Collect all keyframes via BFS from fixed keyframe
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(fixed_kf_id);
    visited.insert(fixed_kf_id);

    while let Some(kf_id) = queue.pop_front() {
        let kf = map.get_keyframe(kf_id)?;

        kf_ids.push(kf_id);
        initial_poses.insert(kf_id, Sim3::from_se3(&kf.pose));

        // Add spanning tree edges
        if let Some(parent_id) = kf.parent_id {
            if visited.contains(&parent_id) {
                if let Some(parent_kf) = map.get_keyframe(parent_id) {
                    let relative = compute_relative_sim3(&parent_kf.pose, &kf.pose);
                    edges.push(PoseGraphEdge {
                        kf_id_i: parent_id,
                        kf_id_j: kf_id,
                        measurement: relative,
                        information: 1.0,
                        edge_type: EdgeType::SpanningTree,
                    });
                }
            }
        }

        // Add children to queue
        for &child_id in &kf.children_ids {
            if !visited.contains(&child_id) {
                visited.insert(child_id);
                queue.push_back(child_id);
            }
        }

        // Add strong covisibility edges
        for (&cov_id, &weight) in kf.covisibility_weights() {
            if weight >= config.min_covisibility_weight && visited.contains(&cov_id) {
                if let Some(cov_kf) = map.get_keyframe(cov_id) {
                    let relative = compute_relative_sim3(&cov_kf.pose, &kf.pose);
                    edges.push(PoseGraphEdge {
                        kf_id_i: cov_id,
                        kf_id_j: kf_id,
                        measurement: relative,
                        information: (weight as f64) / 100.0,
                        edge_type: EdgeType::Covisibility,
                    });
                }
            }
        }
    }

    // Add the loop closure edge
    edges.push(PoseGraphEdge {
        kf_id_i: loop_kf_id,
        kf_id_j: current_kf_id,
        measurement: loop_sim3.clone(),
        information: 100.0, // High weight for loop closure
        edge_type: EdgeType::LoopClosure,
    });

    Some(PoseGraphProblemData {
        kf_ids,
        initial_poses,
        edges,
        fixed_kf_id,
        fix_scale: config.fix_scale,
    })
}

/// PHASE 2: Solve pose graph optimization.
pub fn solve_pose_graph(
    problem: &PoseGraphProblemData,
    config: &PoseGraphConfig,
    should_stop: &dyn Fn() -> bool,
) -> Option<PoseGraphResult> {
    let n_poses = problem.kf_ids.len();
    if n_poses < 2 {
        return None;
    }

    // Build keyframe index mapping
    let kf_to_idx: HashMap<KeyFrameId, usize> = problem
        .kf_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    // Determine which index is fixed
    let fixed_idx = *kf_to_idx.get(&problem.fixed_kf_id)?;

    // Number of parameters per pose (6 for SE3, 7 for Sim3 with scale)
    let params_per_pose = if problem.fix_scale { 6 } else { 7 };

    // Build initial parameter vector (excluding fixed pose)
    let n_optimized = n_poses - 1;
    let n_params = n_optimized * params_per_pose;

    if n_params == 0 {
        return None;
    }

    // Map from keyframe index to parameter index (skipping fixed)
    let mut idx_to_param: HashMap<usize, usize> = HashMap::new();
    let mut param_idx = 0;
    for (kf_idx, _) in problem.kf_ids.iter().enumerate() {
        if kf_idx != fixed_idx {
            idx_to_param.insert(kf_idx, param_idx);
            param_idx += 1;
        }
    }

    // Initialize parameters
    let mut params = DVector::zeros(n_params);
    for (kf_idx, &kf_id) in problem.kf_ids.iter().enumerate() {
        if kf_idx == fixed_idx {
            continue;
        }
        let param_offset = idx_to_param[&kf_idx] * params_per_pose;
        if let Some(sim3) = problem.initial_poses.get(&kf_id) {
            let tangent = sim3.log();
            for i in 0..params_per_pose {
                params[param_offset + i] = tangent[i];
            }
        }
    }

    // Get fixed pose
    let fixed_pose = problem
        .initial_poses
        .get(&problem.fixed_kf_id)
        .cloned()
        .unwrap_or_else(Sim3::identity);

    // Compute initial error
    let initial_error = compute_pose_graph_error(
        &params,
        &problem.edges,
        &kf_to_idx,
        &idx_to_param,
        &fixed_pose,
        fixed_idx,
        params_per_pose,
    );

    // LM optimization
    let mut current_params = params;
    let mut lambda = 1e-3;
    let lambda_up = 10.0;
    let lambda_down = 0.1;
    let min_lambda = 1e-10;
    let max_lambda = 1e10;
    let mut iterations = 0;

    for iter in 0..config.max_iterations {
        if should_stop() {
            break;
        }
        iterations = iter + 1;

        // Compute residuals and Jacobian
        let (residuals, jacobian) = compute_pose_graph_residuals_and_jacobian(
            &current_params,
            &problem.edges,
            &kf_to_idx,
            &idx_to_param,
            &fixed_pose,
            fixed_idx,
            params_per_pose,
        );

        let current_error = residuals.norm_squared();

        // Compute gradient and Hessian approximation
        let gradient = jacobian.transpose() * &residuals;
        let jtj = jacobian.transpose() * &jacobian;

        if gradient.norm() < config.gradient_tolerance {
            break;
        }

        // Damped system
        let mut damped_jtj = jtj.clone();
        for i in 0..n_params {
            damped_jtj[(i, i)] += lambda * damped_jtj[(i, i)].max(1e-6);
        }

        // Solve
        let delta = match damped_jtj.clone().lu().solve(&(-&gradient)) {
            Some(d) => d,
            None => break,
        };

        if delta.norm() < config.param_tolerance * (current_params.norm() + config.param_tolerance) {
            break;
        }

        // Trial step
        let trial_params = &current_params + &delta;
        let trial_error = compute_pose_graph_error(
            &trial_params,
            &problem.edges,
            &kf_to_idx,
            &idx_to_param,
            &fixed_pose,
            fixed_idx,
            params_per_pose,
        );

        if trial_error < current_error {
            current_params = trial_params;
            lambda = (lambda * lambda_down).max(min_lambda);
        } else {
            lambda = (lambda * lambda_up).min(max_lambda);
        }
    }

    // Extract optimized poses
    let final_error = compute_pose_graph_error(
        &current_params,
        &problem.edges,
        &kf_to_idx,
        &idx_to_param,
        &fixed_pose,
        fixed_idx,
        params_per_pose,
    );

    let mut optimized_poses = HashMap::new();

    // Add fixed pose
    optimized_poses.insert(problem.fixed_kf_id, fixed_pose);

    // Extract optimized poses
    for (kf_idx, &kf_id) in problem.kf_ids.iter().enumerate() {
        if kf_idx == fixed_idx {
            continue;
        }
        let param_offset = idx_to_param[&kf_idx] * params_per_pose;
        let mut tangent = [0.0; 7];
        for i in 0..params_per_pose {
            tangent[i] = current_params[param_offset + i];
        }
        if params_per_pose == 6 {
            tangent[6] = 0.0; // log(1.0) = 0
        }
        optimized_poses.insert(kf_id, Sim3::exp(&tangent));
    }

    Some(PoseGraphResult {
        optimized_poses,
        iterations,
        initial_error: initial_error.sqrt(),
        final_error: final_error.sqrt(),
    })
}

/// PHASE 3: Apply optimized poses to the map.
pub fn apply_pose_graph_results(map: &mut Map, results: &PoseGraphResult) -> usize {
    let mut updated = 0;

    for (kf_id, sim3) in &results.optimized_poses {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            if !kf.is_bad {
                kf.pose = sim3.to_se3();
                updated += 1;
            }
        }
    }

    updated
}

/// Compute relative Sim3 between two SE3 poses.
fn compute_relative_sim3(pose_i: &SE3, pose_j: &SE3) -> Sim3 {
    // T_ij = T_i^{-1} * T_j
    let pose_i_inv = pose_i.inverse();
    let relative_se3 = pose_i_inv.compose(pose_j);
    Sim3::from_se3(&relative_se3)
}

/// Compute pose graph error.
fn compute_pose_graph_error(
    params: &DVector<f64>,
    edges: &[PoseGraphEdge],
    kf_to_idx: &HashMap<KeyFrameId, usize>,
    idx_to_param: &HashMap<usize, usize>,
    fixed_pose: &Sim3,
    fixed_idx: usize,
    params_per_pose: usize,
) -> f64 {
    let mut total_error = 0.0;

    for edge in edges {
        let idx_i = *kf_to_idx.get(&edge.kf_id_i).unwrap();
        let idx_j = *kf_to_idx.get(&edge.kf_id_j).unwrap();

        let pose_i = get_pose_from_params(params, idx_i, idx_to_param, fixed_pose, fixed_idx, params_per_pose);
        let pose_j = get_pose_from_params(params, idx_j, idx_to_param, fixed_pose, fixed_idx, params_per_pose);

        // Compute error: log(measurement^{-1} * pose_i^{-1} * pose_j)
        let predicted = pose_i.inverse().compose(&pose_j);
        let error_sim3 = edge.measurement.inverse().compose(&predicted);
        let error_vec = error_sim3.log();

        let error_sq: f64 = error_vec.iter().take(params_per_pose).map(|e| e * e).sum();
        total_error += error_sq * edge.information;
    }

    total_error
}

/// Compute residuals and Jacobian for pose graph.
fn compute_pose_graph_residuals_and_jacobian(
    params: &DVector<f64>,
    edges: &[PoseGraphEdge],
    kf_to_idx: &HashMap<KeyFrameId, usize>,
    idx_to_param: &HashMap<usize, usize>,
    fixed_pose: &Sim3,
    fixed_idx: usize,
    params_per_pose: usize,
) -> (DVector<f64>, DMatrix<f64>) {
    let n_edges = edges.len();
    let n_residuals = n_edges * params_per_pose;
    let n_params = params.len();

    let mut residuals = DVector::zeros(n_residuals);
    let mut jacobian = DMatrix::zeros(n_residuals, n_params);

    for (edge_idx, edge) in edges.iter().enumerate() {
        let idx_i = *kf_to_idx.get(&edge.kf_id_i).unwrap();
        let idx_j = *kf_to_idx.get(&edge.kf_id_j).unwrap();

        let pose_i = get_pose_from_params(params, idx_i, idx_to_param, fixed_pose, fixed_idx, params_per_pose);
        let pose_j = get_pose_from_params(params, idx_j, idx_to_param, fixed_pose, fixed_idx, params_per_pose);

        // Compute error
        let predicted = pose_i.inverse().compose(&pose_j);
        let error_sim3 = edge.measurement.inverse().compose(&predicted);
        let error_vec = error_sim3.log();

        let weight_sqrt = edge.information.sqrt();
        let res_offset = edge_idx * params_per_pose;

        for i in 0..params_per_pose {
            residuals[res_offset + i] = error_vec[i] * weight_sqrt;
        }

        // Numerical Jacobian (simplified)
        let eps = 1e-6;

        // Jacobian w.r.t. pose_i
        if idx_i != fixed_idx {
            let param_idx_i = idx_to_param[&idx_i];
            let col_offset = param_idx_i * params_per_pose;

            for p in 0..params_per_pose {
                let mut params_plus = params.clone();
                let mut params_minus = params.clone();
                params_plus[col_offset + p] += eps;
                params_minus[col_offset + p] -= eps;

                let pose_i_plus = get_pose_from_params(&params_plus, idx_i, idx_to_param, fixed_pose, fixed_idx, params_per_pose);
                let pose_i_minus = get_pose_from_params(&params_minus, idx_i, idx_to_param, fixed_pose, fixed_idx, params_per_pose);

                let pred_plus = pose_i_plus.inverse().compose(&pose_j);
                let pred_minus = pose_i_minus.inverse().compose(&pose_j);

                let err_plus = edge.measurement.inverse().compose(&pred_plus).log();
                let err_minus = edge.measurement.inverse().compose(&pred_minus).log();

                for r in 0..params_per_pose {
                    jacobian[(res_offset + r, col_offset + p)] =
                        (err_plus[r] - err_minus[r]) / (2.0 * eps) * weight_sqrt;
                }
            }
        }

        // Jacobian w.r.t. pose_j
        if idx_j != fixed_idx {
            let param_idx_j = idx_to_param[&idx_j];
            let col_offset = param_idx_j * params_per_pose;

            for p in 0..params_per_pose {
                let mut params_plus = params.clone();
                let mut params_minus = params.clone();
                params_plus[col_offset + p] += eps;
                params_minus[col_offset + p] -= eps;

                let pose_j_plus = get_pose_from_params(&params_plus, idx_j, idx_to_param, fixed_pose, fixed_idx, params_per_pose);
                let pose_j_minus = get_pose_from_params(&params_minus, idx_j, idx_to_param, fixed_pose, fixed_idx, params_per_pose);

                let pred_plus = pose_i.inverse().compose(&pose_j_plus);
                let pred_minus = pose_i.inverse().compose(&pose_j_minus);

                let err_plus = edge.measurement.inverse().compose(&pred_plus).log();
                let err_minus = edge.measurement.inverse().compose(&pred_minus).log();

                for r in 0..params_per_pose {
                    jacobian[(res_offset + r, col_offset + p)] =
                        (err_plus[r] - err_minus[r]) / (2.0 * eps) * weight_sqrt;
                }
            }
        }
    }

    (residuals, jacobian)
}

/// Get Sim3 pose from parameters.
fn get_pose_from_params(
    params: &DVector<f64>,
    idx: usize,
    idx_to_param: &HashMap<usize, usize>,
    fixed_pose: &Sim3,
    fixed_idx: usize,
    params_per_pose: usize,
) -> Sim3 {
    if idx == fixed_idx {
        return fixed_pose.clone();
    }

    let param_idx = idx_to_param[&idx];
    let offset = param_idx * params_per_pose;

    let mut tangent = [0.0; 7];
    for i in 0..params_per_pose {
        tangent[i] = params[offset + i];
    }
    if params_per_pose == 6 {
        tangent[6] = 0.0; // log(1.0) = 0 for fixed scale
    }

    Sim3::exp(&tangent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn test_compute_relative_sim3() {
        let pose_i = SE3::identity();
        let pose_j = SE3 {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::new(1.0, 0.0, 0.0),
        };

        let relative = compute_relative_sim3(&pose_i, &pose_j);

        assert!((relative.translation.x - 1.0).abs() < 1e-10);
        assert!((relative.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pose_graph_config_default() {
        let config = PoseGraphConfig::default();
        assert_eq!(config.max_iterations, 20);
        assert!(config.fix_scale);
    }

    #[test]
    fn test_edge_type() {
        let edge = PoseGraphEdge {
            kf_id_i: KeyFrameId::new(0),
            kf_id_j: KeyFrameId::new(1),
            measurement: Sim3::identity(),
            information: 1.0,
            edge_type: EdgeType::LoopClosure,
        };

        assert_eq!(edge.edge_type, EdgeType::LoopClosure);
    }
}
