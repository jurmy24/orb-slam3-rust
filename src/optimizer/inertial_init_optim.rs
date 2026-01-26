//! IMU Initialization Optimization for Visual-Inertial SLAM.
//!
//! This module implements the optimization step after initial IMU parameter estimation.
//! Following ORB-SLAM3, the optimization refines:
//! - Keyframe velocities
//! - IMU biases (gyroscope and accelerometer)
//! - Gravity direction (Rwg - rotation from gravity to world frame)
//!
//! # Optimization Variants
//!
//! Three variants are provided for different initialization phases:
//! 1. `optimize_inertial_init_full` - Full optimization of all parameters
//! 2. `optimize_inertial_init_bias_only` - Fix Rwg, optimize biases only
//! 3. `optimize_inertial_init_scale_refinement` - Fix biases, refine Rwg only
//!
//! # State Vector
//!
//! The state vector contains:
//! - Velocities: 3D per keyframe (n * 3)
//! - Gyro bias: 3D (shared across all keyframes)
//! - Accel bias: 3D (shared across all keyframes)
//! - Rwg: 3D axis-angle representation
//!
//! Total: n*3 + 3 + 3 + 3 = n*3 + 9 parameters

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, UnitQuaternion, Vector3};
use tracing::{debug, info};

use crate::atlas::map::{KeyFrameId, Map};
use crate::geometry::SE3;
use crate::imu::{ImuBias, PreintegratedState};

/// Gravity magnitude constant (m/s^2).
const GRAVITY_MAGNITUDE: f64 = 9.81;

/// Configuration for inertial initialization optimization.
#[derive(Clone, Debug)]
pub struct InertialInitConfig {
    /// Maximum number of Levenberg-Marquardt iterations.
    pub max_iterations: usize,
    /// Initial LM damping factor.
    pub initial_lambda: f64,
    /// LM damping increase factor when cost increases.
    pub lambda_increase: f64,
    /// LM damping decrease factor when cost decreases.
    pub lambda_decrease: f64,
    /// Minimum lambda (prevents instability).
    pub min_lambda: f64,
    /// Maximum lambda (triggers early exit).
    pub max_lambda: f64,
    /// Convergence threshold for parameter change.
    pub convergence_threshold: f64,
    /// Prior weight for gyro bias (information matrix diagonal).
    pub prior_gyro: f64,
    /// Prior weight for accel bias (information matrix diagonal).
    pub prior_accel: f64,
    /// Weight for gravity magnitude constraint.
    pub gravity_mag_weight: f64,
}

impl Default for InertialInitConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            initial_lambda: 1e-4,
            lambda_increase: 10.0,
            lambda_decrease: 0.1,
            min_lambda: 1e-12,
            max_lambda: 1e8,
            convergence_threshold: 1e-6,
            // Phase 2 priors (t >= 15s) - no strong priors
            prior_gyro: 0.0,
            prior_accel: 0.0,
            gravity_mag_weight: 1e3,
        }
    }
}

impl InertialInitConfig {
    /// Configuration for Phase 0 (t < 5s) with strong priors.
    pub fn phase0() -> Self {
        Self {
            prior_gyro: 1e2,
            prior_accel: 1e5,
            ..Self::default()
        }
    }

    /// Configuration for Phase 1 (5s <= t < 15s) with moderate priors.
    pub fn phase1() -> Self {
        Self {
            prior_gyro: 1.0,
            prior_accel: 1e5,
            ..Self::default()
        }
    }

    /// Configuration for Phase 2 (t >= 15s) with no priors.
    pub fn phase2() -> Self {
        Self::default()
    }

    /// Select configuration based on time since map start.
    pub fn for_time(time_span_seconds: f64) -> Self {
        if time_span_seconds < 5.0 {
            Self::phase0()
        } else if time_span_seconds < 15.0 {
            Self::phase1()
        } else {
            Self::phase2()
        }
    }
}

/// Result of inertial initialization optimization.
#[derive(Clone, Debug)]
pub struct InertialInitOptimResult {
    /// Optimized velocities for each keyframe.
    pub velocities: HashMap<KeyFrameId, Vector3<f64>>,
    /// Optimized gyroscope bias.
    pub gyro_bias: Vector3<f64>,
    /// Optimized accelerometer bias.
    pub accel_bias: Vector3<f64>,
    /// Optimized gravity direction in world frame (rotation from [0,0,-g] to actual gravity).
    pub rwg: UnitQuaternion<f64>,
    /// Gravity vector in world frame.
    pub gravity_world: Vector3<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Initial total cost.
    pub initial_cost: f64,
    /// Final total cost.
    pub final_cost: f64,
    /// Whether optimization converged.
    pub converged: bool,
}

/// Problem data extracted from the map for optimization.
#[derive(Clone)]
pub struct InertialInitProblem {
    /// Keyframe IDs in temporal order.
    pub kf_ids: Vec<KeyFrameId>,
    /// Keyframe poses (T_wc: camera-to-world).
    pub kf_poses: HashMap<KeyFrameId, SE3>,
    /// Initial velocity estimates.
    pub initial_velocities: HashMap<KeyFrameId, Vector3<f64>>,
    /// IMU preintegration between consecutive keyframes.
    /// imu_preint[i] is the preintegration from kf_ids[i] to kf_ids[i+1].
    pub imu_preint: Vec<PreintegratedState>,
    /// Initial gyro bias estimate.
    pub initial_gyro_bias: Vector3<f64>,
    /// Initial accel bias estimate.
    pub initial_accel_bias: Vector3<f64>,
    /// Initial Rwg estimate.
    pub initial_rwg: UnitQuaternion<f64>,
}

impl InertialInitProblem {
    /// Extract problem data from the map.
    pub fn from_map(map: &Map, initial_rwg: UnitQuaternion<f64>) -> Option<Self> {
        let keyframes = map.keyframes_temporal_order();
        if keyframes.len() < 2 {
            return None;
        }

        let kf_ids: Vec<KeyFrameId> = keyframes.iter().map(|kf| kf.id).collect();
        let kf_poses: HashMap<_, _> = keyframes
            .iter()
            .map(|kf| (kf.id, kf.pose.clone()))
            .collect();
        let initial_velocities: HashMap<_, _> = keyframes
            .iter()
            .map(|kf| (kf.id, kf.velocity))
            .collect();

        // Extract IMU preintegration between consecutive keyframes
        let mut imu_preint = Vec::new();
        for i in 1..keyframes.len() {
            let kf = keyframes[i];
            if let Some(ref preint) = kf.imu_preintegrated {
                imu_preint.push(preint.clone());
            } else {
                // Missing preintegration - create identity
                imu_preint.push(PreintegratedState::identity());
            }
        }

        // Get initial bias from the first keyframe
        let first_bias = &keyframes[0].imu_bias;

        Some(Self {
            kf_ids,
            kf_poses,
            initial_velocities,
            imu_preint,
            initial_gyro_bias: first_bias.gyro,
            initial_accel_bias: first_bias.accel,
            initial_rwg,
        })
    }
}

/// State layout for the optimization problem.
struct StateLayout {
    num_keyframes: usize,
}

impl StateLayout {
    fn new(num_keyframes: usize) -> Self {
        Self { num_keyframes }
    }

    /// Total number of parameters.
    fn total_params(&self) -> usize {
        // n*3 velocities + 3 gyro bias + 3 accel bias + 3 rwg
        self.num_keyframes * 3 + 9
    }

    /// Start index for velocity of keyframe i.
    fn vel_start(&self, kf_idx: usize) -> usize {
        kf_idx * 3
    }

    /// Start index for gyro bias.
    fn gyro_bias_start(&self) -> usize {
        self.num_keyframes * 3
    }

    /// Start index for accel bias.
    fn accel_bias_start(&self) -> usize {
        self.num_keyframes * 3 + 3
    }

    /// Start index for Rwg.
    fn rwg_start(&self) -> usize {
        self.num_keyframes * 3 + 6
    }
}

/// Full inertial initialization optimization.
///
/// Optimizes velocities, biases, and Rwg jointly using Levenberg-Marquardt.
///
/// # Arguments
/// * `problem` - Problem data extracted from the map
/// * `config` - Optimization configuration
///
/// # Returns
/// Optimization result with refined parameters
pub fn optimize_inertial_init_full(
    problem: &InertialInitProblem,
    config: &InertialInitConfig,
) -> InertialInitOptimResult {
    let n = problem.kf_ids.len();
    let layout = StateLayout::new(n);
    let num_params = layout.total_params();

    // Initialize state vector
    let mut x = DVector::zeros(num_params);

    // Pack initial velocities
    for (i, kf_id) in problem.kf_ids.iter().enumerate() {
        let vel = problem.initial_velocities.get(kf_id).unwrap();
        let start = layout.vel_start(i);
        x[start] = vel.x;
        x[start + 1] = vel.y;
        x[start + 2] = vel.z;
    }

    // Pack initial biases
    let bg_start = layout.gyro_bias_start();
    x[bg_start] = problem.initial_gyro_bias.x;
    x[bg_start + 1] = problem.initial_gyro_bias.y;
    x[bg_start + 2] = problem.initial_gyro_bias.z;

    let ba_start = layout.accel_bias_start();
    x[ba_start] = problem.initial_accel_bias.x;
    x[ba_start + 1] = problem.initial_accel_bias.y;
    x[ba_start + 2] = problem.initial_accel_bias.z;

    // Pack initial Rwg as axis-angle
    let rwg_start = layout.rwg_start();
    let rwg_aa = problem.initial_rwg.scaled_axis();
    x[rwg_start] = rwg_aa.x;
    x[rwg_start + 1] = rwg_aa.y;
    x[rwg_start + 2] = rwg_aa.z;

    // Compute initial cost
    let (initial_cost, _) = compute_cost_and_residuals(problem, &x, &layout, config, false, false);

    // LM optimization loop
    let mut lambda = config.initial_lambda;
    let mut current_cost = initial_cost;
    let mut converged = false;
    let mut iter = 0;

    for i in 0..config.max_iterations {
        iter = i + 1;

        // Compute Jacobian and residuals
        let (cost, residuals) =
            compute_cost_and_residuals(problem, &x, &layout, config, false, false);
        let jacobian = compute_jacobian(problem, &x, &layout, config, false, false);

        // Compute Hessian approximation: H = J^T * J
        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        // LM update: (H + lambda * diag(H)) * dx = -J^T * r
        let mut h_lm = jtj.clone();
        for j in 0..num_params {
            h_lm[(j, j)] += lambda * (jtj[(j, j)].max(1e-10));
        }

        // Solve for update
        let dx = match h_lm.clone().lu().solve(&(-&jtr)) {
            Some(d) => d,
            None => {
                lambda *= config.lambda_increase;
                continue;
            }
        };

        // Try update
        let x_new = &x + &dx;
        let (new_cost, _) =
            compute_cost_and_residuals(problem, &x_new, &layout, config, false, false);

        if new_cost < cost {
            // Accept update
            x = x_new;
            current_cost = new_cost;
            lambda *= config.lambda_decrease;
            lambda = lambda.max(config.min_lambda);

            // Check convergence
            if dx.norm() < config.convergence_threshold {
                converged = true;
                break;
            }
        } else {
            // Reject update, increase damping
            lambda *= config.lambda_increase;
            if lambda > config.max_lambda {
                break;
            }
        }
    }

    // Unpack results
    let mut velocities = HashMap::new();
    for (i, kf_id) in problem.kf_ids.iter().enumerate() {
        let start = layout.vel_start(i);
        velocities.insert(*kf_id, Vector3::new(x[start], x[start + 1], x[start + 2]));
    }

    let bg_start = layout.gyro_bias_start();
    let gyro_bias = Vector3::new(x[bg_start], x[bg_start + 1], x[bg_start + 2]);

    let ba_start = layout.accel_bias_start();
    let accel_bias = Vector3::new(x[ba_start], x[ba_start + 1], x[ba_start + 2]);

    let rwg_start = layout.rwg_start();
    let rwg_aa = Vector3::new(x[rwg_start], x[rwg_start + 1], x[rwg_start + 2]);
    let rwg = UnitQuaternion::from_scaled_axis(rwg_aa);

    // Compute gravity in world frame
    let g_inertial = Vector3::new(0.0, 0.0, -GRAVITY_MAGNITUDE);
    let gravity_world = rwg * g_inertial;

    debug!(
        "Inertial init optimization: {} iters, cost {:.6} -> {:.6}, converged={}",
        iter, initial_cost, current_cost, converged
    );

    InertialInitOptimResult {
        velocities,
        gyro_bias,
        accel_bias,
        rwg,
        gravity_world,
        iterations: iter,
        initial_cost,
        final_cost: current_cost,
        converged,
    }
}

/// Bias-only optimization (Rwg fixed).
///
/// Used in later phases when gravity direction is well-established.
pub fn optimize_inertial_init_bias_only(
    problem: &InertialInitProblem,
    config: &InertialInitConfig,
) -> InertialInitOptimResult {
    let n = problem.kf_ids.len();
    let layout = StateLayout::new(n);
    let num_params = layout.total_params();

    // Initialize state vector
    let mut x = DVector::zeros(num_params);

    // Pack initial velocities
    for (i, kf_id) in problem.kf_ids.iter().enumerate() {
        let vel = problem.initial_velocities.get(kf_id).unwrap();
        let start = layout.vel_start(i);
        x[start] = vel.x;
        x[start + 1] = vel.y;
        x[start + 2] = vel.z;
    }

    // Pack initial biases
    let bg_start = layout.gyro_bias_start();
    x[bg_start] = problem.initial_gyro_bias.x;
    x[bg_start + 1] = problem.initial_gyro_bias.y;
    x[bg_start + 2] = problem.initial_gyro_bias.z;

    let ba_start = layout.accel_bias_start();
    x[ba_start] = problem.initial_accel_bias.x;
    x[ba_start + 1] = problem.initial_accel_bias.y;
    x[ba_start + 2] = problem.initial_accel_bias.z;

    // Pack Rwg (fixed)
    let rwg_start = layout.rwg_start();
    let rwg_aa = problem.initial_rwg.scaled_axis();
    x[rwg_start] = rwg_aa.x;
    x[rwg_start + 1] = rwg_aa.y;
    x[rwg_start + 2] = rwg_aa.z;

    // Compute initial cost
    let (initial_cost, _) = compute_cost_and_residuals(problem, &x, &layout, config, true, false);

    // LM optimization loop (same as full, but with fixed Rwg flag)
    let mut lambda = config.initial_lambda;
    let mut current_cost = initial_cost;
    let mut converged = false;
    let mut iter = 0;

    for i in 0..config.max_iterations {
        iter = i + 1;

        let (cost, residuals) =
            compute_cost_and_residuals(problem, &x, &layout, config, true, false);
        let jacobian = compute_jacobian(problem, &x, &layout, config, true, false);

        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        let effective_params = num_params - 3; // Rwg is fixed
        let mut h_lm = DMatrix::zeros(effective_params, effective_params);
        let mut jtr_eff = DVector::zeros(effective_params);

        // Copy relevant parts (exclude Rwg)
        for r in 0..effective_params {
            jtr_eff[r] = jtr[r];
            for c in 0..effective_params {
                h_lm[(r, c)] = jtj[(r, c)];
            }
        }

        for j in 0..effective_params {
            h_lm[(j, j)] += lambda * (h_lm[(j, j)].max(1e-10));
        }

        let dx_eff = match h_lm.clone().lu().solve(&(-&jtr_eff)) {
            Some(d) => d,
            None => {
                lambda *= config.lambda_increase;
                continue;
            }
        };

        // Full update (Rwg stays fixed)
        let mut dx = DVector::zeros(num_params);
        for j in 0..effective_params {
            dx[j] = dx_eff[j];
        }

        let x_new = &x + &dx;
        let (new_cost, _) =
            compute_cost_and_residuals(problem, &x_new, &layout, config, true, false);

        if new_cost < cost {
            x = x_new;
            current_cost = new_cost;
            lambda *= config.lambda_decrease;
            lambda = lambda.max(config.min_lambda);

            if dx_eff.norm() < config.convergence_threshold {
                converged = true;
                break;
            }
        } else {
            lambda *= config.lambda_increase;
            if lambda > config.max_lambda {
                break;
            }
        }
    }

    // Unpack results
    let mut velocities = HashMap::new();
    for (i, kf_id) in problem.kf_ids.iter().enumerate() {
        let start = layout.vel_start(i);
        velocities.insert(*kf_id, Vector3::new(x[start], x[start + 1], x[start + 2]));
    }

    let bg_start = layout.gyro_bias_start();
    let gyro_bias = Vector3::new(x[bg_start], x[bg_start + 1], x[bg_start + 2]);

    let ba_start = layout.accel_bias_start();
    let accel_bias = Vector3::new(x[ba_start], x[ba_start + 1], x[ba_start + 2]);

    // Rwg unchanged
    let g_inertial = Vector3::new(0.0, 0.0, -GRAVITY_MAGNITUDE);
    let gravity_world = problem.initial_rwg * g_inertial;

    InertialInitOptimResult {
        velocities,
        gyro_bias,
        accel_bias,
        rwg: problem.initial_rwg,
        gravity_world,
        iterations: iter,
        initial_cost,
        final_cost: current_cost,
        converged,
    }
}

/// Scale/gravity refinement only (biases fixed).
///
/// Used when biases are already well-estimated and we want to refine gravity.
pub fn optimize_inertial_init_scale_refinement(
    problem: &InertialInitProblem,
    config: &InertialInitConfig,
) -> InertialInitOptimResult {
    let n = problem.kf_ids.len();
    let layout = StateLayout::new(n);
    let num_params = layout.total_params();

    // Initialize state vector
    let mut x = DVector::zeros(num_params);

    // Pack initial velocities
    for (i, kf_id) in problem.kf_ids.iter().enumerate() {
        let vel = problem.initial_velocities.get(kf_id).unwrap();
        let start = layout.vel_start(i);
        x[start] = vel.x;
        x[start + 1] = vel.y;
        x[start + 2] = vel.z;
    }

    // Pack biases (fixed)
    let bg_start = layout.gyro_bias_start();
    x[bg_start] = problem.initial_gyro_bias.x;
    x[bg_start + 1] = problem.initial_gyro_bias.y;
    x[bg_start + 2] = problem.initial_gyro_bias.z;

    let ba_start = layout.accel_bias_start();
    x[ba_start] = problem.initial_accel_bias.x;
    x[ba_start + 1] = problem.initial_accel_bias.y;
    x[ba_start + 2] = problem.initial_accel_bias.z;

    // Pack Rwg
    let rwg_start = layout.rwg_start();
    let rwg_aa = problem.initial_rwg.scaled_axis();
    x[rwg_start] = rwg_aa.x;
    x[rwg_start + 1] = rwg_aa.y;
    x[rwg_start + 2] = rwg_aa.z;

    // Compute initial cost
    let (initial_cost, _) = compute_cost_and_residuals(problem, &x, &layout, config, false, true);

    // LM optimization loop (biases fixed)
    let mut lambda = config.initial_lambda;
    let mut current_cost = initial_cost;
    let mut converged = false;
    let mut iter = 0;

    // Active params: velocities (n*3) + rwg (3), skip biases (6)
    let vel_params = n * 3;
    let active_params = vel_params + 3;

    for i in 0..config.max_iterations {
        iter = i + 1;

        let (cost, residuals) =
            compute_cost_and_residuals(problem, &x, &layout, config, false, true);
        let jacobian = compute_jacobian(problem, &x, &layout, config, false, true);

        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        // Extract active parts (velocities and rwg, skip biases)
        let mut h_lm = DMatrix::zeros(active_params, active_params);
        let mut jtr_eff = DVector::zeros(active_params);

        // Copy velocities
        for r in 0..vel_params {
            jtr_eff[r] = jtr[r];
            for c in 0..vel_params {
                h_lm[(r, c)] = jtj[(r, c)];
            }
        }

        // Copy rwg (skipping bias indices)
        let rwg_src = layout.rwg_start();
        for r in 0..3 {
            jtr_eff[vel_params + r] = jtr[rwg_src + r];
            for c in 0..vel_params {
                h_lm[(vel_params + r, c)] = jtj[(rwg_src + r, c)];
                h_lm[(c, vel_params + r)] = jtj[(c, rwg_src + r)];
            }
            for c in 0..3 {
                h_lm[(vel_params + r, vel_params + c)] = jtj[(rwg_src + r, rwg_src + c)];
            }
        }

        for j in 0..active_params {
            h_lm[(j, j)] += lambda * (h_lm[(j, j)].max(1e-10));
        }

        let dx_eff = match h_lm.clone().lu().solve(&(-&jtr_eff)) {
            Some(d) => d,
            None => {
                lambda *= config.lambda_increase;
                continue;
            }
        };

        // Full update (biases stay fixed)
        let mut dx = DVector::zeros(num_params);
        for j in 0..vel_params {
            dx[j] = dx_eff[j];
        }
        for j in 0..3 {
            dx[rwg_src + j] = dx_eff[vel_params + j];
        }

        let x_new = &x + &dx;
        let (new_cost, _) =
            compute_cost_and_residuals(problem, &x_new, &layout, config, false, true);

        if new_cost < cost {
            x = x_new;
            current_cost = new_cost;
            lambda *= config.lambda_decrease;
            lambda = lambda.max(config.min_lambda);

            if dx_eff.norm() < config.convergence_threshold {
                converged = true;
                break;
            }
        } else {
            lambda *= config.lambda_increase;
            if lambda > config.max_lambda {
                break;
            }
        }
    }

    // Unpack results
    let mut velocities = HashMap::new();
    for (i, kf_id) in problem.kf_ids.iter().enumerate() {
        let start = layout.vel_start(i);
        velocities.insert(*kf_id, Vector3::new(x[start], x[start + 1], x[start + 2]));
    }

    let rwg_start = layout.rwg_start();
    let rwg_aa = Vector3::new(x[rwg_start], x[rwg_start + 1], x[rwg_start + 2]);
    let rwg = UnitQuaternion::from_scaled_axis(rwg_aa);

    let g_inertial = Vector3::new(0.0, 0.0, -GRAVITY_MAGNITUDE);
    let gravity_world = rwg * g_inertial;

    InertialInitOptimResult {
        velocities,
        gyro_bias: problem.initial_gyro_bias,
        accel_bias: problem.initial_accel_bias,
        rwg,
        gravity_world,
        iterations: iter,
        initial_cost,
        final_cost: current_cost,
        converged,
    }
}

/// Compute cost and residuals for the optimization problem.
fn compute_cost_and_residuals(
    problem: &InertialInitProblem,
    x: &DVector<f64>,
    layout: &StateLayout,
    config: &InertialInitConfig,
    _fix_rwg: bool,
    _fix_bias: bool,
) -> (f64, DVector<f64>) {
    let n = problem.kf_ids.len();

    // Extract state
    let mut velocities = Vec::with_capacity(n);
    for i in 0..n {
        let start = layout.vel_start(i);
        velocities.push(Vector3::new(x[start], x[start + 1], x[start + 2]));
    }

    let bg_start = layout.gyro_bias_start();
    let gyro_bias = Vector3::new(x[bg_start], x[bg_start + 1], x[bg_start + 2]);

    let ba_start = layout.accel_bias_start();
    let accel_bias = Vector3::new(x[ba_start], x[ba_start + 1], x[ba_start + 2]);

    let rwg_start = layout.rwg_start();
    let rwg_aa = Vector3::new(x[rwg_start], x[rwg_start + 1], x[rwg_start + 2]);
    let rwg = UnitQuaternion::from_scaled_axis(rwg_aa);

    // Gravity in world frame
    let g_inertial = Vector3::new(0.0, 0.0, -GRAVITY_MAGNITUDE);
    let g_w = rwg * g_inertial;

    // Count residuals: 9 per IMU edge + 6 for bias prior (if enabled)
    let num_imu_residuals = (n - 1) * 9;
    let num_bias_residuals = if config.prior_gyro > 0.0 || config.prior_accel > 0.0 {
        6
    } else {
        0
    };
    let total_residuals = num_imu_residuals + num_bias_residuals;

    let mut residuals = DVector::zeros(total_residuals);
    let mut cost = 0.0;

    // IMU preintegration residuals
    // Create a temporary bias for using preintegration accessors
    let current_bias = ImuBias {
        gyro: gyro_bias,
        accel: accel_bias,
    };

    for i in 0..(n - 1) {
        let kf_i_id = problem.kf_ids[i];
        let kf_j_id = problem.kf_ids[i + 1];

        let pose_i = problem.kf_poses.get(&kf_i_id).unwrap();
        let pose_j = problem.kf_poses.get(&kf_j_id).unwrap();

        let vel_i = &velocities[i];
        let vel_j = &velocities[i + 1];

        let preint = &problem.imu_preint[i];
        let dt = preint.dt;

        if dt < 1e-6 {
            continue;
        }

        // Get bias-corrected preintegration values using the helper methods
        let delta_pos = preint.get_delta_position(&current_bias);
        let delta_vel = preint.get_delta_velocity(&current_bias);
        let delta_rot = preint.get_delta_rotation(&current_bias);

        // Expected position increment (with gravity)
        let expected_dp = pose_i.rotation * delta_pos + vel_i * dt + 0.5 * g_w * dt * dt;

        // Actual position increment
        let actual_dp = pose_j.translation - pose_i.translation;

        // Position residual
        let res_p = actual_dp - expected_dp;

        // Expected velocity increment
        let expected_dv = pose_i.rotation * delta_vel + g_w * dt;

        // Actual velocity increment
        let actual_dv = vel_j - vel_i;

        // Velocity residual
        let res_v = actual_dv - expected_dv;

        // Rotation residual (simplified - just use rotation error)
        let expected_rot = pose_i.rotation * delta_rot;
        let rot_error = expected_rot.inverse() * pose_j.rotation;
        let res_r = rot_error.scaled_axis();

        // Pack residuals
        let base_idx = i * 9;
        for j in 0..3 {
            residuals[base_idx + j] = res_p[j];
            residuals[base_idx + 3 + j] = res_v[j];
            residuals[base_idx + 6 + j] = res_r[j];
        }

        cost += res_p.norm_squared() + res_v.norm_squared() + res_r.norm_squared();
    }

    // Bias prior residuals
    if num_bias_residuals > 0 {
        let base_idx = num_imu_residuals;

        if config.prior_gyro > 0.0 {
            let sqrt_info = config.prior_gyro.sqrt();
            for j in 0..3 {
                residuals[base_idx + j] = sqrt_info * gyro_bias[j];
            }
            cost += config.prior_gyro * gyro_bias.norm_squared();
        }

        if config.prior_accel > 0.0 {
            let sqrt_info = config.prior_accel.sqrt();
            for j in 0..3 {
                residuals[base_idx + 3 + j] = sqrt_info * accel_bias[j];
            }
            cost += config.prior_accel * accel_bias.norm_squared();
        }
    }

    (cost, residuals)
}

/// Compute Jacobian matrix using numerical differentiation.
fn compute_jacobian(
    problem: &InertialInitProblem,
    x: &DVector<f64>,
    layout: &StateLayout,
    config: &InertialInitConfig,
    fix_rwg: bool,
    fix_bias: bool,
) -> DMatrix<f64> {
    let n = problem.kf_ids.len();
    let num_params = layout.total_params();

    let num_imu_residuals = (n - 1) * 9;
    let num_bias_residuals = if config.prior_gyro > 0.0 || config.prior_accel > 0.0 {
        6
    } else {
        0
    };
    let num_residuals = num_imu_residuals + num_bias_residuals;

    let mut jacobian = DMatrix::zeros(num_residuals, num_params);
    let eps = 1e-8;

    let (_, residuals_0) = compute_cost_and_residuals(problem, x, layout, config, fix_rwg, fix_bias);

    for i in 0..num_params {
        // Skip fixed parameters
        if fix_rwg && i >= layout.rwg_start() {
            continue;
        }
        if fix_bias && i >= layout.gyro_bias_start() && i < layout.rwg_start() {
            continue;
        }

        let mut x_plus = x.clone();
        x_plus[i] += eps;

        let (_, residuals_plus) =
            compute_cost_and_residuals(problem, &x_plus, layout, config, fix_rwg, fix_bias);

        for j in 0..num_residuals {
            jacobian[(j, i)] = (residuals_plus[j] - residuals_0[j]) / eps;
        }
    }

    jacobian
}

/// Apply inertial optimization results to the map.
pub fn apply_inertial_init_result(map: &mut Map, result: &InertialInitOptimResult) {
    // Update velocities
    for (kf_id, vel) in &result.velocities {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            kf.velocity = *vel;
        }
    }

    // Update biases on all keyframes
    let bias = ImuBias {
        gyro: result.gyro_bias,
        accel: result.accel_bias,
    };

    for kf_id in result.velocities.keys() {
        if let Some(kf) = map.get_keyframe_mut(*kf_id) {
            kf.imu_bias = bias.clone();
        }
    }

    info!(
        "Applied inertial init optimization: gyro_bias=[{:.6}, {:.6}, {:.6}], accel_bias=[{:.6}, {:.6}, {:.6}]",
        result.gyro_bias.x, result.gyro_bias.y, result.gyro_bias.z,
        result.accel_bias.x, result.accel_bias.y, result.accel_bias.z
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_phases() {
        let phase0 = InertialInitConfig::phase0();
        assert!(phase0.prior_gyro > 0.0);
        assert!(phase0.prior_accel > 0.0);

        let phase2 = InertialInitConfig::phase2();
        assert_eq!(phase2.prior_gyro, 0.0);
        assert_eq!(phase2.prior_accel, 0.0);
    }

    #[test]
    fn test_state_layout() {
        let layout = StateLayout::new(5);
        assert_eq!(layout.total_params(), 5 * 3 + 9); // 15 + 9 = 24
        assert_eq!(layout.vel_start(0), 0);
        assert_eq!(layout.vel_start(2), 6);
        assert_eq!(layout.gyro_bias_start(), 15);
        assert_eq!(layout.accel_bias_start(), 18);
        assert_eq!(layout.rwg_start(), 21);
    }
}
