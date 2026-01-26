//! Pose-only optimization with IMU constraints for tracking.
//!
//! Optimizes the current frame's pose, velocity, and biases while keeping
//! map point positions fixed. Uses IMU preintegration from the previous keyframe.
//!
//! This is called during tracking after initial pose estimation to refine
//! the result using IMU constraints.

use nalgebra::{DMatrix, DVector, Vector2, Vector3};
use tracing::debug;

use crate::geometry::SE3;
use crate::imu::{ImuBias, PreintegratedState};
use crate::tracking::frame::CameraModel;

use super::imu_factors::compute_imu_residual;

/// Configuration for pose inertial optimization.
pub struct PoseInertialConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Chi-squared threshold for mono observations (iteration 0).
    pub chi2_mono_init: f64,
    /// Chi-squared threshold for stereo observations (iteration 0).
    pub chi2_stereo_init: f64,
    /// Chi-squared threshold for mono observations (final).
    pub chi2_mono_final: f64,
    /// Chi-squared threshold for stereo observations (final).
    pub chi2_stereo_final: f64,
    /// Weight for IMU prior.
    pub imu_weight: f64,
}

impl Default for PoseInertialConfig {
    fn default() -> Self {
        Self {
            max_iterations: 4,
            chi2_mono_init: 12.0,   // 95% chi2 with 2 DOF, relaxed
            chi2_stereo_init: 15.6, // 95% chi2 with 3 DOF, relaxed
            chi2_mono_final: 5.991, // 95% chi2 with 2 DOF
            chi2_stereo_final: 7.815, // 95% chi2 with 3 DOF
            imu_weight: 1.0,
        }
    }
}

/// Result of pose inertial optimization.
#[derive(Debug)]
pub struct PoseInertialResult {
    /// Optimized pose (T_wc, camera-to-world).
    pub pose: SE3,
    /// Optimized velocity in world frame.
    pub velocity: Vector3<f64>,
    /// Optimized IMU bias.
    pub bias: ImuBias,
    /// Number of inlier observations.
    pub num_inliers: usize,
    /// Total observations.
    pub num_observations: usize,
    /// Number of iterations.
    pub iterations: usize,
}

/// An observation for pose optimization.
pub struct PoseObservation {
    /// Observed 2D pixel coordinates.
    pub uv: Vector2<f64>,
    /// 3D map point position in world frame.
    pub point_world: Vector3<f64>,
    /// Whether this is a stereo observation.
    pub is_stereo: bool,
    /// Observation index (for tracking inliers).
    pub index: usize,
}

/// Optimize frame pose with IMU constraints.
///
/// # Arguments
///
/// * `initial_pose` - Initial pose estimate (T_wc)
/// * `initial_velocity` - Initial velocity estimate (world frame)
/// * `initial_bias` - Initial bias estimate
/// * `prev_kf_pose` - Previous keyframe pose (T_wc)
/// * `prev_kf_velocity` - Previous keyframe velocity
/// * `prev_kf_bias` - Previous keyframe bias
/// * `preintegrated` - IMU preintegration from prev_kf to current frame
/// * `observations` - Visual observations
/// * `camera` - Camera model
/// * `config` - Optimization configuration
///
/// # Returns
///
/// Optimized pose, velocity, bias, and inlier count.
pub fn pose_inertial_optimization(
    initial_pose: &SE3,
    initial_velocity: &Vector3<f64>,
    initial_bias: &ImuBias,
    prev_kf_pose: &SE3,
    prev_kf_velocity: &Vector3<f64>,
    _prev_kf_bias: &ImuBias,
    preintegrated: &PreintegratedState,
    observations: &[PoseObservation],
    camera: &CameraModel,
    config: &PoseInertialConfig,
) -> PoseInertialResult {
    // State layout: [pose(6), velocity(3), gyro_bias(3), accel_bias(3)] = 15 params
    const NUM_PARAMS: usize = 15;

    // Initialize parameters
    let mut params = DVector::zeros(NUM_PARAMS);

    // Pose as axis-angle + translation
    let rot = initial_pose.rotation.scaled_axis();
    params[0] = rot.x;
    params[1] = rot.y;
    params[2] = rot.z;
    params[3] = initial_pose.translation.x;
    params[4] = initial_pose.translation.y;
    params[5] = initial_pose.translation.z;

    // Velocity
    params[6] = initial_velocity.x;
    params[7] = initial_velocity.y;
    params[8] = initial_velocity.z;

    // Biases
    params[9] = initial_bias.gyro.x;
    params[10] = initial_bias.gyro.y;
    params[11] = initial_bias.gyro.z;
    params[12] = initial_bias.accel.x;
    params[13] = initial_bias.accel.y;
    params[14] = initial_bias.accel.z;

    // Track inliers
    let mut inlier_mask = vec![true; observations.len()];
    let mut iterations = 0;

    // Iterative optimization with decreasing thresholds
    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Interpolate chi2 thresholds
        let progress = iter as f64 / (config.max_iterations - 1).max(1) as f64;
        let chi2_mono = config.chi2_mono_init * (1.0 - progress) + config.chi2_mono_final * progress;
        let chi2_stereo = config.chi2_stereo_init * (1.0 - progress) + config.chi2_stereo_final * progress;

        // Compute residuals and Jacobian for inliers only
        let (residuals, jacobian, num_active) = compute_pose_residuals_and_jacobian(
            &params,
            prev_kf_pose,
            prev_kf_velocity,
            preintegrated,
            observations,
            &inlier_mask,
            camera,
            config,
        );

        if num_active < 5 {
            // Too few observations
            break;
        }

        // Solve normal equations
        let gradient = jacobian.transpose() * &residuals;
        let jtj = jacobian.transpose() * &jacobian;

        // Add damping
        let lambda = 1e-3;
        let mut damped_jtj = jtj.clone();
        for i in 0..NUM_PARAMS {
            damped_jtj[(i, i)] += lambda * damped_jtj[(i, i)].max(1e-6);
        }

        let delta = match damped_jtj.lu().solve(&(-&gradient)) {
            Some(d) => d,
            None => break,
        };

        // Update parameters
        params += delta;

        // Update inlier mask based on chi2
        let pose = extract_pose(&params);
        for (i, obs) in observations.iter().enumerate() {
            let err = compute_reprojection_error(&pose, &obs.point_world, &obs.uv, camera);
            let chi2 = err.norm_squared();
            let threshold = if obs.is_stereo { chi2_stereo } else { chi2_mono };
            inlier_mask[i] = chi2 < threshold;
        }
    }

    // Extract final state
    let final_pose = extract_pose(&params);
    let final_velocity = Vector3::new(params[6], params[7], params[8]);
    let final_bias = ImuBias {
        gyro: Vector3::new(params[9], params[10], params[11]),
        accel: Vector3::new(params[12], params[13], params[14]),
    };

    let num_inliers = inlier_mask.iter().filter(|&&x| x).count();

    debug!(
        "[PoseInertialOpt] {} iters, inliers: {}/{}",
        iterations, num_inliers, observations.len()
    );

    PoseInertialResult {
        pose: final_pose,
        velocity: final_velocity,
        bias: final_bias,
        num_inliers,
        num_observations: observations.len(),
        iterations,
    }
}

/// Extract pose from parameter vector.
fn extract_pose(params: &DVector<f64>) -> SE3 {
    let rot = Vector3::new(params[0], params[1], params[2]);
    let trans = Vector3::new(params[3], params[4], params[5]);
    let rotation = nalgebra::UnitQuaternion::from_scaled_axis(rot);
    SE3 { rotation, translation: trans }
}

/// Compute reprojection error.
fn compute_reprojection_error(
    pose_wc: &SE3,
    point_world: &Vector3<f64>,
    observed: &Vector2<f64>,
    camera: &CameraModel,
) -> Vector2<f64> {
    let pose_cw = pose_wc.inverse();
    let p_cam = pose_cw.transform_point(point_world);

    if p_cam.z <= 0.001 {
        return Vector2::new(100.0, 100.0);
    }

    let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
    let v = camera.fy * p_cam.y / p_cam.z + camera.cy;

    Vector2::new(observed.x - u, observed.y - v)
}

/// Compute reprojection error and analytical Jacobian w.r.t. pose.
///
/// Returns (error, jacobian) where jacobian is 2x6 matrix.
/// Jacobian layout: [d/d_rot (3), d/d_trans (3)]
fn compute_reprojection_error_with_jacobian(
    pose_wc: &SE3,
    point_world: &Vector3<f64>,
    observed: &Vector2<f64>,
    camera: &CameraModel,
) -> (Vector2<f64>, nalgebra::SMatrix<f64, 2, 6>) {
    use nalgebra::SMatrix;

    let pose_cw = pose_wc.inverse();
    let p_cam = pose_cw.transform_point(point_world);

    // Handle degenerate case
    if p_cam.z <= 0.001 {
        let error = Vector2::new(100.0, 100.0);
        let jacobian = SMatrix::<f64, 2, 6>::zeros();
        return (error, jacobian);
    }

    let x = p_cam.x;
    let y = p_cam.y;
    let z = p_cam.z;
    let z_inv = 1.0 / z;
    let z_inv_sq = z_inv * z_inv;

    // Projected point
    let u_proj = camera.fx * x * z_inv + camera.cx;
    let v_proj = camera.fy * y * z_inv + camera.cy;
    let error = Vector2::new(observed.x - u_proj, observed.y - v_proj);

    // Jacobian of projection w.r.t. camera-frame point
    // d(u)/d(x) = fx/z, d(u)/d(y) = 0, d(u)/d(z) = -fx*x/z^2
    // d(v)/d(x) = 0, d(v)/d(y) = fy/z, d(v)/d(z) = -fy*y/z^2
    let du_dx = camera.fx * z_inv;
    let du_dz = -camera.fx * x * z_inv_sq;
    let dv_dy = camera.fy * z_inv;
    let dv_dz = -camera.fy * y * z_inv_sq;

    // Jacobian of camera-frame point w.r.t. pose (T_wc)
    // p_cam = R_cw * (p_world - t_wc)
    // where T_wc = (R_wc, t_wc), and R_cw = R_wc^T
    //
    // For small perturbation delta = [delta_rot, delta_trans] to T_wc:
    // d(p_cam)/d(delta_rot) = -R_cw * skew(p_world - t_wc) ≈ -skew(p_cam)
    // d(p_cam)/d(delta_trans) = -R_cw
    //
    // Since error = observed - projected, and we want d(error)/d(pose):
    // d(error)/d(pose) = -d(projected)/d(p_cam) * d(p_cam)/d(pose)
    //
    // The negative sign makes the Jacobian point toward the solution.

    // Jacobian: d(proj)/d(p_cam) is 2x3
    // d(p_cam)/d(rot) uses skew(p_cam) relation
    // For axis-angle parameterization: d(p_cam)/d(delta_rot) ≈ skew(p_cam) (left perturbation on R_cw)
    // But since we perturb T_wc, we need:
    // d(p_cam)/d(delta_rot_wc) = -skew(p_cam) (approximately, for small rotations)
    // d(p_cam)/d(delta_trans_wc) = -R_cw

    // For the reprojection error (obs - proj), the Jacobian w.r.t. pose is:
    // J = -d(proj)/d(p_cam) * d(p_cam)/d(pose)
    // Since error = obs - proj, d(error)/d(pose) = -d(proj)/d(pose)

    // Using standard formulas for pinhole camera Jacobian w.r.t. SE3 pose:
    // J_rot = [fx*xy/z^2,     -(fx + fx*x^2/z^2), fx*y/z,
    //          fy + fy*y^2/z^2, -fy*xy/z^2,        -fy*x/z]
    // J_trans = [-fx/z,  0,     fx*x/z^2,
    //             0,    -fy/z,  fy*y/z^2]
    //
    // Note: Sign convention depends on error definition. Since error = obs - proj,
    // we need d(obs - proj)/d(pose) = -d(proj)/d(pose)

    let xy = x * y;
    let x_sq = x * x;
    let y_sq = y * y;

    // Jacobian of error w.r.t. pose perturbation [rot(3), trans(3)]
    // For error = observed - projected, the Jacobian has opposite sign to projected's Jacobian
    let mut jacobian = SMatrix::<f64, 2, 6>::zeros();

    // d(error_u)/d(rot)
    jacobian[(0, 0)] = -camera.fx * xy * z_inv_sq;
    jacobian[(0, 1)] = camera.fx * (1.0 + x_sq * z_inv_sq);
    jacobian[(0, 2)] = -camera.fx * y * z_inv;

    // d(error_v)/d(rot)
    jacobian[(1, 0)] = -camera.fy * (1.0 + y_sq * z_inv_sq);
    jacobian[(1, 1)] = camera.fy * xy * z_inv_sq;
    jacobian[(1, 2)] = camera.fy * x * z_inv;

    // d(error_u)/d(trans)
    jacobian[(0, 3)] = du_dx;
    jacobian[(0, 4)] = 0.0;
    jacobian[(0, 5)] = du_dz;

    // d(error_v)/d(trans)
    jacobian[(1, 3)] = 0.0;
    jacobian[(1, 4)] = dv_dy;
    jacobian[(1, 5)] = dv_dz;

    (error, jacobian)
}

/// Compute residuals and Jacobian for pose optimization.
fn compute_pose_residuals_and_jacobian(
    params: &DVector<f64>,
    prev_kf_pose: &SE3,
    prev_kf_velocity: &Vector3<f64>,
    preintegrated: &PreintegratedState,
    observations: &[PoseObservation],
    inlier_mask: &[bool],
    camera: &CameraModel,
    config: &PoseInertialConfig,
) -> (DVector<f64>, DMatrix<f64>, usize) {
    const NUM_PARAMS: usize = 15;

    // Count active observations
    let num_active: usize = inlier_mask.iter().filter(|&&x| x).count();
    let num_visual_residuals = num_active * 2;
    let num_imu_residuals = 9;
    let total_residuals = num_visual_residuals + num_imu_residuals;

    let mut residuals = DVector::zeros(total_residuals);
    let mut jacobian = DMatrix::zeros(total_residuals, NUM_PARAMS);

    let pose = extract_pose(params);
    let velocity = Vector3::new(params[6], params[7], params[8]);

    let eps = 1e-6;

    // Visual residuals with analytical Jacobians (5-10x faster than numerical)
    let mut res_idx = 0;
    for (i, obs) in observations.iter().enumerate() {
        if !inlier_mask[i] {
            continue;
        }

        // Use analytical Jacobian for visual residuals
        let (err, vis_jacobian) = compute_reprojection_error_with_jacobian(
            &pose,
            &obs.point_world,
            &obs.uv,
            camera,
        );
        residuals[res_idx] = err.x;
        residuals[res_idx + 1] = err.y;

        // Copy analytical Jacobian for pose parameters (first 6 columns)
        for j in 0..6 {
            jacobian[(res_idx, j)] = vis_jacobian[(0, j)];
            jacobian[(res_idx + 1, j)] = vis_jacobian[(1, j)];
        }
        // Visual residuals don't depend on velocity/bias (columns 6-14 remain zero)

        res_idx += 2;
    }

    // IMU residuals
    let imu_res = compute_imu_residual(prev_kf_pose, prev_kf_velocity, &pose, &velocity, preintegrated);
    let imu_vec = imu_res.as_vector();

    for k in 0..9 {
        residuals[res_idx + k] = imu_vec[k] * config.imu_weight;
    }

    // Numerical Jacobian for IMU residuals w.r.t. current frame state
    for j in 0..NUM_PARAMS {
        let mut params_plus = params.clone();
        params_plus[j] += eps;

        let pose_plus = extract_pose(&params_plus);
        let vel_plus = Vector3::new(params_plus[6], params_plus[7], params_plus[8]);

        let imu_res_plus = compute_imu_residual(prev_kf_pose, prev_kf_velocity, &pose_plus, &vel_plus, preintegrated);
        let imu_vec_plus = imu_res_plus.as_vector();

        for k in 0..9 {
            jacobian[(res_idx + k, j)] = (imu_vec_plus[k] - imu_vec[k]) / eps * config.imu_weight;
        }
    }

    (residuals, jacobian, num_active)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::PreintegratedState;

    #[test]
    fn test_pose_extraction() {
        let mut params = DVector::zeros(15);
        params[0] = 0.1;
        params[1] = 0.2;
        params[2] = 0.3;
        params[3] = 1.0;
        params[4] = 2.0;
        params[5] = 3.0;

        let pose = extract_pose(&params);

        assert!((pose.translation.x - 1.0).abs() < 1e-10);
        assert!((pose.translation.y - 2.0).abs() < 1e-10);
        assert!((pose.translation.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pose_inertial_optimization_no_observations() {
        let pose = SE3::identity();
        let velocity = Vector3::zeros();
        let bias = ImuBias::zero();
        let preint = PreintegratedState::identity();
        let observations = vec![];
        let camera = CameraModel {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            baseline: 0.1,
        };
        let config = PoseInertialConfig::default();

        let result = pose_inertial_optimization(
            &pose, &velocity, &bias,
            &pose, &velocity, &bias,
            &preint, &observations, &camera, &config,
        );

        // Should complete without crashing
        assert_eq!(result.num_observations, 0);
    }
}
