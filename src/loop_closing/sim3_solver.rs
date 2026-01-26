//! Sim3 solver using Horn's method with RANSAC.
//!
//! Computes similarity transformation between two sets of 3D point correspondences.
//! For stereo mode, scale is fixed to 1.0.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use rand::prelude::*;

use crate::geometry::Sim3;

/// Configuration for Sim3 RANSAC solver.
#[derive(Debug, Clone)]
pub struct Sim3SolverConfig {
    /// Maximum number of RANSAC iterations.
    pub max_iterations: usize,
    /// Inlier threshold in meters (point-to-point error).
    pub inlier_threshold: f64,
    /// Minimum number of inliers required.
    pub min_inliers: usize,
    /// Fix scale to 1.0 (true for stereo mode).
    pub fix_scale: bool,
    /// Probability of finding a good model.
    pub probability: f64,
}

impl Default for Sim3SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 300,
            inlier_threshold: 0.075, // 7.5cm
            min_inliers: 15,
            fix_scale: true, // Default to stereo mode
            probability: 0.99,
        }
    }
}

/// Result from Sim3 RANSAC solver.
#[derive(Debug, Clone)]
pub struct Sim3Result {
    /// The computed similarity transformation.
    pub sim3: Sim3,
    /// Indices of inlier correspondences.
    pub inliers: Vec<usize>,
    /// Number of inliers.
    pub num_inliers: usize,
    /// Transformation error (mean squared error of inliers).
    pub mse: f64,
}

/// Compute Sim3 transformation using Horn's method with RANSAC.
///
/// Finds the similarity transformation S such that: points2 ≈ S * points1
///
/// # Arguments
/// * `points1` - Source 3D points (in coordinate frame 1)
/// * `points2` - Target 3D points (in coordinate frame 2)
/// * `config` - RANSAC configuration
///
/// # Returns
/// * `Some(Sim3Result)` if a valid transformation is found
/// * `None` if RANSAC fails to find enough inliers
pub fn compute_sim3_ransac(
    points1: &[Vector3<f64>],
    points2: &[Vector3<f64>],
    config: &Sim3SolverConfig,
) -> Option<Sim3Result> {
    let n = points1.len();
    if n < 3 || n != points2.len() {
        return None;
    }

    if n < config.min_inliers {
        return None;
    }

    let mut rng = rand::thread_rng();
    let mut best_result: Option<Sim3Result> = None;
    let mut best_inliers = 0;

    // Adaptive number of iterations based on inlier ratio
    let mut max_iter = config.max_iterations;

    for iteration in 0..max_iter {
        // Sample 3 random correspondences
        let indices = sample_three_indices(&mut rng, n);

        let sample_pts1: Vec<_> = indices.iter().map(|&i| points1[i]).collect();
        let sample_pts2: Vec<_> = indices.iter().map(|&i| points2[i]).collect();

        // Compute Sim3 from minimal sample using Horn's method
        let sim3 = match compute_sim3_horn(&sample_pts1, &sample_pts2, config.fix_scale) {
            Some(s) => s,
            None => continue,
        };

        // Count inliers
        let (inliers, mse) = find_inliers(points1, points2, &sim3, config.inlier_threshold);

        if inliers.len() > best_inliers {
            best_inliers = inliers.len();
            best_result = Some(Sim3Result {
                sim3,
                num_inliers: inliers.len(),
                inliers,
                mse,
            });

            // Update adaptive iteration count
            if best_inliers >= config.min_inliers {
                let inlier_ratio = best_inliers as f64 / n as f64;
                let updated_iter = compute_adaptive_iterations(inlier_ratio, config.probability, 3);
                max_iter = max_iter.min(iteration + updated_iter);
            }
        }
    }

    // Refine with all inliers if we have a good result
    if let Some(ref mut result) = best_result {
        if result.num_inliers >= config.min_inliers {
            // Collect inlier points
            let inlier_pts1: Vec<_> = result.inliers.iter().map(|&i| points1[i]).collect();
            let inlier_pts2: Vec<_> = result.inliers.iter().map(|&i| points2[i]).collect();

            // Refine Sim3 using all inliers
            if let Some(refined_sim3) =
                compute_sim3_horn(&inlier_pts1, &inlier_pts2, config.fix_scale)
            {
                // Recompute inliers with refined model
                let (new_inliers, new_mse) =
                    find_inliers(points1, points2, &refined_sim3, config.inlier_threshold);

                if new_inliers.len() >= result.num_inliers {
                    result.sim3 = refined_sim3;
                    result.inliers = new_inliers;
                    result.num_inliers = result.inliers.len();
                    result.mse = new_mse;
                }
            }
        }
    }

    // Return only if we meet minimum inlier requirement
    best_result.filter(|r| r.num_inliers >= config.min_inliers)
}

/// Compute Sim3 using Horn's method (closed-form solution).
///
/// Algorithm:
/// 1. Compute centroids of both point sets
/// 2. Center the points
/// 3. Compute scale (if not fixed): s = sqrt(sum(||p2||²) / sum(||p1||²))
/// 4. Compute rotation via SVD of cross-covariance matrix
/// 5. Compute translation: t = c2 - s * R * c1
///
/// Reference: B.K.P. Horn, "Closed-form solution of absolute orientation using unit quaternions"
fn compute_sim3_horn(
    points1: &[Vector3<f64>],
    points2: &[Vector3<f64>],
    fix_scale: bool,
) -> Option<Sim3> {
    let n = points1.len();
    if n < 3 {
        return None;
    }

    // Step 1: Compute centroids
    let centroid1 = compute_centroid(points1);
    let centroid2 = compute_centroid(points2);

    // Step 2: Center the points
    let centered1: Vec<_> = points1.iter().map(|p| p - centroid1).collect();
    let centered2: Vec<_> = points2.iter().map(|p| p - centroid2).collect();

    // Step 3: Compute scale
    let scale = if fix_scale {
        1.0
    } else {
        // s = sqrt(sum(||p2||²) / sum(||p1||²))
        let sum_sq1: f64 = centered1.iter().map(|p| p.norm_squared()).sum();
        let sum_sq2: f64 = centered2.iter().map(|p| p.norm_squared()).sum();

        if sum_sq1 < 1e-10 {
            return None;
        }
        (sum_sq2 / sum_sq1).sqrt()
    };

    // Step 4: Compute rotation via SVD
    // Cross-covariance matrix: H = sum(p1_i * p2_i^T)
    let mut h = Matrix3::zeros();
    for i in 0..n {
        h += centered1[i] * centered2[i].transpose();
    }

    // SVD: H = U * S * V^T
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;

    // R = V * U^T
    let mut rotation_mat = v_t.transpose() * u.transpose();

    // Handle reflection case (det(R) = -1)
    if rotation_mat.determinant() < 0.0 {
        // Flip sign of last column of V
        let mut v = v_t.transpose();
        for i in 0..3 {
            v[(i, 2)] = -v[(i, 2)];
        }
        rotation_mat = v * u.transpose();
    }

    let rotation = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(
        rotation_mat,
    ));

    // Step 5: Compute translation
    // t = c2 - s * R * c1
    let translation = centroid2 - scale * (rotation * centroid1);

    Some(Sim3 {
        rotation,
        translation,
        scale,
    })
}

/// Compute centroid of a set of 3D points.
fn compute_centroid(points: &[Vector3<f64>]) -> Vector3<f64> {
    if points.is_empty() {
        return Vector3::zeros();
    }
    let sum: Vector3<f64> = points.iter().sum();
    sum / points.len() as f64
}

/// Find inliers for a given Sim3 transformation.
fn find_inliers(
    points1: &[Vector3<f64>],
    points2: &[Vector3<f64>],
    sim3: &Sim3,
    threshold: f64,
) -> (Vec<usize>, f64) {
    let threshold_sq = threshold * threshold;
    let mut inliers = Vec::new();
    let mut sum_sq_error = 0.0;

    for (i, (p1, p2)) in points1.iter().zip(points2.iter()).enumerate() {
        let p1_transformed = sim3.transform_point(p1);
        let error_sq = (p1_transformed - p2).norm_squared();

        if error_sq < threshold_sq {
            inliers.push(i);
            sum_sq_error += error_sq;
        }
    }

    let mse = if inliers.is_empty() {
        f64::INFINITY
    } else {
        sum_sq_error / inliers.len() as f64
    };

    (inliers, mse)
}

/// Sample three unique random indices.
fn sample_three_indices(rng: &mut impl Rng, n: usize) -> [usize; 3] {
    let mut indices = [0usize; 3];
    indices[0] = rng.gen_range(0..n);

    loop {
        indices[1] = rng.gen_range(0..n);
        if indices[1] != indices[0] {
            break;
        }
    }

    loop {
        indices[2] = rng.gen_range(0..n);
        if indices[2] != indices[0] && indices[2] != indices[1] {
            break;
        }
    }

    indices
}

/// Compute adaptive number of RANSAC iterations.
fn compute_adaptive_iterations(inlier_ratio: f64, probability: f64, sample_size: usize) -> usize {
    if inlier_ratio <= 0.0 {
        return usize::MAX;
    }
    if inlier_ratio >= 1.0 {
        return 1;
    }

    // k = log(1 - p) / log(1 - w^n)
    // where w = inlier ratio, n = sample size, p = desired probability
    let w_n = inlier_ratio.powi(sample_size as i32);
    let log_denom = (1.0 - w_n).ln();

    if log_denom.abs() < 1e-10 {
        return 1;
    }

    let k = (1.0 - probability).ln() / log_denom;
    (k.ceil() as usize).max(1)
}

/// Compute Sim3 from matched map points (convenience function for loop closing).
///
/// This is the main entry point used by the loop closing module.
/// Takes matched 3D points from two keyframes and computes the relative Sim3.
pub fn compute_sim3_from_matches(
    current_points: &[Vector3<f64>],
    loop_points: &[Vector3<f64>],
    fix_scale: bool,
) -> Option<Sim3Result> {
    let config = Sim3SolverConfig {
        fix_scale,
        ..Default::default()
    };
    compute_sim3_ransac(current_points, loop_points, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_horn_identity() {
        // Same points should give identity transform
        let points: Vec<_> = (0..10)
            .map(|i| Vector3::new(i as f64, (i * 2) as f64, (i * 3) as f64))
            .collect();

        let sim3 = compute_sim3_horn(&points, &points, true).unwrap();

        assert_relative_eq!(sim3.scale, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sim3.translation.norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_horn_pure_translation() {
        let points1: Vec<_> = (0..10)
            .map(|i| Vector3::new(i as f64, (i * 2) as f64, (i * 3) as f64))
            .collect();

        let translation = Vector3::new(5.0, -3.0, 2.0);
        let points2: Vec<_> = points1.iter().map(|p| p + translation).collect();

        let sim3 = compute_sim3_horn(&points1, &points2, true).unwrap();

        assert_relative_eq!(sim3.scale, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sim3.translation, translation, epsilon = 1e-10);
    }

    #[test]
    fn test_horn_rotation() {
        let points1: Vec<_> = (0..10)
            .map(|i| Vector3::new((i + 1) as f64, ((i + 1) * 2) as f64, ((i + 1) * 3) as f64))
            .collect();

        // 90 degree rotation around Z axis
        let rotation = UnitQuaternion::from_axis_angle(
            &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            std::f64::consts::FRAC_PI_2,
        );

        let points2: Vec<_> = points1.iter().map(|p| rotation * p).collect();

        let sim3 = compute_sim3_horn(&points1, &points2, true).unwrap();

        assert_relative_eq!(sim3.scale, 1.0, epsilon = 1e-10);

        // Check that the rotation is correct by transforming points
        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let p1_transformed = sim3.transform_point(p1);
            assert_relative_eq!(p1_transformed, *p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_horn_with_scale() {
        let points1: Vec<_> = (0..10)
            .map(|i| Vector3::new((i + 1) as f64, ((i + 1) * 2) as f64, ((i + 1) * 3) as f64))
            .collect();

        let scale = 2.5;
        let points2: Vec<_> = points1.iter().map(|p| p * scale).collect();

        // With fix_scale = false
        let sim3 = compute_sim3_horn(&points1, &points2, false).unwrap();

        assert_relative_eq!(sim3.scale, scale, epsilon = 1e-10);

        // Check transformation
        for (p1, p2) in points1.iter().zip(points2.iter()) {
            let p1_transformed = sim3.transform_point(p1);
            assert_relative_eq!(p1_transformed, *p2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ransac_with_outliers() {
        let mut rng = rand::thread_rng();

        // Generate inlier correspondences (simple translation)
        let translation = Vector3::new(1.0, 2.0, 3.0);
        let n_inliers = 50;
        let n_outliers = 10;

        let mut points1 = Vec::new();
        let mut points2 = Vec::new();

        // Add inliers
        for _ in 0..n_inliers {
            let p1 = Vector3::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            );
            let p2 = p1 + translation;
            points1.push(p1);
            points2.push(p2);
        }

        // Add outliers
        for _ in 0..n_outliers {
            let p1 = Vector3::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            );
            let p2 = Vector3::new(
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
                rng.gen_range(-10.0..10.0),
            );
            points1.push(p1);
            points2.push(p2);
        }

        let config = Sim3SolverConfig {
            fix_scale: true,
            min_inliers: 20,
            ..Default::default()
        };

        let result = compute_sim3_ransac(&points1, &points2, &config).unwrap();

        // Should find most inliers
        assert!(result.num_inliers >= n_inliers - 5);
        assert_relative_eq!(result.sim3.translation, translation, epsilon = 0.1);
    }

    #[test]
    fn test_ransac_insufficient_points() {
        let points1 = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)];
        let points2 = vec![Vector3::new(1.0, 2.0, 3.0), Vector3::new(4.0, 5.0, 6.0)];

        let config = Sim3SolverConfig::default();
        let result = compute_sim3_ransac(&points1, &points2, &config);

        assert!(result.is_none());
    }
}
