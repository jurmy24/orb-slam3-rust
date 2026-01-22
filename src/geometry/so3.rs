//! SO(3) Lie group utilities for IMU preintegration.
//!
//! Provides skew-symmetric matrix construction and the right Jacobian Jr(φ)
//! following the convention in Forster et al. "IMU Preintegration on Manifold".

use nalgebra::{Matrix3, Vector3};

/// Small angle threshold for numerical stability.
const SMALL_ANGLE_THRESHOLD: f64 = 1e-6;

/// Constructs the skew-symmetric matrix [v]× such that [v]× u = v × u.
///
/// ```text
/// [v]× = |  0   -v_z   v_y |
///        |  v_z   0   -v_x |
///        | -v_y  v_x    0  |
/// ```
#[inline]
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

/// Computes the right Jacobian Jr(φ) of SO(3).
///
/// The right Jacobian relates the derivative of the exponential map to the
/// Lie algebra element:
///
/// ```text
/// Jr(φ) = I - (1 - cos|φ|)/|φ|² [φ]× + (|φ| - sin|φ|)/|φ|³ [φ]×²
/// ```
///
/// For small angles (|φ| < ε):
/// ```text
/// Jr(φ) ≈ I - 0.5 [φ]×
/// ```
pub fn right_jacobian_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();

    if theta < SMALL_ANGLE_THRESHOLD {
        // First-order approximation for small angles
        return Matrix3::identity() - 0.5 * skew(phi);
    }

    let theta_sq = theta * theta;
    let theta_cu = theta_sq * theta;
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    let skew_phi = skew(phi);
    let skew_phi_sq = skew_phi * skew_phi;

    // Jr(φ) = I - (1 - cos θ)/θ² [φ]× + (θ - sin θ)/θ³ [φ]×²
    Matrix3::identity()
        - ((1.0 - cos_theta) / theta_sq) * skew_phi
        + ((theta - sin_theta) / theta_cu) * skew_phi_sq
}

/// Computes the inverse of the right Jacobian Jr⁻¹(φ).
///
/// ```text
/// Jr⁻¹(φ) = I + 0.5 [φ]× + (1/|φ|² - (1 + cos|φ|)/(2|φ| sin|φ|)) [φ]×²
/// ```
///
/// For small angles:
/// ```text
/// Jr⁻¹(φ) ≈ I + 0.5 [φ]× + 1/12 [φ]×²
/// ```
pub fn right_jacobian_so3_inv(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta = phi.norm();

    if theta < SMALL_ANGLE_THRESHOLD {
        // Second-order approximation for small angles
        let skew_phi = skew(phi);
        return Matrix3::identity() + 0.5 * skew_phi + (1.0 / 12.0) * skew_phi * skew_phi;
    }

    let theta_sq = theta * theta;
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();

    let skew_phi = skew(phi);
    let skew_phi_sq = skew_phi * skew_phi;

    // Coefficient for [φ]×² term
    let coeff = 1.0 / theta_sq - (1.0 + cos_theta) / (2.0 * theta * sin_theta);

    Matrix3::identity() + 0.5 * skew_phi + coeff * skew_phi_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_skew_cross_product() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let u = Vector3::new(4.0, 5.0, 6.0);

        let cross_direct = v.cross(&u);
        let cross_skew = skew(&v) * u;

        assert_relative_eq!(cross_direct, cross_skew, epsilon = 1e-12);
    }

    #[test]
    fn test_skew_antisymmetric() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let skew_v = skew(&v);

        assert_relative_eq!(skew_v, -skew_v.transpose(), epsilon = 1e-12);
    }

    #[test]
    fn test_right_jacobian_identity_at_zero() {
        let phi = Vector3::zeros();
        let jr = right_jacobian_so3(&phi);

        assert_relative_eq!(jr, Matrix3::identity(), epsilon = 1e-10);
    }

    #[test]
    fn test_right_jacobian_inverse_identity_at_zero() {
        let phi = Vector3::zeros();
        let jr_inv = right_jacobian_so3_inv(&phi);

        assert_relative_eq!(jr_inv, Matrix3::identity(), epsilon = 1e-10);
    }

    #[test]
    fn test_right_jacobian_inverse_relationship() {
        // Jr(φ) * Jr⁻¹(φ) should equal I
        let phi = Vector3::new(0.1, 0.2, 0.3);
        let jr = right_jacobian_so3(&phi);
        let jr_inv = right_jacobian_so3_inv(&phi);

        let product = jr * jr_inv;
        assert_relative_eq!(product, Matrix3::identity(), epsilon = 1e-10);
    }

    #[test]
    fn test_right_jacobian_small_angle_consistency() {
        // Test that small angle formula matches general formula near transition
        let phi_small = Vector3::new(1e-7, 1e-7, 1e-7);
        let phi_medium = Vector3::new(1e-5, 1e-5, 1e-5);

        let jr_small = right_jacobian_so3(&phi_small);
        let jr_medium = right_jacobian_so3(&phi_medium);

        // Both should be very close to identity for such small angles
        assert_relative_eq!(jr_small, Matrix3::identity(), epsilon = 1e-5);
        assert_relative_eq!(jr_medium, Matrix3::identity(), epsilon = 1e-4);
    }
}
