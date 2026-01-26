//! Sim3: 7-DOF similarity transformation (rotation + translation + scale).
//!
//! For stereo/stereo-inertial SLAM, scale is typically fixed to 1.0 since depth
//! is observable from the stereo baseline. This implementation follows ORB-SLAM3's
//! approach of using Sim3 with a `fix_scale` flag.

use nalgebra::{Matrix3, Matrix4, Rotation3, UnitQuaternion, Vector3};

use super::SE3;

/// 7-DOF Similarity transformation: rotation + translation + scale.
///
/// Transforms points as: p' = s * R * p + t
///
/// For stereo mode, scale is always 1.0 (equivalent to SE3).
#[derive(Debug, Clone, PartialEq)]
pub struct Sim3 {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
    pub scale: f64,
}

impl Sim3 {
    /// Identity transformation (no rotation, no translation, scale = 1).
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    /// Construct from rotation matrix, translation, and scale.
    pub fn from_rts(rotation: Matrix3<f64>, translation: Vector3<f64>, scale: f64) -> Self {
        let rot3 = Rotation3::from_matrix_unchecked(rotation);
        Self {
            rotation: UnitQuaternion::from_rotation_matrix(&rot3),
            translation,
            scale,
        }
    }

    /// Construct from SE3 with scale = 1.0.
    pub fn from_se3(se3: &SE3) -> Self {
        Self {
            rotation: se3.rotation,
            translation: se3.translation,
            scale: 1.0,
        }
    }

    /// Construct from SE3 with explicit scale.
    pub fn from_se3_with_scale(se3: &SE3, scale: f64) -> Self {
        Self {
            rotation: se3.rotation,
            translation: se3.translation,
            scale,
        }
    }

    /// Construct from quaternion (w, x, y, z), translation, and scale.
    pub fn from_quaternion(
        qw: f64,
        qx: f64,
        qy: f64,
        qz: f64,
        translation: Vector3<f64>,
        scale: f64,
    ) -> Self {
        let rotation =
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
        Self {
            rotation,
            translation,
            scale,
        }
    }

    /// Construct from homogeneous 4x4 matrix of form [s*R | t; 0 | 1].
    ///
    /// The scale is extracted from the rotation block's norm.
    pub fn from_matrix(mat: Matrix4<f64>) -> Self {
        let sr_block = mat.fixed_view::<3, 3>(0, 0).into_owned();
        let translation = Vector3::new(mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]);

        // Extract scale from first column norm (assuming uniform scaling)
        let scale = sr_block.column(0).norm();

        // Extract rotation by dividing by scale
        let rotation_mat = if scale > 1e-10 {
            sr_block / scale
        } else {
            Matrix3::identity()
        };

        let rot3 = Rotation3::from_matrix_unchecked(rotation_mat);
        Self {
            rotation: UnitQuaternion::from_rotation_matrix(&rot3),
            translation,
            scale,
        }
    }

    /// Convert to SE3, ignoring scale.
    ///
    /// Note: This drops scale information. For stereo mode where scale = 1.0,
    /// this is lossless.
    pub fn to_se3(&self) -> SE3 {
        SE3 {
            rotation: self.rotation,
            translation: self.translation,
        }
    }

    /// Convert to homogeneous 4x4 matrix of form [s*R | t; 0 0 0 1].
    pub fn to_matrix(&self) -> Matrix4<f64> {
        let rot_mat = self.rotation.to_rotation_matrix().into_inner();
        let sr = rot_mat * self.scale;

        Matrix4::new(
            sr[(0, 0)],
            sr[(0, 1)],
            sr[(0, 2)],
            self.translation.x,
            sr[(1, 0)],
            sr[(1, 1)],
            sr[(1, 2)],
            self.translation.y,
            sr[(2, 0)],
            sr[(2, 1)],
            sr[(2, 2)],
            self.translation.z,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Inverse transformation.
    ///
    /// For Sim3: S = [sR | t; 0 | 1]
    /// Inverse: S^{-1} = [(1/s)R^T | -(1/s)R^T*t; 0 | 1]
    pub fn inverse(&self) -> Self {
        let rot_inv = self.rotation.inverse();
        let scale_inv = 1.0 / self.scale;
        let t_inv = -(rot_inv * self.translation) * scale_inv;
        Self {
            rotation: rot_inv,
            translation: t_inv,
            scale: scale_inv,
        }
    }

    /// Compose two Sim3 transforms: self ∘ other.
    ///
    /// For S1 = [s1*R1 | t1] and S2 = [s2*R2 | t2]:
    /// S1 ∘ S2 = [s1*s2*R1*R2 | s1*R1*t2 + t1]
    pub fn compose(&self, other: &Sim3) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
            translation: self.scale * (self.rotation * other.translation) + self.translation,
            scale: self.scale * other.scale,
        }
    }

    /// Transform a single point: p' = s * R * p + t.
    pub fn transform_point(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.scale * (self.rotation * p) + self.translation
    }

    /// Transform multiple points.
    pub fn transform_points(&self, pts: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        pts.iter().map(|p| self.transform_point(p)).collect()
    }

    /// Get the rotation matrix (without scale).
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        self.rotation.to_rotation_matrix().into_inner()
    }

    /// Get the scaled rotation matrix (s * R).
    pub fn scaled_rotation_matrix(&self) -> Matrix3<f64> {
        self.rotation.to_rotation_matrix().into_inner() * self.scale
    }

    /// Check if this is essentially an SE3 transform (scale ≈ 1.0).
    pub fn is_se3(&self, tolerance: f64) -> bool {
        (self.scale - 1.0).abs() < tolerance
    }

    /// Log map: Convert to 7-element vector [rotation_vec (3), translation (3), log_scale (1)].
    ///
    /// This is useful for optimization where we want to work in the tangent space.
    pub fn log(&self) -> [f64; 7] {
        let rot_vec = self.rotation.scaled_axis();
        [
            rot_vec.x,
            rot_vec.y,
            rot_vec.z,
            self.translation.x,
            self.translation.y,
            self.translation.z,
            self.scale.ln(),
        ]
    }

    /// Exponential map: Construct from 7-element tangent vector.
    pub fn exp(tangent: &[f64; 7]) -> Self {
        let rot_vec = Vector3::new(tangent[0], tangent[1], tangent[2]);
        let rotation = UnitQuaternion::from_scaled_axis(rot_vec);
        let translation = Vector3::new(tangent[3], tangent[4], tangent[5]);
        let scale = tangent[6].exp();
        Self {
            rotation,
            translation,
            scale,
        }
    }

    /// Apply a small update in the tangent space.
    ///
    /// This is used during optimization to update the current estimate.
    pub fn retract(&self, delta: &[f64; 7]) -> Self {
        let delta_sim3 = Self::exp(delta);
        self.compose(&delta_sim3)
    }
}

impl Default for Sim3 {
    fn default() -> Self {
        Self::identity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let sim3 = Sim3::identity();
        assert_eq!(sim3.scale, 1.0);
        assert_eq!(sim3.translation, Vector3::zeros());

        let p = Vector3::new(1.0, 2.0, 3.0);
        let p_transformed = sim3.transform_point(&p);
        assert_relative_eq!(p_transformed, p, epsilon = 1e-10);
    }

    #[test]
    fn test_from_se3() {
        let se3 = SE3::from_quaternion(1.0, 0.0, 0.0, 0.0, Vector3::new(1.0, 2.0, 3.0));
        let sim3 = Sim3::from_se3(&se3);

        assert_eq!(sim3.scale, 1.0);
        assert_eq!(sim3.translation, se3.translation);
        assert_eq!(sim3.rotation, se3.rotation);
    }

    #[test]
    fn test_to_se3() {
        let sim3 = Sim3 {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::new(1.0, 2.0, 3.0),
            scale: 2.0,
        };
        let se3 = sim3.to_se3();

        assert_eq!(se3.translation, sim3.translation);
        assert_eq!(se3.rotation, sim3.rotation);
    }

    #[test]
    fn test_inverse() {
        let sim3 = Sim3 {
            rotation: UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
                std::f64::consts::FRAC_PI_2,
            ),
            translation: Vector3::new(1.0, 2.0, 3.0),
            scale: 2.0,
        };

        let sim3_inv = sim3.inverse();
        let composed = sim3.compose(&sim3_inv);

        assert_relative_eq!(composed.scale, 1.0, epsilon = 1e-10);
        assert_relative_eq!(composed.translation.norm(), 0.0, epsilon = 1e-10);

        // Also verify with a point
        let p = Vector3::new(1.0, 2.0, 3.0);
        let p_transformed = sim3.transform_point(&p);
        let p_back = sim3_inv.transform_point(&p_transformed);
        assert_relative_eq!(p_back, p, epsilon = 1e-10);
    }

    #[test]
    fn test_compose() {
        let s1 = Sim3 {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::new(1.0, 0.0, 0.0),
            scale: 2.0,
        };
        let s2 = Sim3 {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::new(0.0, 1.0, 0.0),
            scale: 3.0,
        };

        let composed = s1.compose(&s2);

        // Scale should multiply
        assert_relative_eq!(composed.scale, 6.0, epsilon = 1e-10);

        // Translation: s1.t + s1.s * R1 * s2.t = (1,0,0) + 2*(0,1,0) = (1,2,0)
        assert_relative_eq!(composed.translation, Vector3::new(1.0, 2.0, 0.0), epsilon = 1e-10);
    }

    #[test]
    fn test_transform_point_with_scale() {
        let sim3 = Sim3 {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::new(1.0, 0.0, 0.0),
            scale: 2.0,
        };

        let p = Vector3::new(1.0, 1.0, 1.0);
        let p_transformed = sim3.transform_point(&p);

        // p' = 2 * I * (1,1,1) + (1,0,0) = (3, 2, 2)
        assert_relative_eq!(p_transformed, Vector3::new(3.0, 2.0, 2.0), epsilon = 1e-10);
    }

    #[test]
    fn test_to_from_matrix() {
        let sim3 = Sim3 {
            rotation: UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(Vector3::new(1.0, 0.0, 0.0)),
                std::f64::consts::FRAC_PI_4,
            ),
            translation: Vector3::new(1.0, 2.0, 3.0),
            scale: 2.5,
        };

        let mat = sim3.to_matrix();
        let sim3_reconstructed = Sim3::from_matrix(mat);

        assert_relative_eq!(sim3.scale, sim3_reconstructed.scale, epsilon = 1e-10);
        assert_relative_eq!(
            sim3.translation,
            sim3_reconstructed.translation,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            sim3.rotation.coords,
            sim3_reconstructed.rotation.coords,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let sim3 = Sim3 {
            rotation: UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(Vector3::new(0.0, 1.0, 0.0)),
                0.5,
            ),
            translation: Vector3::new(1.0, 2.0, 3.0),
            scale: 1.5,
        };

        let log_vec = sim3.log();
        let sim3_reconstructed = Sim3::exp(&log_vec);

        assert_relative_eq!(sim3.scale, sim3_reconstructed.scale, epsilon = 1e-10);
        assert_relative_eq!(
            sim3.translation,
            sim3_reconstructed.translation,
            epsilon = 1e-10
        );
        assert_relative_eq!(
            sim3.rotation.coords,
            sim3_reconstructed.rotation.coords,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_is_se3() {
        let sim3_scale1 = Sim3::from_se3(&SE3::identity());
        assert!(sim3_scale1.is_se3(1e-10));

        let sim3_scaled = Sim3 {
            scale: 1.5,
            ..Sim3::identity()
        };
        assert!(!sim3_scaled.is_se3(0.1));
    }
}
