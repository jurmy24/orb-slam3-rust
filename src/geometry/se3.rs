use nalgebra::{Isometry3, Matrix3, Matrix4, Rotation3, Translation3, UnitQuaternion, Vector3};

/// Rigid body transform in SE(3).
#[derive(Debug, Clone, PartialEq)]
pub struct SE3 {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

impl SE3 {
    /// Identity transform.
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Construct from rotation matrix and translation.
    pub fn from_rt(rotation: Matrix3<f64>, translation: Vector3<f64>) -> Self {
        let rot3 = Rotation3::from_matrix_unchecked(rotation);
        Self {
            rotation: UnitQuaternion::from_rotation_matrix(&rot3),
            translation,
        }
    }

    /// Construct from quaternion (w, x, y, z) and translation.
    pub fn from_quaternion(qw: f64, qx: f64, qy: f64, qz: f64, translation: Vector3<f64>) -> Self {
        let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
        Self {
            rotation,
            translation,
        }
    }

    /// Construct from homogeneous 4x4 matrix.
    pub fn from_matrix(mat: Matrix4<f64>) -> Self {
        let rotation_mat = mat.fixed_view::<3, 3>(0, 0).into_owned();
        let translation = Vector3::new(mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]);
        let rot3 = Rotation3::from_matrix_unchecked(rotation_mat);
        Self {
            rotation: UnitQuaternion::from_rotation_matrix(&rot3),
            translation,
        }
    }

    /// Convert to homogeneous 4x4 matrix.
    pub fn to_matrix(&self) -> Matrix4<f64> {
        let iso: Isometry3<f64> =
            Isometry3::from_parts(Translation3::from(self.translation), self.rotation);
        iso.to_homogeneous()
    }

    /// Inverse transform.
    pub fn inverse(&self) -> Self {
        let rot_inv = self.rotation.inverse();
        let t_inv = -(rot_inv * self.translation);
        Self {
            rotation: rot_inv,
            translation: t_inv,
        }
    }

    /// Compose two transforms (self @ other).
    pub fn compose(&self, other: &SE3) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
            translation: self.rotation * other.translation + self.translation,
        }
    }

    /// Transform a single point.
    pub fn transform_point(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * p + self.translation
    }

    /// Transform multiple points.
    pub fn transform_points(&self, pts: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        pts.iter().map(|p| self.transform_point(p)).collect()
    }
}
