//! PnP (Perspective-n-Point) solver using OpenCV.

use anyhow::Result;
use nalgebra::{Matrix3, Vector3};
use opencv::calib3d;
use opencv::core::{Mat, Point2f, Point3d};
use opencv::prelude::*;

use crate::geometry::SE3;
use crate::tracking::frame::CameraModel;

/// Result of PnP solving.
pub struct PnPResult {
    /// Estimated camera pose (T_wc).
    pub pose: SE3,
    /// Inlier mask per correspondence.
    pub inlier_mask: Vec<bool>,
    /// Reprojection error per correspondence (in pixels).
    pub reproj_errors: Vec<f64>,
}

/// Solve PnP with RANSAC given 3D-2D correspondences.
///
/// # Arguments
/// * `points3d` - 3D world points
/// * `points2d` - Corresponding 2D image points
/// * `camera` - Camera intrinsics
/// * `prior` - Optional initial pose guess (enables extrinsic guess)
pub fn solve_pnp_ransac(
    points3d: &[Vector3<f64>],
    points2d: &[Point2f],
    camera: &CameraModel,
    prior: Option<&SE3>,
) -> Result<SE3> {
    // Convert to Point3d for OpenCV
    let pts3d: Vec<Point3d> = points3d
        .iter()
        .map(|p| Point3d::new(p.x, p.y, p.z))
        .collect();
    let obj_points = Mat::from_slice(&pts3d)?.try_clone()?;
    let img_points = Mat::from_slice(points2d)?.try_clone()?;

    let camera_matrix = Mat::from_slice_2d(&[
        [camera.fx, 0.0, camera.cx],
        [0.0, camera.fy, camera.cy],
        [0.0, 0.0, 1.0],
    ])?
    .try_clone()?;
    let dist_coeffs = Mat::zeros(1, 5, opencv::core::CV_64F)?.to_mat()?;

    let mut rvec = Mat::default();
    let mut tvec = Mat::default();
    let mut use_extrinsic_guess = false;

    if let Some(prior_pose) = prior {
        // OpenCV expects T_cw (world-to-camera), but our poses are T_wc (camera-to-world)
        // So we need to invert the prior before passing to OpenCV
        let prior_cw = prior_pose.inverse();
        let rot_mat = prior_cw.rotation.to_rotation_matrix().into_inner();
        rvec = rotation_matrix_to_rvec(rot_mat)?;
        tvec = Mat::from_slice(&[
            prior_cw.translation.x,
            prior_cw.translation.y,
            prior_cw.translation.z,
        ])?
        .try_clone()?;
        use_extrinsic_guess = !rvec.empty();
    }

    let mut inliers = Mat::default();
    calib3d::solve_pnp_ransac(
        &obj_points,
        &img_points,
        &camera_matrix,
        &dist_coeffs,
        &mut rvec,
        &mut tvec,
        use_extrinsic_guess,
        100,  // iterations
        8.0,  // reprojection error
        0.99, // confidence
        &mut inliers,
        calib3d::SOLVEPNP_ITERATIVE,
    )?;

    let mut rot_mat = Mat::default();
    calib3d::rodrigues(&rvec, &mut rot_mat, &mut opencv::core::no_array())?;
    let rotation = mat3_to_matrix3(&rot_mat)?;
    let translation = Vector3::new(
        *tvec.at::<f64>(0i32)?,
        *tvec.at::<f64>(1i32)?,
        *tvec.at::<f64>(2i32)?,
    );

    // OpenCV returns T_cw (world-to-camera), but our convention is T_wc (camera-to-world)
    // So we need to invert the result
    let pose_cw = SE3::from_rt(rotation, translation);
    Ok(pose_cw.inverse())
}

/// Solve PnP with RANSAC and also return inlier mask and reprojection errors.
pub fn solve_pnp_ransac_detailed(
    points3d: &[Vector3<f64>],
    points2d: &[Point2f],
    camera: &CameraModel,
    prior: Option<&SE3>,
) -> Result<PnPResult> {
    let pose = solve_pnp_ransac(points3d, points2d, camera, prior)?;

    // Compute reprojection errors for each correspondence.
    let mut reproj_errors = Vec::with_capacity(points3d.len());
    let mut inlier_mask = Vec::with_capacity(points3d.len());

    for (p3, p2) in points3d.iter().zip(points2d.iter()) {
        // Transform to camera frame.
        let pose_cw = pose.inverse();
        let p_cam = pose_cw.transform_point(p3);
        if p_cam.z <= 0.0 {
            reproj_errors.push(f64::INFINITY);
            inlier_mask.push(false);
            continue;
        }
        let u = camera.fx * p_cam.x / p_cam.z + camera.cx;
        let v = camera.fy * p_cam.y / p_cam.z + camera.cy;
        let du = u - p2.x as f64;
        let dv = v - p2.y as f64;
        let err = (du * du + dv * dv).sqrt();
        reproj_errors.push(err);
        inlier_mask.push(err < 8.0);
    }

    Ok(PnPResult {
        pose,
        inlier_mask,
        reproj_errors,
    })
}

/// Convert rotation matrix to Rodrigues vector.
fn rotation_matrix_to_rvec(rot: Matrix3<f64>) -> Result<Mat> {
    let rot_mat = Mat::from_slice(rot.as_slice())?.try_clone()?;
    let rot_mat_reshaped = rot_mat.reshape(1, 3)?.try_clone()?;
    let mut rvec = Mat::default();
    calib3d::rodrigues(&rot_mat_reshaped, &mut rvec, &mut opencv::core::no_array())?;
    Ok(rvec)
}

/// Convert OpenCV 3x3 Mat to nalgebra Matrix3.
fn mat3_to_matrix3(mat: &Mat) -> Result<Matrix3<f64>> {
    let mut arr = [0.0f64; 9];
    for i in 0..9 {
        arr[i] = *mat.at::<f64>(i as i32)?;
    }
    Ok(Matrix3::from_row_slice(&arr))
}
