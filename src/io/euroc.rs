use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use csv::ReaderBuilder;
use nalgebra::{Matrix3, Matrix4, Vector3};
use opencv::prelude::*;
use opencv::{imgcodecs, imgcodecs::IMREAD_GRAYSCALE};
use serde::Deserialize;
use tracing::warn;

use crate::geometry::SE3;
use crate::imu::ImuSample;

#[derive(Debug, Clone)]
pub struct ImageEntry {
    pub timestamp_ns: u64,
    pub filename: String,
}

#[derive(Debug, Clone)]
pub struct StereoImagePair {
    pub left: Mat,
    pub right: Mat,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone)]
pub struct ImuEntry {
    pub timestamp_ns: u64,
    pub sample: ImuSample,
}

#[derive(Debug, Clone)]
pub struct GroundTruthEntry {
    pub timestamp_ns: u64,
    pub pose: SE3, // position + orientation
    pub velocity: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
    pub accel_bias: Vector3<f64>,
}

#[derive(Debug, Clone)]
pub struct StereoCalibration {
    pub k_left: Matrix3<f64>,
    pub k_right: Matrix3<f64>,
    pub t_cam0_body: SE3,
    pub t_cam1_body: SE3,
    pub t_cam1_cam0: SE3,
    pub baseline: f64,
}

#[derive(Debug)]
pub struct EurocDataset {
    dataset_path: PathBuf,
    pub cam0_entries: Vec<ImageEntry>,
    pub cam1_entries: Vec<ImageEntry>,
    pub imu_entries: Vec<ImuEntry>,
    pub groundtruth: Vec<GroundTruthEntry>,
    pub calibration: StereoCalibration,
}

impl EurocDataset {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let cam0_entries = load_image_list(root.join("cam0/data.csv"))?;
        let cam1_entries = load_image_list(root.join("cam1/data.csv"))?;

        if cam0_entries.len() != cam1_entries.len() {
            bail!("cam0 and cam1 have different number of frames");
        }

        let imu_entries = load_imu_list(root.join("imu0/data.csv"))?;
        // Ground truth is optional - some datasets might not have it
        let groundtruth = load_groundtruth_list(root.join("state_groundtruth_estimate0/data.csv"))
            .unwrap_or_else(|e| {
                warn!("Could not load ground truth: {}. Continuing without it.", e);
                Vec::new()
            });
        let calib = load_stereo_calibration(&root)?;

        Ok(Self {
            dataset_path: root,
            cam0_entries,
            cam1_entries,
            imu_entries,
            groundtruth,
            calibration: calib,
        })
    }

    pub fn len(&self) -> usize {
        self.cam0_entries.len()
    }

    pub fn frame_timestamp(&self, idx: usize) -> Option<u64> {
        self.cam0_entries.get(idx).map(|e| e.timestamp_ns)
    }

    pub fn stereo_pair(&self, idx: usize) -> Result<StereoImagePair> {
        let left_entry = self
            .cam0_entries
            .get(idx)
            .with_context(|| format!("No left frame at index {}", idx))?;
        let right_entry = self
            .cam1_entries
            .get(idx)
            .with_context(|| format!("No right frame at index {}", idx))?;

        // Expect synchronized timestamps
        let timestamp_ns = left_entry.timestamp_ns;

        let left_path = self
            .dataset_path
            .join("cam0/data")
            .join(&left_entry.filename);
        let right_path = self
            .dataset_path
            .join("cam1/data")
            .join(&right_entry.filename);

        let left = imgcodecs::imread(left_path.to_str().unwrap(), IMREAD_GRAYSCALE)
            .with_context(|| format!("Failed to read left image {:?}", left_path))?;
        let right = imgcodecs::imread(right_path.to_str().unwrap(), IMREAD_GRAYSCALE)
            .with_context(|| format!("Failed to read right image {:?}", right_path))?;

        Ok(StereoImagePair {
            left,
            right,
            timestamp_ns,
        })
    }

    /// Get IMU samples between two timestamps (inclusive).
    pub fn imu_between(&self, t_ns_start: u64, t_ns_end: u64) -> Vec<ImuEntry> {
        self.imu_entries
            .iter()
            .filter(|e| e.timestamp_ns >= t_ns_start && e.timestamp_ns <= t_ns_end)
            .cloned()
            .collect()
    }

    /// Get all ground truth positions up to (and including) the given timestamp.
    /// Transforms from body/IMU frame to camera frame (cam0) and aligns to SLAM origin.
    /// Uses binary search for O(log n) lookup instead of O(n) scan.
    ///
    /// Note: GT pose is body pose in reference/world frame. We transform the full pose
    /// from body to camera: T_world_cam = T_world_body * T_body_cam = T_world_body * T_cam0_body^-1
    /// Then we subtract the first GT position so trajectories are origin-aligned.
    pub fn groundtruth_positions_until(&self, timestamp_ns: u64) -> Vec<Vector3<f64>> {
        if self.groundtruth.is_empty() {
            return Vec::new();
        }

        // Transform from body frame to camera frame: T_cam0_body
        // GT pose is T_world_body, we need T_world_cam = T_world_body * T_body_cam0
        let t_body_cam0 = self.calibration.t_cam0_body.inverse();

        // Get the first GT position (to align with SLAM origin)
        let first_gt_cam = self.groundtruth[0].pose.compose(&t_body_cam0).translation;

        // Use binary search to find the cutoff point efficiently
        let cutoff_idx = self
            .groundtruth
            .partition_point(|gt| gt.timestamp_ns <= timestamp_ns);

        self.groundtruth[..cutoff_idx]
            .iter()
            .map(|gt| {
                // Transform full pose from body to camera, then extract camera position in world frame
                // T_world_cam = T_world_body * T_body_cam0
                let t_world_cam = gt.pose.compose(&t_body_cam0);
                // Subtract first position to align with SLAM origin
                t_world_cam.translation - first_gt_cam
            })
            .collect()
    }
}

fn load_image_list(csv_path: PathBuf) -> Result<Vec<ImageEntry>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .comment(Some(b'#'))
        .from_path(&csv_path)
        .with_context(|| format!("Failed to open {}", csv_path.display()))?;

    let mut entries = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() < 2 {
            continue;
        }
        let ts: u64 = rec[0].trim().parse()?;
        let filename = rec[1].trim().to_string();
        entries.push(ImageEntry {
            timestamp_ns: ts,
            filename,
        });
    }
    Ok(entries)
}

fn load_imu_list(csv_path: PathBuf) -> Result<Vec<ImuEntry>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .comment(Some(b'#'))
        .from_path(&csv_path)
        .with_context(|| format!("Failed to open {}", csv_path.display()))?;

    let mut entries = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() < 7 {
            continue;
        }
        let ts: u64 = rec[0].trim().parse()?;
        let gyro = Vector3::new(
            rec[1].trim().parse()?,
            rec[2].trim().parse()?,
            rec[3].trim().parse()?,
        );
        let accel = Vector3::new(
            rec[4].trim().parse()?,
            rec[5].trim().parse()?,
            rec[6].trim().parse()?,
        );
        entries.push(ImuEntry {
            timestamp_ns: ts,
            sample: ImuSample {
                timestamp_s: ts as f64 * 1e-9,
                accel,
                gyro,
            },
        });
    }
    Ok(entries)
}

fn load_groundtruth_list(csv_path: PathBuf) -> Result<Vec<GroundTruthEntry>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .comment(Some(b'#'))
        .from_path(&csv_path)
        .with_context(|| format!("Failed to open {}", csv_path.display()))?;

    let mut entries = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        // CSV format: timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z,
        //             v_RS_R_x, v_RS_R_y, v_RS_R_z, b_w_RS_S_x, b_w_RS_S_y, b_w_RS_S_z,
        //             b_a_RS_S_x, b_a_RS_S_y, b_a_RS_S_z
        if rec.len() < 17 {
            continue;
        }
        let ts: u64 = rec[0].trim().parse()?;

        // Position (p_RS_R_x/y/z)
        let position = Vector3::new(
            rec[1].trim().parse()?,
            rec[2].trim().parse()?,
            rec[3].trim().parse()?,
        );

        // Orientation quaternion (q_RS_w/x/y/z) - w-first format
        let qw: f64 = rec[4].trim().parse()?;
        let qx: f64 = rec[5].trim().parse()?;
        let qy: f64 = rec[6].trim().parse()?;
        let qz: f64 = rec[7].trim().parse()?;
        let pose = SE3::from_quaternion(qw, qx, qy, qz, position);

        // Velocity (v_RS_R_x/y/z)
        let velocity = Vector3::new(
            rec[8].trim().parse()?,
            rec[9].trim().parse()?,
            rec[10].trim().parse()?,
        );

        // Gyro bias (b_w_RS_S_x/y/z)
        let gyro_bias = Vector3::new(
            rec[11].trim().parse()?,
            rec[12].trim().parse()?,
            rec[13].trim().parse()?,
        );

        // Accel bias (b_a_RS_S_x/y/z)
        let accel_bias = Vector3::new(
            rec[14].trim().parse()?,
            rec[15].trim().parse()?,
            rec[16].trim().parse()?,
        );

        entries.push(GroundTruthEntry {
            timestamp_ns: ts,
            pose,
            velocity,
            gyro_bias,
            accel_bias,
        });
    }
    Ok(entries)
}

/// EuRoC T_BS transform format: has cols, rows, data fields
#[derive(Debug, Deserialize)]
struct TransformYaml {
    data: Vec<f64>,
}

/// EuRoC camera sensor.yaml format
#[derive(Debug, Deserialize)]
struct CameraYaml {
    #[serde(rename = "T_BS")]
    t_bs: TransformYaml,
    /// [fx, fy, cx, cy] intrinsics
    intrinsics: Vec<f64>,
}

fn load_stereo_calibration(root: &Path) -> Result<StereoCalibration> {
    let cam0_yaml = root.join("cam0/sensor.yaml");
    let cam1_yaml = root.join("cam1/sensor.yaml");

    let cam0: CameraYaml = serde_yaml::from_reader(
        File::open(&cam0_yaml).with_context(|| format!("Failed to open {:?}", cam0_yaml))?,
    )?;
    let cam1: CameraYaml = serde_yaml::from_reader(
        File::open(&cam1_yaml).with_context(|| format!("Failed to open {:?}", cam1_yaml))?,
    )?;

    let k_left = intrinsics_to_matrix3(&cam0.intrinsics)?;
    let k_right = intrinsics_to_matrix3(&cam1.intrinsics)?;

    let t_cam0_body = transform_from(&cam0.t_bs.data)?;
    let t_cam1_body = transform_from(&cam1.t_bs.data)?;

    // Transform from cam0 to cam1: T_c1_c0 = T_c1_b * T_b_c0
    let t_cam0_body_inv = t_cam0_body.inverse();
    let t_cam1_cam0 = t_cam1_body.compose(&t_cam0_body_inv);
    let baseline = t_cam1_cam0.translation.norm();

    Ok(StereoCalibration {
        k_left,
        k_right,
        t_cam0_body,
        t_cam1_body,
        t_cam1_cam0,
        baseline,
    })
}

/// Convert EuRoC intrinsics [fx, fy, cx, cy] to 3x3 camera matrix (K-matrix for pinhole cameras)
/// For a 3D point in camera coordinates (x, y, z), you normalize it by dividing x, y, and z by z
/// and then apply K (u, v, 1) = K(x/z, y/z, 1) to get the 2D pixel coordinates (u, v)
/// This is what the K-matrix is used for
fn intrinsics_to_matrix3(intrinsics: &[f64]) -> Result<Matrix3<f64>> {
    if intrinsics.len() != 4 {
        bail!(
            "Expected 4 intrinsics [fx, fy, cx, cy], got {}",
            intrinsics.len()
        );
    }
    let fx = intrinsics[0];
    let fy = intrinsics[1];
    let cx = intrinsics[2];
    let cy = intrinsics[3];
    Ok(Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0))
}

fn transform_from(data: &[f64]) -> Result<SE3> {
    if data.len() != 16 {
        bail!("Expected 16 elements for transform, got {}", data.len());
    }
    let mat = Matrix4::from_row_slice(data);
    Ok(SE3::from_matrix(mat))
}
