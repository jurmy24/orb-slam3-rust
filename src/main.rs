use anyhow::Result;
use nalgebra::Vector3;

use rust_vslam::atlas::atlas::Atlas;
use rust_vslam::io::euroc::EurocDataset;
use rust_vslam::tracking::Tracker;
use rust_vslam::tracking::frame::{CameraModel, StereoProcessor};
use rust_vslam::tracking::result::TrackingResult;
use rust_vslam::viz::rerun::RerunVisualizer;

fn main() -> Result<()> {
    let dataset_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/euroc/MH_01_easy/mav0".to_string());

    println!("Loading EuRoC dataset from: {}", dataset_path);
    let dataset = EurocDataset::new(&dataset_path)?;
    println!(
        "Loaded {} stereo frames, {} IMU samples",
        dataset.len(),
        dataset.imu_entries.len()
    );

    // Only set up left camera model since right is only used for match finding
    // TODO: Might need to rectify the images from the dataset so the cameras share identical intrinsics
    let cam =
        CameraModel::from_k_and_baseline(dataset.calibration.k_left, dataset.calibration.baseline);

    let mut stereo = StereoProcessor::new(cam, 1200)?;
    let mut tracker = Tracker::new(cam)?;
    let mut atlas = Atlas::new();
    let viz = RerunVisualizer::new("rust-orb-slam3-stereo-inertial");

    // Iterates over stereo frames (not IMU samples!)
    for i in 0..dataset.len() {
        let pair = dataset.stereo_pair(i)?;

        // Collect IMU between current and next frame
        let t_start = pair.timestamp_ns;
        let t_end = if i + 1 < dataset.len() {
            dataset.cam0_entries[i + 1].timestamp_ns
        } else {
            pair.timestamp_ns // Set = to t_start so no IMU samples are found for last frame
        };
        let imu_between = dataset.imu_between(t_start, t_end);

        // Process stereo frame
        let stereo_frame = stereo.process(&pair.left, &pair.right, pair.timestamp_ns)?;

        // Run tracker
        let result: TrackingResult =
            tracker.process_frame(stereo_frame.clone(), &imu_between, &mut atlas)?;

        // Log everything to Rerun
        viz.set_time(pair.timestamp_ns); // Set the current timestamp for all subsequent logs
        viz.log_stereo_frame(&stereo_frame, &pair.left, &pair.right);
        viz.log_pose(&result.pose);
        viz.log_trajectory(
            &tracker
                .trajectory
                .iter()
                .map(|p| p.translation)
                .collect::<Vec<Vector3<f64>>>(),
        );
        viz.log_tracking_metrics(&result.metrics);
        viz.log_timing(&result.timing);
        viz.log_tracking_state(result.state, i, atlas.active_map_index());

        // Progress indicator
        if i % 100 == 0 {
            println!("Processed frame {}/{}", i, dataset.len());
        }
    }

    println!("Done! Processed {} frames", dataset.len());
    Ok(())
}
