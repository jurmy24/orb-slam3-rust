use anyhow::Result;
use nalgebra::Vector3;

use rust_vslam::io::euroc::EurocDataset;
use rust_vslam::system::SlamSystem;
use rust_vslam::tracking::frame::{CameraModel, StereoFrame, StereoProcessor};
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

    // Set up camera model
    let cam =
        CameraModel::from_k_and_baseline(dataset.calibration.k_left, dataset.calibration.baseline);

    let mut stereo = StereoProcessor::new(cam, 1200)?;
    let mut slam_system = SlamSystem::new(cam)?;
    let viz = RerunVisualizer::new("rust-orb-slam3-stereo-inertial");

    // Iterates over stereo frames (not IMU samples!)
    for i in 0..dataset.len() {
        let pair = dataset.stereo_pair(i)?;

        // Collect IMU between current and next frame
        let t_start = pair.timestamp_ns;
        let t_end = if i + 1 < dataset.len() {
            dataset.cam0_entries[i + 1].timestamp_ns
        } else {
            pair.timestamp_ns
        };
        let imu_between = dataset.imu_between(t_start, t_end);

        // Process stereo frame
        let stereo_frame: StereoFrame =
            stereo.process(&pair.left, &pair.right, pair.timestamp_ns)?;

        // Run SLAM system (tracking + keyframe creation)
        let result: TrackingResult =
            slam_system.process_frame(stereo_frame.clone(), &imu_between)?;

        // Log everything to Rerun
        viz.set_time(pair.timestamp_ns);
        viz.log_stereo_frame(&stereo_frame, &pair.left, &pair.right);
        viz.log_pose(&result.pose);
        viz.log_trajectory(
            &slam_system
                .trajectory()
                .iter()
                .map(|p| p.translation)
                .collect::<Vec<Vector3<f64>>>(),
        );
        viz.log_tracking_metrics(&result.metrics);
        viz.log_timing(&result.timing);

        // Log tracking state with map info
        {
            let shared = slam_system.shared_state();
            let atlas = shared.atlas.read();
            viz.log_tracking_state(result.state, i, atlas.active_map_index());

            // Log map statistics periodically
            if i % 100 == 0 {
                let map = atlas.active_map();
                println!(
                    "Frame {}/{}: {} keyframes, {} map points, state={:?}",
                    i,
                    dataset.len(),
                    map.num_keyframes(),
                    map.num_map_points(),
                    result.state
                );
            }
        }

        // Log map points for visualization
        if i % 10 == 0 {
            let shared = slam_system.shared_state();
            let atlas = shared.atlas.read();
            let map = atlas.active_map();
            let points: Vec<Vector3<f64>> = map.map_points().map(|mp| mp.position).collect();
            viz.log_map_points(&points);
        }
    }

    println!("Done! Processed {} frames", dataset.len());

    // Shutdown cleanly (joins Local Mapping thread)
    slam_system.shutdown();

    Ok(())
}
