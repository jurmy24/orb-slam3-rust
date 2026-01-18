use anyhow::Result;
use nalgebra::Vector3;

use rust_vslam::estimator::vi_estimator::VisualInertialEstimator;
use rust_vslam::frontend::camera::CameraModel;
use rust_vslam::frontend::stereo::StereoProcessor;
use rust_vslam::io::euroc::EurocDataset;
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

    let cam =
        CameraModel::from_k_and_baseline(dataset.calibration.k_left, dataset.calibration.baseline);

    let mut stereo = StereoProcessor::new(cam, 1200)?;
    let mut estimator = VisualInertialEstimator::new(cam)?;
    let viz = RerunVisualizer::new("rust-vslam");

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
        let stereo_frame = stereo.process(&pair.left, &pair.right, pair.timestamp_ns)?;

        // Run visual-inertial estimator
        let pose = estimator.process_frame(stereo_frame.clone(), &imu_between)?;

        // Log everything to Rerun
        viz.set_time(pair.timestamp_ns);
        viz.log_stereo_frame(&stereo_frame, &pair.left, &pair.right);
        viz.log_pose(&pose);
        viz.log_trajectory(
            &estimator
                .trajectory
                .iter()
                .map(|p| p.translation)
                .collect::<Vec<Vector3<f64>>>(),
        );

        // Progress indicator
        if i % 100 == 0 {
            println!("Processed frame {}/{}", i, dataset.len());
        }
    }

    println!("Done! Processed {} frames", dataset.len());
    Ok(())
}
