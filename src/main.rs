use anyhow::Result;
use nalgebra::Vector3;
use tracing::{debug, info, warn};

use rust_vslam::io::euroc::EurocDataset;
use rust_vslam::system::SlamSystem;
use rust_vslam::tracking::frame::{CameraModel, StereoFrame, StereoProcessor};
use rust_vslam::tracking::result::TrackingResult;
use rust_vslam::viz::rerun::RerunVisualizer;

fn main() -> Result<()> {
    // Initialize tracing subscriber with environment filter
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();
    let dataset_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/euroc/MH_01_easy/mav0".to_string());

    let dataset = EurocDataset::new(&dataset_path)?;
    debug!(
        "Loaded {} stereo frames, {} IMU samples, {} ground truth entries",
        dataset.len(),
        dataset.imu_entries.len(),
        dataset.groundtruth.len()
    );
    // Print the camera baseline and calibration matrix
    debug!("Camera baseline: {}", dataset.calibration.baseline);
    debug!(
        "Camera calibration matrix: {:?}",
        dataset.calibration.k_left
    );

    let cam =
        CameraModel::from_k_and_baseline(dataset.calibration.k_left, dataset.calibration.baseline);
    let mut stereo = StereoProcessor::new(cam, 1200)?;
    let mut slam_system = SlamSystem::new(cam)?;
    let mut viz = RerunVisualizer::new("rust-orb-slam3-stereo-inertial");

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

        // Status bar (needs FPS before map access)
        let fps = viz.fps();

        // Image feed (doesn't need map access)
        viz.log_image_feed(&pair.left);

        // Annotated features
        viz.log_annotated_features(
            &stereo_frame,
            &result.matches,
            &result.matches.inlier_indices,
            &result.matches.outlier_indices,
        );

        // 3D visualization (commented out for testing GT only)
        // viz.log_camera_pose(&result.pose);
        // let est_trajectory: Vec<Vector3<f64>> = slam_system
        //     .trajectory()
        //     .iter()
        //     .map(|p| p.translation)
        //     .collect();
        // viz.log_trajectory(&est_trajectory);

        // Ground truth trajectory (up to current frame time)
        // Note: We transform GT from body frame to camera frame to match SLAM coordinate system
        let gt_positions = dataset.groundtruth_positions_until(pair.timestamp_ns);
        viz.log_groundtruth_trajectory(&gt_positions);

        // Debug output periodically
        if i % 100 == 0 {
            debug!(
                "Frame {} (ts={}): GT trajectory: {} points",
                i,
                pair.timestamp_ns,
                gt_positions.len()
            );
            // if !est_trajectory.is_empty() {
            //     let first_est = est_trajectory.first().unwrap();
            //     let last_est = est_trajectory.last().unwrap();
            //     debug!(
            //         "  Est: first=[{:.2}, {:.2}, {:.2}], last=[{:.2}, {:.2}, {:.2}], dist={:.2}m",
            //         first_est.x,
            //         first_est.y,
            //         first_est.z,
            //         last_est.x,
            //         last_est.y,
            //         last_est.z,
            //         (last_est - first_est).norm()
            //     );
            // }
            if !gt_positions.is_empty() {
                let first_gt = gt_positions.first().unwrap();
                let last_gt = gt_positions.last().unwrap();
                debug!(
                    "  GT: first=[{:.2}, {:.2}, {:.2}], last=[{:.2}, {:.2}, {:.2}], dist={:.2}m",
                    first_gt.x,
                    first_gt.y,
                    first_gt.z,
                    last_gt.x,
                    last_gt.y,
                    last_gt.z,
                    (last_gt - first_gt).norm()
                );
            } else {
                // Check why GT is empty
                if !dataset.groundtruth.is_empty() {
                    let first_gt_ts = dataset.groundtruth[0].timestamp_ns;
                    let last_gt_ts =
                        dataset.groundtruth[dataset.groundtruth.len() - 1].timestamp_ns;
                    warn!(
                        "  GT empty! Frame ts={}, GT range=[{}, {}]",
                        pair.timestamp_ns, first_gt_ts, last_gt_ts
                    );
                }
            }
        }

        // Temporal plots
        viz.log_temporal_plots(&result.metrics, &result.timing);
        viz.log_bias_magnitude(result.imu_bias.as_ref());

        // Get map state for visualization (keeping atlas lock scoped)
        {
            let shared = slam_system.shared_state();
            let atlas = shared.atlas.read();
            let map = atlas.active_map();
            let _map_id = atlas.active_map_index();
            let imu_state = map.imu_init_state();

            // Status bar
            let n_matched = result.matches.matched_map_points.len();
            viz.log_status_bar(result.state, imu_state, &result.metrics, n_matched, fps);

            // Map ID
            // viz.log_map_id(map_id);

            // Collect and log local map points
            // let local_points: Vec<Vector3<f64>> = result
            //     .matches
            //     .local_map_point_ids
            //     .iter()
            //     .filter_map(|mp_id| map.get_map_point(*mp_id).map(|mp| mp.position))
            //     .collect();
            // viz.log_local_map_points(&local_points);

            // Log keyframes periodically
            // if i % 10 == 0 {
            //     let keyframes: Vec<_> = map.keyframes().collect();
            //     viz.log_keyframes(&keyframes);
            // }

            // Full map points with LOD (less frequently)
            // if i % 5 == 0 {
            //     let all_points: Vec<Vector3<f64>> =
            //         map.map_points().map(|mp| mp.position).collect();
            //     viz.log_map_points_lod(&all_points, result.pose.translation);
            // }

            // Log map statistics periodically
            if i % 100 == 0 {
                info!(
                    "Frame {}/{}: {} keyframes, {} map points, state={:?}, IMU={:?}",
                    i,
                    dataset.len(),
                    map.num_keyframes(),
                    map.num_map_points(),
                    result.state,
                    imu_state
                );
            }
        }
    }

    info!("Done! Processed {} frames", dataset.len());

    // Shutdown cleanly (joins Local Mapping thread)
    slam_system.shutdown();

    Ok(())
}
