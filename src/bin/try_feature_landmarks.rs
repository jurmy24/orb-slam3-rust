use anyhow::{Context, Result};
use opencv::{core, features2d, highgui, prelude::*, videoio};

fn main() -> Result<()> {
    println!("--- Initializing vSLAM Feature Tracker ---");

    // 1. Initialize Camera
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .context("Could not open camera. Check permissions or connection.")?;

    if !videoio::VideoCapture::is_opened(&cam)? {
        anyhow::bail!("Camera is reported as closed by the system.");
    }

    // 2. Setup ORB Feature Detector
    // Parameters: (nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold)
    let mut orb = features2d::ORB::create(
        1000, // Detect up to 1000 landmarks
        1.2,  // Image pyramid scale
        8,    // Number of pyramid levels
        31,   // Edge threshold
        0,    // First level
        2,    // WTA_K
        features2d::ORB_ScoreType::HARRIS_SCORE,
        31, // Patch size
        20, // FAST threshold
    )?;

    // 3. Create GUI Window
    highgui::named_window("vSLAM: Feature Landmarks", highgui::WINDOW_AUTOSIZE)?;

    let mut frame = Mat::default();
    let mut gray_frame = Mat::default();
    let mut output_frame = Mat::default();

    println!("Success! Press 'q' to quit.");

    loop {
        // Capture frame
        cam.read(&mut frame)?;
        if frame.empty() {
            continue;
        }

        // SLAM usually works on Grayscale images to save processing power
        opencv::imgproc::cvt_color(
            &frame,
            &mut gray_frame,
            opencv::imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_APPROX,
        )?;

        // 4. Detect Landmarks (Keypoints)
        let mut keypoints = core::Vector::<core::KeyPoint>::new();
        let mut descriptors = Mat::default();
        let mask = Mat::default();

        // This is the core "Front-end" step
        orb.detect_and_compute(&gray_frame, &mask, &mut keypoints, &mut descriptors, false)?;

        // 5. Draw Landmarks
        // We draw them on the color 'frame' for better visualization
        features2d::draw_keypoints(
            &frame,
            &keypoints,
            &mut output_frame,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0), // Bright red
            features2d::DrawMatchesFlags::DEFAULT,
        )?;

        // Update the display
        highgui::imshow("vSLAM: Feature Landmarks", &output_frame)?;

        // Handle Exit
        let key = highgui::wait_key(1)?;
        if key == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
