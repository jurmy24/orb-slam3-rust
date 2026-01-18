use anyhow::{Context, Result};
use opencv::{
    highgui,    // For GUI windows
    prelude::*, // Provides traits like MatTrait (needed to work with matrices)
    videoio,    // For camera/video input
};

fn main() -> Result<()> {
    println!("Starting camera tests");

    // 1. Initialize the camera
    // 0 is usually the default built-in webcam
    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)
        .context("Failed to open camera. Is it plugged in?")?;

    // Check if camera opened successfully
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    // 2. Create an OpenCV window
    highgui::named_window("vSLAM: Camera Feed", highgui::WINDOW_AUTOSIZE)?;

    // 3. Prepare a buffer for the frames
    let mut frame = Mat::default();

    println!("Starting camera feed... Press 'q' to quit.");

    loop {
        // Read a new frame from the camera
        cam.read(&mut frame)?;

        if frame.size()?.width > 0 {
            // Display the image in the window
            highgui::imshow("vSLAM: Camera Feed", &frame)?;
        }

        // Wait for 10ms and check if 'q' was pressed
        // ASCII for 'q' is 113
        let key = highgui::wait_key(10)?;
        if key == 113 {
            break;
        }
    }

    Ok(())
}
