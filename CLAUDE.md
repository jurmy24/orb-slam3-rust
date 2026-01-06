# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Visual SLAM (Simultaneous Localization and Mapping) implementation in Rust using OpenCV bindings. The project implements computer vision algorithms for tracking camera position and building 3D maps from camera feeds.

## Build Commands

### Environment Setup (macOS)

Before building, set these environment variables to locate libclang and OpenCV:

```bash
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$DYLD_LIBRARY_PATH"
export PATH="$(brew --prefix llvm)/bin:$PATH"
```

### Build & Run

```bash
# Build the project
cargo build

# Build release version
cargo build --release

# Run the main binary (placeholder)
cargo run

# Run camera feed test
cargo run --bin start-camera

# Run feature detection/landmark tracking
cargo run --bin feature-landmarks
```

## Project Architecture

### Binary Structure

The project is organized with multiple binaries for different VSLAM components:

- **src/main.rs**: Main entry point (currently a placeholder)
- **src/bin/start-camera.rs**: Camera initialization and feed display
- **src/bin/feature-landmarks.rs**: ORB feature detection and keypoint visualization

### VSLAM Pipeline

The codebase implements the "front-end" of a VSLAM system:

1. **Camera Input**: Captures frames from webcam (VideoCapture with device 0)
2. **Grayscale Conversion**: Converts BGR to grayscale for efficient processing
3. **Feature Detection**: Uses ORB (Oriented FAST and Rotated BRIEF) detector
   - Configured to detect up to 1000 keypoints
   - Uses image pyramid with 8 levels, 1.2 scale factor
   - HARRIS_SCORE for keypoint scoring
4. **Visualization**: Draws detected keypoints on the frame in red

### Key Dependencies

- **opencv** (0.98.1): Core computer vision operations, feature detection, camera I/O
- **nalgebra** (0.34.1): Linear algebra (for future pose estimation and 3D reconstruction)
- **rerun** (0.28.1): Visualization toolkit (likely for future 3D map visualization)
- **anyhow** (1.0.100): Error handling

### OpenCV Usage Patterns

All binaries follow a consistent pattern:

1. Initialize camera with `videoio::VideoCapture::new(0, videoio::CAP_ANY)`
2. Check camera status with `VideoCapture::is_opened()`
3. Create named window with `highgui::named_window()`
4. Main loop: read frames, process, display with `highgui::imshow()`
5. Exit on 'q' key press (ASCII 113)

### Future Architecture Notes

- The nalgebra and rerun dependencies suggest planned features:
  - Camera pose estimation (using nalgebra for 3D transformations)
  - 3D map visualization (using rerun for point clouds and trajectories)
- The "front-end" (feature detection) is implemented; "back-end" (bundle adjustment, loop closure) is not yet present
