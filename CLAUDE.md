# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Recreate the **stereo-inertial** mode of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) in Rust. The original C++ implementation is in `/original/` for reference. We are testing exclusively on the **EuRoC MAV Dataset**.

**Scope**: Only stereo-inertial SLAM. Ignore monocular, RGB-D, and non-inertial modes.

## Build & Run

```bash
# macOS environment setup (required for OpenCV bindings)
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$DYLD_LIBRARY_PATH"
export PATH="$(brew --prefix llvm)/bin:$PATH"

# Build and run
cargo build --release
cargo run --release                           # Uses data/euroc/MH_01_easy/mav0
cargo run --release -- <path-to-dataset>      # Custom dataset path

# Run tests
cargo test
cargo test --release
```

## Project Structure

```
orb-slam3-rust/
├── src/
│   ├── system/           # Thread orchestration (SlamSystem, SharedState)
│   ├── tracking/         # Front-end: frame processing, pose estimation
│   │   └── frame/        # Camera models, ORB features, stereo matching
│   ├── local_mapping/    # Back-end: keyframe processing, triangulation
│   ├── optimizer/        # Bundle adjustment (Levenberg-Marquardt)
│   ├── atlas/            # Map data structures (KeyFrame, MapPoint, Map)
│   ├── imu/              # IMU preintegration
│   ├── geometry/         # SE3, SO3, PnP solvers
│   ├── io/               # EuRoC dataset loading
│   └── viz/              # Rerun visualization
└── data/
    ├── euroc/            # EuRoC dataset sequences
    └── ORBvoc.txt        # DBoW2 ORB vocabulary (k=10, L=6, ~970k words)
```

## Implementation Status

### Implemented (~75%)
- **Tracking**: ORB extraction, stereo matching, IMU preintegration, motion model, PnP-RANSAC pose estimation, track-local-map
- **Local Mapping**: Keyframe processing, stereo triangulation, map point culling, keyframe culling, IMU initialization
- **Optimizer**: Levenberg-Marquardt local BA, IMU preintegration factors
- **Data Structures**: KeyFrame, MapPoint, Map, covisibility graph, spanning tree
- **Vocabulary**: DBoW2 vocabulary loading, BowVector/FeatureVector computation, accelerated feature matching
- **Visualization**: Real-time Rerun viewer with trajectories, map points, feature matches

### Not Yet Implemented
- **Loop Closing**: Loop detection via BoW, Sim3 optimization, pose-graph correction, global BA
- **Multi-Map**: Atlas with multiple maps, map merging

## Key Architecture Decisions

**Threading Model**:
- Tracking runs in main thread
- LocalMapping runs in spawned background thread
- Communication via `crossbeam-channel` (bounded, capacity 5)
- Shared state via `Arc<RwLock<Atlas>>`

**No g2o Available**: The original uses g2o for optimization. We use the `levenberg-marquardt` crate instead. Custom optimization code goes in `src/optimizer/`.

**Dependencies**:
- `nalgebra` - Linear algebra and Lie groups
- `opencv` - Feature detection, stereo matching, PnP
- `levenberg-marquardt` - Optimization solver
- `rerun` - Visualization

## C++ to Rust Mapping

| C++ File | Rust Equivalent |
|----------|-----------------|
| `System.cc` | `src/system/slam_system.rs` |
| `Tracking.cc` | `src/tracking/tracker.rs` |
| `LocalMapping.cc` | `src/local_mapping/local_mapper.rs` |
| `LoopClosing.cc` | **NOT IMPLEMENTED** |
| `Optimizer.cc` | `src/optimizer/local_ba_lm.rs`, `local_ba.rs` |
| `ImuTypes.cc` | `src/imu/preintegration.rs` |
| `Frame.cc` | `src/tracking/frame/`, `tracking_frame.rs` |
| `KeyFrame.cc` | `src/atlas/map/keyframe.rs` |
| `MapPoint.cc` | `src/atlas/map/map_point.rs` |
| `Map.cc` | `src/atlas/map/map.rs` |
| `Atlas.cc` | `src/atlas/atlas.rs` |
| `KeyFrameDatabase.cc` | `src/atlas/keyframe_db.rs` |
| `ORBextractor.cc` | Uses OpenCV ORB directly |
| `ORBmatcher.cc` | `src/tracking/frame/stereo.rs` (partial) |
| `ORBVocabulary` (DBoW2) | `src/vocabulary/mod.rs` |
| `G2oTypes.cc` | `src/optimizer/imu_factors.rs` |

## Original C++ Architecture Reference

### Thread Communication
```
Main Thread (Tracking)
    │
    ▼ [KeyFrame queue - mutex protected]
LocalMapping Thread
    │
    ▼ [KeyFrame queue - mutex protected]
LoopClosing Thread
    │
    ▼ [Spawns GBA thread when loop detected]
```

### Stereo-Inertial Pipeline Per Frame
1. **GrabImageStereo**: Extract ORB features, compute stereo matches and depths
2. **GrabImuData**: Queue IMU measurements
3. **PreintegrateIMU**: Integrate IMU between frames with bias Jacobians
4. **PredictStateIMU**: Use IMU to predict initial pose guess
5. **TrackWithMotionModel**: Match features, optimize pose with IMU factor
6. **TrackLocalMap**: Refine with more map points from covisible keyframes
7. **NeedNewKeyFrame**: Decide if current frame becomes keyframe
8. **CreateNewKeyFrame**: Insert into LocalMapping queue

### IMU Preintegration (ImuTypes.cc)
```cpp
// Integrates gyro/accel between frames
IntegrateNewMeasurement(acc, gyro, dt):
    dR = dR * Exp(gyro * dt)           // Rotation delta
    dV = dV + dR * acc * dt            // Velocity delta
    dP = dP + dV * dt + 0.5 * dR * acc * dt²  // Position delta
    // Update Jacobians: JRg, JVg, JVa, JPg, JPa
    // Update covariance matrix
```

### Local Bundle Adjustment Structure
- **Optimize**: Recent keyframe poses, velocities, biases + observed map points
- **Fix**: Older keyframes that observe the same points (provide constraints)
- **Edges**: Reprojection errors + IMU preintegration constraints

## Development Notes

- Reference C++ code is in `/original/src/` and `/original/include/`
- The largest/most complex files: `Optimizer.cc` (193KB), `Tracking.cc` (138KB), `LoopClosing.cc` (97KB)
- For algorithms not available as Rust crates, implement in a separate file in the appropriate module
- Use `tracing` for structured logging (env-filter compatible)
