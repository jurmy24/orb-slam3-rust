# rust-vslam

A Rust implementation of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), a Visual-Inertial SLAM system. This one focuses exclusively on Stereo Vision + IMU data. 

## Overview

This project implements the algorithms described in the [ORB-SLAM3 paper](https://arxiv.org/pdf/2007.11898) (Campos et al., 2021) in Rust, targeting the [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) which provides:

- **Stereo camera images** from a global shutter camera rig
- **IMU measurements** from a synchronized inertial measurement unit
- **Ground truth poses** for evaluation

## References

- [ORB-SLAM3 Repository](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [ORB-SLAM3 Paper (arXiv)](https://arxiv.org/pdf/2007.11898)
- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

## Build & Run

```bash
# Build the project
cargo build

# Build release version (recommended for performance)
cargo build --release

# Run the main binary
cargo run --release
```

> [NOTE]: You might need to run these to tell the Rust compiler where libclang and OpenCV are located:

```bash
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$DYLD_LIBRARY_PATH"
export PATH="$(brew --prefix llvm)/bin:$PATH"
```