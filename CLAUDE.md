# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust-based implementation of [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), a versatile Visual-Inertial SLAM system. The implementation follows the algorithms described in the [ORB-SLAM3 paper](https://arxiv.org/pdf/2007.11898).

**Target Dataset**: [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) - a benchmark for visual-inertial odometry featuring stereo cameras and IMU data collected from a micro aerial vehicle.

**Key Features**:
- Stereo camera support for depth estimation
- IMU integration for visual-inertial odometry
- ORB feature detection and matching

## ORB-SLAM3 Architecture

The system consists of four main components (from Figure 1 of the paper):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRACKING                                        │
│  ┌───────┐   ┌───────────┐   ┌─────────────────────┐   ┌─────────────────┐  │
│  │ Frame │──▶│Extract ORB│──▶│ Initial Pose Est.   │──▶│ Track Local Map │  │
│  └───────┘   └───────────┘   │ (last frame/reloc/  │   └────────┬────────┘  │
│  ┌───────┐   ┌───────────┐   │  map creation)      │            │           │
│  │  IMU  │──▶│IMU Integr.│──▶└─────────────────────┘   ┌────────▼────────┐  │
│  └───────┘   └───────────┘                             │New KF Decision  │  │
│                                                        └────────┬────────┘  │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │ KeyFrame
┌─────────────────────────────────────────────────────────────────┼───────────┐
│                              ATLAS                              │           │
│  ┌────────────────────┐  ┌─────────────┐  ┌─────────────┐       │           │
│  │ DBoW2 KEYFRAME DB  │  │ Active Map  │  │Non-active   │       │           │
│  │ ┌────────────────┐ │  │ ┌─────────┐ │  │   Map       │       │           │
│  │ │Visual Vocabul. │ │  │ │MapPoints│ │  │ ┌─────────┐ │       │           │
│  │ └────────────────┘ │  │ ├─────────┤ │  │ │MapPoints│ │       │           │
│  │ ┌────────────────┐ │  │ │KeyFrames│ │  │ ├─────────┤ │       │           │
│  │ │Recognition DB  │ │  │ ├─────────┤ │  │ │KeyFrames│ │       │           │
│  │ └────────────────┘ │  │ │Covisib. │ │  │ ├─────────┤ │       │           │
│  └────────────────────┘  │ │  Graph  │ │  │ │Covisib. │ │       │           │
│                          │ ├─────────┤ │  │ │ Graph   │ │       │           │
│                          │ │Spanning │ │  │ ├─────────┤ │       │           │
│                          │ │  Tree   │ │  │ │Spanning │ │       │           │
│                          │ └─────────┘ │  │ │  Tree   │ │       │           │
│                          └─────────────┘  │ └─────────┘ │       │           │
│                                           └─────────────┘       │           │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │
┌────────────────────────────────────────────┐  ┌─────────────────┼───────────┐
│         LOOP & MAP MERGING                 │  │          LOCAL MAPPING      │
│  ┌──────────────────────────────────────┐  │  │  ┌─────────────▼─────────┐  │
│  │ Place Recognition                    │  │  │  │ KeyFrame Insertion    │  │
│  │  ┌──────────────┐ ┌───────────────┐  │  │  │  ├───────────────────────┤  │
│  │  │Database Query│ │Compute Sim3/  │  │  │  │  │ Recent MapPts Culling │  │
│  │  └──────────────┘ │    SE3        │  │  │  │  ├───────────────────────┤  │
│  │                   └───────────────┘  │  │  │  │ New Points Creation   │  │
│  └──────────────────────────────────────┘  │  │  ├───────────────────────┤  │
│  ┌──────────────────────────────────────┐  │  │  │ Local BA              │  │
│  │ Loop Correction                      │  │  │  ├───────────────────────┤  │
│  │  ┌───────────┐ ┌───────────────────┐ │  │  │  │ IMU Initialization    │  │
│  │  │Loop Fusion│ │Optimize Essential │ │  │  │  ├───────────────────────┤  │
│  │  └───────────┘ │     Graph         │ │  │  │  │ Local KF Culling      │  │
│  │                └───────────────────┘ │  │  │  ├───────────────────────┤  │
│  └──────────────────────────────────────┘  │  │  │ IMU Scale Refinement  │  │
│  ┌──────────────────────────────────────┐  │  │  └───────────────────────┘  │
│  │ Map Merging                          │  │  └─────────────────────────────┘
│  │  ┌──────────┐ ┌──────────┐ ┌──────┐  │  │
│  │  │Merge Maps│ │Welding BA│ │Opt.  │  │  │
│  │  └──────────┘ └──────────┘ │E.Graph│ │  │
│  │                            └──────┘  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│      FULL BA       │
│ ┌────────────────┐ │
│ │   Map Update   │ │
│ │   + Full BA    │ │
│ └────────────────┘ │
└────────────────────┘
```

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| **TRACKING** | | |
| Extract ORB | ✅ Done | `src/frontend/features.rs` |
| IMU Integration | ✅ Done | `src/math/preintegration.rs` |
| Initial Pose Estimation | ✅ Done | `src/estimator/vi_estimator.rs` (PnP + IMU prior) |
| Track Local Map | ❌ Missing | Currently only tracks previous frame |
| New KeyFrame Decision | ❌ Missing | |
| **ATLAS** | | |
| Active Map (MapPoints, KeyFrames) | ❌ Missing | |
| Covisibility Graph | ❌ Missing | |
| Spanning Tree | ❌ Missing | |
| Non-active Maps | ❌ Missing | |
| DBoW2 Visual Vocabulary | ❌ Missing | |
| **LOCAL MAPPING** | | |
| KeyFrame Insertion | ❌ Missing | |
| MapPoints Culling | ❌ Missing | |
| New Points Creation | ❌ Missing | |
| Local Bundle Adjustment | ❌ Missing | |
| IMU Initialization | ❌ Missing | |
| IMU Scale Refinement | ❌ Missing | |
| **LOOP & MAP MERGING** | | |
| Place Recognition | ❌ Missing | |
| Loop Fusion | ❌ Missing | |
| Essential Graph Optimization | ❌ Missing | |
| Map Merging | ❌ Missing | |
| **FULL BA** | ❌ Missing | |

## Development Guidelines

- **Plan before implementing**: Always create a todo list before starting work on a feature or fix
- **Write clean, concise code**: Favor readability and simplicity over cleverness
- **Avoid over-engineering**: Only implement what's needed for the current task
- **Follow Rust idioms**: Use proper error handling with `Result`/`Option`, leverage the type system

## Build Commands

### Environment Setup (macOS)

Before building, set these environment variables to locate libclang and OpenCV:

```bash
export LIBCLANG_PATH="$(brew --prefix llvm)/lib"
export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$DYLD_LIBRARY_PATH"
export PATH="$(brew --prefix llvm)/bin:$PATH"
```


### References

- [ORB-SLAM3 Repository](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [ORB-SLAM3 Paper](https://arxiv.org/pdf/2007.11898) - Campos et al., "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM"
- [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
