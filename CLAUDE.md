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
This section describes what the original C++ implementation of stereo-inertial ORB-SLAM3 looked like. Please keep this as your ultimate goal when helping me develop this in rust. It should be noted that as I'm not developing in C++ or Python, there are some things (like g2o) that I don't have access to. When you encounter this you should look for alternative rust crates or search for the source code of the specific eg. optimizer you're looking for in g2o to implement it yourself in rust. Any things like this should be implemented in a separate file to show clearly that its part of what would be a crate. 

  Overarching Sections - What, Why, How

  System (Orchestrator)

  - What: Main entry point that initializes and coordinates all SLAM components, managing thread lifecycle and sensor configuration
  - Why: Provides a clean API for users to feed sensor data and retrieve poses; abstracts away multi-threaded complexity
  - How: Creates Tracking, LocalMapping, LoopClosing objects; spawns threads for mapping/loop closing; routes TrackStereo() calls through the pipeline

  Tracking

  - What: Real-time front-end that processes each stereo+IMU frame to estimate current camera pose
  - Why: Must run at frame-rate to provide continuous localization; first line of defense against tracking loss
  - How: Extracts ORB features from left/right images, performs stereo matching for depth, preintegrates IMU measurements, predicts pose with IMU, refines with visual matching against local map

  LocalMapping

  - What: Background thread that builds and refines the local map by processing new keyframes
  - Why: Offloads expensive map building from tracking thread to maintain real-time performance; maintains map quality through optimization
  - How: Triangulates new 3D points from keyframe features, runs local bundle adjustment, culls redundant keyframes/map points, handles IMU initialization and scale refinement

  LoopClosing

  - What: Background thread that detects when the camera revisits a previously mapped area and corrects accumulated drift
  - Why: Visual-inertial odometry drifts over time; loop closures provide absolute constraints to bound this error
  - How: Queries keyframe database with Bag-of-Words vectors, validates geometric consistency with Sim3/SE3 optimization, applies pose-graph correction, triggers global bundle adjustment

  IMU Preintegration

  - What: Mathematical framework that efficiently combines high-rate IMU measurements (200Hz) between visual frames (30Hz)
  - Why: Directly integrating IMU would require re-integration when biases are updated; preintegration allows bias correction without re-processing
  - How: Integrates rotation, velocity, position deltas with Jacobians w.r.t. biases; updates can be applied analytically via first-order Taylor expansion

  Optimizer

  - What: Non-linear optimization engine using g2o that jointly optimizes poses, map points, velocities, and IMU biases
  - Why: Sensor measurements are noisy; least-squares optimization finds the most consistent state estimate given all constraints
  - How: Constructs factor graph with visual reprojection edges and IMU preintegration edges; solves using Levenberg-Marquardt or Gauss-Newton

  Atlas (Multi-Map Manager)

  - What: Container that manages one or more maps and enables map switching/merging
  - Why: Supports scenarios where tracking is lost and a new map must be started; enables map merging when revisiting old areas
  - How: Stores collection of Map objects with their keyframes and map points; provides methods to create new maps and merge existing ones

  KeyFrame & MapPoint

  - What: Core data structures representing selected frames (with full feature/pose info) and 3D landmarks in the world
  - Why: Not all frames are stored (memory); keyframes capture essential geometry; map points enable localization against the map
  - How: Keyframes store ORB features, BoW vectors, poses, IMU states, covisibility graph; MapPoints store 3D position, descriptor, observation list

  ---
  Thread Architecture & Communication

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                              MAIN THREAD                                     │
  │  ┌────────────────────────────────────────────────────────────────────────┐ │
  │  │                           TRACKING                                      │ │
  │  │  ┌──────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐ │ │
  │  │  │  Stereo  │──▶│IMU Preinteg. │──▶│ Pose Predict│──▶│Track LocalMap│ │ │
  │  │  │ Feature  │   │ & Prediction │   │ (IMU prior) │   │ (visual opt) │ │ │
  │  │  │ Extract  │   └──────────────┘   └─────────────┘   └──────────────┘ │ │
  │  │  └──────────┘                                              │           │ │
  │  └────────────────────────────────────────────────────────────│───────────┘ │
  │                                                               │             │
  │                              ▼ (new keyframe?)                │             │
  │                    ┌─────────────────────┐                    │             │
  │                    │  KeyFrame Queue     │◀───────────────────┘             │
  │                    │  (thread-safe)      │                                  │
  │                    └─────────┬───────────┘                                  │
  └──────────────────────────────│──────────────────────────────────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         │
  ┌─────────────────────────────┐                     │
  │     LOCAL MAPPING THREAD    │                     │
  │  ┌───────────────────────┐  │                     │
  │  │Process New KeyFrame   │  │                     │
  │  │- Add to covisibility  │  │                     │
  │  │- Create new MapPoints │  │                     │
  │  └───────────┬───────────┘  │                     │
  │              ▼              │                     │
  │  ┌───────────────────────┐  │                     │
  │  │ Local Bundle Adjust   │  │                     │
  │  │ (Inertial BA if IMU)  │  │                     │
  │  └───────────┬───────────┘  │                     │
  │              ▼              │                     │
  │  ┌───────────────────────┐  │    ┌────────────────────────────────┐
  │  │ IMU Initialization    │  │    │      LOOP CLOSING THREAD       │
  │  │ & Scale Refinement    │  │    │  ┌──────────────────────────┐  │
  │  └───────────┬───────────┘  │    │  │ Detect Loop Candidates   │  │
  │              ▼              │    │  │ (BoW query)              │  │
  │  ┌───────────────────────┐  │    │  └───────────┬──────────────┘  │
  │  │ KeyFrame Culling      │──┼───▶│              ▼                 │
  │  └───────────────────────┘  │    │  ┌──────────────────────────┐  │
  └─────────────────────────────┘    │  │ Compute Sim3 Transform   │  │
                                     │  └───────────┬──────────────┘  │
                                     │              ▼                 │
                                     │  ┌──────────────────────────┐  │
                                     │  │ Correct Loop / Merge Map │  │
                                     │  └───────────┬──────────────┘  │
                                     │              ▼                 │
                                     │  ┌──────────────────────────┐  │
                                     │  │ Global Bundle Adjustment │  │
                                     │  │ (spawns separate thread) │  │
                                     │  └──────────────────────────┘  │
                                     └────────────────────────────────┘

  COMMUNICATION MECHANISMS:
  ═════════════════════════
  1. Tracking → LocalMapping: KeyFrame queue (mutex-protected list)
  2. LocalMapping → LoopClosing: KeyFrame queue (mutex-protected list)
  3. LoopClosing → LocalMapping: Stop flag for GBA (atomic bool)
  4. All → Atlas: Shared map access (mutex-protected)
  5. GBA completion: Flag signals completion, results copied atomically

  SHARED DATA STRUCTURES:
  ═══════════════════════
  • Atlas: Contains all Maps, KeyFrames, MapPoints
  • KeyFrameDatabase: BoW index for place recognition
  • Covisibility Graph: Links KeyFrames sharing MapPoints
  • Spanning Tree: Hierarchical KeyFrame organization

  ---
  Step-by-Step Process (Pseudocode)

  Main Orchestrating File: src/System.cc (initialization) + src/Tracking.cc (per-frame processing)

  ═══════════════════════════════════════════════════════════════════════════════
                          SYSTEM INITIALIZATION (System.cc)
  ═══════════════════════════════════════════════════════════════════════════════

  1. LOAD_VOCABULARY(vocabulary_file)
     └── Load DBoW2 ORB vocabulary for place recognition

  2. LOAD_SETTINGS(settings_file)
     ├── Parse camera intrinsics (fx, fy, cx, cy) for left & right
     ├── Parse stereo baseline and T_c1_c2 (right-to-left transform)
     ├── Parse IMU calibration: T_body_camera, noise parameters, frequency
     └── Parse ORB extractor params: nFeatures, nLevels, scaleFactor

  3. CREATE_COMPONENTS()
     ├── Atlas = new Atlas()                    // Multi-map container
     ├── KeyFrameDatabase = new KFDatabase()    // BoW-based retrieval
     ├── Tracker = new Tracking(...)            // Front-end processor
     ├── LocalMapper = new LocalMapping(...)    // Map builder
     └── LoopCloser = new LoopClosing(...)      // Drift corrector

  4. SPAWN_THREADS()
     ├── LocalMappingThread = thread(LocalMapper.Run)
     ├── LoopClosingThread = thread(LoopCloser.Run)
     └── [Optional] ViewerThread = thread(Viewer.Run)

  5. CONNECT_COMPONENTS()
     ├── Tracker.SetLocalMapper(LocalMapper)
     ├── Tracker.SetLoopClosing(LoopCloser)
     ├── LocalMapper.SetLoopCloser(LoopCloser)
     └── LoopCloser.SetLocalMapper(LocalMapper)

  ═══════════════════════════════════════════════════════════════════════════════
                       PER-FRAME PROCESSING (Tracking.cc)
  ═══════════════════════════════════════════════════════════════════════════════

  FOR EACH FRAME (imLeft, imRight, timestamp, imuMeasurements):

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 1: GRAB_IMAGE_STEREO(imLeft, imRight, timestamp)                       │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │   1.1 Convert images to grayscale if needed                                 │
  │   1.2 Create Frame object:                                                  │
  │       ├── Extract ORB features from LEFT image (N keypoints)                │
  │       ├── Extract ORB features from RIGHT image                             │
  │       ├── COMPUTE_STEREO_MATCHES():                                         │
  │       │   ├── For each left keypoint, search in right image (epipolar)      │
  │       │   ├── Match by ORB descriptor distance (Hamming)                    │
  │       │   ├── Sub-pixel refinement with parabola fitting                    │
  │       │   └── Compute depth: depth = baseline * fx / disparity              │
  │       ├── Assign keypoints to grid cells (for fast search)                  │
  │       └── Store: mvKeys, mvKeysRight, mvDepth, mDescriptors                 │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 2: GRAB_IMU_DATA(imuMeasurements)                                      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │   2.1 Add all IMU points to queue: mlQueueImuData.push_back(...)            │
  │   2.2 Each IMU::Point contains: (ax, ay, az, wx, wy, wz, timestamp)         │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ STEP 3: TRACK()  [Main tracking logic]                                      │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                             │
  │   3.1 PREINTEGRATE_IMU():                                                   │
  │       ├── Get IMU measurements between last frame and current frame         │
  │       ├── Create new Preintegrated object with current bias estimate        │
  │       ├── FOR EACH IMU measurement:                                         │
  │       │   └── preintegrated.IntegrateNewMeasurement(acc, gyro, dt)          │
  │       │       ├── Update deltaR (rotation) using gyro                       │
  │       │       ├── Update dV (velocity) using acc rotated by deltaR          │
  │       │       ├── Update dP (position) using dV                             │
  │       │       ├── Update Jacobians (JRg, JVg, JVa, JPg, JPa)                │
  │       │       └── Update covariance matrix C                                │
  │       └── Store preintegrated in current Frame                              │
  │                                                                             │
  │   3.2 PREDICT_STATE_IMU():                                                  │
  │       ├── Get last frame pose (R, p), velocity (v), bias (b)                │
  │       ├── Get preintegrated deltas (dR, dV, dP)                             │
  │       ├── Predict current pose:                                             │
  │       │   R_curr = R_last * dR                                              │
  │       │   v_curr = v_last + gravity*dt + R_last*dV                          │
  │       │   p_curr = p_last + v_last*dt + 0.5*gravity*dt² + R_last*dP         │
  │       └── Set as initial guess for visual optimization                      │
  │                                                                             │
  │   3.3 IF (NOT INITIALIZED):                                                 │
  │       └── STEREO_INITIALIZATION():                                          │
  │           ├── Set first frame as KeyFrame                                   │
  │           ├── For each stereo-matched feature with valid depth:             │
  │           │   └── Create MapPoint at 3D position (triangulated)             │
  │           ├── Insert KeyFrame into Atlas and LocalMapping queue             │
  │           └── Set state = OK                                                │
  │                                                                             │
  │   3.4 ELSE (TRACKING):                                                      │
  │       │                                                                     │
  │       ├── 3.4.1 TRACK_WITH_MOTION_MODEL() or TRACK_REFERENCE_KF():          │
  │       │   ├── Project local map points into current frame                   │
  │       │   ├── Search for matches in feature grid (guided by IMU prediction) │
  │       │   ├── POSE_OPTIMIZATION() or POSE_INERTIAL_OPTIMIZATION():          │
  │       │   │   ├── Create g2o graph with pose vertex                         │
  │       │   │   ├── Add reprojection edges for each matched MapPoint          │
  │       │   │   ├── [Inertial] Add IMU edge linking to previous frame         │
  │       │   │   ├── Run optimization (4 iterations)                           │
  │       │   │   ├── Mark outliers by chi-square test                          │
  │       │   │   └── Return optimized pose                                     │
  │       │   └── Check inlier count → if too few, tracking may be lost         │
  │       │                                                                     │
  │       ├── 3.4.2 TRACK_LOCAL_MAP():                                          │
  │       │   ├── UPDATE_LOCAL_MAP():                                           │
  │       │   │   ├── Find covisible keyframes (share MapPoints)                │
  │       │   │   └── Collect all MapPoints observed by local keyframes         │
  │       │   ├── SEARCH_LOCAL_POINTS():                                        │
  │       │   │   ├── Project each local MapPoint into current frame            │
  │       │   │   ├── Check if in frustum and not already matched               │
  │       │   │   └── Search for descriptor match in predicted location         │
  │       │   ├── POSE_OPTIMIZATION() again with more matches                   │
  │       │   └── Update tracking statistics                                    │
  │       │                                                                     │
  │       └── 3.4.3 NEED_NEW_KEYFRAME():                                        │
  │           ├── Check conditions:                                             │
  │           │   ├── Enough frames since last keyframe (temporal)              │
  │           │   ├── Local mapping not busy (queue not too full)               │
  │           │   ├── Current frame tracks enough points                        │
  │           │   ├── Enough new points observed (not in last KF)               │
  │           │   └── [Inertial] Enough IMU measurements accumulated            │
  │           └── IF conditions met:                                            │
  │               └── CREATE_NEW_KEYFRAME():                                    │
  │                   ├── Create KeyFrame from current Frame                    │
  │                   ├── Copy pose, features, IMU state                        │
  │                   ├── Insert into LocalMapping queue                        │
  │                   └── Update reference keyframe                             │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘

  RETURN current_pose (SE3)

  ═══════════════════════════════════════════════════════════════════════════════
                       LOCAL MAPPING THREAD (LocalMapping.cc)
  ═══════════════════════════════════════════════════════════════════════════════

  WHILE (not shutdown):

     WAIT for new keyframe in queue

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ PROCESS_NEW_KEYFRAME():                                                 │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   1. Compute BoW vector for keyframe (for place recognition)            │
     │   2. Update covisibility graph:                                         │
     │      ├── Find keyframes that observe same MapPoints                     │
     │      └── Add edges weighted by shared observations                      │
     │   3. Update spanning tree (parent-child relationships)                  │
     │   4. Associate MapPoints with keyframe observations                     │
     │   5. Insert into KeyFrameDatabase (for loop detection)                  │
     └─────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ CREATE_NEW_MAP_POINTS():                                                │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   1. Get N best covisible keyframes                                     │
     │   2. FOR EACH covisible keyframe pair (KF1, KF2):                       │
     │      ├── Search for unmatched feature correspondences                   │
     │      ├── Check epipolar constraint                                      │
     │      ├── Triangulate 3D point                                           │
     │      ├── Check reprojection error and parallax                          │
     │      └── Create MapPoint if valid                                       │
     └─────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ MAP_POINT_CULLING():                                                    │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   Remove MapPoints that:                                                │
     │   ├── Have low observation ratio (seen < 25% of expected times)         │
     │   └── Have been observed by < 3 keyframes after creation                │
     └─────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ LOCAL_BUNDLE_ADJUSTMENT() / LOCAL_INERTIAL_BA():                        │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   1. Collect local keyframes (recent + covisible)                       │
     │   2. Collect fixed keyframes (neighbors of local, but not optimized)    │
     │   3. Collect all MapPoints observed by local keyframes                  │
     │   4. Build g2o graph:                                                   │
     │      ├── Add pose vertices for each keyframe                            │
     │      ├── Add 3D point vertices for each MapPoint                        │
     │      ├── Add reprojection edges (KeyFrame-MapPoint)                     │
     │      └── [Inertial] Add IMU edges between consecutive keyframes:        │
     │          ├── IMU preintegration constraint                              │
     │          ├── Velocity vertices                                          │
     │          └── Bias vertices (with random walk prior)                     │
     │   5. Run optimization (5-10 iterations)                                 │
     │   6. Remove outlier observations                                        │
     │   7. Update poses, points, velocities, biases                           │
     └─────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ [INERTIAL] IMU_INITIALIZATION() - Called early in sequence:             │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   1. Collect keyframes from last N seconds                              │
     │   2. Estimate gravity direction from accelerometer (static assumption)  │
     │   3. Initialize velocity estimates                                      │
     │   4. INERTIAL_OPTIMIZATION():                                           │
     │      ├── Optimize: gravity direction, scale, gyro bias, accel bias      │
     │      ├── Fix visual structure, vary IMU parameters                      │
     │      └── Iterate until convergence                                      │
     │   5. Apply scale correction to map (for monocular-inertial)             │
     │   6. Mark IMU as initialized                                            │
     └─────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ KEYFRAME_CULLING():                                                     │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   FOR EACH local keyframe:                                              │
     │   ├── Count observations seen by ≥3 other keyframes                     │
     │   ├── IF >90% of points are redundant:                                  │
     │   │   └── Mark keyframe as bad (remove from active set)                 │
     │   └── [Inertial] Don't cull if needed for IMU chain                     │
     └─────────────────────────────────────────────────────────────────────────┘

     Insert keyframe into LoopClosing queue

  ═══════════════════════════════════════════════════════════════════════════════
                        LOOP CLOSING THREAD (LoopClosing.cc)
  ═══════════════════════════════════════════════════════════════════════════════

  WHILE (not shutdown):

     WAIT for new keyframe in queue

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ DETECT_COMMON_REGIONS() / NEW_DETECT_COMMON_REGIONS():                  │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   1. Query KeyFrameDatabase with current KF's BoW vector                │
     │   2. Get candidate keyframes with similar appearance                    │
     │   3. Filter by covisibility (exclude recent neighbors)                  │
     │   4. FOR EACH candidate:                                                │
     │      ├── Match ORB features between current and candidate               │
     │      ├── IF enough matches:                                             │
     │      │   ├── Compute Sim3 transform (includes scale for mono)           │
     │      │   ├── RANSAC to remove outliers                                  │
     │      │   └── Refine with more correspondences                           │
     │      └── IF geometric verification passes → loop detected!              │
     └─────────────────────────────────────────────────────────────────────────┘

     IF loop detected:

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ CORRECT_LOOP():                                                         │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   1. Request LocalMapping to pause (set stop flag)                      │
     │   2. Wait for LocalMapping to acknowledge                               │
     │   3. Compute corrected poses for current keyframe's neighborhood        │
     │   4. Propagate correction through covisibility graph                    │
     │   5. Fuse duplicate MapPoints (same 3D point, different observations)   │
     │   6. OPTIMIZE_ESSENTIAL_GRAPH():                                        │
     │      ├── Pose-graph optimization (only poses, not points)               │
     │      ├── Loop edge provides strong constraint                           │
     │      ├── [Inertial] 4-DoF optimization (fix roll/pitch from gravity)    │
     │      └── Distribute error across entire trajectory                      │
     │   7. Update all MapPoint positions based on corrected KF poses          │
     │   8. Resume LocalMapping                                                │
     └─────────────────────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────────────────────────────────────┐
     │ RUN_GLOBAL_BUNDLE_ADJUSTMENT() - Spawned in separate thread:            │
     ├─────────────────────────────────────────────────────────────────────────┤
     │   1. Stop LocalMapping                                                  │
     │   2. FULL_INERTIAL_BA() or GLOBAL_BUNDLE_ADJUSTMENT():                  │
     │      ├── Optimize ALL keyframe poses                                    │
     │      ├── Optimize ALL MapPoint positions                                │
     │      ├── [Inertial] Optimize ALL velocities and biases                  │
     │      └── Run for many iterations (20+)                                  │
     │   3. Merge optimized results into map                                   │
     │   4. Signal GBA completion                                              │
     │   5. Resume LocalMapping                                                │
     └─────────────────────────────────────────────────────────────────────────┘

  ═══════════════════════════════════════════════════════════════════════════════
                              SHUTDOWN (System.cc)
  ═══════════════════════════════════════════════════════════════════════════════

  1. Signal all threads to stop
  2. Wait for thread completion (join)
  3. Save trajectory to file if requested
  4. Save map to file if requested
  5. Clean up resources

  ---
  Key Files Summary
  ┌───────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────┐
  │                       File                        │                                      Role                                      │
  ├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ src/System.cc                                     │ Main orchestrator - initialization, thread spawning, user API                  │
  ├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ src/Tracking.cc                                   │ Per-frame processing - feature extraction, IMU preintegration, pose estimation │
  ├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ src/LocalMapping.cc                               │ Map building - new points, local BA, IMU initialization                        │
  ├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ src/LoopClosing.cc                                │ Drift correction - loop detection, pose-graph optimization, global BA          │
  ├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ src/Optimizer.cc                                  │ All g2o-based optimizations - local BA, inertial BA, pose optimization         │
  ├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ src/ImuTypes.cc                                   │ IMU preintegration math                                                        │
  ├───────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────┤
  │ Examples/Stereo-Inertial/stereo_inertial_euroc.cc │ Example showing how to use the system

## Development Guidelines

- **Plan before implementing**: Always create a todo list before starting work on a feature or fix
- **Write clean, concise code**: Favor readability and simplicity over cleverness
- **Avoid over-engineering**: Only implement what's needed for the current task
- **Follow Rust best practices**: If you're unsure, check what is recommended for Rust in your implementation