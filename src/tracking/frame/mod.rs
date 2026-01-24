//! Frame-level processing: camera model, ORB feature detection, stereo matching.

pub mod camera;
pub mod stereo;

pub use camera::CameraModel;
pub use stereo::{
    FeatureSet, NN_RATIO, StereoFrame, StereoProcessor, TH_HIGH, TH_LOW, descriptor_distance,
};
