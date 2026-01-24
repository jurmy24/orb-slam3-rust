//! Geometry utilities: SE3 transforms, SO3 Lie group operations, PnP solving, frame conversions.

pub mod frames;
pub mod pnp;
pub mod se3;
pub mod so3;

pub use frames::{
    body_pose_to_viz, body_position_to_viz, camera_pose_to_viz, camera_position_to_viz,
    FrameConverter,
};
pub use pnp::{PnPResult, solve_pnp_ransac, solve_pnp_ransac_detailed};
pub use se3::SE3;
pub use so3::{right_jacobian_so3, right_jacobian_so3_inv, skew};
