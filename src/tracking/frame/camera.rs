use nalgebra::Matrix3;

#[derive(Debug, Clone, Copy)]
pub struct CameraModel {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub baseline: f64,
}

impl CameraModel {
    pub fn from_k_and_baseline(k: Matrix3<f64>, baseline: f64) -> Self {
        Self {
            fx: k[(0, 0)],
            fy: k[(1, 1)],
            cx: k[(0, 2)],
            cy: k[(1, 2)],
            baseline,
        }
    }
}
