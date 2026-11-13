use std::f64::consts::PI;

use cs_core::great_circle_distance;
use serde::{Deserialize, Serialize};

/// Supported initial scalar field presets for CLI / configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InitialKind {
    #[default]
    GaussianBell,
    CosineBell,
    TwinGaussian,
    Constant,
}

/// Abstract scalar field on the sphere `phi(lon, lat, xyz)`.
pub trait ScalarField: Send + Sync {
    fn name(&self) -> &'static str;
    fn eval(&self, lon: f64, lat: f64, xyz: [f64; 3], radius: f64) -> f64;
}

/// Uniform constant field `phi = value` for free-stream preservation tests.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConstantField {
    pub value: f64,
}

impl Default for ConstantField {
    fn default() -> Self {
        Self { value: 1.0 }
    }
}

impl ScalarField for ConstantField {
    fn name(&self) -> &'static str {
        "constant"
    }

    fn eval(&self, _lon: f64, _lat: f64, _xyz: [f64; 3], _radius: f64) -> f64 {
        self.value
    }
}

/// Smooth Gaussian Bell tracer matching `P2_new.ipynb`:
/// `phi = height * exp(-dist^2 / (2 * sigma^2))`, with `sigma = R * sigma_ratio`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GaussianBell {
    pub center_lon: f64,
    pub center_lat: f64,
    pub height: f64,
    pub sigma_ratio: f64,
}

impl Default for GaussianBell {
    fn default() -> Self {
        Self {
            center_lon: 0.0,
            center_lat: 0.0,
            height: 1.0,
            sigma_ratio: 1.0 / 3.0,
        }
    }
}

impl ScalarField for GaussianBell {
    fn name(&self) -> &'static str {
        "gaussian_bell"
    }

    fn eval(&self, lon: f64, lat: f64, _xyz: [f64; 3], radius: f64) -> f64 {
        let sigma = radius * self.sigma_ratio;
        let dist = great_circle_distance(lon, lat, self.center_lon, self.center_lat, radius);
        self.height * (-(dist * dist) / (2.0 * sigma * sigma)).exp()
    }
}

/// Williamson et al. (1992) Test Case 1 Cosine Bell (`C^1` continuous with compact support).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CosineBell {
    pub center_lon: f64,
    pub center_lat: f64,
    pub height: f64,
    pub radius_ratio: f64,
}

impl Default for CosineBell {
    fn default() -> Self {
        Self {
            center_lon: 1.5 * PI,
            center_lat: 0.0,
            height: 1.0,
            radius_ratio: 1.0 / 3.0,
        }
    }
}

impl ScalarField for CosineBell {
    fn name(&self) -> &'static str {
        "cosine_bell"
    }

    fn eval(&self, lon: f64, lat: f64, _xyz: [f64; 3], radius: f64) -> f64 {
        let r_bell = radius * self.radius_ratio;
        let dist = great_circle_distance(lon, lat, self.center_lon, self.center_lat, radius);
        if dist < r_bell {
            0.5 * self.height * (1.0 + (PI * dist / r_bell).cos())
        } else {
            0.0
        }
    }
}

/// Twin symmetric Gaussian bells suited for Nair & Lauritzen (2010) Case 4 deformational flow.
///
/// Placed at `(5*pi/6, 0)` and `(-5*pi/6, 0)` by default (`lon = pi +- pi/6`), providing a smooth
/// `C^inf` dual-vortex tracer distribution that undergoes strong filamentation at `t = T/2`
/// before returning identically at `t = T`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TwinGaussianBells {
    pub lon1: f64,
    pub lat1: f64,
    pub lon2: f64,
    pub lat2: f64,
    pub height: f64,
    pub sigma_ratio: f64,
}

impl Default for TwinGaussianBells {
    fn default() -> Self {
        Self {
            lon1: 5.0 * PI / 6.0,
            lat1: 0.0,
            lon2: -5.0 * PI / 6.0,
            lat2: 0.0,
            height: 1.0,
            // Matches Nair & Lauritzen (2010) Case 4 Gaussian hill decay b0 = 5 -> 1 / sqrt(2 * b0)
            sigma_ratio: 1.0 / 10.0_f64.sqrt(),
        }
    }
}

impl ScalarField for TwinGaussianBells {
    fn name(&self) -> &'static str {
        "twin_gaussian"
    }

    fn eval(&self, lon: f64, lat: f64, _xyz: [f64; 3], radius: f64) -> f64 {
        let sigma = radius * self.sigma_ratio;
        let denom = 2.0 * sigma * sigma;
        let d1 = great_circle_distance(lon, lat, self.lon1, self.lat1, radius);
        let d2 = great_circle_distance(lon, lat, self.lon2, self.lat2, radius);
        self.height * ((-d1 * d1 / denom).exp() + (-d2 * d2 / denom).exp())
    }
}
