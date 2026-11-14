//! Spherical tangential wind fields for transport benchmarks.

use std::f64::consts::PI;

use cubesphere_core::{cross3, lon_lat_wind_to_3d};
use serde::{Deserialize, Serialize};

/// Supported wind field presets for CLI / configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WindKind {
    /// Steady rigid-body spherical rotation ([`SolidBodyRotation`]).
    #[default]
    SolidBody,
    /// Time-dependent Nair & Lauritzen (2010) Case 4 deformational flow ([`NairLauritzenFlow`]).
    Deformational,
}

/// Optional closed-form parameter descriptor enabling GPU backends to evaluate winds
/// directly inside device kernels with zero host-to-device updates per stage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindDescriptor {
    /// Steady solid-body rotation parameters matching [`SolidBodyRotation`].
    SolidBody {
        /// Rotation axis inclination angle in radians (`0` = equatorial, `pi/2` = polar).
        alpha0: f64,
        /// Optional equatorial speed `u0`; `None` uses `2 * pi * R` (unit period `T = 1`).
        u0: Option<f64>,
    },
    /// Deformational flow parameters matching [`NairLauritzenFlow`].
    NairLauritzen {
        /// Full deformational return period `T` in physical time units.
        period: f64,
        /// Dimensionless vortex amplitude factor `k * T / R`.
        amplitude_factor: f64,
    },
}

/// Abstract 3D spherical tangential wind field `u(t, r)`.
///
/// Evaluating directly in 3D Cartesian space avoids any coordinate singularity
/// at the geographic North/South poles when mapping onto Cubed-Sphere panels.
pub trait WindField: Send + Sync {
    /// Short snake-case identifier of this wind field.
    fn name(&self) -> &'static str;

    /// Returns `true` if velocity varies explicitly with time `t`.
    fn is_time_dependent(&self) -> bool {
        false
    }

    /// Optional closed-form descriptor for GPU offload / JIT specialization.
    fn descriptor(&self) -> Option<WindDescriptor> {
        None
    }

    /// Evaluates the 3D Cartesian wind vector `[ux, uy, uz]` at time `t` and spherical location.
    fn velocity_3d(&self, t: f64, lon: f64, lat: f64, xyz: [f64; 3], radius: f64) -> [f64; 3];

    /// Conservative upper bound on physical wind speed `||u||` across the sphere.
    fn max_physical_speed(&self, radius: f64) -> f64;

    /// Characteristic flow period (e.g., time of 1 full cycle), if applicable.
    fn characteristic_period(&self, radius: f64) -> f64;
}

/// Williamson et al. (1992) Test Case 1: Solid Body Rotation on a Sphere.
///
/// Uses singularity-free 3D rigid rotation `u = omega x r` with angular velocity
/// `omega = (u0 / R) * (-sin(alpha0), 0, cos(alpha0))`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SolidBodyRotation {
    /// Inclination angle `alpha0` in radians (`0` = equatorial, `pi/2` = polar, `pi/4` = oblique).
    pub alpha0: f64,
    /// Optional explicit equatorial speed `u0`. When `None`, uses `2 * pi * R` (unit period `T = 1`).
    pub u0: Option<f64>,
}

impl Default for SolidBodyRotation {
    fn default() -> Self {
        Self {
            alpha0: std::f64::consts::FRAC_PI_4,
            u0: None,
        }
    }
}

impl SolidBodyRotation {
    /// Creates a solid-body rotation wind with axis inclination `alpha0` (radians)
    /// and optional equatorial speed `u0` (`None` defaults to `2 * pi * R`, unit period).
    pub fn new(alpha0: f64, u0: Option<f64>) -> Self {
        Self { alpha0, u0 }
    }

    /// Returns the equatorial speed `u0` on a sphere of `radius`, defaulting to `2 * pi * radius`.
    #[inline]
    pub fn effective_u0(&self, radius: f64) -> f64 {
        self.u0.unwrap_or(2.0 * PI * radius)
    }

    /// Angular velocity vector `omega` such that `velocity_3d = omega x r`.
    #[inline]
    pub fn angular_velocity(&self, radius: f64) -> [f64; 3] {
        let rate = self.effective_u0(radius) / radius;
        let (sa, ca) = self.alpha0.sin_cos();
        [-rate * sa, 0.0, rate * ca]
    }
}

impl WindField for SolidBodyRotation {
    fn name(&self) -> &'static str {
        "solid_body_rotation"
    }

    fn descriptor(&self) -> Option<WindDescriptor> {
        Some(WindDescriptor::SolidBody {
            alpha0: self.alpha0,
            u0: self.u0,
        })
    }

    fn velocity_3d(&self, _t: f64, _lon: f64, _lat: f64, xyz: [f64; 3], radius: f64) -> [f64; 3] {
        cross3(self.angular_velocity(radius), xyz)
    }

    fn max_physical_speed(&self, radius: f64) -> f64 {
        self.effective_u0(radius)
    }

    fn characteristic_period(&self, radius: f64) -> f64 {
        2.0 * PI * radius / self.effective_u0(radius)
    }
}

/// Nair & Lauritzen (2010) Case 4: Time-dependent non-divergent deformational flow
/// superimposed on zonal solid-body background rotation.
///
/// With `lambda' = lambda - 2*pi*t/T` and `k = amplitude_factor * R / T`:
/// - `u_lon = k * sin^2(lambda') * sin(2*lat) * cos(pi*t/T) + (2*pi*R/T) * cos(lat)`
/// - `v_lat = k * sin(2*lambda') * cos(lat) * cos(pi*t/T)`
///
/// The velocity is strictly divergence-free everywhere and returns every fluid parcel
/// back to its initial position at `t = T` (default `T = 5.0`, `amplitude_factor = 10.0`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NairLauritzenFlow {
    /// Full deformational return period `T` (default `5.0`).
    pub period: f64,
    /// Dimensionless vortex strength multiplier (default `10.0`, giving `k = 2*R` for `T = 5`).
    pub amplitude_factor: f64,
}

impl Default for NairLauritzenFlow {
    fn default() -> Self {
        Self {
            period: 5.0,
            amplitude_factor: 10.0,
        }
    }
}

impl NairLauritzenFlow {
    /// Creates a Nair & Lauritzen Case 4 flow with return `period` and dimensionless `amplitude_factor`.
    ///
    /// # Panics
    ///
    /// Panics if `period <= 0.0` (or `NaN`).
    pub fn new(period: f64, amplitude_factor: f64) -> Self {
        assert!(period > 0.0, "flow period must be positive");
        Self {
            period,
            amplitude_factor,
        }
    }
}

impl WindField for NairLauritzenFlow {
    fn name(&self) -> &'static str {
        "nair_lauritzen_case4"
    }

    fn is_time_dependent(&self) -> bool {
        true
    }

    fn descriptor(&self) -> Option<WindDescriptor> {
        Some(WindDescriptor::NairLauritzen {
            period: self.period,
            amplitude_factor: self.amplitude_factor,
        })
    }

    fn velocity_3d(&self, t: f64, lon: f64, lat: f64, _xyz: [f64; 3], radius: f64) -> [f64; 3] {
        let t_inv = 1.0 / self.period;
        let k = self.amplitude_factor * radius * t_inv;
        let bg = 2.0 * PI * radius * t_inv;

        let lon_prime = lon - 2.0 * PI * t * t_inv;
        let time_cos = (PI * t * t_inv).cos();

        let slp = lon_prime.sin();
        let u_lon = k * slp * slp * (2.0 * lat).sin() * time_cos + bg * lat.cos();
        let v_lat = k * (2.0 * lon_prime).sin() * lat.cos() * time_cos;

        lon_lat_wind_to_3d(lon, lat, u_lon, v_lat)
    }

    fn max_physical_speed(&self, radius: f64) -> f64 {
        let k = self.amplitude_factor * radius / self.period;
        let bg = 2.0 * PI * radius / self.period;
        1.15 * k + bg
    }

    fn characteristic_period(&self, _radius: f64) -> f64 {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubesphere_core::{dot3, xyz_to_lon_lat};

    #[test]
    fn test_solid_body_matches_spherical_formula_and_is_tangent() {
        let radius = 1.5;
        for alpha0 in [
            0.0,
            0.37,
            std::f64::consts::FRAC_PI_4,
            std::f64::consts::FRAC_PI_2,
        ] {
            let wind = SolidBodyRotation::new(alpha0, None);
            let u0 = wind.effective_u0(radius);
            for &lat in &[-1.1_f64, -0.3, 0.0, 0.6, 1.4] {
                for &lon in &[-2.4_f64, -0.5, 0.0, 1.2, 2.9] {
                    let xyz = [
                        radius * lat.cos() * lon.cos(),
                        radius * lat.cos() * lon.sin(),
                        radius * lat.sin(),
                    ];
                    let v3 = wind.velocity_3d(0.0, lon, lat, xyz, radius);
                    // Must be orthogonal to position vector r
                    assert!(dot3(v3, xyz).abs() < 1e-13);

                    let u_sph =
                        u0 * (lat.cos() * alpha0.cos() + lat.sin() * lon.cos() * alpha0.sin());
                    let v_sph = -u0 * lon.sin() * alpha0.sin();
                    let v_ref = lon_lat_wind_to_3d(lon, lat, u_sph, v_sph);
                    for c in 0..3 {
                        assert!((v3[c] - v_ref[c]).abs() < 1e-13);
                    }
                }
            }
        }
    }

    #[test]
    fn test_nair_lauritzen_poles_and_symmetry() {
        let radius = 2.0;
        let flow = NairLauritzenFlow::default();

        // Vanishes smoothly at North and South poles
        for &z in &[radius, -radius] {
            let (lon, lat) = xyz_to_lon_lat([0.0, 0.0, z]);
            let v = flow.velocity_3d(0.7, lon, lat, [0.0, 0.0, z], radius);
            let mag = dot3(v, v).sqrt();
            assert!(mag < 1e-12, "pole speed={mag}");
        }

        // At t = T/2, cos(pi * t / T) == 0 -> pure solid-body background rotation
        let xyz = [radius, 0.0, 0.0];
        let v_mid = flow.velocity_3d(flow.period * 0.5, 0.0, 0.0, xyz, radius);
        let expected_zonal = 2.0 * PI * radius / flow.period;
        assert!((v_mid[0]).abs() < 1e-13);
        assert!((v_mid[1] - expected_zonal).abs() < 1e-13);
        assert!((v_mid[2]).abs() < 1e-13);
    }
}
