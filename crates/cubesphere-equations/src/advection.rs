//! Spherical scalar advection equation and analytical solution/wind evaluation.

use std::sync::Arc;

use cubesphere_core::{cross3, dot3, xyz_to_lon_lat, CubedSphereMesh, FloatScalar};

use crate::initial::ScalarField;
use crate::wind::{SolidBodyRotation, WindField};

/// Abstract PDE interface enabling future extension beyond scalar transport
/// (e.g., tracer coupled systems or shallow water equations).
pub trait Equation: Send + Sync {
    /// Short snake-case identifier of the PDE system (for example `"scalar_advection"`).
    fn name(&self) -> &'static str;
    /// Number of prognostic scalar variables carried at each mesh node.
    fn n_vars(&self) -> usize;
    /// Returns `true` if the PDE coefficients vary explicitly with physical time `t`.
    fn is_time_dependent(&self) -> bool;
}

/// Spherical scalar advection equation `d(phi)/dt + div(phi * u) = 0`.
///
/// Couples a prescribed [`WindField`] and initial [`ScalarField`] to evaluate initial states,
/// analytical reference solutions (when available), and 3D or contravariant winds on a mesh.
pub struct ScalarAdvection {
    wind: Arc<dyn WindField>,
    initial: Arc<dyn ScalarField>,
    /// Optional angular velocity info allowing exact closed-form evaluation at arbitrary `t`
    /// under solid-body rotation.
    solid_body_params: Option<SolidBodyRotation>,
}

impl ScalarAdvection {
    /// Constructs a scalar advection problem from a spherical wind field and initial tracer field.
    ///
    /// [`Self::evaluate_exact`] only returns values at integer multiples of the wind's
    /// characteristic period; use [`Self::solid_body`] when closed-form rotation at arbitrary `t`
    /// is desired.
    pub fn new(wind: Arc<dyn WindField>, initial: Arc<dyn ScalarField>) -> Self {
        Self {
            wind,
            initial,
            solid_body_params: None,
        }
    }

    /// Constructs a scalar advection problem under steady [`SolidBodyRotation`].
    ///
    /// Stores the rotation parameters so [`Self::evaluate_exact`] can evaluate the analytical
    /// solution at any time `t` via backward Rodrigues rotation.
    pub fn solid_body(wind: SolidBodyRotation, initial: Arc<dyn ScalarField>) -> Self {
        Self {
            wind: Arc::new(wind),
            initial,
            solid_body_params: Some(wind),
        }
    }

    /// Returns a shared reference to the prescribed spherical wind field.
    pub fn wind(&self) -> &dyn WindField {
        self.wind.as_ref()
    }

    /// Returns a cloned `Arc` handle to the wind field for sharing with compute backends.
    pub fn wind_arc(&self) -> Arc<dyn WindField> {
        Arc::clone(&self.wind)
    }

    /// Returns a shared reference to the initial scalar field `phi(t = 0)`.
    pub fn initial(&self) -> &dyn ScalarField {
        self.initial.as_ref()
    }

    /// Evaluates initial tracer values across all mesh nodes in target precision `T`.
    ///
    /// Evaluates in `f64` using `mesh.lon_f64`, `mesh.lat_f64`, and `mesh.xyz` in `[elem][j][i]`
    /// node order before converting each sample to `T`.
    pub fn evaluate_initial<T: FloatScalar>(&self, mesh: &CubedSphereMesh<T>) -> Vec<T> {
        let r = mesh.config.radius;
        let mut out = vec![T::ZERO; mesh.total_nodes];
        for (idx, dst) in out.iter_mut().enumerate() {
            let xyz = [
                mesh.xyz[0][idx].to_f64(),
                mesh.xyz[1][idx].to_f64(),
                mesh.xyz[2][idx].to_f64(),
            ];
            let val = self
                .initial
                .eval(mesh.lon_f64[idx], mesh.lat_f64[idx], xyz, r);
            *dst = T::from_f64(val);
        }
        out
    }

    /// Evaluates exact analytical solution at time `t`, if available.
    ///
    /// - For [`SolidBodyRotation`], uses Rodrigues' backward rotation at arbitrary `t`.
    /// - For time-dependent periodic flows (such as [`crate::NairLauritzenFlow`]), returns the
    ///   exact initial profile when `t` is an integer multiple of `characteristic_period`.
    pub fn evaluate_exact<T: FloatScalar>(
        &self,
        mesh: &CubedSphereMesh<T>,
        t: f64,
    ) -> Option<Vec<T>> {
        let r = mesh.config.radius;

        if let Some(sb) = self.solid_body_params {
            let omega = sb.angular_velocity(r);
            let omega_mag = dot3(omega, omega).sqrt();
            if omega_mag == 0.0 || t.abs() < 1e-14 {
                return Some(self.evaluate_initial(mesh));
            }
            let axis = [
                omega[0] / omega_mag,
                omega[1] / omega_mag,
                omega[2] / omega_mag,
            ];
            let angle = -omega_mag * t; // backward along characteristic
            let (sin_a, cos_a) = angle.sin_cos();

            let mut exact = vec![T::ZERO; mesh.total_nodes];
            for (idx, dst) in exact.iter_mut().enumerate() {
                let p = [
                    mesh.xyz[0][idx].to_f64(),
                    mesh.xyz[1][idx].to_f64(),
                    mesh.xyz[2][idx].to_f64(),
                ];
                let k_cross_p = cross3(axis, p);
                let k_dot_p = dot3(axis, p);
                let p0 = [
                    p[0] * cos_a + k_cross_p[0] * sin_a + axis[0] * k_dot_p * (1.0 - cos_a),
                    p[1] * cos_a + k_cross_p[1] * sin_a + axis[1] * k_dot_p * (1.0 - cos_a),
                    p[2] * cos_a + k_cross_p[2] * sin_a + axis[2] * k_dot_p * (1.0 - cos_a),
                ];
                let (lon0, lat0) = xyz_to_lon_lat(p0);
                *dst = T::from_f64(self.initial.eval(lon0, lat0, p0, r));
            }
            return Some(exact);
        }

        let period = self.wind.characteristic_period(r);
        if period > 0.0 {
            let cycles = t / period;
            if (cycles - cycles.round()).abs() < 1e-9 {
                return Some(self.evaluate_initial(mesh));
            }
        }
        None
    }

    /// Evaluates 3D Cartesian wind vectors `[ux, uy, uz]` in `f64` at every node for time `t`.
    ///
    /// Output length is `mesh.total_nodes` in `[elem][j][i]` node order.
    pub fn evaluate_wind_3d<T: FloatScalar>(
        &self,
        mesh: &CubedSphereMesh<T>,
        t: f64,
    ) -> Vec<[f64; 3]> {
        let r = mesh.config.radius;
        let mut wind_3d = Vec::with_capacity(mesh.total_nodes);
        for idx in 0..mesh.total_nodes {
            let xyz = [
                mesh.xyz[0][idx].to_f64(),
                mesh.xyz[1][idx].to_f64(),
                mesh.xyz[2][idx].to_f64(),
            ];
            wind_3d.push(
                self.wind
                    .velocity_3d(t, mesh.lon_f64[idx], mesh.lat_f64[idx], xyz, r),
            );
        }
        wind_3d
    }

    /// Projects analytical wind onto contravariant components `(u^alpha, u^beta)` at time `t`.
    ///
    /// Writes `mesh.total_nodes` values in `[elem][j][i]` order in precision `T` using the
    /// panel `(alpha, beta) in [-pi/4, pi/4]^2` contravariant basis vectors.
    ///
    /// # Panics
    ///
    /// Panics if `u_contra.len() != mesh.total_nodes` or `v_contra.len() != mesh.total_nodes`.
    pub fn evaluate_contravariant_wind<T: FloatScalar>(
        &self,
        mesh: &CubedSphereMesh<T>,
        t: f64,
        u_contra: &mut [T],
        v_contra: &mut [T],
    ) {
        let wind_3d = self.evaluate_wind_3d(mesh, t);
        mesh.project_wind_3d(&wind_3d, u_contra, v_contra);
    }
}

impl Equation for ScalarAdvection {
    fn name(&self) -> &'static str {
        "scalar_advection"
    }

    fn n_vars(&self) -> usize {
        1
    }

    fn is_time_dependent(&self) -> bool {
        self.wind.is_time_dependent()
    }
}
