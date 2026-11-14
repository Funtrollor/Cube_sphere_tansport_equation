//! Five-stage fourth-order low-storage Runge-Kutta integration and angular CFL estimates.
use std::time::Instant;

use cubesphere_core::{CubedSphereMesh, FloatScalar};
use cubesphere_equations::{Equation, ScalarAdvection};
use serde::{Deserialize, Serialize};

use crate::backend::ComputeBackend;

/// Carpenter & Kennedy (1994) 5-stage, 4th-order Low-Storage Runge-Kutta (2N-storage) coefficients.
/// Residual multipliers `du = A[stage] * du + dt * rhs`, in stage order 0 through 4.
pub const RK45_A: [f64; 5] = [
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
];

/// Solution-update multipliers `u += B[stage] * du` for Carpenter-Kennedy LSRK45.
/// Dimensionless `f64` coefficients in stage order 0 through 4.
pub const RK45_B: [f64; 5] = [
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
];

/// Dimensionless RHS evaluation offsets `t_stage = t + C[stage] * dt` for LSRK45.
/// Stored in `f64` in stage order 0 through 4, beginning at zero.
pub const RK45_C: [f64; 5] = [
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
];

/// Time integration configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// CFL safety number (typically `0.3`..`0.8`; default `0.5`).
    pub cfl: f64,
    /// Number of characteristic periods to simulate when `t_final` is `None`.
    pub periods: f64,
    /// Optional final physical time, using the same time unit as the equation's wind.
    pub t_final: Option<f64>,
    /// Target intermediate-frame count, used to choose a step stride rather than an exact count.
    /// Positive values also save initial and final frames; zero leaves `snapshots` empty.
    pub save_frames: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            cfl: 0.5,
            periods: 1.0,
            t_final: None,
            save_frames: 0,
        }
    }
}

/// Captured solution snapshot at a given step/time.
#[derive(Debug, Clone)]
pub struct Snapshot<T: FloatScalar> {
    /// Number of completed full time steps; zero denotes the initial condition.
    pub step: usize,
    /// Physical time of this frame, in the wind model's time unit.
    pub time: f64,
    /// Downloaded nodal scalar values in mesh order `[elem][j][i]`.
    pub phi: Vec<T>,
}

/// Result returned by running a simulation through LSRK45.
#[derive(Debug, Clone)]
pub struct SimulationOutput<T: FloatScalar> {
    /// Nominal physical step after capping at the requested final time; the last step may be shorter.
    /// Zero for a zero-duration simulation.
    pub dt: f64,
    /// Number of completed full LSRK45 steps, each comprising five stages.
    pub steps: usize,
    /// Reached physical time from compensated accumulation, in the wind model's time unit.
    pub t_final: f64,
    /// Wall-clock seconds for stepping, snapshot downloads, and the final download.
    pub elapsed_secs: f64,
    /// Sampled maximum `|u^alpha| + |u^beta|` in radians per time unit; zero for zero duration.
    pub max_contra_speed: f64,
    /// Initial scalar field evaluated on the host in mesh order `[elem][j][i]`.
    pub initial: Vec<T>,
    /// Final scalar field downloaded to the host in mesh order `[elem][j][i]`.
    pub final_state: Vec<T>,
    /// Saved frames including endpoints when enabled; empty if saving is disabled or duration is zero.
    pub snapshots: Vec<Snapshot<T>>,
}

/// Computes the maximum contravariant angular speed `max(|u^alpha| + |u^beta|)` across all nodes.
///
/// Angular rates are radians per wind-model time unit, including the mesh's radius scaling.
/// Steady winds are evaluated at zero; time-dependent winds at 0, 1/4, 1/2, 3/4, and one
/// characteristic period. This sampled estimate need not bound every intervening time.
pub fn compute_max_contravariant_speed<T: FloatScalar>(
    mesh: &CubedSphereMesh<T>,
    equation: &ScalarAdvection,
) -> f64 {
    let r = mesh.config.radius;
    let period = equation.wind().characteristic_period(r);
    let sample_times: &[f64] = if equation.is_time_dependent() {
        &[0.0, 0.25 * period, 0.5 * period, 0.75 * period, period]
    } else {
        &[0.0]
    };

    let mut u = vec![T::ZERO; mesh.total_nodes];
    let mut v = vec![T::ZERO; mesh.total_nodes];
    let mut max_speed = 0.0_f64;

    for &t in sample_times {
        equation.evaluate_contravariant_wind(mesh, t, &mut u, &mut v);
        for idx in 0..mesh.total_nodes {
            let s = u[idx].to_f64().abs() + v[idx].to_f64().abs();
            if s > max_speed {
                max_speed = s;
            }
        }
    }
    max_speed
}

/// Multi-element Spectral Element CFL time step:
/// `dt = CFL * pi / (2 * K * p^2 * nu_max)`.
/// `nu_max` is `max_contra_speed` in radians/time; returns `fallback_dt` unchanged if it is `<= 1e-30`.
/// Inputs are not otherwise validated; `K` and `p` come from the mesh.
pub fn compute_cfl_dt<T: FloatScalar>(
    mesh: &CubedSphereMesh<T>,
    max_contra_speed: f64,
    cfl: f64,
    fallback_dt: f64,
) -> f64 {
    if max_contra_speed <= 1e-30 {
        return fallback_dt;
    }
    let k = mesh.num_elem_per_face as f64;
    let p = mesh.reference.order as f64;
    cfl * std::f64::consts::PI / (2.0 * k * p * p * max_contra_speed)
}

#[inline]
fn neumaier_add(sum: &mut f64, comp: &mut f64, delta: f64) {
    let next = *sum + delta;
    if sum.abs() >= delta.abs() {
        *comp += (*sum - next) + delta;
    } else {
        *comp += (delta - next) + *sum;
    }
    *sum = next;
}

/// Executes a full scalar transport simulation using 5-stage LSRK45 on the given [`ComputeBackend`],
/// optionally overriding the CFL step size with an explicit `fixed_dt`.
/// Initializes from `equation`; the backend must evaluate the matching advection RHS.
/// Time starts at zero, uses the wind model's units, and shortens the last step to the target.
/// # Panics
/// Panics if the derived final time is negative or NaN, or `fixed_dt` is not finite and positive.
/// # Errors
/// Propagates backend allocation, upload, time-step, and download errors.
pub fn run_lsrk45_with_dt<T, B>(
    backend: &B,
    equation: &ScalarAdvection,
    config: &SimulationConfig,
    fixed_dt: Option<f64>,
) -> Result<SimulationOutput<T>, B::Error>
where
    T: FloatScalar,
    B: ComputeBackend<T>,
{
    let mesh = backend.mesh();
    let t_final = config.t_final.unwrap_or_else(|| {
        config.periods * equation.wind().characteristic_period(mesh.config.radius)
    });
    assert!(t_final >= 0.0, "t_final must be non-negative");
    if let Some(dt) = fixed_dt {
        assert!(
            dt.is_finite() && dt > 0.0,
            "fixed_dt must be positive and finite"
        );
    }

    let initial = equation.evaluate_initial(mesh);
    if t_final == 0.0 {
        return Ok(SimulationOutput {
            dt: 0.0,
            steps: 0,
            t_final: 0.0,
            elapsed_secs: 0.0,
            max_contra_speed: 0.0,
            initial: initial.clone(),
            final_state: initial,
            snapshots: Vec::new(),
        });
    }

    let max_contra_speed = compute_max_contravariant_speed(mesh, equation);
    let step_dt = fixed_dt
        .unwrap_or_else(|| compute_cfl_dt(mesh, max_contra_speed, config.cfl, t_final))
        .min(t_final);

    let mut u = backend.upload(&initial)?;
    let mut du = backend.allocate_zeros(mesh.total_nodes)?;
    let mut rhs = backend.allocate_zeros(mesh.total_nodes)?;

    let estimated_steps = (t_final / step_dt).ceil().max(1.0) as usize;
    let frame_stride = estimated_steps
        .checked_div(config.save_frames)
        .map_or(usize::MAX, |s| s.max(1));

    let mut snapshots = Vec::new();
    if config.save_frames > 0 {
        snapshots.push(Snapshot {
            step: 0,
            time: 0.0,
            phi: initial.clone(),
        });
    }

    let start = Instant::now();
    let tol = 1e-12 * t_final.max(1.0);
    let mut t_sum = 0.0_f64;
    let mut t_comp = 0.0_f64;
    let mut step = 0usize;

    while t_sum + t_comp < t_final - tol {
        let current_time = t_sum + t_comp;
        let dt_actual = if current_time + step_dt >= t_final - tol {
            (t_final - t_sum) - t_comp
        } else {
            step_dt
        };

        backend.lsrk45_step(&mut u, &mut du, &mut rhs, current_time, dt_actual)?;
        neumaier_add(&mut t_sum, &mut t_comp, dt_actual);
        step += 1;

        let exact_time = t_sum + t_comp;
        if config.save_frames > 0
            && step.is_multiple_of(frame_stride)
            && exact_time < t_final - 1e-12
        {
            snapshots.push(Snapshot {
                step,
                time: exact_time,
                phi: backend.download(&u)?,
            });
        }
    }

    let final_state = backend.download(&u)?;
    let elapsed_secs = start.elapsed().as_secs_f64();
    let final_time = t_sum + t_comp;

    if config.save_frames > 0 {
        snapshots.push(Snapshot {
            step,
            time: final_time,
            phi: final_state.clone(),
        });
    }

    Ok(SimulationOutput {
        dt: step_dt,
        steps: step,
        t_final: final_time,
        elapsed_secs,
        max_contra_speed,
        initial,
        final_state,
        snapshots,
    })
}

/// Executes a full scalar transport simulation using 5-stage LSRK45 on the given [`ComputeBackend`].
/// Uses [`run_lsrk45_with_dt`] with a CFL-selected step; backend and equation must describe the same RHS.
/// # Panics
/// Panics if the derived final time is negative or NaN.
/// # Errors
/// Propagates backend allocation, upload, time-step, and download errors.
pub fn run_lsrk45_simulation<T, B>(
    backend: &B,
    equation: &ScalarAdvection,
    config: &SimulationConfig,
) -> Result<SimulationOutput<T>, B::Error>
where
    T: FloatScalar,
    B: ComputeBackend<T>,
{
    run_lsrk45_with_dt(backend, equation, config, None)
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_4;
    use std::sync::Arc;

    use super::*;
    use cubesphere_core::{MeshConfig, ProjectionKind};
    use cubesphere_equations::{GaussianBell, SolidBodyRotation};

    #[test]
    fn test_cfl_dt_is_radius_invariant_for_unit_period_flow() {
        let eq = ScalarAdvection::solid_body(
            SolidBodyRotation::new(FRAC_PI_4, None),
            Arc::new(GaussianBell::default()),
        );
        let radii = [1.0, 2.5, 6_371_000.0];
        let mut dts = Vec::new();
        let mut speeds = Vec::new();

        for &radius in &radii {
            let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig {
                radius,
                num_elem_per_face: 3,
                poly_order: 4,
                projection: ProjectionKind::Equiangular,
            });
            let nu = compute_max_contravariant_speed(&mesh, &eq);
            let dt = compute_cfl_dt(&mesh, nu, 0.5, 1.0);
            speeds.push(nu);
            dts.push(dt);
        }

        for i in 1..radii.len() {
            assert!(
                (speeds[i] - speeds[0]).abs() / speeds[0] < 1e-13,
                "max contravariant angular speed varied with R: {} vs {}",
                speeds[0],
                speeds[i]
            );
            assert!(
                (dts[i] - dts[0]).abs() / dts[0] < 1e-13,
                "CFL dt varied with R: {} vs {}",
                dts[0],
                dts[i]
            );
        }
    }

    #[test]
    fn test_cfl_fallback_and_rk45_stages_monotonic() {
        let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig::default());
        assert_eq!(compute_cfl_dt(&mesh, 0.0, 0.5, 0.25), 0.25);
        assert_eq!(RK45_C[0], 0.0);
        for i in 1..RK45_C.len() {
            assert!(RK45_C[i] > RK45_C[i - 1] && RK45_C[i] < 1.0);
        }
    }

    #[test]
    fn test_neumaier_time_accumulation_zero_drift() {
        for &(t_final, step_dt, expected_steps) in &[
            (1.0_f64, 1.0_f64 / 24_576.0, 24_576_usize),
            (1.0_f64, 8.0e-5_f64, 12_500_usize),
        ] {
            let tol = 1e-12 * t_final;
            let mut t_sum = 0.0;
            let mut t_comp = 0.0;
            let mut steps = 0usize;
            let mut last_dt = 0.0;

            while t_sum + t_comp < t_final - tol {
                let current_time = t_sum + t_comp;
                let dt_actual = if current_time + step_dt >= t_final - tol {
                    (t_final - t_sum) - t_comp
                } else {
                    step_dt
                };
                last_dt = dt_actual;
                neumaier_add(&mut t_sum, &mut t_comp, dt_actual);
                steps += 1;
            }

            assert_eq!(steps, expected_steps, "step_dt={step_dt}");
            assert!(
                ((t_sum + t_comp) - t_final).abs() < 1e-15,
                "step_dt={step_dt}: final time drift={:.3e}",
                ((t_sum + t_comp) - t_final).abs()
            );
            assert!(
                (last_dt - step_dt).abs() < 1e-15,
                "step_dt={step_dt}: final step perturbed by {:.3e}",
                (last_dt - step_dt).abs()
            );
        }
    }
}
