//! Advect a Gaussian bell once around the sphere along a tilted solid-body rotation
//! and compare the result with the exact solution.
//!
//! ```bash
//! cargo run --release -p cubesphere-backend-cpu --example solid_body
//! ```

use std::f64::consts::FRAC_PI_4;
use std::sync::Arc;

use cubesphere_backend_cpu::CpuBackend;
use cubesphere_core::{CubedSphereMesh, MeshConfig, ProjectionKind};
use cubesphere_equations::{GaussianBell, ScalarAdvection, SolidBodyRotation};
use cubesphere_io::compute_diagnostics;
use cubesphere_solver::{run_lsrk45_with_dt, NumericalFlux, SimulationConfig};

fn main() {
    // 6 * 8 * 8 elements of polynomial order 4 on the unit sphere.
    let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
        radius: 1.0,
        num_elem_per_face: 8,
        poly_order: 4,
        projection: ProjectionKind::Equiangular,
    }));

    // Rotation axis tilted by 45 degrees, so the bell crosses cube corners.
    let equation = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_4, None),
        Arc::new(GaussianBell::default()),
    );
    let backend =
        CpuBackend::from_equation(Arc::clone(&mesh), &equation).with_flux(NumericalFlux::Upwind);

    // One full revolution with CFL 0.5 (the defaults); `None` = derive dt from the CFL number.
    let out = run_lsrk45_with_dt(&backend, &equation, &SimulationConfig::default(), None)
        .expect("CPU backend never fails");

    let exact = equation.evaluate_exact(&mesh, out.t_final);
    let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, exact.as_deref());

    println!("steps          : {} (dt = {:.3e})", out.steps, out.dt);
    println!("L2 relative    : {:.4e}", m.l2_rel);
    println!("L-inf          : {:.4e}", m.l_inf);
    println!("mass rel. error: {:.3e}", m.mass_rel_err);
    println!("energy change  : {:+.3e}", m.energy_rel_change);
}
