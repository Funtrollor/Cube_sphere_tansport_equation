use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::sync::Arc;

use cs_backend_cpu::CpuBackend;
use cs_core::{CubedSphereMesh, MeshConfig, ProjectionKind};
use cs_equations::{
    ConstantField, GaussianBell, NairLauritzenFlow, ScalarAdvection, SolidBodyRotation,
    TwinGaussianBells,
};
use cs_io::compute_diagnostics;
use cs_solver::{run_lsrk45_simulation, run_lsrk45_with_dt, SimulationConfig};

#[test]
fn test_freestream_preservation_constant_field() {
    // Under conservative Split-Form, for phi == 1 the semi-discrete RHS reduces to `-div_h(u)`,
    // which converges spectrally to machine zero as resolution `(K, p)` increases.
    let cases = [
        (1, 20, ProjectionKind::Equiangular, 5e-12, 5e-13),
        (4, 8, ProjectionKind::Equiangular, 1e-7, 5e-9),
        (4, 12, ProjectionKind::Equiangular, 5e-12, 1e-13),
        (4, 8, ProjectionKind::Equidistant, 2e-6, 2e-7),
        (4, 12, ProjectionKind::Equidistant, 2e-10, 2e-11),
    ];

    for (k, p, projection, max_linf, max_l2_rel) in cases {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.5,
            num_elem_per_face: k,
            poly_order: p,
            projection,
        }));
        let eq = ScalarAdvection::solid_body(
            SolidBodyRotation::new(FRAC_PI_4, None),
            Arc::new(ConstantField { value: 1.0 }),
        );
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
        let out = run_lsrk45_simulation(
            &backend,
            &eq,
            &SimulationConfig {
                cfl: 0.5,
                periods: 0.5,
                t_final: None,
                save_frames: 0,
            },
        )
        .unwrap();

        let exact = eq.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));
        assert!(
            m.l_inf < max_linf,
            "K={k}, p={p}, {projection:?}: freestream L_inf={:.3e} >= {max_linf:.3e}",
            m.l_inf
        );
        assert!(
            m.l2_rel < max_l2_rel,
            "K={k}, p={p}, {projection:?}: freestream L2_rel={:.3e} >= {max_l2_rel:.3e}",
            m.l2_rel
        );
        let max_mass_err = if p >= 20 { 5e-14 } else { 1e-14 };
        assert!(
            m.mass_rel_err < max_mass_err,
            "K={k}, p={p}, {projection:?}: mass_rel_err={:.3e} >= {max_mass_err:.3e}",
            m.mass_rel_err
        );
    }
}

#[test]
fn test_exact_discrete_global_mass_conservation() {
    // Verify SBP-SAT conserves global discrete mass to machine precision even on coarse grids
    // (where spatial truncation error is non-trivial) and across both projections / flows.
    let solid_cases = [
        (1, 6, 1.0, FRAC_PI_4, ProjectionKind::Equiangular),
        (3, 3, 1.7320508075688772, FRAC_PI_4, ProjectionKind::Equiangular),
        (4, 4, 2.5, FRAC_PI_2, ProjectionKind::Equidistant),
    ];

    for (k, p, radius, alpha0, projection) in solid_cases {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius,
            num_elem_per_face: k,
            poly_order: p,
            projection,
        }));
        let eq = ScalarAdvection::solid_body(
            SolidBodyRotation::new(alpha0, None),
            Arc::new(GaussianBell::default()),
        );
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
        let out = run_lsrk45_simulation(
            &backend,
            &eq,
            &SimulationConfig {
                cfl: 0.5,
                periods: 0.5,
                t_final: None,
                save_frames: 0,
            },
        )
        .unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, None);
        assert!(
            m.mass_rel_err < 1e-14,
            "Solid K={k}, p={p}, {projection:?}: mass_rel_err={:.3e}",
            m.mass_rel_err
        );
    }

    // Time-dependent Nair & Lauritzen deformational flow must also conserve discrete mass to ~1e-14.
    let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
        radius: 1.0,
        num_elem_per_face: 3,
        poly_order: 4,
        projection: ProjectionKind::Equiangular,
    }));
    let eq = ScalarAdvection::new(
        Arc::new(NairLauritzenFlow::default()),
        Arc::new(TwinGaussianBells::default()),
    );
    let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
    let out = run_lsrk45_simulation(
        &backend,
        &eq,
        &SimulationConfig {
            cfl: 0.5,
            periods: 0.5,
            t_final: None,
            save_frames: 0,
        },
    )
    .unwrap();
    let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, None);
    assert!(
        m.mass_rel_err < 1e-14,
        "Deformational K=3, p=4: mass_rel_err={:.3e}",
        m.mass_rel_err
    );
}

#[test]
fn test_single_element_regression_vs_notebook() {
    // Baseline comparison against P2_new.ipynb (`K=1`, `alpha0=0`, `R=1`, `period=1`):
    // Notebook recorded L2_abs: p=8 -> 8.63e-3, p=12 -> 1.61e-4, p=16 -> 1.46e-6.
    let eq = ScalarAdvection::solid_body(
        SolidBodyRotation::new(0.0, None),
        Arc::new(GaussianBell::default()),
    );
    let sim_cfg = SimulationConfig {
        cfl: 0.5,
        periods: 1.0,
        t_final: None,
        save_frames: 0,
    };

    let mut prev_l2 = f64::INFINITY;
    for (p, max_l2_abs) in [(8, 1.5e-2), (12, 3.0e-4), (16, 3.0e-6), (20, 2.5e-8)] {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.0,
            num_elem_per_face: 1,
            poly_order: p,
            projection: ProjectionKind::Equiangular,
        }));
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
        let out = run_lsrk45_simulation(&backend, &eq, &sim_cfg).unwrap();
        let exact = eq.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));

        assert!(
            m.l2_abs < max_l2_abs,
            "p={p}: l2_abs={:.3e} >= {max_l2_abs:.3e}",
            m.l2_abs
        );
        assert!(m.l2_abs < prev_l2);
        assert!(m.mass_rel_err < 1e-14, "p={p}: mass_rel_err={:.3e}", m.mass_rel_err);
        prev_l2 = m.l2_abs;
    }
}

#[test]
fn test_multi_element_h_refinement_and_polar_p_refinement() {
    // 1. h-refinement across K = 2, 4, 8 at p = 4 (oblique alpha0 = pi/4 over cube corners)
    let eq_oblique = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_4, None),
        Arc::new(GaussianBell::default()),
    );
    let sim_cfg = SimulationConfig {
        cfl: 0.5,
        periods: 1.0,
        t_final: None,
        save_frames: 0,
    };

    let mut errs = Vec::new();
    for k in [2, 4, 8] {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.0,
            num_elem_per_face: k,
            poly_order: 4,
            projection: ProjectionKind::Equiangular,
        }));
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq_oblique);
        let out = run_lsrk45_simulation(&backend, &eq_oblique, &sim_cfg).unwrap();
        let exact = eq_oblique.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));
        assert!(m.mass_rel_err < 1e-14);
        errs.push(m.l2_rel);
    }

    let order_2_4 = (errs[0] / errs[1]).ln() / 2.0_f64.ln();
    let order_4_8 = (errs[1] / errs[2]).ln() / 2.0_f64.ln();
    assert!(order_2_4 > 4.2, "K 2->4 order={order_2_4:.2}");
    assert!(order_4_8 > 4.6, "K 4->8 order={order_4_8:.2}");

    // 2. Polar p-refinement (alpha0 = pi/2 directly over North and South poles!) at K = 2
    let eq_polar = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_2, None),
        Arc::new(GaussianBell::default()),
    );
    let mut prev = f64::INFINITY;
    for p in [4, 6, 8, 10, 12] {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.0,
            num_elem_per_face: 2,
            poly_order: p,
            projection: ProjectionKind::Equiangular,
        }));
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq_polar);
        let out = run_lsrk45_simulation(&backend, &eq_polar, &sim_cfg).unwrap();
        let exact = eq_polar.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));
        assert!(m.l2_rel < prev);
        prev = m.l2_rel;
    }
    assert!(prev < 2e-7, "polar K=2, p=12 l2_rel={prev:.3e}");
}

#[test]
fn test_nair_lauritzen_full_cycle_return_and_convergence() {
    let eq = ScalarAdvection::new(
        Arc::new(NairLauritzenFlow::default()),
        Arc::new(TwinGaussianBells::default()),
    );
    let sim_cfg = SimulationConfig {
        cfl: 0.5,
        periods: 1.0,
        t_final: None,
        save_frames: 0,
    };

    let mut l2_errs = Vec::new();
    for k in [4, 8] {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.0,
            num_elem_per_face: k,
            poly_order: 5,
            projection: ProjectionKind::Equiangular,
        }));
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
        let out = run_lsrk45_simulation(&backend, &eq, &sim_cfg).unwrap();

        let exact = eq.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));
        assert!(m.mass_rel_err < 1e-14, "K={k}: mass_rel_err={:.3e}", m.mass_rel_err);
        l2_errs.push(m.l2_rel);
    }

    assert!(
        l2_errs[1] < 0.05 && l2_errs[1] < 0.25 * l2_errs[0],
        "Nair-Lauritzen K=4->{:.3e}, K=8->{:.3e} did not converge cleanly",
        l2_errs[0],
        l2_errs[1]
    );
}

#[test]
fn test_f32_vs_f64_parity() {
    let cfg = MeshConfig {
        radius: 1.0,
        num_elem_per_face: 4,
        poly_order: 4,
        projection: ProjectionKind::Equiangular,
    };
    let eq = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_4, None),
        Arc::new(GaussianBell::default()),
    );
    let sim_cfg = SimulationConfig {
        cfl: 0.5,
        periods: 0.25,
        t_final: None,
        save_frames: 0,
    };

    let mesh64 = Arc::new(CubedSphereMesh::<f64>::from_config(cfg));
    let out64 = run_lsrk45_simulation(&CpuBackend::from_equation(Arc::clone(&mesh64), &eq), &eq, &sim_cfg).unwrap();

    let mesh32 = Arc::new(CubedSphereMesh::<f32>::from_config(cfg));
    let out32 = run_lsrk45_simulation(&CpuBackend::from_equation(Arc::clone(&mesh32), &eq), &eq, &sim_cfg).unwrap();

    let max_diff = out64
        .final_state
        .iter()
        .zip(out32.final_state.iter())
        .map(|(&a, &b)| (a - (b as f64)).abs())
        .fold(0.0_f64, f64::max);

    assert!(max_diff < 5e-6, "f32 vs f64 max_diff={max_diff:.3e}");
}

#[test]
fn test_spatial_fourth_order_convergence_p3() {
    // At polynomial degree p = 3, h-refinement yields theoretical O(h^{p+1}) = O(h^4) accuracy.
    let eq = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_4, None),
        Arc::new(GaussianBell::default()),
    );
    let sim_cfg = SimulationConfig {
        cfl: 0.5,
        periods: 1.0,
        t_final: None,
        save_frames: 0,
    };

    let mut l2_errs = Vec::new();
    for k in [2, 4, 8, 16] {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.0,
            num_elem_per_face: k,
            poly_order: 3,
            projection: ProjectionKind::Equiangular,
        }));
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
        let out = run_lsrk45_simulation(&backend, &eq, &sim_cfg).unwrap();
        let exact = eq.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));
        assert!(m.mass_rel_err < 1e-14, "p=3 K={k}: mass_rel_err={:.3e}", m.mass_rel_err);
        l2_errs.push(m.l2_rel);
    }

    let order_4_8 = (l2_errs[1] / l2_errs[2]).ln() / 2.0_f64.ln();
    let order_8_16 = (l2_errs[2] / l2_errs[3]).ln() / 2.0_f64.ln();
    assert!(order_4_8 > 4.0, "p=3 K 4->8 order={order_4_8:.3}");
    assert!(
        (3.90..4.15).contains(&order_8_16),
        "p=3 K 8->16 order={order_8_16:.3} expected ~4.00"
    );
}

#[test]
fn test_temporal_fourth_order_convergence_lsrk45() {
    // At fixed spatially resolved mesh (K=1, p=28), halving explicit dt confirms O(dt^4) scaling
    // of Carpenter-Kennedy 5-stage LSRK45 together with Neumaier-compensated endpoint timing.
    let eq = ScalarAdvection::solid_body(
        SolidBodyRotation::new(0.0, None),
        Arc::new(GaussianBell::default()),
    );
    let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
        radius: 1.0,
        num_elem_per_face: 1,
        poly_order: 28,
        projection: ProjectionKind::Equiangular,
    }));
    let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
    let sim_cfg = SimulationConfig {
        cfl: 0.5,
        periods: 0.25,
        t_final: None,
        save_frames: 0,
    };

    let dts = [2.5e-4, 1.25e-4, 6.25e-5];
    let mut l2_errs = Vec::with_capacity(dts.len());

    for &dt in &dts {
        let out = run_lsrk45_with_dt(&backend, &eq, &sim_cfg, Some(dt)).unwrap();
        assert!(
            (out.t_final - 0.25).abs() < 1e-15,
            "dt={dt}: t_final drift={:.3e}",
            (out.t_final - 0.25).abs()
        );
        let exact = eq.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));
        assert!(m.mass_rel_err < 1e-14, "dt={dt}: mass_rel_err={:.3e}", m.mass_rel_err);
        l2_errs.push(m.l2_rel);
    }

    let order_0_1 = (l2_errs[0] / l2_errs[1]).ln() / 2.0_f64.ln();
    let order_1_2 = (l2_errs[1] / l2_errs[2]).ln() / 2.0_f64.ln();
    assert!(
        (3.85..4.15).contains(&order_0_1),
        "LSRK45 temporal order (2.5e-4 -> 1.25e-4)={order_0_1:.3} expected ~4.00"
    );
    assert!(
        (3.85..4.15).contains(&order_1_2),
        "LSRK45 temporal order (1.25e-4 -> 6.25e-5)={order_1_2:.3} expected ~4.00"
    );
}

