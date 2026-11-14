use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
use std::sync::Arc;

use cubesphere_backend_cpu::CpuBackend;
use cubesphere_core::{CubedSphereMesh, MeshConfig, ProjectionKind};
use cubesphere_equations::{
    ConstantField, GaussianBell, NairLauritzenFlow, ScalarAdvection, SolidBodyRotation,
    TwinGaussianBells, WindField,
};
use cubesphere_io::{compute_diagnostics, compute_energy};
use cubesphere_solver::{
    run_lsrk45_simulation, run_lsrk45_with_dt, ComputeBackend, NumericalFlux, SimulationConfig,
};

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
        (
            3,
            3,
            1.7320508075688772,
            FRAC_PI_4,
            ProjectionKind::Equiangular,
        ),
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
        assert!(
            m.mass_rel_err < 1e-14,
            "p={p}: mass_rel_err={:.3e}",
            m.mass_rel_err
        );
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
        assert!(
            m.mass_rel_err < 1e-14,
            "K={k}: mass_rel_err={:.3e}",
            m.mass_rel_err
        );
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
    let out64 = run_lsrk45_simulation(
        &CpuBackend::from_equation(Arc::clone(&mesh64), &eq),
        &eq,
        &sim_cfg,
    )
    .unwrap();

    let mesh32 = Arc::new(CubedSphereMesh::<f32>::from_config(cfg));
    let out32 = run_lsrk45_simulation(
        &CpuBackend::from_equation(Arc::clone(&mesh32), &eq),
        &eq,
        &sim_cfg,
    )
    .unwrap();

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
        assert!(
            m.mass_rel_err < 1e-14,
            "p=3 K={k}: mass_rel_err={:.3e}",
            m.mass_rel_err
        );
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
        assert!(
            m.mass_rel_err < 1e-14,
            "dt={dt}: mass_rel_err={:.3e}",
            m.mass_rel_err
        );
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

/// Deterministic pseudo-random values in `[lo, hi)` (64-bit LCG; avoids a `rand` dependency).
fn pseudo_random_field(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (s >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * unit
        })
        .collect()
}

/// Discrete SBP mass-matrix inner product `<a, b>_M = sum_k w_i w_j sqrt(g)_k a_k b_k`.
fn mass_inner(mesh: &CubedSphereMesh<f64>, a: &[f64], b: &[f64]) -> f64 {
    let prod: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    mesh.integrate(&prod)
}

fn apply_rhs(backend: &CpuBackend<f64>, phi: &[f64], t: f64) -> Vec<f64> {
    let state = phi.to_vec();
    let mut rhs = vec![0.0; phi.len()];
    backend.compute_advection_rhs(&state, &mut rhs, t).unwrap();
    rhs
}

#[test]
fn test_flux_family_energy_structure_of_semi_discrete_operator() {
    // Energy analysis of `d/dt phi = L phi` in the SBP norm `M` (cf. DG_energy_estimate_handout):
    //   dE/dt = phi^T (M L + L^T M) phi.
    // Split-form volume terms contribute only a *diagonal* part to `M L + L^T M`, and the
    // alpha-flux SAT contributes `-alpha * sum_interfaces |sqrt(g) u_n| w (q_out - q_in)^2`
    // plus diagonal terms. Hence, exactly (to round-off) and for any field:
    //   (1) central: `M L + L^T M` is diagonal -> fields with disjoint supports never exchange energy;
    //   (2) upwind - central rate = -sum |c| [[q]]^2 <= 0, and it is linear in alpha;
    //   (3) continuous fields have no interface jumps, so every flux gives the same RHS.
    let cases: [(&str, MeshConfig, Arc<dyn WindField>, f64); 3] = [
        (
            "solid-body 45deg, equiangular",
            MeshConfig {
                radius: 1.3,
                num_elem_per_face: 3,
                poly_order: 4,
                projection: ProjectionKind::Equiangular,
            },
            Arc::new(SolidBodyRotation::new(FRAC_PI_4, None)),
            0.0,
        ),
        (
            "solid-body over poles, equidistant",
            MeshConfig {
                radius: 1.0,
                num_elem_per_face: 2,
                poly_order: 5,
                projection: ProjectionKind::Equidistant,
            },
            Arc::new(SolidBodyRotation::new(FRAC_PI_2, None)),
            0.0,
        ),
        (
            "nair-lauritzen t=0.37, equiangular",
            MeshConfig {
                radius: 1.0,
                num_elem_per_face: 2,
                poly_order: 4,
                projection: ProjectionKind::Equiangular,
            },
            Arc::new(NairLauritzenFlow::default()),
            0.37,
        ),
    ];

    for (label, cfg, wind, t) in cases {
        let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(cfg));
        let backend = |flux| CpuBackend::new(Arc::clone(&mesh), Arc::clone(&wind)).with_flux(flux);
        let central = backend(NumericalFlux::Central);
        let upwind = backend(NumericalFlux::Upwind);
        let blended = backend(NumericalFlux::Blended(0.5));
        let n = mesh.total_nodes;
        let n_elem = mesh.n_nodes_elem;
        let norm = |v: &[f64]| mass_inner(&mesh, v, v).sqrt();

        // (1) Positive fields supported on even (phi) / odd (psi) elements, which share many
        //     intra-panel and cross-panel edges.
        let mut phi = pseudo_random_field(n, 11, 0.5, 1.5);
        let mut psi = pseudo_random_field(n, 23, 0.5, 1.5);
        for idx in 0..n {
            if (idx / n_elem).is_multiple_of(2) {
                psi[idx] = 0.0;
            } else {
                phi[idx] = 0.0;
            }
        }
        let cross_rate = |b: &CpuBackend<f64>| {
            mass_inner(&mesh, &phi, &apply_rhs(b, &psi, t))
                + mass_inner(&mesh, &psi, &apply_rhs(b, &phi, t))
        };
        let scale = norm(&phi) * norm(&apply_rhs(&central, &psi, t))
            + norm(&psi) * norm(&apply_rhs(&central, &phi, t));
        let cross_central = cross_rate(&central);
        let cross_upwind = cross_rate(&upwind);
        assert!(
            cross_central.abs() < 1e-12 * scale,
            "{label}: central flux exchanges energy between disjoint supports: {cross_central:.3e} (scale {scale:.3e})"
        );
        // Upwind couples them through `+|c| phi_a psi_b > 0`, so the check above is not vacuous.
        assert!(
            cross_upwind > 1e-3 * scale,
            "{label}: expected positive upwind interface coupling, got {cross_upwind:.3e} (scale {scale:.3e})"
        );

        // (2) Interface dissipation on a discontinuous field.
        let q = pseudo_random_field(n, 7, -1.0, 1.0);
        let rate = |b: &CpuBackend<f64>| 2.0 * mass_inner(&mesh, &q, &apply_rhs(b, &q, t));
        let rate_central = rate(&central);
        let diss_upwind = rate(&upwind) - rate_central;
        let diss_blended = rate(&blended) - rate_central;
        assert!(
            diss_upwind < 0.0,
            "{label}: upwind must add non-positive energy rate, got {diss_upwind:.3e}"
        );
        assert!(
            (diss_blended - 0.5 * diss_upwind).abs() < 1e-10 * diss_upwind.abs(),
            "{label}: alpha=0.5 dissipation {diss_blended:.6e} != 0.5 * upwind {diss_upwind:.6e}"
        );

        // (3) Globally continuous polynomial field: all fluxes agree to round-off.
        let smooth: Vec<f64> = (0..n)
            .map(|k| {
                let (x, y, z) = (mesh.xyz[0][k], mesh.xyz[1][k], mesh.xyz[2][k]);
                z + 0.3 * x + 0.1 * x * y
            })
            .collect();
        let r_central = apply_rhs(&central, &smooth, t);
        let r_upwind = apply_rhs(&upwind, &smooth, t);
        let max_rhs = r_central.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let max_diff = r_central
            .iter()
            .zip(&r_upwind)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_diff < 1e-12 * max_rhs,
            "{label}: continuous field RHS differs between fluxes by {max_diff:.3e} (max |rhs| {max_rhs:.3e})"
        );
    }
}

#[test]
fn test_flux_energy_evolution_upwind_dissipates_central_conserves() {
    // Under-resolved multi-element Gaussian (K=4, p=3) crossing panel corners for one period.
    let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
        radius: 1.0,
        num_elem_per_face: 4,
        poly_order: 3,
        projection: ProjectionKind::Equiangular,
    }));
    let eq = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_4, None),
        Arc::new(GaussianBell::default()),
    );
    let sim_cfg = SimulationConfig {
        cfl: 0.5,
        periods: 1.0,
        t_final: None,
        save_frames: 10,
    };

    let run = |flux: NumericalFlux| {
        let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq).with_flux(flux);
        let out = run_lsrk45_simulation(&backend, &eq, &sim_cfg).unwrap();
        let exact = eq.evaluate_exact(&mesh, out.t_final).unwrap();
        let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));
        let history: Vec<f64> = out
            .snapshots
            .iter()
            .map(|s| compute_energy(&mesh, &s.phi))
            .collect();
        assert!(
            m.mass_rel_err < 1e-14,
            "{flux}: mass_rel_err={:.3e}",
            m.mass_rel_err
        );
        (m, history)
    };

    let (m_up, hist_up) = run(NumericalFlux::Upwind);
    let (m_bl, _) = run(NumericalFlux::Blended(0.5));
    let (m_c, _) = run(NumericalFlux::Central);

    // Upwind: energy decays monotonically and noticeably on this coarse grid.
    assert!(hist_up.len() >= 10);
    for (k, w) in hist_up.windows(2).enumerate() {
        assert!(
            w[1] <= w[0],
            "upwind energy increased between snapshots {k} and {}: {:.12e} -> {:.12e}",
            k + 1,
            w[0],
            w[1]
        );
    }
    assert!(
        m_up.energy_rel_change < -1e-3,
        "upwind energy change {:.3e} should be clearly dissipative",
        m_up.energy_rel_change
    );

    // Central: interface coupling is energy neutral; the small remainder comes from the discrete
    // divergence of the interpolated wind and LSRK45, and is far below the upwind dissipation.
    assert!(
        m_c.energy_rel_change.abs() < 0.02 * m_up.energy_rel_change.abs(),
        "central energy change {:.3e} vs upwind {:.3e}",
        m_c.energy_rel_change,
        m_up.energy_rel_change
    );
    // Blended flux dissipates in between.
    assert!(
        m_up.energy_rel_change < m_bl.energy_rel_change
            && m_bl.energy_rel_change < m_c.energy_rel_change,
        "expected E change ordering upwind < blended < central, got {:.3e}, {:.3e}, {:.3e}",
        m_up.energy_rel_change,
        m_bl.energy_rel_change,
        m_c.energy_rel_change
    );
    // Central remains stable and accurate (it trades dissipation for dispersion error).
    assert!(m_c.l2_rel < 0.05, "central L2_rel={:.3e}", m_c.l2_rel);
    assert!(m_up.l2_rel < 0.05, "upwind L2_rel={:.3e}", m_up.l2_rel);
}
