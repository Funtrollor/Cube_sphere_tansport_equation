use std::env;
use std::f64::consts::{FRAC_PI_4, PI};
use std::process;
use std::sync::Arc;
use std::time::Instant;

use cubesphere_backend_cpu::CpuBackend;
use cubesphere_core::{CubedSphereMesh, FloatScalar, MeshConfig, ProjectionKind};
use cubesphere_equations::{
    ConstantField, CosineBell, GaussianBell, InitialKind, NairLauritzenFlow, ScalarAdvection,
    ScalarField, SolidBodyRotation, TwinGaussianBells, WindKind,
};
use cubesphere_io::{
    compute_diagnostics, compute_energy, format_convergence_table,
    format_temporal_convergence_table, populate_convergence_orders,
    populate_temporal_convergence_orders, write_convergence_csv, write_json, write_pvd_series,
    write_vtu, ConvergenceRow, ErrorMetrics,
};
use cubesphere_solver::{run_lsrk45_with_dt, ComputeBackend, NumericalFlux, SimulationConfig};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Precision {
    F64,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
enum ConvergeMode {
    PRefinement,
    HRefinement,
    TRefinement,
}

#[derive(Debug, Clone)]
struct CliOptions {
    k: usize,
    order: usize,
    radius: f64,
    projection: ProjectionKind,
    wind: WindKind,
    initial: Option<InitialKind>,
    alpha0: f64,
    cfl: f64,
    periods: f64,
    dt: Option<f64>,
    sigma_ratio: Option<f64>,
    precision: Precision,
    flux: NumericalFlux,
    vtk_out: Option<String>,
    pvd_dir: Option<String>,
    frames: usize,
    json_out: Option<String>,
    csv_out: Option<String>,
    mode: ConvergeMode,
    orders: Vec<usize>,
    ks: Vec<usize>,
    dts: Vec<f64>,
    bench_steps: usize,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            k: 4,
            order: 6,
            radius: 1.0,
            projection: ProjectionKind::Equiangular,
            wind: WindKind::SolidBody,
            initial: None,
            alpha0: FRAC_PI_4,
            cfl: 0.5,
            periods: 1.0,
            dt: None,
            sigma_ratio: None,
            precision: Precision::F64,
            flux: NumericalFlux::Upwind,
            vtk_out: None,
            pvd_dir: None,
            frames: 0,
            json_out: None,
            csv_out: None,
            mode: ConvergeMode::PRefinement,
            orders: vec![8, 10, 12, 14, 16, 18, 20, 22],
            ks: vec![2, 4, 8, 16],
            dts: vec![1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4],
            bench_steps: 100,
        }
    }
}

fn print_usage() {
    println!(
        "Cubed-Sphere Spectral Element Transport Solver (v{})

USAGE:
    cubesphere <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    info        Inspect multi-element mesh statistics and surface area quadrature
    run         Execute a full LSRK45 transport simulation and print diagnostics
    converge    Run h-refinement, p-refinement, or temporal dt-refinement convergence study
    bench       Benchmark parallel CPU RHS + LSRK45 throughput

COMMON OPTIONS:
    --k <N>                         Elements per face edge (default: 4; total = 6*K*K)
    --order <p>                     Polynomial order p per element (default: 6)
    --radius <R>                    Sphere radius (default: 1.0)
    --projection <kind>             equiangular | equidistant (default: equiangular)
    --wind <kind>                   solid-body | deformational (default: solid-body)
    --initial <kind>                gaussian | cosine | twin-gaussian | constant
    --alpha0 <rad>                  Solid-body inclination angle in radians (default: pi/4)
    --alpha0-deg <deg>              Solid-body inclination angle in degrees
    --cfl <CFL>                     CFL factor (default: 0.5)
    --periods <P>                   Number of flow periods (default: 1.0)
    --dt <step>                     Explicit fixed time step dt overriding CFL (for `run`)
    --sigma-ratio <ratio>           Gaussian width ratio sigma/R (optional override)
    --precision <f64|f32>           Floating-point scalar type (default: f64)
    --flux <kind>                   Interface flux: upwind | central | alpha in [0,1] (default: upwind)

OUTPUT / CONVERGENCE / BENCH OPTIONS:
    --vtk-out <file.vtu>            Write final solution + exact/error fields to VTK .vtu
    --pvd-dir <dir>                 Write time-series .vtu/.pvd frames (use with --frames)
    --frames <N>                    Number of intermediate snapshots to record (default: 0)
    --json-out <file.json>          Write JSON metrics report
    --csv-out <file.csv>            Write convergence CSV report
    --mode <p|h|t>                  Convergence refinement mode: p, h, or t (default: p)
    --orders <list>                 Comma-separated p orders (default: 8,10,12,14,16,18,20,22)
    --ks <list>                     Comma-separated K counts (default: 2,4,8,16)
    --dts <list>                    Comma-separated dt values for --mode t (default: 1e-3,5e-4,2.5e-4,1.25e-4)
    --steps <N>                     Benchmark step count (default: 100)
    -h, --help                      Print this help message",
        env!("CARGO_PKG_VERSION")
    );
}

fn parse_list(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|e| format!("invalid integer '{part}': {e}"))
        })
        .collect()
}

fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<f64>()
                .map_err(|e| format!("invalid float '{part}': {e}"))
        })
        .collect()
}

fn parse_args(args: &[String]) -> Result<CliOptions, String> {
    let mut opts = CliOptions::default();
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let next_val = |idx: usize| -> Result<&str, String> {
            args.get(idx + 1)
                .map(|s| s.as_str())
                .ok_or_else(|| format!("missing value for flag {flag}"))
        };

        match flag {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "--k" => {
                opts.k = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --k: {e}"))?;
                i += 2;
            }
            "--order" => {
                opts.order = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --order: {e}"))?;
                i += 2;
            }
            "--radius" => {
                opts.radius = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --radius: {e}"))?;
                i += 2;
            }
            "--projection" => {
                opts.projection = match next_val(i)? {
                    "equiangular" => ProjectionKind::Equiangular,
                    "equidistant" => ProjectionKind::Equidistant,
                    other => return Err(format!("unknown --projection '{other}'")),
                };
                i += 2;
            }
            "--wind" => {
                opts.wind = match next_val(i)? {
                    "solid-body" | "solid_body" => WindKind::SolidBody,
                    "deformational" | "nair-lauritzen" | "nair_lauritzen" => {
                        WindKind::Deformational
                    }
                    other => return Err(format!("unknown --wind '{other}'")),
                };
                i += 2;
            }
            "--initial" => {
                opts.initial = Some(match next_val(i)? {
                    "gaussian" | "gaussian-bell" => InitialKind::GaussianBell,
                    "cosine" | "cosine-bell" => InitialKind::CosineBell,
                    "twin-gaussian" | "twin" => InitialKind::TwinGaussian,
                    "constant" | "const" => InitialKind::Constant,
                    other => return Err(format!("unknown --initial '{other}'")),
                });
                i += 2;
            }
            "--alpha0" => {
                opts.alpha0 = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --alpha0: {e}"))?;
                i += 2;
            }
            "--alpha0-deg" => {
                let deg: f64 = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --alpha0-deg: {e}"))?;
                opts.alpha0 = deg * PI / 180.0;
                i += 2;
            }
            "--cfl" => {
                opts.cfl = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --cfl: {e}"))?;
                i += 2;
            }
            "--periods" => {
                opts.periods = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --periods: {e}"))?;
                i += 2;
            }
            "--dt" => {
                let dt_val: f64 = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --dt: {e}"))?;
                opts.dt = Some(dt_val);
                i += 2;
            }
            "--sigma-ratio" => {
                let sr: f64 = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --sigma-ratio: {e}"))?;
                opts.sigma_ratio = Some(sr);
                i += 2;
            }
            "--precision" => {
                opts.precision = match next_val(i)? {
                    "f64" => Precision::F64,
                    "f32" => Precision::F32,
                    other => return Err(format!("unknown --precision '{other}'")),
                };
                i += 2;
            }
            "--flux" => {
                opts.flux = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --flux: {e}"))?;
                i += 2;
            }
            "--vtk-out" => {
                opts.vtk_out = Some(next_val(i)?.to_string());
                i += 2;
            }
            "--pvd-dir" => {
                opts.pvd_dir = Some(next_val(i)?.to_string());
                i += 2;
            }
            "--frames" => {
                opts.frames = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --frames: {e}"))?;
                i += 2;
            }
            "--json-out" => {
                opts.json_out = Some(next_val(i)?.to_string());
                i += 2;
            }
            "--csv-out" => {
                opts.csv_out = Some(next_val(i)?.to_string());
                i += 2;
            }
            "--mode" => {
                opts.mode = match next_val(i)? {
                    "p" | "p-refinement" => ConvergeMode::PRefinement,
                    "h" | "h-refinement" => ConvergeMode::HRefinement,
                    "t" | "dt" | "temporal" | "t-refinement" => ConvergeMode::TRefinement,
                    other => return Err(format!("unknown --mode '{other}'")),
                };
                i += 2;
            }
            "--orders" => {
                opts.orders = parse_list(next_val(i)?)?;
                i += 2;
            }
            "--ks" => {
                opts.ks = parse_list(next_val(i)?)?;
                i += 2;
            }
            "--dts" => {
                opts.dts = parse_f64_list(next_val(i)?)?;
                i += 2;
            }
            "--steps" => {
                opts.bench_steps = next_val(i)?
                    .parse()
                    .map_err(|e| format!("invalid --steps: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unrecognized option '{other}'")),
        }
    }
    Ok(opts)
}

/// Discrete energy `E(t) = \int phi^2 dS` sampled at a recorded snapshot.
#[derive(Debug, Clone, Copy, Serialize)]
struct EnergySample {
    step: usize,
    time: f64,
    energy: f64,
}

/// JSON report of `run`: the diagnostic metrics (flattened at the top level, as before),
/// the interface flux, and the energy history when `--frames` is used.
#[derive(Debug, Serialize)]
struct RunReport {
    flux: NumericalFlux,
    #[serde(flatten)]
    metrics: ErrorMetrics,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    energy_history: Vec<EnergySample>,
}

fn build_equation(opts: &CliOptions) -> ScalarAdvection {
    let initial_kind = opts.initial.unwrap_or(match opts.wind {
        WindKind::SolidBody => InitialKind::GaussianBell,
        WindKind::Deformational => InitialKind::TwinGaussian,
    });

    let initial: Arc<dyn ScalarField> = match initial_kind {
        InitialKind::GaussianBell => {
            let mut bell = GaussianBell::default();
            if let Some(sr) = opts.sigma_ratio {
                bell.sigma_ratio = sr;
            }
            Arc::new(bell)
        }
        InitialKind::CosineBell => Arc::new(CosineBell::default()),
        InitialKind::TwinGaussian => {
            let mut twin = TwinGaussianBells::default();
            if let Some(sr) = opts.sigma_ratio {
                twin.sigma_ratio = sr;
            }
            Arc::new(twin)
        }
        InitialKind::Constant => Arc::new(ConstantField::default()),
    };

    match opts.wind {
        WindKind::SolidBody => {
            ScalarAdvection::solid_body(SolidBodyRotation::new(opts.alpha0, None), initial)
        }
        WindKind::Deformational => {
            ScalarAdvection::new(Arc::new(NairLauritzenFlow::default()), initial)
        }
    }
}

fn cmd_info(opts: &CliOptions) {
    let cfg = MeshConfig {
        radius: opts.radius,
        num_elem_per_face: opts.k,
        poly_order: opts.order,
        projection: opts.projection,
    };
    let mesh = CubedSphereMesh::<f64>::from_config(cfg);
    let area = mesh.surface_area();
    let exact_area = 4.0 * PI * opts.radius * opts.radius;
    let rel_err = (area - exact_area).abs() / exact_area;

    println!("=== Cubed-Sphere Mesh Info ===");
    println!("Projection        : {:?}", cfg.projection);
    println!("Radius (R)        : {:.6}", cfg.radius);
    println!("Elements / edge K : {}", cfg.num_elem_per_face);
    println!("Total elements    : {} (6 x K^2)", mesh.n_elems);
    println!(
        "Polynomial order  : p = {} ({} x {} LGL nodes)",
        cfg.poly_order, mesh.n_nodes_1d, mesh.n_nodes_1d
    );
    println!("Total DOFs        : {}", mesh.total_nodes);
    println!("Quadrature area   : {area:.12}");
    println!("Exact area (4piR2): {exact_area:.12}");
    println!("Area rel error    : {rel_err:.3e}");
}

fn run_typed<T: FloatScalar>(opts: &CliOptions, label: &str) {
    let cfg = MeshConfig {
        radius: opts.radius,
        num_elem_per_face: opts.k,
        poly_order: opts.order,
        projection: opts.projection,
    };
    let mesh = Arc::new(CubedSphereMesh::<T>::from_config(cfg));
    let equation = build_equation(opts);
    let backend = CpuBackend::from_equation(Arc::clone(&mesh), &equation).with_flux(opts.flux);

    let sim_cfg = SimulationConfig {
        cfl: opts.cfl,
        periods: opts.periods,
        t_final: None,
        save_frames: opts.frames,
    };

    let output =
        run_lsrk45_with_dt(&backend, &equation, &sim_cfg, opts.dt).expect("CPU backend failed");
    let exact = equation.evaluate_exact(&mesh, output.t_final);
    let metrics = compute_diagnostics(
        &mesh,
        &output.initial,
        &output.final_state,
        exact.as_deref(),
    );
    let energy_history: Vec<EnergySample> = output
        .snapshots
        .iter()
        .map(|s| EnergySample {
            step: s.step,
            time: s.time,
            energy: compute_energy(&mesh, &s.phi),
        })
        .collect();

    let dof_stages = (mesh.total_nodes as f64) * (output.steps as f64) * 5.0;
    let mdofs_per_sec = if output.elapsed_secs > 0.0 {
        dof_stages / (output.elapsed_secs * 1e6)
    } else {
        0.0
    };

    println!("=== Simulation Summary ({label}) ===");
    println!(
        "Mesh              : K={}, p={}, {} elems, {} DOFs ({:?})",
        opts.k, opts.order, mesh.n_elems, mesh.total_nodes, opts.projection
    );
    println!(
        "Physics           : wind={}, initial={}, periods={}, flux={}",
        equation.wind().name(),
        equation.initial().name(),
        opts.periods,
        opts.flux
    );
    println!(
        "Time integration  : dt={:.6e}, steps={}, T_final={:.6}",
        output.dt, output.steps, output.t_final
    );
    println!(
        "Performance       : {:.4}s elapsed ({:.2} MDOF-stages/s)",
        output.elapsed_secs, mdofs_per_sec
    );
    println!(
        "Mass conservation : M(0)={:.12e}, M(T)={:.12e}, rel_err={:.3e}",
        metrics.mass_initial, metrics.mass_final, metrics.mass_rel_err
    );
    println!(
        "Energy (int phi^2): E(0)={:.12e}, E(T)={:.12e}, rel_change={:+.3e}",
        metrics.energy_initial, metrics.energy_final, metrics.energy_rel_change
    );
    println!(
        "Extrema           : init=[{:.6}, {:.6}], final=[{:.6}, {:.6}]",
        metrics.min_initial, metrics.max_initial, metrics.min_final, metrics.max_final
    );
    if !metrics.l2_rel.is_nan() {
        println!(
            "Accuracy errors   : L1_rel={:.4e}, L2_rel={:.4e}, L2_abs={:.4e}, Linf={:.4e}",
            metrics.l1_rel, metrics.l2_rel, metrics.l2_abs, metrics.l_inf
        );
    }

    if !energy_history.is_empty() {
        println!("Energy history    : {} snapshots", energy_history.len());
        for s in &energy_history {
            println!(
                "    step {:>8}  t={:.6}  E={:.12e}  E/E(0)-1={:+.3e}",
                s.step,
                s.time,
                s.energy,
                s.energy / metrics.energy_initial - 1.0
            );
        }
    }

    if let Some(vtk_path) = &opts.vtk_out {
        let mut fields: Vec<(&str, &[T])> = vec![
            ("phi", &output.final_state),
            ("initial", &output.initial),
            ("sqrt_g", &mesh.sqrt_g),
        ];
        let err_vec: Option<Vec<T>> = exact.as_ref().map(|ex| {
            output
                .final_state
                .iter()
                .zip(ex.iter())
                .map(|(&f, &e)| f - e)
                .collect()
        });
        if let Some(ex) = exact.as_ref() {
            fields.push(("exact", ex));
        }
        if let Some(ev) = err_vec.as_ref() {
            fields.push(("error", ev));
        }
        write_vtu(vtk_path, &mesh, &fields).expect("failed writing VTU");
        println!("Wrote VTK output  : {vtk_path}");
    }

    if let Some(pvd_dir) = &opts.pvd_dir {
        if !output.snapshots.is_empty() {
            let frames: Vec<(f64, &[T])> = output
                .snapshots
                .iter()
                .map(|s| (s.time, s.phi.as_slice()))
                .collect();
            write_pvd_series(pvd_dir, "frame", &mesh, &frames).expect("failed writing PVD series");
            println!(
                "Wrote PVD series  : {pvd_dir}/frame.pvd ({} frames)",
                frames.len()
            );
        }
    }

    if let Some(json_path) = &opts.json_out {
        let report = RunReport {
            flux: opts.flux,
            metrics,
            energy_history,
        };
        write_json(json_path, &report).expect("failed writing JSON");
        println!("Wrote JSON report : {json_path}");
    }
}

fn cmd_converge(opts: &CliOptions) {
    let equation = build_equation(opts);
    let sim_cfg = SimulationConfig {
        cfl: opts.cfl,
        periods: opts.periods,
        t_final: None,
        save_frames: 0,
    };

    println!(
        "=== Convergence Study ({:?}, wind={}, initial={}, {:?}, alpha0={:.3} rad, flux={}) ===",
        opts.mode,
        equation.wind().name(),
        equation.initial().name(),
        opts.projection,
        opts.alpha0,
        opts.flux
    );

    let total_start = Instant::now();
    let mut rows = Vec::new();

    match opts.mode {
        ConvergeMode::TRefinement => {
            let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
                radius: opts.radius,
                num_elem_per_face: opts.k,
                poly_order: opts.order,
                projection: opts.projection,
            }));
            let backend =
                CpuBackend::from_equation(Arc::clone(&mesh), &equation).with_flux(opts.flux);
            for &dt in &opts.dts {
                let out = run_lsrk45_with_dt(&backend, &equation, &sim_cfg, Some(dt))
                    .expect("CPU failed");
                let exact = equation
                    .evaluate_exact(&mesh, out.t_final)
                    .expect("convergence test requires analytical exact solution at t_final");
                let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));

                rows.push(ConvergenceRow {
                    k: opts.k,
                    poly_order: opts.order,
                    total_dofs: mesh.total_nodes,
                    dt: out.dt,
                    steps: out.steps,
                    elapsed_secs: out.elapsed_secs,
                    l1_rel: m.l1_rel,
                    l2_abs: m.l2_abs,
                    l2_rel: m.l2_rel,
                    order_l2: None,
                    l_inf: m.l_inf,
                    order_linf: None,
                    mass_rel_err: m.mass_rel_err,
                });
            }
            populate_temporal_convergence_orders(&mut rows);
            println!("\n{}", format_temporal_convergence_table(&rows));
        }
        ConvergeMode::PRefinement | ConvergeMode::HRefinement => {
            let cases: Vec<(usize, usize)> = match opts.mode {
                ConvergeMode::PRefinement => opts.orders.iter().map(|&p| (opts.k, p)).collect(),
                ConvergeMode::HRefinement => opts.ks.iter().map(|&k| (k, opts.order)).collect(),
                ConvergeMode::TRefinement => unreachable!(),
            };
            rows.reserve(cases.len());

            for (k, p) in cases {
                let mesh = Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig {
                    radius: opts.radius,
                    num_elem_per_face: k,
                    poly_order: p,
                    projection: opts.projection,
                }));
                let backend =
                    CpuBackend::from_equation(Arc::clone(&mesh), &equation).with_flux(opts.flux);
                let out =
                    run_lsrk45_with_dt(&backend, &equation, &sim_cfg, opts.dt).expect("CPU failed");
                let exact = equation
                    .evaluate_exact(&mesh, out.t_final)
                    .expect("convergence test requires analytical exact solution at t_final");
                let m = compute_diagnostics(&mesh, &out.initial, &out.final_state, Some(&exact));

                rows.push(ConvergenceRow {
                    k,
                    poly_order: p,
                    total_dofs: mesh.total_nodes,
                    dt: out.dt,
                    steps: out.steps,
                    elapsed_secs: out.elapsed_secs,
                    l1_rel: m.l1_rel,
                    l2_abs: m.l2_abs,
                    l2_rel: m.l2_rel,
                    order_l2: None,
                    l_inf: m.l_inf,
                    order_linf: None,
                    mass_rel_err: m.mass_rel_err,
                });
            }

            let vary_by_k = matches!(opts.mode, ConvergeMode::HRefinement);
            populate_convergence_orders(&mut rows, vary_by_k);
            println!("\n{}", format_convergence_table(&rows));
        }
    }

    println!(
        "Total wall-clock time across {} runs: {:.3}s",
        rows.len(),
        total_start.elapsed().as_secs_f64()
    );

    if let Some(csv_path) = &opts.csv_out {
        write_convergence_csv(csv_path, &rows).expect("failed writing CSV");
        println!("Wrote CSV report: {csv_path}");
    }
}

fn bench_typed<T: FloatScalar>(opts: &CliOptions, label: &str) {
    let mesh = Arc::new(CubedSphereMesh::<T>::from_config(MeshConfig {
        radius: opts.radius,
        num_elem_per_face: opts.k,
        poly_order: opts.order,
        projection: opts.projection,
    }));
    let eq = build_equation(opts);
    let backend = CpuBackend::from_equation(Arc::clone(&mesh), &eq).with_flux(opts.flux);

    let init = eq.evaluate_initial(&mesh);
    let mut u = backend.upload(&init).unwrap();
    let mut du = backend.allocate_zeros(mesh.total_nodes).unwrap();
    let mut rhs = backend.allocate_zeros(mesh.total_nodes).unwrap();

    let dt = 1e-4;
    // Warmup
    backend
        .lsrk45_step(&mut u, &mut du, &mut rhs, 0.0, dt)
        .unwrap();

    let start = Instant::now();
    for step in 0..opts.bench_steps {
        backend
            .lsrk45_step(&mut u, &mut du, &mut rhs, (step as f64) * dt, dt)
            .unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let _out = backend.download(&u).unwrap();

    let stages = (opts.bench_steps as f64) * 5.0;
    let total_dof_stages = (mesh.total_nodes as f64) * stages;
    println!(
        "[{label}] K={:<2}, p={:<2}, DOFs={:<7} | {:>4} steps ({:>5.0} stages) in {:>7.4}s | {:>7.3} ms/step | {:>7.2} MDOF-stages/s",
        opts.k,
        opts.order,
        mesh.total_nodes,
        opts.bench_steps,
        stages,
        elapsed,
        (elapsed * 1e3) / (opts.bench_steps as f64),
        total_dof_stages / (elapsed * 1e6)
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    let subcmd = args[1].as_str();
    if subcmd == "-h" || subcmd == "--help" {
        print_usage();
        return;
    }

    let opts = match parse_args(&args[2..]) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("Error: {err}\nRun `cubesphere --help` for usage.");
            process::exit(1);
        }
    };

    match subcmd {
        "info" => cmd_info(&opts),
        "run" => match opts.precision {
            Precision::F64 => run_typed::<f64>(&opts, "f64"),
            Precision::F32 => run_typed::<f32>(&opts, "f32"),
        },
        "converge" => cmd_converge(&opts),
        "bench" => {
            println!(
                "=== CPU Rayon Throughput Benchmark ({:?}) ===",
                opts.projection
            );
            match opts.precision {
                Precision::F64 => bench_typed::<f64>(&opts, "f64"),
                Precision::F32 => bench_typed::<f32>(&opts, "f32"),
            }
        }
        other => {
            eprintln!("Unknown subcommand '{other}'. Run `cubesphere --help` for usage.");
            process::exit(1);
        }
    }
}
