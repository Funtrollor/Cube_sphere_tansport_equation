use std::f64::consts::FRAC_PI_4;
use std::sync::Arc;
use std::time::Instant;

use cs_backend_cpu::CpuBackend;
use cs_backend_cuda::{CudaBackend, CudaScalar};
use cs_core::{CubedSphereMesh, MeshConfig, ProjectionKind};
use cs_equations::{
    GaussianBell, NairLauritzenFlow, ScalarAdvection, SolidBodyRotation, TwinGaussianBells,
};
use cs_io::compute_diagnostics;
use cs_solver::{ComputeBackend, NumericalFlux, SimulationConfig, run_lsrk45_simulation};

fn verify_full_simulation_parity<T: CudaScalar>(
    label: &str,
    cfg: MeshConfig,
    equation: &ScalarAdvection,
    sim_cfg: &SimulationConfig,
    flux: NumericalFlux,
    max_parity_tol: f64,
    max_mass_tol: f64,
) {
    let mesh = Arc::new(CubedSphereMesh::<T>::from_config(cfg));
    let cpu = CpuBackend::from_equation(Arc::clone(&mesh), equation).with_flux(flux);
    let gpu = CudaBackend::from_equation(0, Arc::clone(&mesh), equation)
        .expect("failed initializing CudaBackend")
        .with_flux(flux);

    let cpu_out = run_lsrk45_simulation(&cpu, equation, sim_cfg).expect("CPU run failed");
    let gpu_out = run_lsrk45_simulation(&gpu, equation, sim_cfg).expect("GPU run failed");

    let max_diff = cpu_out
        .final_state
        .iter()
        .zip(gpu_out.final_state.iter())
        .map(|(&c, &g)| (c.to_f64() - g.to_f64()).abs())
        .fold(0.0_f64, f64::max);

    let exact = equation.evaluate_exact(&mesh, gpu_out.t_final);
    let metrics = compute_diagnostics(&mesh, &gpu_out.initial, &gpu_out.final_state, exact.as_deref());

    let dof_stages = (mesh.total_nodes as f64) * (gpu_out.steps as f64) * 5.0;
    let cpu_mdofs = dof_stages / (cpu_out.elapsed_secs.max(1e-9) * 1e6);
    let gpu_mdofs = dof_stages / (gpu_out.elapsed_secs.max(1e-9) * 1e6);
    let speedup = cpu_out.elapsed_secs / gpu_out.elapsed_secs.max(1e-9);

    println!(
        "[{label}] flux={flux}, K={}, p={}, DOFs={}, steps={} | max|GPU-CPU|={max_diff:.3e} | mass_err={:.3e} | CPU={:.4}s ({cpu_mdofs:.2} MDOF-stages/s) | GPU={:.4}s ({gpu_mdofs:.2} MDOF-stages/s) | {speedup:.2}x",
        cfg.num_elem_per_face,
        cfg.poly_order,
        mesh.total_nodes,
        gpu_out.steps,
        metrics.mass_rel_err,
        cpu_out.elapsed_secs,
        gpu_out.elapsed_secs,
    );

    assert!(
        max_diff < max_parity_tol,
        "{label}: GPU vs CPU mismatch max_diff={max_diff:.3e} >= {max_parity_tol:.3e}"
    );
    assert!(
        metrics.mass_rel_err < max_mass_tol,
        "{label}: GPU mass conservation error={:.3e} >= {max_mass_tol:.3e}",
        metrics.mass_rel_err
    );
}

fn benchmark_throughput<T: CudaScalar>(label: &str, k: usize, p: usize, steps: usize) {
    let cfg = MeshConfig {
        radius: 1.0,
        num_elem_per_face: k,
        poly_order: p,
        projection: ProjectionKind::Equiangular,
    };
    let mesh = Arc::new(CubedSphereMesh::<T>::from_config(cfg));
    let eq = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_4, None),
        Arc::new(GaussianBell::default()),
    );
    let init = eq.evaluate_initial(&mesh);
    let dt = 1e-4;

    // CPU timing
    let cpu = CpuBackend::from_equation(Arc::clone(&mesh), &eq);
    let mut cpu_u = cpu.upload(&init).unwrap();
    let mut cpu_du = cpu.allocate_zeros(mesh.total_nodes).unwrap();
    let mut cpu_rhs = cpu.allocate_zeros(mesh.total_nodes).unwrap();
    cpu.lsrk45_step(&mut cpu_u, &mut cpu_du, &mut cpu_rhs, 0.0, dt).unwrap();

    let cpu_t0 = Instant::now();
    for s in 0..steps {
        cpu.lsrk45_step(&mut cpu_u, &mut cpu_du, &mut cpu_rhs, (s as f64) * dt, dt)
            .unwrap();
    }
    let cpu_elapsed = cpu_t0.elapsed().as_secs_f64();

    // GPU timing (warmed up + stream synchronized before stopping timer)
    let gpu = CudaBackend::from_equation(0, Arc::clone(&mesh), &eq).unwrap();
    let mut gpu_u = gpu.upload(&init).unwrap();
    let mut gpu_du = gpu.allocate_zeros(mesh.total_nodes).unwrap();
    let mut gpu_rhs = gpu.allocate_zeros(mesh.total_nodes).unwrap();
    gpu.lsrk45_step(&mut gpu_u, &mut gpu_du, &mut gpu_rhs, 0.0, dt).unwrap();
    gpu.synchronize().unwrap();

    let gpu_t0 = Instant::now();
    for s in 0..steps {
        gpu.lsrk45_step(&mut gpu_u, &mut gpu_du, &mut gpu_rhs, (s as f64) * dt, dt)
            .unwrap();
    }
    gpu.synchronize().unwrap();
    let gpu_elapsed = gpu_t0.elapsed().as_secs_f64();

    let dof_stages = (mesh.total_nodes as f64) * (steps as f64) * 5.0;
    let cpu_mdofs = dof_stages / (cpu_elapsed.max(1e-9) * 1e6);
    let gpu_mdofs = dof_stages / (gpu_elapsed.max(1e-9) * 1e6);
    let speedup = cpu_elapsed / gpu_elapsed.max(1e-9);

    println!(
        "[BENCH {label}] K={k:<2} p={p:<2} DOFs={:<7} ({steps} steps) | CPU={cpu_elapsed:.4}s ({cpu_mdofs:>7.2} MDOF-stages/s) | GPU={gpu_elapsed:.4}s ({gpu_mdofs:>8.2} MDOF-stages/s) | Speedup={speedup:>6.2}x",
        mesh.total_nodes
    );
}

fn main() {
    println!("=== Cubed-Sphere Transport: cuda-oxide GPU vs CPU Parity & Benchmark ===");

    let solid_eq = ScalarAdvection::solid_body(
        SolidBodyRotation::new(FRAC_PI_4, None),
        Arc::new(GaussianBell::default()),
    );
    verify_full_simulation_parity::<f64>(
        "SolidBody f64",
        MeshConfig {
            radius: 1.0,
            num_elem_per_face: 4,
            poly_order: 4,
            projection: ProjectionKind::Equiangular,
        },
        &solid_eq,
        &SimulationConfig {
            cfl: 0.5,
            periods: 1.0,
            t_final: None,
            save_frames: 0,
        },
        NumericalFlux::Upwind,
        1e-13,
        1e-14,
    );

    let deform_eq = ScalarAdvection::new(
        Arc::new(NairLauritzenFlow::default()),
        Arc::new(TwinGaussianBells::default()),
    );
    verify_full_simulation_parity::<f64>(
        "NairLauritzen f64",
        MeshConfig {
            radius: 1.0,
            num_elem_per_face: 4,
            poly_order: 5,
            projection: ProjectionKind::Equiangular,
        },
        &deform_eq,
        &SimulationConfig {
            cfl: 0.5,
            periods: 1.0,
            t_final: None,
            save_frames: 0,
        },
        NumericalFlux::Upwind,
        1e-12,
        1e-14,
    );

    verify_full_simulation_parity::<f64>(
        "SolidBody f64 central",
        MeshConfig {
            radius: 1.0,
            num_elem_per_face: 4,
            poly_order: 4,
            projection: ProjectionKind::Equiangular,
        },
        &solid_eq,
        &SimulationConfig {
            cfl: 0.5,
            periods: 1.0,
            t_final: None,
            save_frames: 0,
        },
        NumericalFlux::Central,
        1e-13,
        1e-14,
    );

    verify_full_simulation_parity::<f64>(
        "NairLauritzen f64 blended",
        MeshConfig {
            radius: 1.0,
            num_elem_per_face: 4,
            poly_order: 5,
            projection: ProjectionKind::Equiangular,
        },
        &deform_eq,
        &SimulationConfig {
            cfl: 0.5,
            periods: 1.0,
            t_final: None,
            save_frames: 0,
        },
        NumericalFlux::Blended(0.5),
        1e-12,
        1e-14,
    );

    verify_full_simulation_parity::<f32>(
        "SolidBody f32",
        MeshConfig {
            radius: 1.0,
            num_elem_per_face: 8,
            poly_order: 4,
            projection: ProjectionKind::Equiangular,
        },
        &solid_eq,
        &SimulationConfig {
            cfl: 0.5,
            periods: 0.5,
            t_final: None,
            save_frames: 0,
        },
        NumericalFlux::Upwind,
        2e-5,
        1e-5,
    );

    println!("\n=== High-Resolution Throughput Comparison (RTX GPU vs CPU Rayon) ===");
    benchmark_throughput::<f64>("f64", 16, 4, 200);
    benchmark_throughput::<f32>("f32", 16, 4, 200);
    benchmark_throughput::<f64>("f64", 32, 4, 200);
    benchmark_throughput::<f32>("f32", 32, 4, 200);
    println!("\nALL GPU PARITY CHECKS PASSED.");
}
