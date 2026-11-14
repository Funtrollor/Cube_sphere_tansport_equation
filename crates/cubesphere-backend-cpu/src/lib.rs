#![warn(missing_docs)]
//! Parallel Rayon CPU backend implementing [`cubesphere_solver::ComputeBackend`] with split-form divergence and SBP-SAT.
//!
//! ```rust
//! use cubesphere_backend_cpu::CpuBackend;
//! use cubesphere_core::{CubedSphereMesh, MeshConfig};
//! use cubesphere_equations::{GaussianBell, ScalarAdvection, SolidBodyRotation};
//! use cubesphere_solver::ComputeBackend;
//! let mesh = std::sync::Arc::new(CubedSphereMesh::<f64>::from_config(MeshConfig::default()));
//! let eq = ScalarAdvection::solid_body(SolidBodyRotation::default(), std::sync::Arc::new(GaussianBell::default()));
//! let cpu = CpuBackend::from_equation(mesh.clone(), &eq);
//! let (mut u, mut du, mut rhs) = (eq.evaluate_initial(&mesh), vec![0.0; mesh.total_nodes], vec![0.0; mesh.total_nodes]);
//! let m0 = mesh.integrate(&u);
//! cpu.lsrk45_step(&mut u, &mut du, &mut rhs, 0.0, 1e-4).unwrap();
//! assert!((mesh.integrate(&u) - m0).abs() < 1e-13);
//! ```

pub mod backend;

pub use backend::CpuBackend;
