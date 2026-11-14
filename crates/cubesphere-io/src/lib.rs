#![warn(missing_docs)]
//! Surface-quadrature diagnostics, convergence reporting, and VTK output.
//!
//! Evaluates `L1`/`L2`/`L_inf` error norms, mass conservation, and discrete energy on
//! `cubesphere_core::CubedSphereMesh`, and writes CSV, JSON, and `.vtu`/`.pvd` files.
//!
//! ```rust
//! use cubesphere_core::{CubedSphereMesh, MeshConfig};
//! use cubesphere_io::{compute_diagnostics, compute_energy};
//! let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig::default());
//! let ones = vec![1.0_f64; mesh.total_nodes];
//! let energy = compute_energy(&mesh, &ones);
//! let metrics = compute_diagnostics(&mesh, &ones, &ones, Some(&ones));
//! let exact_area = 4.0 * std::f64::consts::PI * mesh.config.radius.powi(2);
//! assert!((energy - exact_area).abs() / exact_area < 1e-10);
//! assert!(metrics.l2_rel < 1e-15 && metrics.mass_rel_err < 1e-15);
//! ```

pub mod diagnostics;
pub mod report;
pub mod vtk;

pub use diagnostics::{compute_diagnostics, compute_energy, ErrorMetrics};
pub use report::{
    format_convergence_table, format_temporal_convergence_table, populate_convergence_orders,
    populate_temporal_convergence_orders, write_convergence_csv, write_json, ConvergenceRow,
};
pub use vtk::{write_pvd_series, write_vtu};
