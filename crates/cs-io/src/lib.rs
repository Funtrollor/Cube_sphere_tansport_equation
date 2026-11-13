pub mod diagnostics;
pub mod report;
pub mod vtk;

pub use diagnostics::{compute_diagnostics, ErrorMetrics};
pub use report::{
    format_convergence_table, format_temporal_convergence_table, populate_convergence_orders,
    populate_temporal_convergence_orders, write_convergence_csv, write_json, ConvergenceRow,
};
pub use vtk::{write_pvd_series, write_vtu};
