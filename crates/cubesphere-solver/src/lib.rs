#![warn(missing_docs)]
//! Backend-independent LSRK45 time integration, CFL estimates, and SBP-SAT flux choices.
//! Uses `cubesphere-core` meshes and `cubesphere-equations` scalar advection models;
//! workspace compute backends implement `ComputeBackend` to advance their own buffers.
//!
//! ```rust
//! use cubesphere_solver::NumericalFlux;
//! let flux: NumericalFlux = "0.25".parse().unwrap();
//! assert_eq!(flux.upwind_weight(), 0.25);
//! assert!("1.5".parse::<NumericalFlux>().is_err());
//! ```
pub mod backend;
pub mod flux;
pub mod lsrk45;

pub use backend::ComputeBackend;
pub use flux::NumericalFlux;
pub use lsrk45::{
    compute_cfl_dt, compute_max_contravariant_speed, run_lsrk45_simulation, run_lsrk45_with_dt,
    SimulationConfig, SimulationOutput, Snapshot, RK45_A, RK45_B, RK45_C,
};
