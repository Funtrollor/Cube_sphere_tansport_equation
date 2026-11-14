#![warn(missing_docs)]
//! Physical PDE definitions, spherical wind fields, and initial tracer fields.
//!
//! Builds on `cubesphere_core` meshes to provide analytical initial conditions, exact
//! solutions, and 3D or contravariant velocities for `cubesphere_solver` backends.
//!
//! ```rust
//! use std::sync::Arc;
//! use cubesphere_core::{CubedSphereMesh, MeshConfig};
//! use cubesphere_equations::{GaussianBell, ScalarAdvection, SolidBodyRotation};
//! let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig::default());
//! let eq = ScalarAdvection::solid_body(SolidBodyRotation::default(), Arc::new(GaussianBell::default()));
//! let phi0 = eq.evaluate_initial(&mesh);
//! let phi1 = eq.evaluate_exact(&mesh, 1.0).unwrap();
//! let max_diff = phi0.iter().zip(&phi1).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
//! assert!(max_diff < 1e-14);
//! ```

pub mod advection;
pub mod initial;
pub mod wind;

pub use advection::{Equation, ScalarAdvection};
pub use initial::{
    ConstantField, CosineBell, GaussianBell, InitialKind, ScalarField, TwinGaussianBells,
};
pub use wind::{NairLauritzenFlow, SolidBodyRotation, WindDescriptor, WindField, WindKind};
