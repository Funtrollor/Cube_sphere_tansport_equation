#![warn(missing_docs)]
//! LGL reference operators, cubed-sphere geometry, and spectral-element mesh topology.
//! Geometry is precomputed in `f64` and converted to `FloatScalar` runtime buffers for
//! the workspace's transport equations and compute backends.
//!
//! ```rust
//! use cubesphere_core::{CubedSphereMesh, MeshConfig};
//! let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig {
//!     radius: 2.0, num_elem_per_face: 4, poly_order: 8, ..MeshConfig::default()
//! });
//! let exact = 4.0 * std::f64::consts::PI * 2.0_f64.powi(2);
//! assert!((mesh.surface_area() - exact).abs() / exact < 1e-12);
//! ```
pub mod basis;
pub mod geometry;
pub mod mesh;
pub mod scalar;

pub use basis::ReferenceElement;
pub use geometry::{
    cross3, dot3, great_circle_distance, lon_lat_wind_to_3d, xyz_to_lon_lat, CubeFace, EdgeSide,
    EquiangularProjection, EquidistantProjection, PanelConnection, PointMetric, Projection,
    PANEL_CONNECTIVITY,
};
pub use mesh::{CubedSphereMesh, EdgeNeighbor, MeshConfig, ProjectionKind};
pub use scalar::FloatScalar;
