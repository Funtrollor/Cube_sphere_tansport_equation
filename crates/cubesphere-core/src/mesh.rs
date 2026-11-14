//! Tensor-product spectral elements on six faces, with geometry and SBP-SAT edge neighbors.
use serde::{Deserialize, Serialize};

use crate::basis::ReferenceElement;
use crate::geometry::{
    dot3, CubeFace, EdgeSide, EquiangularProjection, EquidistantProjection, Projection,
    PANEL_CONNECTIVITY,
};
use crate::scalar::FloatScalar;

/// Supported Cubed-Sphere projection kinds for configuration and CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ProjectionKind {
    /// Uniform angular coordinates, `x = tan(alpha)`, `y = tan(beta)`.
    #[default]
    Equiangular,
    /// Uniform cube-plane coordinates, `x = 4*alpha/pi`, `y = 4*beta/pi`.
    Equidistant,
}

/// Configuration for generating a multi-element Cubed-Sphere mesh.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MeshConfig {
    /// Positive sphere radius `R`, setting the length unit of Cartesian geometry.
    pub radius: f64,
    /// Number of spectral elements per face edge (`K`), producing `6 * K * K` elements total.
    pub num_elem_per_face: usize,
    /// Polynomial order `p` inside each element (`(p + 1) x (p + 1)` LGL nodes).
    pub poly_order: usize,
    /// Projection mapping strategy.
    pub projection: ProjectionKind,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            radius: 1.0,
            num_elem_per_face: 8,
            poly_order: 4,
            projection: ProjectionKind::Equiangular,
        }
    }
}

/// Neighbor metadata for one edge of one spectral element.
///
/// `#[repr(C)]` ensures identical memory layout for GPU buffer transfer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeNeighbor {
    /// Global element index across this edge, in face/beta/alpha order.
    pub nbr_elem: u32,
    /// `EdgeSide` encoded as `u8` (0=West, 1=East, 2=South, 3=North).
    pub nbr_side: u8,
    /// `1` if the neighbor edge quadrature nodes must be reversed, `0` otherwise.
    pub reverse: u8,
    /// Two padding bytes, initialized to zero for the C-compatible buffer layout.
    pub _pad: [u8; 2],
}

impl EdgeNeighbor {
    /// Decodes the neighboring edge's local coordinate direction.
    /// # Panics
    /// Panics if `nbr_side` is outside `0..4`.
    #[inline]
    pub const fn side(&self) -> EdgeSide {
        EdgeSide::from_index(self.nbr_side as usize)
    }

    /// Whether edge node `m` must be read as neighbor node `p - m` (`reverse != 0`).
    #[inline]
    pub const fn is_reversed(&self) -> bool {
        self.reverse != 0
    }
}

/// Multi-element Cubed-Sphere mesh with contiguous `[elem][j][i]` layout.
///
/// Node ordering inside element `e` is `idx = e * n_nodes_elem + j * n_nodes_1d + i`,
/// where `i in 0..=p` corresponds to `alpha/xi` and `j in 0..=p` corresponds to `beta/eta`.
/// Faces cover `alpha, beta` in `[-pi/4, pi/4]` radians; element boundaries retain duplicate nodes.
/// Geometry is computed in `f64`, then converted to `T`; selected `f64` arrays are retained.
#[derive(Debug, Clone)]
pub struct CubedSphereMesh<T: FloatScalar> {
    /// Configuration supplied at construction, retaining the original `f64` radius.
    pub config: MeshConfig,
    /// Sphere radius converted to runtime scalar `T`.
    pub radius: T,
    /// Elements per panel edge, `K`; each face has `K * K` elements.
    pub num_elem_per_face: usize,
    /// Total number of elements, `6 * K * K`.
    pub n_elems: usize,
    /// Nodes per element coordinate direction, `p + 1`.
    pub n_nodes_1d: usize,
    /// Nodes per tensor-product element, `(p + 1)^2`.
    pub n_nodes_elem: usize,
    /// Length of each nodal field, `n_elems * n_nodes_elem`, including duplicated edge nodes.
    pub total_nodes: usize,
    /// Mapping derivative `d(xi)/d(alpha) = d(eta)/d(beta) = 4 * K / pi`.
    pub map_factor: T,
    /// Element half-width `d(alpha)/d(xi) = d(beta)/d(eta) = pi / (4 * K)` in radians.
    pub quad_scale: T,
    /// Scaled 1D quadrature weights `w_i = (pi / (4K)) * w_ref_i` for SAT/integration.
    pub weights_1d: Vec<T>,
    /// The same `pi/(4K) * w_ref` weights retained in `f64`, length `n_nodes_1d`.
    pub weights_1d_f64: Vec<f64>,
    /// Unscaled reference nodes, weights, and differentiation matrix on `[-1, 1]`.
    pub reference: ReferenceElement<T>,
    /// Flattened coordinates in 3D Cartesian space `[X, Y, Z]`, each of length `total_nodes`.
    pub xyz: [Vec<T>; 3],
    /// Longitude in radians in `[-pi, pi]`, in `[elem][j][i]` order, length `total_nodes`.
    pub lon: Vec<T>,
    /// Latitude in radians in `[-pi/2, pi/2]`, in `[elem][j][i]` order, length `total_nodes`.
    pub lat: Vec<T>,
    /// Longitude in radians before conversion to `T`, in the same nodal order as `lon`.
    pub lon_f64: Vec<f64>,
    /// Latitude in radians before conversion to `T`, in the same nodal order as `lat`.
    pub lat_f64: Vec<f64>,
    /// Area factor `sqrt(g) = J(alpha, beta)` in squared radius units, length `total_nodes`.
    /// Excludes `quad_scale^2`; integration pairs it with two scaled `weights_1d` factors.
    pub sqrt_g: Vec<T>,
    /// Area factors retained in `f64` before conversion to `T`, in `[elem][j][i]` order.
    pub sqrt_g_f64: Vec<f64>,
    /// Dual basis `e^alpha` in inverse length units, stored as `[X, Y, Z]` component arrays.
    /// Each array has `total_nodes` entries; dotting physical wind yields `d(alpha)/dt`.
    pub contra_alpha: [Vec<T>; 3],
    /// Dual basis `e^beta` in inverse length units, stored as `[X, Y, Z]` component arrays.
    /// Each array has `total_nodes` entries; dotting physical wind yields `d(beta)/dt`.
    pub contra_beta: [Vec<T>; 3],
    /// Flattened neighbor table of length `n_elems * 4`, indexed as `elem * 4 + side`.
    pub neighbors: Vec<EdgeNeighbor>,
}

impl<T: FloatScalar> CubedSphereMesh<T> {
    /// Builds geometry and topology using the projection selected by `config.projection`.
    /// # Panics
    /// Panics unless radius is positive, `num_elem_per_face >= 1`, and `poly_order >= 1`.
    pub fn from_config(config: MeshConfig) -> Self {
        match config.projection {
            ProjectionKind::Equiangular => Self::new(config, &EquiangularProjection),
            ProjectionKind::Equidistant => Self::new(config, &EquidistantProjection),
        }
    }

    /// Builds a mesh with the supplied projection, independently of `config.projection`.
    /// Geometry uses `f64` before conversion to `T`; topology uses Staircase face connections.
    /// # Panics
    /// Panics on invalid radius, element count, or order; see [`Self::from_config`].
    pub fn new<P: Projection>(config: MeshConfig, projection: &P) -> Self {
        assert!(config.radius > 0.0, "sphere radius must be positive");
        assert!(
            config.num_elem_per_face >= 1,
            "num_elem_per_face must be >= 1"
        );
        assert!(config.poly_order >= 1, "poly_order must be >= 1");

        let k = config.num_elem_per_face;
        let p = config.poly_order;
        let n_nodes_1d = p + 1;
        let n_nodes_elem = n_nodes_1d * n_nodes_1d;
        let n_elems = 6 * k * k;
        let total_nodes = n_elems * n_nodes_elem;

        let reference = ReferenceElement::<T>::new(p);

        let pi4 = std::f64::consts::FRAC_PI_4;
        let elem_half_width = pi4 / (k as f64); // pi / (4K)
        let map_factor_f64 = 1.0 / elem_half_width; // 4K / pi

        let weights_1d_f64: Vec<f64> = reference
            .weights_f64
            .iter()
            .map(|&w| w * elem_half_width)
            .collect();
        let weights_1d: Vec<T> = weights_1d_f64.iter().map(|&w| T::from_f64(w)).collect();

        let mut x = vec![T::ZERO; total_nodes];
        let mut y = vec![T::ZERO; total_nodes];
        let mut z = vec![T::ZERO; total_nodes];
        let mut lon = vec![T::ZERO; total_nodes];
        let mut lat = vec![T::ZERO; total_nodes];
        let mut lon_f64 = vec![0.0; total_nodes];
        let mut lat_f64 = vec![0.0; total_nodes];
        let mut sqrt_g = vec![T::ZERO; total_nodes];
        let mut sqrt_g_f64 = vec![0.0; total_nodes];

        let mut ca_x = vec![T::ZERO; total_nodes];
        let mut ca_y = vec![T::ZERO; total_nodes];
        let mut ca_z = vec![T::ZERO; total_nodes];
        let mut cb_x = vec![T::ZERO; total_nodes];
        let mut cb_y = vec![T::ZERO; total_nodes];
        let mut cb_z = vec![T::ZERO; total_nodes];

        let mut neighbors = Vec::with_capacity(n_elems * 4);

        for face in CubeFace::ALL {
            for e_beta in 0..k {
                let beta_center = -pi4 + (2 * e_beta + 1) as f64 * elem_half_width;
                for e_alpha in 0..k {
                    let alpha_center = -pi4 + (2 * e_alpha + 1) as f64 * elem_half_width;
                    let elem = Self::compute_elem_id(k, face, e_alpha, e_beta);
                    let base = elem * n_nodes_elem;

                    for j in 0..n_nodes_1d {
                        let beta = beta_center + elem_half_width * reference.nodes_f64[j];
                        for i in 0..n_nodes_1d {
                            let alpha = alpha_center + elem_half_width * reference.nodes_f64[i];
                            let m = projection.point_metric(face, alpha, beta, config.radius);
                            let idx = base + j * n_nodes_1d + i;

                            x[idx] = T::from_f64(m.xyz[0]);
                            y[idx] = T::from_f64(m.xyz[1]);
                            z[idx] = T::from_f64(m.xyz[2]);
                            lon_f64[idx] = m.lon;
                            lat_f64[idx] = m.lat;
                            lon[idx] = T::from_f64(m.lon);
                            lat[idx] = T::from_f64(m.lat);
                            sqrt_g_f64[idx] = m.sqrt_g;
                            sqrt_g[idx] = T::from_f64(m.sqrt_g);

                            ca_x[idx] = T::from_f64(m.contra_alpha[0]);
                            ca_y[idx] = T::from_f64(m.contra_alpha[1]);
                            ca_z[idx] = T::from_f64(m.contra_alpha[2]);
                            cb_x[idx] = T::from_f64(m.contra_beta[0]);
                            cb_y[idx] = T::from_f64(m.contra_beta[1]);
                            cb_z[idx] = T::from_f64(m.contra_beta[2]);
                        }
                    }

                    for side in EdgeSide::ALL {
                        neighbors.push(Self::compute_neighbor(k, face, e_alpha, e_beta, side));
                    }
                }
            }
        }

        Self {
            config,
            radius: T::from_f64(config.radius),
            num_elem_per_face: k,
            n_elems,
            n_nodes_1d,
            n_nodes_elem,
            total_nodes,
            map_factor: T::from_f64(map_factor_f64),
            quad_scale: T::from_f64(elem_half_width),
            weights_1d,
            weights_1d_f64,
            reference,
            xyz: [x, y, z],
            lon,
            lat,
            lon_f64,
            lat_f64,
            sqrt_g,
            sqrt_g_f64,
            contra_alpha: [ca_x, ca_y, ca_z],
            contra_beta: [cb_x, cb_y, cb_z],
            neighbors,
        }
    }

    /// Flattens panel coordinates as `(face.index() * k + e_beta) * k + e_alpha`.
    /// Expects `e_alpha, e_beta < k`; no coordinate bounds checking is performed.
    #[inline]
    pub const fn compute_elem_id(k: usize, face: CubeFace, e_alpha: usize, e_beta: usize) -> usize {
        (face.index() * k + e_beta) * k + e_alpha
    }

    /// Inverts an element index into `(face, e_alpha, e_beta)`, with alpha varying fastest.
    /// # Panics
    /// Panics if `elem >= n_elems` for an unmodified mesh.
    #[inline]
    pub const fn elem_coords(&self, elem: usize) -> (CubeFace, usize, usize) {
        let k = self.num_elem_per_face;
        let kk = k * k;
        let face_idx = elem / kk;
        let rem = elem % kk;
        let e_beta = rem / k;
        let e_alpha = rem % k;
        (CubeFace::from_index(face_idx), e_alpha, e_beta)
    }

    /// Flattens `(elem, i, j)` as `elem * n_nodes_elem + j * n_nodes_1d + i`.
    /// `i` indexes alpha and `j` beta; no coordinate bounds checking is performed.
    #[inline]
    pub const fn node_idx(&self, elem: usize, i: usize, j: usize) -> usize {
        elem * self.n_nodes_elem + j * self.n_nodes_1d + i
    }

    /// Looks up the adjacent element, its edge side, and any edge-node reversal.
    /// # Panics
    /// Panics if `elem * 4 + side.index()` is outside the neighbor table.
    #[inline]
    pub fn neighbor(&self, elem: usize, side: EdgeSide) -> EdgeNeighbor {
        self.neighbors[elem * 4 + side.index()]
    }

    fn compute_neighbor(
        k: usize,
        face: CubeFace,
        e_alpha: usize,
        e_beta: usize,
        side: EdgeSide,
    ) -> EdgeNeighbor {
        // Interior edges within the same cube panel
        match side {
            EdgeSide::West if e_alpha > 0 => {
                return EdgeNeighbor {
                    nbr_elem: Self::compute_elem_id(k, face, e_alpha - 1, e_beta) as u32,
                    nbr_side: EdgeSide::East as u8,
                    reverse: 0,
                    _pad: [0; 2],
                };
            }
            EdgeSide::East if e_alpha + 1 < k => {
                return EdgeNeighbor {
                    nbr_elem: Self::compute_elem_id(k, face, e_alpha + 1, e_beta) as u32,
                    nbr_side: EdgeSide::West as u8,
                    reverse: 0,
                    _pad: [0; 2],
                };
            }
            EdgeSide::South if e_beta > 0 => {
                return EdgeNeighbor {
                    nbr_elem: Self::compute_elem_id(k, face, e_alpha, e_beta - 1) as u32,
                    nbr_side: EdgeSide::North as u8,
                    reverse: 0,
                    _pad: [0; 2],
                };
            }
            EdgeSide::North if e_beta + 1 < k => {
                return EdgeNeighbor {
                    nbr_elem: Self::compute_elem_id(k, face, e_alpha, e_beta + 1) as u32,
                    nbr_side: EdgeSide::South as u8,
                    reverse: 0,
                    _pad: [0; 2],
                };
            }
            _ => {}
        }

        // Cross-panel edges
        let conn = PANEL_CONNECTIVITY[face.index()][side.index()];
        let t = match side {
            EdgeSide::West | EdgeSide::East => e_beta,
            EdgeSide::South | EdgeSide::North => e_alpha,
        };
        let t_nbr = if conn.reverse { k - 1 - t } else { t };

        let (nbr_a, nbr_b) = match conn.nbr_side {
            EdgeSide::West => (0, t_nbr),
            EdgeSide::East => (k - 1, t_nbr),
            EdgeSide::South => (t_nbr, 0),
            EdgeSide::North => (t_nbr, k - 1),
        };

        EdgeNeighbor {
            nbr_elem: Self::compute_elem_id(k, conn.nbr_face, nbr_a, nbr_b) as u32,
            nbr_side: conn.nbr_side as u8,
            reverse: if conn.reverse { 1 } else { 0 },
            _pad: [0; 2],
        }
    }

    /// Integrates a `[elem][j][i]` nodal field as `sum(w_i * w_j * sqrt(g) * field)` in `f64`.
    /// Uses retained `f64` geometry and weights; result has field units times squared radius units.
    /// # Panics
    /// Panics unless `field.len() == total_nodes`.
    pub fn integrate(&self, field: &[T]) -> f64 {
        assert_eq!(field.len(), self.total_nodes);
        let n = self.n_nodes_1d;
        let mut total = 0.0;

        for e in 0..self.n_elems {
            let base = e * self.n_nodes_elem;
            let mut elem_sum = 0.0;
            for j in 0..n {
                let wj = self.weights_1d_f64[j];
                let mut row_sum = 0.0;
                for i in 0..n {
                    let idx = base + j * n + i;
                    row_sum += self.weights_1d_f64[i] * self.sqrt_g_f64[idx] * field[idx].to_f64();
                }
                elem_sum += wj * row_sum;
            }
            total += elem_sum;
        }
        total
    }

    /// Computes total surface area via numerical quadrature (exact theory: `4 * pi * R^2`).
    pub fn surface_area(&self) -> f64 {
        let ones = vec![T::ONE; self.total_nodes];
        self.integrate(&ones)
    }

    /// Projects 3D Cartesian wind vectors onto contravariant velocity components `(u^alpha, u^beta)`.
    /// All slices use `[elem][j][i]` order; physical length/time winds yield angular rates in radians/time.
    /// # Panics
    /// Panics unless each slice has length `total_nodes`.
    pub fn project_wind_3d(&self, wind_3d: &[[f64; 3]], u_contra: &mut [T], v_contra: &mut [T]) {
        assert_eq!(wind_3d.len(), self.total_nodes);
        assert_eq!(u_contra.len(), self.total_nodes);
        assert_eq!(v_contra.len(), self.total_nodes);

        for idx in 0..self.total_nodes {
            let w = wind_3d[idx];
            let ca = [
                self.contra_alpha[0][idx].to_f64(),
                self.contra_alpha[1][idx].to_f64(),
                self.contra_alpha[2][idx].to_f64(),
            ];
            let cb = [
                self.contra_beta[0][idx].to_f64(),
                self.contra_beta[1][idx].to_f64(),
                self.contra_beta[2][idx].to_f64(),
            ];
            u_contra[idx] = T::from_f64(dot3(w, ca));
            v_contra[idx] = T::from_f64(dot3(w, cb));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_surface_area_multi_element() {
        for (k, p, tol) in [(1, 16, 1e-13), (2, 10, 1e-13), (4, 8, 1e-13), (8, 5, 1e-12)] {
            let radius = 1.7320508075688772;
            let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig {
                radius,
                num_elem_per_face: k,
                poly_order: p,
                projection: ProjectionKind::Equiangular,
            });
            let area = mesh.surface_area();
            let exact = 4.0 * std::f64::consts::PI * radius * radius;
            let rel_err = (area - exact).abs() / exact;
            assert!(
                rel_err < tol,
                "K={k}, p={p}: rel_err={rel_err:.3e} >= {tol:.3e}"
            );
        }
    }

    #[test]
    fn test_equidistant_surface_area() {
        let radius = 2.0;
        let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius,
            num_elem_per_face: 4,
            poly_order: 8,
            projection: ProjectionKind::Equidistant,
        });
        let area = mesh.surface_area();
        let exact = 4.0 * std::f64::consts::PI * radius * radius;
        assert!((area - exact).abs() / exact < 1e-12);
    }

    #[test]
    fn test_all_multi_element_edges_coincide_and_invert_symmetrically() {
        for k in [1, 2, 3, 5] {
            let p = 4;
            let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig {
                radius: 1.0,
                num_elem_per_face: k,
                poly_order: p,
                projection: ProjectionKind::Equiangular,
            });
            let n = mesh.n_nodes_1d;

            let edge_node_idx = |elem: usize, side: EdgeSide, m: usize| -> usize {
                match side {
                    EdgeSide::West => mesh.node_idx(elem, 0, m),
                    EdgeSide::East => mesh.node_idx(elem, p, m),
                    EdgeSide::South => mesh.node_idx(elem, m, 0),
                    EdgeSide::North => mesh.node_idx(elem, m, p),
                }
            };

            for elem in 0..mesh.n_elems {
                for side in EdgeSide::ALL {
                    let nbr = mesh.neighbor(elem, side);
                    let nbr_elem = nbr.nbr_elem as usize;
                    let nbr_side = nbr.side();

                    // Symmetry check: neighbor's neighbor must point back to us with same reverse flag.
                    let back = mesh.neighbor(nbr_elem, nbr_side);
                    assert_eq!(back.nbr_elem as usize, elem);
                    assert_eq!(back.side(), side);
                    assert_eq!(back.is_reversed(), nbr.is_reversed());

                    for m in 0..n {
                        let m_nbr = if nbr.is_reversed() { n - 1 - m } else { m };
                        let i1 = edge_node_idx(elem, side, m);
                        let i2 = edge_node_idx(nbr_elem, nbr_side, m_nbr);
                        let d2 = (mesh.xyz[0][i1] - mesh.xyz[0][i2]).powi(2)
                            + (mesh.xyz[1][i1] - mesh.xyz[1][i2]).powi(2)
                            + (mesh.xyz[2][i1] - mesh.xyz[2][i2]).powi(2);
                        assert!(
                            d2.sqrt() < 1e-14,
                            "K={k}: elem {elem} {:?} node {m} vs nbr {nbr_elem} {:?} node {m_nbr}: dist={}",
                            side,
                            nbr_side,
                            d2.sqrt()
                        );
                    }
                }
            }
        }
    }
}
