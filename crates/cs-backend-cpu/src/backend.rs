use std::convert::Infallible;
use std::sync::Arc;

use rayon::prelude::*;

use cs_core::{dot3, CubedSphereMesh, EdgeSide, FloatScalar};
use cs_equations::{ScalarAdvection, WindField};
use cs_solver::{ComputeBackend, NumericalFlux};

/// Thread-local scratch memory per Rayon worker thread to avoid heap allocations in RHS calls.
struct ElementScratch<T: FloatScalar> {
    u_a: Vec<T>,
    u_b: Vec<T>,
    fa: Vec<T>,
    fb: Vec<T>,
    ga: Vec<T>,
    gb: Vec<T>,
    d_fb_db: Vec<T>,
    d_phi_db: Vec<T>,
    d_gb_db: Vec<T>,
}

impl<T: FloatScalar> ElementScratch<T> {
    fn new(n_nodes_elem: usize, n_nodes_1d: usize) -> Self {
        Self {
            u_a: vec![T::ZERO; n_nodes_elem],
            u_b: vec![T::ZERO; n_nodes_elem],
            fa: vec![T::ZERO; n_nodes_elem],
            fb: vec![T::ZERO; n_nodes_elem],
            ga: vec![T::ZERO; n_nodes_elem],
            gb: vec![T::ZERO; n_nodes_elem],
            d_fb_db: vec![T::ZERO; n_nodes_1d],
            d_phi_db: vec![T::ZERO; n_nodes_1d],
            d_gb_db: vec![T::ZERO; n_nodes_1d],
        }
    }
}

/// High-performance pure-Rust CPU backend using Rayon across spectral elements.
///
/// Key optimizations:
/// - Single parallel pass per RHS evaluation (Volume Split-Form + SAT penalty)
///   with zero shared writes (`rhs` partitioned disjointly per element).
/// - Precomputed steady-state contravariant wind when `!wind.is_time_dependent()`,
///   and allocation-free thread-local evaluation when time-dependent.
/// - Unit-stride loops over `xi_i` aligned for AVX2 / NEON auto-vectorization.
///
/// The interface coupling defaults to the upwind flux; use [`CpuBackend::with_flux`]
/// to select central or blended alpha-fluxes.
pub struct CpuBackend<T: FloatScalar> {
    mesh: Arc<CubedSphereMesh<T>>,
    wind: Arc<dyn WindField>,
    time_dependent: bool,
    steady_u_contra: Vec<T>,
    steady_v_contra: Vec<T>,
    flux: NumericalFlux,
    upwind_weight: T,
}

impl<T: FloatScalar> CpuBackend<T> {
    pub fn new(mesh: Arc<CubedSphereMesh<T>>, wind: Arc<dyn WindField>) -> Self {
        let time_dependent = wind.is_time_dependent();
        let (steady_u_contra, steady_v_contra) = if !time_dependent {
            let mut u = vec![T::ZERO; mesh.total_nodes];
            let mut v = vec![T::ZERO; mesh.total_nodes];
            let r = mesh.config.radius;
            let wind_3d: Vec<[f64; 3]> = (0..mesh.total_nodes)
                .map(|idx| {
                    let xyz = [
                        mesh.xyz[0][idx].to_f64(),
                        mesh.xyz[1][idx].to_f64(),
                        mesh.xyz[2][idx].to_f64(),
                    ];
                    wind.velocity_3d(0.0, mesh.lon_f64[idx], mesh.lat_f64[idx], xyz, r)
                })
                .collect();
            mesh.project_wind_3d(&wind_3d, &mut u, &mut v);
            (u, v)
        } else {
            (Vec::new(), Vec::new())
        };

        let flux = NumericalFlux::default();
        Self {
            mesh,
            wind,
            time_dependent,
            steady_u_contra,
            steady_v_contra,
            flux,
            upwind_weight: T::from_f64(flux.upwind_weight()),
        }
    }

    pub fn from_equation(mesh: Arc<CubedSphereMesh<T>>, equation: &ScalarAdvection) -> Self {
        Self::new(mesh, equation.wind_arc())
    }

    /// Selects the interface numerical flux used by the SAT penalty.
    pub fn with_flux(mut self, flux: NumericalFlux) -> Self {
        self.flux = flux;
        self.upwind_weight = T::from_f64(flux.upwind_weight());
        self
    }

    pub fn flux(&self) -> NumericalFlux {
        self.flux
    }
}

#[inline]
fn edge_local_idx(side: EdgeSide, n: usize, m: usize) -> usize {
    match side {
        EdgeSide::West => m * n,
        EdgeSide::East => m * n + (n - 1),
        EdgeSide::South => m,
        EdgeSide::North => (n - 1) * n + m,
    }
}

impl<T: FloatScalar> ComputeBackend<T> for CpuBackend<T> {
    type Buffer = Vec<T>;
    type Error = Infallible;

    fn name(&self) -> &'static str {
        "cpu_rayon"
    }

    fn mesh(&self) -> &CubedSphereMesh<T> {
        &self.mesh
    }

    fn allocate_zeros(&self, len: usize) -> Result<Self::Buffer, Self::Error> {
        Ok(vec![T::ZERO; len])
    }

    fn upload(&self, data: &[T]) -> Result<Self::Buffer, Self::Error> {
        Ok(data.to_vec())
    }

    fn download(&self, buf: &Self::Buffer) -> Result<Vec<T>, Self::Error> {
        Ok(buf.clone())
    }

    fn compute_advection_rhs(
        &self,
        state: &Self::Buffer,
        rhs: &mut Self::Buffer,
        t: f64,
    ) -> Result<(), Self::Error> {
        let mesh = &self.mesh;
        let n = mesh.n_nodes_1d;
        let n_elem_nodes = mesh.n_nodes_elem;
        assert_eq!(state.len(), mesh.total_nodes);
        assert_eq!(rhs.len(), mesh.total_nodes);

        let d = &mesh.reference.diff_matrix;
        let map_factor = mesh.map_factor;
        let inv_w0 = T::ONE / mesh.weights_1d[0];
        let inv_wn = T::ONE / mesh.weights_1d[n - 1];
        let radius = mesh.config.radius;
        let upwind_weight = self.upwind_weight;

        rhs.par_chunks_mut(n_elem_nodes)
            .enumerate()
            .for_each_init(
                || ElementScratch::new(n_elem_nodes, n),
                |scratch, (elem, rhs_elem)| {
                    let base = elem * n_elem_nodes;
                    let phi_elem = &state[base..base + n_elem_nodes];
                    let j_elem = &mesh.sqrt_g[base..base + n_elem_nodes];

                    // Obtain contravariant velocity slice `(u_a, u_b)` for this element
                    let (u_a, u_b): (&[T], &[T]) = if !self.time_dependent {
                        (
                            &self.steady_u_contra[base..base + n_elem_nodes],
                            &self.steady_v_contra[base..base + n_elem_nodes],
                        )
                    } else {
                        for k in 0..n_elem_nodes {
                            let idx = base + k;
                            let xyz = [
                                mesh.xyz[0][idx].to_f64(),
                                mesh.xyz[1][idx].to_f64(),
                                mesh.xyz[2][idx].to_f64(),
                            ];
                            let w3 = self.wind.velocity_3d(
                                t,
                                mesh.lon_f64[idx],
                                mesh.lat_f64[idx],
                                xyz,
                                radius,
                            );
                            let ca = [
                                mesh.contra_alpha[0][idx].to_f64(),
                                mesh.contra_alpha[1][idx].to_f64(),
                                mesh.contra_alpha[2][idx].to_f64(),
                            ];
                            let cb = [
                                mesh.contra_beta[0][idx].to_f64(),
                                mesh.contra_beta[1][idx].to_f64(),
                                mesh.contra_beta[2][idx].to_f64(),
                            ];
                            scratch.u_a[k] = T::from_f64(dot3(w3, ca));
                            scratch.u_b[k] = T::from_f64(dot3(w3, cb));
                        }
                        (&scratch.u_a, &scratch.u_b)
                    };

                    // Precompute flux & metric products inside element
                    for k in 0..n_elem_nodes {
                        let jk = j_elem[k];
                        let ga = jk * u_a[k];
                        let gb = jk * u_b[k];
                        let phik = phi_elem[k];
                        scratch.ga[k] = ga;
                        scratch.gb[k] = gb;
                        scratch.fa[k] = ga * phik;
                        scratch.fb[k] = gb * phik;
                    }

                    // 1. Volume Split-Form Divergence (conservative)
                    for j in 0..n {
                        // Compute d/d(beta) for row `j` across all `i in 0..n` using unit-stride AXPY
                        scratch.d_fb_db.fill(T::ZERO);
                        scratch.d_phi_db.fill(T::ZERO);
                        scratch.d_gb_db.fill(T::ZERO);

                        for m in 0..n {
                            let djm = d[j * n + m];
                            let m_row = m * n;
                            for i in 0..n {
                                scratch.d_fb_db[i] += djm * scratch.fb[m_row + i];
                                scratch.d_phi_db[i] += djm * phi_elem[m_row + i];
                                scratch.d_gb_db[i] += djm * scratch.gb[m_row + i];
                            }
                        }

                        let j_row = j * n;
                        for i in 0..n {
                            let d_row = &d[i * n..(i + 1) * n];
                            let mut d_fa_da = T::ZERO;
                            let mut d_phi_da = T::ZERO;
                            let mut d_ga_da = T::ZERO;
                            for m in 0..n {
                                let dim = d_row[m];
                                d_fa_da += dim * scratch.fa[j_row + m];
                                d_phi_da += dim * phi_elem[j_row + m];
                                d_ga_da += dim * scratch.ga[j_row + m];
                            }

                            let k = j_row + i;
                            let inv_j = T::ONE / j_elem[k];
                            let term1 = map_factor * (d_fa_da + scratch.d_fb_db[i]) * inv_j;
                            let term2 = map_factor
                                * (u_a[k] * d_phi_da + u_b[k] * scratch.d_phi_db[i]);
                            let term3 = phi_elem[k]
                                * map_factor
                                * (d_ga_da + scratch.d_gb_db[i])
                                * inv_j;
                            rhs_elem[k] = -T::HALF * (term1 + term2 + term3);
                        }
                    }

                    // 2. Conservative alpha-flux SAT Boundary Penalty (no `sqrt_g` in denominator)
                    for side in EdgeSide::ALL {
                        let nbr = mesh.neighbor(elem, side);
                        let nbr_base = (nbr.nbr_elem as usize) * n_elem_nodes;
                        let nbr_side = nbr.side();
                        let rev = nbr.is_reversed();
                        let inv_w_edge = match side {
                            EdgeSide::West | EdgeSide::South => inv_w0,
                            EdgeSide::East | EdgeSide::North => inv_wn,
                        };

                        for m in 0..n {
                            let local_idx = edge_local_idx(side, n, m);
                            let m_nbr = if rev { n - 1 - m } else { m };
                            let nbr_local_idx = edge_local_idx(nbr_side, n, m_nbr);

                            let vn = match side {
                                EdgeSide::West => -u_a[local_idx],
                                EdgeSide::East => u_a[local_idx],
                                EdgeSide::South => -u_b[local_idx],
                                EdgeSide::North => u_b[local_idx],
                            };

                            // alpha = 1 (upwind): <= 0 on inflow, 0 on outflow; alpha = 0: central
                            let switch = T::HALF * (vn - upwind_weight * vn.abs());
                            let q_in = phi_elem[local_idx];
                            let q_out = state[nbr_base + nbr_local_idx];
                            rhs_elem[local_idx] += -switch * (q_out - q_in) * inv_w_edge;
                        }
                    }
                },
            );

        Ok(())
    }

    fn lsrk_stage_update(
        &self,
        u: &mut Self::Buffer,
        du: &mut Self::Buffer,
        rhs: &Self::Buffer,
        rk_a: T,
        rk_b: T,
        dt: T,
    ) -> Result<(), Self::Error> {
        u.par_iter_mut()
            .zip(du.par_iter_mut())
            .zip(rhs.par_iter())
            .for_each(|((u_i, du_i), &r_i)| {
                let new_du = rk_a * *du_i + dt * r_i;
                *du_i = new_du;
                *u_i += rk_b * new_du;
            });
        Ok(())
    }
}
