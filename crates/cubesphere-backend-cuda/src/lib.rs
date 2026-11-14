#![allow(clippy::too_many_arguments)]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc, Mutex};

use cuda_core::{
    CudaContext, CudaStream, DeviceBuffer, DeviceCopy, DriverError, LaunchConfig1D,
    LaunchContractError,
};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::{EmbeddedModuleError, cuda_module};

use cubesphere_core::{CubedSphereMesh, EdgeNeighbor, FloatScalar, dot3};
use cubesphere_equations::{ScalarAdvection, WindDescriptor, WindField};
use cubesphere_solver::{ComputeBackend, NumericalFlux};

const BLOCK_THREADS: u32 = 256;

/// Unified error type returned by [`CudaBackend`].
#[derive(Debug, Clone)]
pub enum CudaError {
    Driver(String),
    EmbeddedModule(String),
    LaunchContract(String),
    InvalidConfiguration(String),
}

impl Display for CudaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(msg) => write!(f, "CUDA driver error: {msg}"),
            Self::EmbeddedModule(msg) => write!(f, "CUDA embedded module error: {msg}"),
            Self::LaunchContract(msg) => write!(f, "CUDA launch contract error: {msg}"),
            Self::InvalidConfiguration(msg) => write!(f, "CUDA backend configuration error: {msg}"),
        }
    }
}

impl Error for CudaError {}

impl From<DriverError> for CudaError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value.to_string())
    }
}

impl From<EmbeddedModuleError> for CudaError {
    fn from(value: EmbeddedModuleError) -> Self {
        Self::EmbeddedModule(value.to_string())
    }
}

impl From<LaunchContractError> for CudaError {
    fn from(value: LaunchContractError) -> Self {
        Self::LaunchContract(value.to_string())
    }
}

/// GPU-side `#[repr(C)]` representation matching [`cubesphere_core::EdgeNeighbor`].
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EdgeNeighborGpu {
    pub nbr_elem: u32,
    pub nbr_side: u8,
    pub reverse: u8,
    pub _pad: [u8; 2],
}

impl From<EdgeNeighbor> for EdgeNeighborGpu {
    #[inline]
    fn from(n: EdgeNeighbor) -> Self {
        Self {
            nbr_elem: n.nbr_elem,
            nbr_side: n.nbr_side,
            reverse: n.reverse,
            _pad: [0; 2],
        }
    }
}

// SAFETY: `#[repr(C)]` struct containing only primitive integers where all-zero is valid.
unsafe impl DeviceCopy for EdgeNeighborGpu {}

/// Encodes one edge neighbor record into a primitive `u64` for coalesced `ld.global.u64` access.
#[inline]
pub fn pack_neighbor(n: EdgeNeighbor) -> u64 {
    (n.nbr_elem as u64) | ((n.nbr_side as u64) << 32) | ((n.reverse as u64) << 40)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn unpack_neighbor(packed: u64) -> (usize, usize, bool) {
        let nbr_elem = (packed & 0xFFFF_FFFF) as usize;
        let nbr_side = ((packed >> 32) & 0xFF) as usize;
        let reverse = ((packed >> 40) & 0xFF) != 0;
        (nbr_elem, nbr_side, reverse)
    }

    #[inline(always)]
    fn edge_local_offset(side: usize, n_1d: usize, m: usize) -> usize {
        match side {
            0 => m * n_1d,              // West: i = 0
            1 => m * n_1d + (n_1d - 1), // East: i = n_1d - 1
            2 => m,                     // South: j = 0
            _ => (n_1d - 1) * n_1d + m, // North: j = n_1d - 1
        }
    }

    #[inline(always)]
    fn sat_edge_flux_f64(
        q_in: f64,
        vn: f64,
        upwind_weight: f64,
        inv_w_edge: f64,
        m: usize,
        n_1d: usize,
        n_elem: usize,
        packed_nbr: u64,
        state: &[f64],
    ) -> f64 {
        // alpha-flux `switch = 0.5 * (vn - alpha * |vn|)` without requiring std `abs()`
        // (alpha = 1: upwind, exactly `vn` on inflow and 0 on outflow; alpha = 0: central).
        let abs_vn = if vn < 0.0 { -vn } else { vn };
        let switch = 0.5_f64 * (vn - upwind_weight * abs_vn);
        if switch == 0.0 {
            return 0.0;
        }
        let (nbr_elem, nbr_side, rev) = unpack_neighbor(packed_nbr);
        let m_nbr = if rev { n_1d - 1 - m } else { m };
        let nbr_local = edge_local_offset(nbr_side, n_1d, m_nbr);
        let q_out = state[nbr_elem * n_elem + nbr_local];
        -switch * (q_out - q_in) * inv_w_edge
    }

    #[inline(always)]
    fn sat_edge_flux_f32(
        q_in: f32,
        vn: f32,
        upwind_weight: f32,
        inv_w_edge: f32,
        m: usize,
        n_1d: usize,
        n_elem: usize,
        packed_nbr: u64,
        state: &[f32],
    ) -> f32 {
        let abs_vn = if vn < 0.0 { -vn } else { vn };
        let switch = 0.5_f32 * (vn - upwind_weight * abs_vn);
        if switch == 0.0 {
            return 0.0;
        }
        let (nbr_elem, nbr_side, rev) = unpack_neighbor(packed_nbr);
        let m_nbr = if rev { n_1d - 1 - m } else { m };
        let nbr_local = edge_local_offset(nbr_side, n_1d, m_nbr);
        let q_out = state[nbr_elem * n_elem + nbr_local];
        -switch * (q_out - q_in) * inv_w_edge
    }

    /// Fused Split-Form Volume Divergence + alpha-flux SAT Boundary Penalty (`f64`).
    ///
    /// Uses 1 thread per DOF (`tid in 0..total_nodes`), eliminating corner synchronization hazards
    /// and atomic contention across shared element boundaries and corners.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn split_form_and_sat_f64(
        state: &[f64],
        u_a: &[f64],
        u_b: &[f64],
        sqrt_g: &[f64],
        diff_mat: &[f64],
        neighbors_packed: &[u64],
        mut rhs: DisjointSlice<f64>,
        n_nodes_1d: u32,
        map_factor: f64,
        inv_w0: f64,
        inv_wn: f64,
        upwind_weight: f64,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let n_1d = n_nodes_1d as usize;
        if n_1d < 2 {
            return;
        }

        if let Some(rhs_slot) = rhs.get_mut(idx) {
            let n_elem = n_1d * n_1d;
            let elem = tid / n_elem;
            let rem = tid % n_elem;
            let j = rem / n_1d;
            let i = rem % n_1d;
            let base = elem * n_elem;
            let row_base = base + j * n_1d;

            let mut d_fa_da = 0.0_f64;
            let mut d_phi_da = 0.0_f64;
            let mut d_ga_da = 0.0_f64;
            let d_row_i = i * n_1d;
            for m in 0..n_1d {
                let d_im = diff_mat[d_row_i + m];
                let k = row_base + m;
                let phi_m = state[k];
                let ga_m = sqrt_g[k] * u_a[k];
                d_fa_da += d_im * (ga_m * phi_m);
                d_phi_da += d_im * phi_m;
                d_ga_da += d_im * ga_m;
            }

            let mut d_fb_db = 0.0_f64;
            let mut d_phi_db = 0.0_f64;
            let mut d_gb_db = 0.0_f64;
            let d_row_j = j * n_1d;
            for m in 0..n_1d {
                let d_jm = diff_mat[d_row_j + m];
                let k = base + m * n_1d + i;
                let phi_m = state[k];
                let gb_m = sqrt_g[k] * u_b[k];
                d_fb_db += d_jm * (gb_m * phi_m);
                d_phi_db += d_jm * phi_m;
                d_gb_db += d_jm * gb_m;
            }

            let phi_k = state[tid];
            let ua_k = u_a[tid];
            let ub_k = u_b[tid];
            let inv_j = 1.0_f64 / sqrt_g[tid];

            let term1 = map_factor * (d_fa_da + d_fb_db) * inv_j;
            let term2 = map_factor * (ua_k * d_phi_da + ub_k * d_phi_db);
            let term3 = phi_k * map_factor * (d_ga_da + d_gb_db) * inv_j;
            let mut total_rhs = -0.5_f64 * (term1 + term2 + term3);

            let nbr_base = elem * 4;
            if i == 0 {
                total_rhs += sat_edge_flux_f64(
                    phi_k,
                    -ua_k,
                    upwind_weight,
                    inv_w0,
                    j,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base],
                    state,
                );
            }
            if i + 1 == n_1d {
                total_rhs += sat_edge_flux_f64(
                    phi_k,
                    ua_k,
                    upwind_weight,
                    inv_wn,
                    j,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base + 1],
                    state,
                );
            }
            if j == 0 {
                total_rhs += sat_edge_flux_f64(
                    phi_k,
                    -ub_k,
                    upwind_weight,
                    inv_w0,
                    i,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base + 2],
                    state,
                );
            }
            if j + 1 == n_1d {
                total_rhs += sat_edge_flux_f64(
                    phi_k,
                    ub_k,
                    upwind_weight,
                    inv_wn,
                    i,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base + 3],
                    state,
                );
            }

            *rhs_slot = total_rhs;
        }
    }

    /// Fused Split-Form Volume Divergence + alpha-flux SAT Boundary Penalty (`f32`).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn split_form_and_sat_f32(
        state: &[f32],
        u_a: &[f32],
        u_b: &[f32],
        sqrt_g: &[f32],
        diff_mat: &[f32],
        neighbors_packed: &[u64],
        mut rhs: DisjointSlice<f32>,
        n_nodes_1d: u32,
        map_factor: f32,
        inv_w0: f32,
        inv_wn: f32,
        upwind_weight: f32,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        let n_1d = n_nodes_1d as usize;
        if n_1d < 2 {
            return;
        }

        if let Some(rhs_slot) = rhs.get_mut(idx) {
            let n_elem = n_1d * n_1d;
            let elem = tid / n_elem;
            let rem = tid % n_elem;
            let j = rem / n_1d;
            let i = rem % n_1d;
            let base = elem * n_elem;
            let row_base = base + j * n_1d;

            let mut d_fa_da = 0.0_f32;
            let mut d_phi_da = 0.0_f32;
            let mut d_ga_da = 0.0_f32;
            let d_row_i = i * n_1d;
            for m in 0..n_1d {
                let d_im = diff_mat[d_row_i + m];
                let k = row_base + m;
                let phi_m = state[k];
                let ga_m = sqrt_g[k] * u_a[k];
                d_fa_da += d_im * (ga_m * phi_m);
                d_phi_da += d_im * phi_m;
                d_ga_da += d_im * ga_m;
            }

            let mut d_fb_db = 0.0_f32;
            let mut d_phi_db = 0.0_f32;
            let mut d_gb_db = 0.0_f32;
            let d_row_j = j * n_1d;
            for m in 0..n_1d {
                let d_jm = diff_mat[d_row_j + m];
                let k = base + m * n_1d + i;
                let phi_m = state[k];
                let gb_m = sqrt_g[k] * u_b[k];
                d_fb_db += d_jm * (gb_m * phi_m);
                d_phi_db += d_jm * phi_m;
                d_gb_db += d_jm * gb_m;
            }

            let phi_k = state[tid];
            let ua_k = u_a[tid];
            let ub_k = u_b[tid];
            let inv_j = 1.0_f32 / sqrt_g[tid];

            let term1 = map_factor * (d_fa_da + d_fb_db) * inv_j;
            let term2 = map_factor * (ua_k * d_phi_da + ub_k * d_phi_db);
            let term3 = phi_k * map_factor * (d_ga_da + d_gb_db) * inv_j;
            let mut total_rhs = -0.5_f32 * (term1 + term2 + term3);

            let nbr_base = elem * 4;
            if i == 0 {
                total_rhs += sat_edge_flux_f32(
                    phi_k,
                    -ua_k,
                    upwind_weight,
                    inv_w0,
                    j,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base],
                    state,
                );
            }
            if i + 1 == n_1d {
                total_rhs += sat_edge_flux_f32(
                    phi_k,
                    ua_k,
                    upwind_weight,
                    inv_wn,
                    j,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base + 1],
                    state,
                );
            }
            if j == 0 {
                total_rhs += sat_edge_flux_f32(
                    phi_k,
                    -ub_k,
                    upwind_weight,
                    inv_w0,
                    i,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base + 2],
                    state,
                );
            }
            if j + 1 == n_1d {
                total_rhs += sat_edge_flux_f32(
                    phi_k,
                    ub_k,
                    upwind_weight,
                    inv_wn,
                    i,
                    n_1d,
                    n_elem,
                    neighbors_packed[nbr_base + 3],
                    state,
                );
            }

            *rhs_slot = total_rhs;
        }
    }

    /// Transcendental-free analytical evaluation of one contravariant wind component (`u_a` or `u_b`)
    /// for Nair & Lauritzen (2010) Case 4 via trigonometric addition identities (`f64`).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn update_nair_lauritzen_comp_f64(
        sin_2lon: &[f64],
        cos_2lon: &[f64],
        u_bg: &[f64],
        cu: &[f64],
        cv: &[f64],
        mut u_out: DisjointSlice<f64>,
        c2wt: f64,
        s2wt: f64,
        kt: f64,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        if let Some(slot) = u_out.get_mut(idx) {
            let s2l = sin_2lon[tid];
            let c2l = cos_2lon[tid];
            let sin2_lon_prime = 0.5_f64 * (1.0_f64 - c2l * c2wt - s2l * s2wt);
            let sin_2lon_prime = s2l * c2wt - c2l * s2wt;
            *slot = u_bg[tid] + kt * (sin2_lon_prime * cu[tid] + sin_2lon_prime * cv[tid]);
        }
    }

    /// Transcendental-free analytical evaluation of one contravariant wind component for Nair & Lauritzen Case 4 (`f32`).
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn update_nair_lauritzen_comp_f32(
        sin_2lon: &[f32],
        cos_2lon: &[f32],
        u_bg: &[f32],
        cu: &[f32],
        cv: &[f32],
        mut u_out: DisjointSlice<f32>,
        c2wt: f32,
        s2wt: f32,
        kt: f32,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        if let Some(slot) = u_out.get_mut(idx) {
            let s2l = sin_2lon[tid];
            let c2l = cos_2lon[tid];
            let sin2_lon_prime = 0.5_f32 * (1.0_f32 - c2l * c2wt - s2l * s2wt);
            let sin_2lon_prime = s2l * c2wt - c2l * s2wt;
            *slot = u_bg[tid] + kt * (sin2_lon_prime * cu[tid] + sin_2lon_prime * cv[tid]);
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn lsrk_update_du_f64(
        mut du: DisjointSlice<f64>,
        rhs: &[f64],
        rk_a: f64,
        dt: f64,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        if let Some(du_slot) = du.get_mut(idx) {
            *du_slot = rk_a * *du_slot + dt * rhs[tid];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn lsrk_update_u_f64(mut u: DisjointSlice<f64>, du: &[f64], rk_b: f64) {
        let idx = thread::index_1d();
        let tid = idx.get();
        if let Some(u_slot) = u.get_mut(idx) {
            *u_slot += rk_b * du[tid];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn lsrk_update_du_f32(
        mut du: DisjointSlice<f32>,
        rhs: &[f32],
        rk_a: f32,
        dt: f32,
    ) {
        let idx = thread::index_1d();
        let tid = idx.get();
        if let Some(du_slot) = du.get_mut(idx) {
            *du_slot = rk_a * *du_slot + dt * rhs[tid];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn lsrk_update_u_f32(mut u: DisjointSlice<f32>, du: &[f32], rk_b: f32) {
        let idx = thread::index_1d();
        let tid = idx.get();
        if let Some(u_slot) = u.get_mut(idx) {
            *u_slot += rk_b * du[tid];
        }
    }
}

type RhsDispatchFn<T> = Box<
    dyn Fn(
            &CudaStream,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &DeviceBuffer<u64>,
            &mut DeviceBuffer<T>,
            u32,
            T,
            T,
            T,
            T,
        ) -> Result<(), CudaError>
        + Send
        + Sync,
>;

type WindCompDispatchFn<T> = Box<
    dyn Fn(
            &CudaStream,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &DeviceBuffer<T>,
            &mut DeviceBuffer<T>,
            T,
            T,
            T,
        ) -> Result<(), CudaError>
        + Send
        + Sync,
>;

type LsrkDispatchFn<T> = Box<
    dyn Fn(&CudaStream, &mut DeviceBuffer<T>, &mut DeviceBuffer<T>, &DeviceBuffer<T>, T, T, T) -> Result<(), CudaError>
        + Send
        + Sync,
>;

/// Sealed scalar specialization trait dispatching generic [`CudaBackend<T>`]
/// operations onto concrete non-generic `#[kernel]` entry points (`f64` / `f32`).
pub trait CudaScalar: FloatScalar + DeviceCopy {
    fn prepare_kernels(
        module: &kernels::LoadedModule,
        grid_blocks: u32,
    ) -> Result<(RhsDispatchFn<Self>, WindCompDispatchFn<Self>, LsrkDispatchFn<Self>), CudaError>;
}

impl CudaScalar for f64 {
    fn prepare_kernels(
        module: &kernels::LoadedModule,
        grid_blocks: u32,
    ) -> Result<(RhsDispatchFn<Self>, WindCompDispatchFn<Self>, LsrkDispatchFn<Self>), CudaError> {
        let cfg = LaunchConfig1D::new(grid_blocks, BLOCK_THREADS, 0);

        let mod_rhs = module.clone();
        let prep_rhs = mod_rhs.prepare_split_form_and_sat_f64(cfg)?;
        let rhs_fn: RhsDispatchFn<f64> = Box::new(
            move |stream, state, u_a, u_b, sqrt_g, diff_mat, nbrs, rhs, n1d, mf, iw0, iwn, uw| {
                mod_rhs
                    .split_form_and_sat_f64(
                        stream, &prep_rhs, state, u_a, u_b, sqrt_g, diff_mat, nbrs, rhs, n1d, mf,
                        iw0, iwn, uw,
                    )
                    .map_err(CudaError::from)
            },
        );

        let mod_wind = module.clone();
        let prep_wind = mod_wind.prepare_update_nair_lauritzen_comp_f64(cfg)?;
        let wind_fn: WindCompDispatchFn<f64> = Box::new(
            move |stream, s2l, c2l, u_bg, cu, cv, u_out, c2wt, s2wt, kt| {
                mod_wind
                    .update_nair_lauritzen_comp_f64(
                        stream, &prep_wind, s2l, c2l, u_bg, cu, cv, u_out, c2wt, s2wt, kt,
                    )
                    .map_err(CudaError::from)
            },
        );

        let mod_lsrk = module.clone();
        let prep_du = mod_lsrk.prepare_lsrk_update_du_f64(cfg)?;
        let prep_u = mod_lsrk.prepare_lsrk_update_u_f64(cfg)?;
        let lsrk_fn: LsrkDispatchFn<f64> = Box::new(move |stream, u, du, rhs, rk_a, rk_b, dt| {
            mod_lsrk.lsrk_update_du_f64(stream, &prep_du, du, rhs, rk_a, dt)?;
            mod_lsrk
                .lsrk_update_u_f64(stream, &prep_u, u, du, rk_b)
                .map_err(CudaError::from)
        });

        Ok((rhs_fn, wind_fn, lsrk_fn))
    }
}

impl CudaScalar for f32 {
    fn prepare_kernels(
        module: &kernels::LoadedModule,
        grid_blocks: u32,
    ) -> Result<(RhsDispatchFn<Self>, WindCompDispatchFn<Self>, LsrkDispatchFn<Self>), CudaError> {
        let cfg = LaunchConfig1D::new(grid_blocks, BLOCK_THREADS, 0);

        let mod_rhs = module.clone();
        let prep_rhs = mod_rhs.prepare_split_form_and_sat_f32(cfg)?;
        let rhs_fn: RhsDispatchFn<f32> = Box::new(
            move |stream, state, u_a, u_b, sqrt_g, diff_mat, nbrs, rhs, n1d, mf, iw0, iwn, uw| {
                mod_rhs
                    .split_form_and_sat_f32(
                        stream, &prep_rhs, state, u_a, u_b, sqrt_g, diff_mat, nbrs, rhs, n1d, mf,
                        iw0, iwn, uw,
                    )
                    .map_err(CudaError::from)
            },
        );

        let mod_wind = module.clone();
        let prep_wind = mod_wind.prepare_update_nair_lauritzen_comp_f32(cfg)?;
        let wind_fn: WindCompDispatchFn<f32> = Box::new(
            move |stream, s2l, c2l, u_bg, cu, cv, u_out, c2wt, s2wt, kt| {
                mod_wind
                    .update_nair_lauritzen_comp_f32(
                        stream, &prep_wind, s2l, c2l, u_bg, cu, cv, u_out, c2wt, s2wt, kt,
                    )
                    .map_err(CudaError::from)
            },
        );

        let mod_lsrk = module.clone();
        let prep_du = mod_lsrk.prepare_lsrk_update_du_f32(cfg)?;
        let prep_u = mod_lsrk.prepare_lsrk_update_u_f32(cfg)?;
        let lsrk_fn: LsrkDispatchFn<f32> = Box::new(move |stream, u, du, rhs, rk_a, rk_b, dt| {
            mod_lsrk.lsrk_update_du_f32(stream, &prep_du, du, rhs, rk_a, dt)?;
            mod_lsrk
                .lsrk_update_u_f32(stream, &prep_u, u, du, rk_b)
                .map_err(CudaError::from)
        });

        Ok((rhs_fn, wind_fn, lsrk_fn))
    }
}

struct DynamicWindBuffers<T> {
    u_a: DeviceBuffer<T>,
    u_b: DeviceBuffer<T>,
    last_t: Option<f64>,
    host_scratch_u: Vec<T>,
    host_scratch_v: Vec<T>,
    host_wind_3d: Vec<[f64; 3]>,
}

enum DeviceWindState<T> {
    Steady {
        u_a: DeviceBuffer<T>,
        u_b: DeviceBuffer<T>,
    },
    NairLauritzenDevice {
        period: f64,
        amplitude_factor: f64,
        sin_2lon: DeviceBuffer<T>,
        cos_2lon: DeviceBuffer<T>,
        u_bg_a: DeviceBuffer<T>,
        u_bg_b: DeviceBuffer<T>,
        cu_a: DeviceBuffer<T>,
        cv_a: DeviceBuffer<T>,
        cu_b: DeviceBuffer<T>,
        cv_b: DeviceBuffer<T>,
        dyn_buf: Mutex<DynamicWindBuffers<T>>,
    },
    HostEvaluatedDynamic {
        wind: Arc<dyn WindField>,
        dyn_buf: Mutex<DynamicWindBuffers<T>>,
    },
}

/// GPU compute backend powered by NVIDIA `cuda-oxide` (`NVIDIA/cuda-rust`).
///
/// Pre-allocates and retains all geometric metrics, LGL differentiation matrices,
/// neighbor topology tables, and wind coefficients in device memory so every time
/// step executes on-GPU with zero host-device copies during integration.
pub struct CudaBackend<T: CudaScalar> {
    mesh: Arc<CubedSphereMesh<T>>,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    sqrt_g_dev: DeviceBuffer<T>,
    diff_mat_dev: DeviceBuffer<T>,
    neighbors_dev: DeviceBuffer<u64>,
    wind_state: DeviceWindState<T>,
    rhs_dispatch: RhsDispatchFn<T>,
    wind_comp_dispatch: WindCompDispatchFn<T>,
    lsrk_dispatch: LsrkDispatchFn<T>,
    inv_w0: T,
    inv_wn: T,
    flux: NumericalFlux,
    upwind_weight: T,
}

impl<T: CudaScalar> CudaBackend<T> {
    /// Creates a new `CudaBackend` on CUDA device `device_ordinal` (typically `0`).
    pub fn new(
        device_ordinal: usize,
        mesh: Arc<CubedSphereMesh<T>>,
        wind: Arc<dyn WindField>,
    ) -> Result<Self, CudaError> {
        if mesh.total_nodes == 0 {
            return Err(CudaError::InvalidConfiguration(
                "mesh must contain at least one node".to_string(),
            ));
        }

        let context = CudaContext::new(device_ordinal)?;
        let stream = context.default_stream();

        // SAFETY: this package owns the embedded artifact bundle compiled for `mod kernels`.
        let module = unsafe { kernels::load(&context)? };
        let grid_blocks = (mesh.total_nodes as u32).div_ceil(BLOCK_THREADS);
        let (rhs_dispatch, wind_comp_dispatch, lsrk_dispatch) =
            T::prepare_kernels(&module, grid_blocks)?;

        let sqrt_g_dev = DeviceBuffer::from_host(&stream, &mesh.sqrt_g)?;
        let diff_mat_dev = DeviceBuffer::from_host(&stream, &mesh.reference.diff_matrix)?;
        let packed_neighbors: Vec<u64> =
            mesh.neighbors.iter().copied().map(pack_neighbor).collect();
        let neighbors_dev = DeviceBuffer::from_host(&stream, &packed_neighbors)?;

        let wind_state = Self::build_wind_state(&stream, &mesh, &wind)?;
        let n = mesh.n_nodes_1d;
        let inv_w0 = T::ONE / mesh.weights_1d[0];
        let inv_wn = T::ONE / mesh.weights_1d[n - 1];

        stream.synchronize()?;

        Ok(Self {
            mesh,
            context,
            stream,
            sqrt_g_dev,
            diff_mat_dev,
            neighbors_dev,
            wind_state,
            rhs_dispatch,
            wind_comp_dispatch,
            lsrk_dispatch,
            inv_w0,
            inv_wn,
            flux: NumericalFlux::default(),
            upwind_weight: T::from_f64(NumericalFlux::default().upwind_weight()),
        })
    }

    /// Selects the interface numerical flux used by the SAT penalty (default: upwind).
    pub fn with_flux(mut self, flux: NumericalFlux) -> Self {
        self.flux = flux;
        self.upwind_weight = T::from_f64(flux.upwind_weight());
        self
    }

    pub fn flux(&self) -> NumericalFlux {
        self.flux
    }

    pub fn from_equation(
        device_ordinal: usize,
        mesh: Arc<CubedSphereMesh<T>>,
        equation: &ScalarAdvection,
    ) -> Result<Self, CudaError> {
        Self::new(device_ordinal, mesh, equation.wind_arc())
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Blocks the calling CPU thread until all previously enqueued GPU kernels complete.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.stream.synchronize().map_err(CudaError::from)
    }

    fn build_wind_state(
        stream: &Arc<CudaStream>,
        mesh: &CubedSphereMesh<T>,
        wind: &Arc<dyn WindField>,
    ) -> Result<DeviceWindState<T>, CudaError> {
        let total = mesh.total_nodes;
        let r = mesh.config.radius;

        if !wind.is_time_dependent() {
            let mut u = vec![T::ZERO; total];
            let mut v = vec![T::ZERO; total];
            let wind_3d: Vec<[f64; 3]> = (0..total)
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
            let u_a = DeviceBuffer::from_host(stream, &u)?;
            let u_b = DeviceBuffer::from_host(stream, &v)?;
            return Ok(DeviceWindState::Steady { u_a, u_b });
        }

        if let Some(WindDescriptor::NairLauritzen {
            period,
            amplitude_factor,
        }) = wind.descriptor()
        {
            let bg = 2.0 * PI * r / period;
            let mut sin_2lon_h = Vec::with_capacity(total);
            let mut cos_2lon_h = Vec::with_capacity(total);
            let mut u_bg_a_h = Vec::with_capacity(total);
            let mut u_bg_b_h = Vec::with_capacity(total);
            let mut cu_a_h = Vec::with_capacity(total);
            let mut cv_a_h = Vec::with_capacity(total);
            let mut cu_b_h = Vec::with_capacity(total);
            let mut cv_b_h = Vec::with_capacity(total);

            for idx in 0..total {
                let lon = mesh.lon_f64[idx];
                let lat = mesh.lat_f64[idx];
                let (sl, cl) = lon.sin_cos();
                let (st, ct) = lat.sin_cos();
                let e_lon = [-sl, cl, 0.0];
                let e_lat = [-st * cl, -st * sl, ct];

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

                let e_lon_ca = dot3(e_lon, ca);
                let e_lat_ca = dot3(e_lat, ca);
                let e_lon_cb = dot3(e_lon, cb);
                let e_lat_cb = dot3(e_lat, cb);

                let (s2l, c2l) = (2.0 * lon).sin_cos();
                let s2t = (2.0 * lat).sin();

                sin_2lon_h.push(T::from_f64(s2l));
                cos_2lon_h.push(T::from_f64(c2l));
                u_bg_a_h.push(T::from_f64(bg * ct * e_lon_ca));
                u_bg_b_h.push(T::from_f64(bg * ct * e_lon_cb));
                cu_a_h.push(T::from_f64(s2t * e_lon_ca));
                cv_a_h.push(T::from_f64(ct * e_lat_ca));
                cu_b_h.push(T::from_f64(s2t * e_lon_cb));
                cv_b_h.push(T::from_f64(ct * e_lat_cb));
            }

            let dyn_buf = DynamicWindBuffers {
                u_a: DeviceBuffer::<T>::zeroed(stream, total)?,
                u_b: DeviceBuffer::<T>::zeroed(stream, total)?,
                last_t: None,
                host_scratch_u: Vec::new(),
                host_scratch_v: Vec::new(),
                host_wind_3d: Vec::new(),
            };

            return Ok(DeviceWindState::NairLauritzenDevice {
                period,
                amplitude_factor,
                sin_2lon: DeviceBuffer::from_host(stream, &sin_2lon_h)?,
                cos_2lon: DeviceBuffer::from_host(stream, &cos_2lon_h)?,
                u_bg_a: DeviceBuffer::from_host(stream, &u_bg_a_h)?,
                u_bg_b: DeviceBuffer::from_host(stream, &u_bg_b_h)?,
                cu_a: DeviceBuffer::from_host(stream, &cu_a_h)?,
                cv_a: DeviceBuffer::from_host(stream, &cv_a_h)?,
                cu_b: DeviceBuffer::from_host(stream, &cu_b_h)?,
                cv_b: DeviceBuffer::from_host(stream, &cv_b_h)?,
                dyn_buf: Mutex::new(dyn_buf),
            });
        }

        let dyn_buf = DynamicWindBuffers {
            u_a: DeviceBuffer::<T>::zeroed(stream, total)?,
            u_b: DeviceBuffer::<T>::zeroed(stream, total)?,
            last_t: None,
            host_scratch_u: vec![T::ZERO; total],
            host_scratch_v: vec![T::ZERO; total],
            host_wind_3d: vec![[0.0; 3]; total],
        };

        Ok(DeviceWindState::HostEvaluatedDynamic {
            wind: Arc::clone(wind),
            dyn_buf: Mutex::new(dyn_buf),
        })
    }
}

impl<T: CudaScalar> ComputeBackend<T> for CudaBackend<T> {
    type Buffer = DeviceBuffer<T>;
    type Error = CudaError;

    fn name(&self) -> &'static str {
        "cuda_oxide"
    }

    fn mesh(&self) -> &CubedSphereMesh<T> {
        &self.mesh
    }

    fn allocate_zeros(&self, len: usize) -> Result<Self::Buffer, Self::Error> {
        DeviceBuffer::<T>::zeroed(&self.stream, len).map_err(CudaError::from)
    }

    fn upload(&self, data: &[T]) -> Result<Self::Buffer, Self::Error> {
        DeviceBuffer::from_host(&self.stream, data).map_err(CudaError::from)
    }

    fn download(&self, buf: &Self::Buffer) -> Result<Vec<T>, Self::Error> {
        buf.to_host_vec(&self.stream).map_err(CudaError::from)
    }

    fn compute_advection_rhs(
        &self,
        state: &Self::Buffer,
        rhs: &mut Self::Buffer,
        t: f64,
    ) -> Result<(), Self::Error> {
        let total = self.mesh.total_nodes;
        if state.len() != total || rhs.len() != total {
            return Err(CudaError::InvalidConfiguration(format!(
                "buffer lengths (state={}, rhs={}) do not match mesh total_nodes={}",
                state.len(),
                rhs.len(),
                total
            )));
        }

        let n_1d = self.mesh.n_nodes_1d as u32;
        match &self.wind_state {
            DeviceWindState::Steady { u_a, u_b } => (self.rhs_dispatch)(
                &self.stream,
                state,
                u_a,
                u_b,
                &self.sqrt_g_dev,
                &self.diff_mat_dev,
                &self.neighbors_dev,
                rhs,
                n_1d,
                self.mesh.map_factor,
                self.inv_w0,
                self.inv_wn,
                self.upwind_weight,
            ),
            DeviceWindState::NairLauritzenDevice {
                period,
                amplitude_factor,
                sin_2lon,
                cos_2lon,
                u_bg_a,
                u_bg_b,
                cu_a,
                cv_a,
                cu_b,
                cv_b,
                dyn_buf,
            } => {
                let mut guard = dyn_buf.lock().map_err(|_| {
                    CudaError::InvalidConfiguration("dynamic wind mutex poisoned".to_string())
                })?;
                if guard.last_t != Some(t) {
                    let t_inv = 1.0 / *period;
                    let wt2 = 4.0 * PI * t * t_inv;
                    let (s2wt, c2wt) = wt2.sin_cos();
                    let kt = (*amplitude_factor * self.mesh.config.radius * t_inv)
                        * (PI * t * t_inv).cos();

                    let c2wt_t = T::from_f64(c2wt);
                    let s2wt_t = T::from_f64(s2wt);
                    let kt_t = T::from_f64(kt);

                    let db = &mut *guard;
                    (self.wind_comp_dispatch)(
                        &self.stream,
                        sin_2lon,
                        cos_2lon,
                        u_bg_a,
                        cu_a,
                        cv_a,
                        &mut db.u_a,
                        c2wt_t,
                        s2wt_t,
                        kt_t,
                    )?;
                    (self.wind_comp_dispatch)(
                        &self.stream,
                        sin_2lon,
                        cos_2lon,
                        u_bg_b,
                        cu_b,
                        cv_b,
                        &mut db.u_b,
                        c2wt_t,
                        s2wt_t,
                        kt_t,
                    )?;
                    db.last_t = Some(t);
                }

                (self.rhs_dispatch)(
                    &self.stream,
                    state,
                    &guard.u_a,
                    &guard.u_b,
                    &self.sqrt_g_dev,
                    &self.diff_mat_dev,
                    &self.neighbors_dev,
                    rhs,
                    n_1d,
                    self.mesh.map_factor,
                    self.inv_w0,
                    self.inv_wn,
                    self.upwind_weight,
                )
            }
            DeviceWindState::HostEvaluatedDynamic { wind, dyn_buf } => {
                let mut guard = dyn_buf.lock().map_err(|_| {
                    CudaError::InvalidConfiguration("dynamic wind mutex poisoned".to_string())
                })?;
                if guard.last_t != Some(t) {
                    let r = self.mesh.config.radius;
                    let db = &mut *guard;
                    for (idx, dst) in db.host_wind_3d.iter_mut().enumerate() {
                        let xyz = [
                            self.mesh.xyz[0][idx].to_f64(),
                            self.mesh.xyz[1][idx].to_f64(),
                            self.mesh.xyz[2][idx].to_f64(),
                        ];
                        *dst = wind.velocity_3d(
                            t,
                            self.mesh.lon_f64[idx],
                            self.mesh.lat_f64[idx],
                            xyz,
                            r,
                        );
                    }
                    self.mesh.project_wind_3d(
                        &db.host_wind_3d,
                        &mut db.host_scratch_u,
                        &mut db.host_scratch_v,
                    );
                    db.u_a.copy_from_host(&self.stream, &db.host_scratch_u)?;
                    db.u_b.copy_from_host(&self.stream, &db.host_scratch_v)?;
                    db.last_t = Some(t);
                }

                (self.rhs_dispatch)(
                    &self.stream,
                    state,
                    &guard.u_a,
                    &guard.u_b,
                    &self.sqrt_g_dev,
                    &self.diff_mat_dev,
                    &self.neighbors_dev,
                    rhs,
                    n_1d,
                    self.mesh.map_factor,
                    self.inv_w0,
                    self.inv_wn,
                    self.upwind_weight,
                )
            }
        }
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
        let total = self.mesh.total_nodes;
        if u.len() != total || du.len() != total || rhs.len() != total {
            return Err(CudaError::InvalidConfiguration(format!(
                "LSRK buffer length mismatch (u={}, du={}, rhs={}, expected={})",
                u.len(),
                du.len(),
                rhs.len(),
                total
            )));
        }
        (self.lsrk_dispatch)(&self.stream, u, du, rhs, rk_a, rk_b, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_neighbor_gpu_layout_and_packing() {
        assert_eq!(std::mem::size_of::<EdgeNeighborGpu>(), 8);
        assert_eq!(std::mem::align_of::<EdgeNeighborGpu>(), 4);

        let n = EdgeNeighbor {
            nbr_elem: 123_456,
            nbr_side: 3,
            reverse: 1,
            _pad: [0; 2],
        };
        let gpu_n = EdgeNeighborGpu::from(n);
        assert_eq!(gpu_n.nbr_elem, 123_456);
        assert_eq!(gpu_n.nbr_side, 3);
        assert_eq!(gpu_n.reverse, 1);

        let packed = pack_neighbor(n);
        assert_eq!((packed & 0xFFFF_FFFF) as u32, 123_456);
        assert_eq!(((packed >> 32) & 0xFF) as u8, 3);
        assert_eq!(((packed >> 40) & 0xFF) as u8, 1);
    }
}
