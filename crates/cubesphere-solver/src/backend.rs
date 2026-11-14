//! Compute and buffer interface used by the backend-independent LSRK45 driver.
use std::error::Error;

use cubesphere_core::{CubedSphereMesh, FloatScalar};

/// Hardware-agnostic compute and memory backend trait.
///
/// Both the pure-Rust Rayon CPU backend (`cubesphere-backend-cpu`) and the NVIDIA `cuda-oxide`
/// GPU backend (`cubesphere-backend-cuda`) implement this trait so time integration, tests,
/// and diagnostics are shared verbatim.
pub trait ComputeBackend<T: FloatScalar> {
    /// Backend-owned storage for `T` scalars; nodal fields use mesh order `[elem][j][i]`.
    type Buffer;
    /// Allocation, transfer, or computation failure reported by the implementation.
    type Error: Error + Send + Sync + 'static;

    /// Human-readable implementation identifier for reporting the compute device or backend.
    fn name(&self) -> &'static str;
    /// Mesh defining geometry, quadrature, and the ordering of all nodal buffers.
    fn mesh(&self) -> &CubedSphereMesh<T>;

    /// Allocates `len` scalars initialized to `T::ZERO` in backend memory.
    /// # Errors
    /// Returns the implementation's allocation or initialization error.
    fn allocate_zeros(&self, len: usize) -> Result<Self::Buffer, Self::Error>;
    /// Copies host scalars into a new backend buffer, preserving their order.
    /// # Errors
    /// Returns the implementation's allocation or host-to-backend transfer error.
    fn upload(&self, data: &[T]) -> Result<Self::Buffer, Self::Error>;
    /// Copies a backend buffer into a host vector, preserving its scalar order.
    /// # Errors
    /// Returns the implementation's backend-to-host transfer error.
    fn download(&self, buf: &Self::Buffer) -> Result<Vec<T>, Self::Error>;

    /// Evaluates full semi-discrete RHS:
    /// `rhs = -VolumeSplitDivergence(state, t) + SAT_BoundaryPenalty(state, t)`.
    /// `t` is physical time; both buffers contain `mesh().total_nodes` scalars in mesh order.
    /// # Errors
    /// Returns the implementation's RHS evaluation error.
    fn compute_advection_rhs(
        &self,
        state: &Self::Buffer,
        rhs: &mut Self::Buffer,
        t: f64,
    ) -> Result<(), Self::Error>;

    /// Performs one fused Carpenter-Kennedy LSRK45 stage update:
    /// `du[k] = rk_a * du[k] + dt * rhs[k]`
    /// `u[k]  = u[k] + rk_b * du[k]`
    /// All buffers share mesh order; `dt` uses the physical time unit of the RHS.
    /// # Errors
    /// Returns the implementation's stage-update error.
    fn lsrk_stage_update(
        &self,
        u: &mut Self::Buffer,
        du: &mut Self::Buffer,
        rhs: &Self::Buffer,
        rk_a: T,
        rk_b: T,
        dt: T,
    ) -> Result<(), Self::Error>;

    /// Executes an entire 5-stage LSRK45 step `[t, t + dt]` directly on backend memory.
    /// Evaluates the RHS at `t + RK45_C[stage] * dt`, converting coefficients and `dt` to `T`.
    /// # Errors
    /// Propagates the first RHS evaluation or stage-update error.
    fn lsrk45_step(
        &self,
        u: &mut Self::Buffer,
        du: &mut Self::Buffer,
        rhs: &mut Self::Buffer,
        t: f64,
        dt: f64,
    ) -> Result<(), Self::Error> {
        let dt_t = T::from_f64(dt);
        for stage in 0..5 {
            let stage_t = t + crate::lsrk45::RK45_C[stage] * dt;
            self.compute_advection_rhs(u, rhs, stage_t)?;
            self.lsrk_stage_update(
                u,
                du,
                rhs,
                T::from_f64(crate::lsrk45::RK45_A[stage]),
                T::from_f64(crate::lsrk45::RK45_B[stage]),
                dt_t,
            )?;
        }
        Ok(())
    }
}
