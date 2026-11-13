use std::error::Error;

use cs_core::{CubedSphereMesh, FloatScalar};

/// Hardware-agnostic compute and memory backend trait.
///
/// Both the pure-Rust Rayon CPU backend (`cs-backend-cpu`) and the NVIDIA `cuda-oxide`
/// GPU backend (`cs-backend-cuda`) implement this trait so time integration, tests,
/// and diagnostics are shared verbatim.
pub trait ComputeBackend<T: FloatScalar> {
    type Buffer;
    type Error: Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn mesh(&self) -> &CubedSphereMesh<T>;

    fn allocate_zeros(&self, len: usize) -> Result<Self::Buffer, Self::Error>;
    fn upload(&self, data: &[T]) -> Result<Self::Buffer, Self::Error>;
    fn download(&self, buf: &Self::Buffer) -> Result<Vec<T>, Self::Error>;

    /// Evaluates full semi-discrete RHS:
    /// `rhs = -VolumeSplitDivergence(state, t) + SAT_BoundaryPenalty(state, t)`.
    fn compute_advection_rhs(
        &self,
        state: &Self::Buffer,
        rhs: &mut Self::Buffer,
        t: f64,
    ) -> Result<(), Self::Error>;

    /// Performs one fused Carpenter-Kennedy LSRK45 stage update:
    /// `du[k] = rk_a * du[k] + dt * rhs[k]`
    /// `u[k]  = u[k] + rk_b * du[k]`
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
