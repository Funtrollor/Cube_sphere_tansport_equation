//! LGL surface-quadrature error norms, mass conservation, and discrete energy diagnostics.

use serde::{Deserialize, Serialize};

use cubesphere_core::{CubedSphereMesh, FloatScalar};

/// Comprehensive diagnostic metrics computed using full `f64` surface quadrature.
///
/// Error norms (`l1_rel`, `l2_abs`, `l2_rel`, `l_inf`, `l_inf_rel`) are `NaN` when no exact
/// reference field is supplied to [`compute_diagnostics`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Relative `L1` error `integral(|phi - exact|) / integral(|exact|)` (falls back to absolute when denominator `<= 1e-30`).
    pub l1_rel: f64,
    /// Absolute `L2` error `sqrt(integral((phi - exact)^2))` using `f64` LGL quadrature.
    pub l2_abs: f64,
    /// Relative `L2` error `l2_abs / sqrt(integral(exact^2))` (falls back to `l2_abs` when denominator `<= 1e-30`).
    pub l2_rel: f64,
    /// Pointwise `L_inf` error `max |phi - exact|` over all mesh nodes.
    pub l_inf: f64,
    /// Relative `L_inf` error `l_inf / max |exact|` (falls back to `l_inf` when `max |exact| <= 1e-30`).
    pub l_inf_rel: f64,
    /// Discrete global mass `integral(initial dS)` evaluated in `f64`.
    pub mass_initial: f64,
    /// Discrete global mass `integral(final_state dS)` evaluated in `f64`.
    pub mass_final: f64,
    /// Relative global mass conservation error `|mass_final - mass_initial| / |mass_initial|` (absolute if `<= 1e-30`).
    pub mass_rel_err: f64,
    /// Minimum nodal value of the initial field (converted to `f64`).
    pub min_initial: f64,
    /// Maximum nodal value of the initial field (converted to `f64`).
    pub max_initial: f64,
    /// Minimum nodal value of the final field (converted to `f64`).
    pub min_final: f64,
    /// Maximum nodal value of the final field (converted to `f64`).
    pub max_final: f64,
    /// Discrete energy `E = integral(phi^2 dS)` of the initial state.
    pub energy_initial: f64,
    /// Discrete energy `E = integral(phi^2 dS)` of the final state.
    pub energy_final: f64,
    /// Signed relative energy change `(E(T) - E(0)) / E(0)` (negative = dissipated).
    pub energy_rel_change: f64,
}

/// Discrete energy `E = integral_{S^2}(phi^2 dS)` using the same `f64` LGL surface quadrature
/// (`w_i w_j sqrt(g)`) as the SBP mass matrix, so it matches the norm of the energy estimate.
///
/// Squares are formed in `f64` so `f32` states do not lose precision in the diagnostic.
/// `field` must have length `mesh.total_nodes` in `[elem][j][i]` order.
///
/// # Panics
///
/// Panics if `field.len() != mesh.total_nodes`.
pub fn compute_energy<T: FloatScalar>(mesh: &CubedSphereMesh<T>, field: &[T]) -> f64 {
    assert_eq!(field.len(), mesh.total_nodes);
    let n = mesh.n_nodes_1d;
    let mut total = 0.0;
    for e in 0..mesh.n_elems {
        let base = e * mesh.n_nodes_elem;
        for j in 0..n {
            let wj = mesh.weights_1d_f64[j];
            let mut row_sum = 0.0;
            for i in 0..n {
                let idx = base + j * n + i;
                let v = field[idx].to_f64();
                row_sum += mesh.weights_1d_f64[i] * mesh.sqrt_g_f64[idx] * v * v;
            }
            total += wj * row_sum;
        }
    }
    total
}

/// Signed relative change `(after - before) / before`, falling back to the absolute change
/// when `before` is (numerically) zero.
fn signed_rel_change(before: f64, after: f64) -> f64 {
    if before.abs() > 1e-30 {
        (after - before) / before.abs()
    } else {
        after - before
    }
}

/// Evaluates L1/L2/Linf error norms, extrema, discrete mass conservation, and discrete energy.
///
/// All slices must have length `mesh.total_nodes` in `[elem][j][i]` order and are promoted to
/// `f64` during quadrature; if `exact` is `None`, the error norm fields are `NaN`.
///
/// # Panics
///
/// Panics if `initial`, `final_state`, or `exact` (when `Some`) has length different from `mesh.total_nodes`.
pub fn compute_diagnostics<T: FloatScalar>(
    mesh: &CubedSphereMesh<T>,
    initial: &[T],
    final_state: &[T],
    exact: Option<&[T]>,
) -> ErrorMetrics {
    assert_eq!(initial.len(), mesh.total_nodes);
    assert_eq!(final_state.len(), mesh.total_nodes);

    let mass_initial = mesh.integrate(initial);
    let mass_final = mesh.integrate(final_state);
    let mass_rel_err = if mass_initial.abs() > 1e-30 {
        (mass_final - mass_initial).abs() / mass_initial.abs()
    } else {
        (mass_final - mass_initial).abs()
    };

    let (mut min_initial, mut max_initial) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_final, mut max_final) = (f64::INFINITY, f64::NEG_INFINITY);

    for idx in 0..mesh.total_nodes {
        let vi = initial[idx].to_f64();
        let vf = final_state[idx].to_f64();
        min_initial = min_initial.min(vi);
        max_initial = max_initial.max(vi);
        min_final = min_final.min(vf);
        max_final = max_final.max(vf);
    }

    let (l1_rel, l2_abs, l2_rel, l_inf, l_inf_rel) = if let Some(exact_slice) = exact {
        assert_eq!(exact_slice.len(), mesh.total_nodes);
        let n = mesh.n_nodes_1d;

        let mut int_abs_diff = 0.0;
        let mut int_abs_exact = 0.0;
        let mut int_sq_diff = 0.0;
        let mut int_sq_exact = 0.0;
        let mut l_inf = 0.0_f64;
        let mut max_exact = 0.0_f64;

        for e in 0..mesh.n_elems {
            let base = e * mesh.n_nodes_elem;
            for j in 0..n {
                let wj = mesh.weights_1d_f64[j];
                for i in 0..n {
                    let idx = base + j * n + i;
                    let w = wj * mesh.weights_1d_f64[i] * mesh.sqrt_g_f64[idx];
                    let vf = final_state[idx].to_f64();
                    let ve = exact_slice[idx].to_f64();
                    let d = (vf - ve).abs();

                    int_abs_diff += w * d;
                    int_abs_exact += w * ve.abs();
                    int_sq_diff += w * d * d;
                    int_sq_exact += w * ve * ve;

                    if d > l_inf {
                        l_inf = d;
                    }
                    if ve.abs() > max_exact {
                        max_exact = ve.abs();
                    }
                }
            }
        }

        let l2_abs = int_sq_diff.sqrt();
        let norm_exact_l2 = int_sq_exact.sqrt();
        let l2_rel = if norm_exact_l2 > 1e-30 {
            l2_abs / norm_exact_l2
        } else {
            l2_abs
        };
        let l1_rel = if int_abs_exact > 1e-30 {
            int_abs_diff / int_abs_exact
        } else {
            int_abs_diff
        };
        let l_inf_rel = if max_exact > 1e-30 {
            l_inf / max_exact
        } else {
            l_inf
        };

        (l1_rel, l2_abs, l2_rel, l_inf, l_inf_rel)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };

    let energy_initial = compute_energy(mesh, initial);
    let energy_final = compute_energy(mesh, final_state);

    ErrorMetrics {
        l1_rel,
        l2_abs,
        l2_rel,
        l_inf,
        l_inf_rel,
        mass_initial,
        mass_final,
        mass_rel_err,
        min_initial,
        max_initial,
        min_final,
        max_final,
        energy_initial,
        energy_final,
        energy_rel_change: signed_rel_change(energy_initial, energy_final),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubesphere_core::{MeshConfig, ProjectionKind};

    #[test]
    fn test_diagnostics_metrics_and_uniform_perturbation() {
        let mesh = CubedSphereMesh::<f64>::from_config(MeshConfig {
            radius: 1.5,
            num_elem_per_face: 2,
            poly_order: 4,
            projection: ProjectionKind::Equiangular,
        });
        let initial = vec![1.0_f64; mesh.total_nodes];
        let perturbed = vec![1.02_f64; mesh.total_nodes];

        let m_with_exact = compute_diagnostics(&mesh, &initial, &perturbed, Some(&initial));
        assert!((m_with_exact.l_inf - 0.02).abs() < 1e-13);
        assert!((m_with_exact.l_inf_rel - 0.02).abs() < 1e-13);
        assert!((m_with_exact.l1_rel - 0.02).abs() < 1e-13);
        assert!((m_with_exact.l2_rel - 0.02).abs() < 1e-13);
        assert!((m_with_exact.mass_rel_err - 0.02).abs() < 1e-13);
        assert!((m_with_exact.min_final - 1.02).abs() < 1e-13);

        // Energy of phi == 1 is the (quadrature) surface area; scaling by 1.02 scales it by 1.02^2.
        let area = mesh.surface_area();
        assert!((m_with_exact.energy_initial - area).abs() / area < 1e-14);
        assert!((m_with_exact.energy_rel_change - (1.02_f64 * 1.02 - 1.0)).abs() < 1e-13);

        let m_no_exact = compute_diagnostics(&mesh, &initial, &perturbed, None);
        assert!(m_no_exact.l2_rel.is_nan());
        assert!(m_no_exact.l_inf.is_nan());
        assert!((m_no_exact.mass_rel_err - 0.02).abs() < 1e-13);
        assert!(m_no_exact.energy_rel_change > 0.0);

        // Signed: shrinking the field reports a negative (dissipative) change.
        let shrunk = vec![0.9_f64; mesh.total_nodes];
        let m_shrunk = compute_diagnostics(&mesh, &initial, &shrunk, None);
        assert!((m_shrunk.energy_rel_change - (0.81 - 1.0)).abs() < 1e-13);
    }

    #[test]
    fn test_energy_matches_l2_norm_squared_and_f32_precision() {
        let cfg = MeshConfig {
            radius: 1.0,
            num_elem_per_face: 3,
            poly_order: 5,
            projection: ProjectionKind::Equidistant,
        };
        let mesh = CubedSphereMesh::<f64>::from_config(cfg);
        let field: Vec<f64> = (0..mesh.total_nodes)
            .map(|idx| mesh.xyz[2][idx] + 0.3 * mesh.xyz[0][idx])
            .collect();
        let squared: Vec<f64> = field.iter().map(|v| v * v).collect();
        let e = compute_energy(&mesh, &field);
        assert!((e - mesh.integrate(&squared)).abs() < 1e-13 * e);
        // \int (z + 0.3 x)^2 dS over the unit sphere = (1 + 0.09) * 4 pi / 3
        let exact = 1.09 * 4.0 * std::f64::consts::PI / 3.0;
        assert!((e - exact).abs() / exact < 1e-6, "E={e}, exact={exact}");

        let mesh32 = CubedSphereMesh::<f32>::from_config(cfg);
        let field32: Vec<f32> = field.iter().map(|&v| v as f32).collect();
        let e32 = compute_energy(&mesh32, &field32);
        assert!((e32 - e).abs() / e < 1e-6, "f32 energy {e32} vs f64 {e}");
    }
}
