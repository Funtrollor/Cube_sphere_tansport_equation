use serde::{Deserialize, Serialize};

use cs_core::{CubedSphereMesh, FloatScalar};

/// Comprehensive diagnostic metrics computed using full `f64` surface quadrature.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub l1_rel: f64,
    pub l2_abs: f64,
    pub l2_rel: f64,
    pub l_inf: f64,
    pub l_inf_rel: f64,
    pub mass_initial: f64,
    pub mass_final: f64,
    pub mass_rel_err: f64,
    pub min_initial: f64,
    pub max_initial: f64,
    pub min_final: f64,
    pub max_final: f64,
}

/// Evaluates L1/L2/Linf error norms, extrema, and discrete mass conservation.
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cs_core::{MeshConfig, ProjectionKind};

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

        let m_no_exact = compute_diagnostics(&mesh, &initial, &perturbed, None);
        assert!(m_no_exact.l2_rel.is_nan());
        assert!(m_no_exact.l_inf.is_nan());
        assert!((m_no_exact.mass_rel_err - 0.02).abs() < 1e-13);
    }
}
