//! LGL quadrature and nodal differentiation on the reference interval `[-1, 1]`.
use crate::scalar::FloatScalar;

/// 1D Legendre-Gauss-Lobatto (LGL) reference element operators on `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct ReferenceElement<T: FloatScalar> {
    /// Polynomial degree `p >= 1` of the nodal interpolant.
    pub order: usize,
    /// Number of LGL nodes, `p + 1`, including both endpoints.
    pub n_nodes: usize,
    /// Sorted LGL nodes in `[-1, 1]`, length `n_nodes`.
    pub nodes: Vec<T>,
    /// LGL quadrature weights on `[-1, 1]`, length `n_nodes`, summing to 2.
    pub weights: Vec<T>,
    /// Row-major nodal differentiation matrix `D[i * n_nodes + j] = dL_j(xi_i) / dxi`.
    pub diff_matrix: Vec<T>,
    /// Exact `f64` nodes kept for geometry precomputation.
    pub nodes_f64: Vec<f64>,
    /// Exact `f64` weights kept for diagnostics integration.
    pub weights_f64: Vec<f64>,
}

impl<T: FloatScalar> ReferenceElement<T> {
    /// Computes LGL nodes, weights, and differentiation in `f64`, then converts to `T`.
    /// # Panics
    /// Panics if `order < 1`.
    pub fn new(order: usize) -> Self {
        assert!(order >= 1, "polynomial order must be >= 1");
        let (nodes_f64, weights_f64) = lgl_nodes_and_weights(order);
        let diff_f64 = lgl_differentiation_matrix(&nodes_f64);
        let n_nodes = order + 1;

        Self {
            order,
            n_nodes,
            nodes: nodes_f64.iter().map(|&v| T::from_f64(v)).collect(),
            weights: weights_f64.iter().map(|&v| T::from_f64(v)).collect(),
            diff_matrix: diff_f64.iter().map(|&v| T::from_f64(v)).collect(),
            nodes_f64,
            weights_f64,
        }
    }
}

/// Evaluates Legendre polynomial `P_n(x)` and its first derivative `P'_n(x)`
/// via the stable three-term recurrence.
pub fn legendre_poly_and_deriv(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    let mut p_prev = 1.0;
    let mut p_curr = x;
    let mut dp_prev = 0.0;
    let mut dp_curr = 1.0;

    for k in 1..n {
        let kf = k as f64;
        let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
        let dp_next = ((2.0 * kf + 1.0) * (p_curr + x * dp_curr) - kf * dp_prev) / (kf + 1.0);
        p_prev = p_curr;
        p_curr = p_next;
        dp_prev = dp_curr;
        dp_curr = dp_next;
    }

    (p_curr, dp_curr)
}

/// Computes `(p + 1)` Legendre-Gauss-Lobatto nodes in `[-1, 1]` and their quadrature weights.
/// Weights sum to 2 and integrate polynomials through degree `2 * order - 1` exactly.
/// # Panics
/// Panics if `order < 1`.
pub fn lgl_nodes_and_weights(order: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(order >= 1, "polynomial order must be >= 1");
    let n = order + 1;
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    nodes[0] = -1.0;
    nodes[order] = 1.0;
    let end_w = 2.0 / ((order * (order + 1)) as f64);
    weights[0] = end_w;
    weights[order] = end_w;

    if order == 1 {
        return (nodes, weights);
    }

    let pf = order as f64;
    let half = order / 2;

    for k in 1..=half {
        if order.is_multiple_of(2) && k == half {
            nodes[k] = 0.0;
            let (p0, _) = legendre_poly_and_deriv(order, 0.0);
            weights[k] = 2.0 / ((order * (order + 1)) as f64 * p0 * p0);
            continue;
        }

        // Initial guess: Chebyshev-Gauss-Lobatto node in (-1, 0)
        let mut x = -(std::f64::consts::PI * (k as f64) / pf).cos();

        for _ in 0..64 {
            let (p, dp) = legendre_poly_and_deriv(order, x);
            // From Legendre ODE: (1 - x^2) P''_p = 2x P'_p - p(p+1) P_p
            let d2p = (2.0 * x * dp - pf * (pf + 1.0) * p) / (1.0 - x * x);
            let dx = -dp / d2p;
            x += dx;
            if dx.abs() <= 1e-16 * (1.0 + x.abs()) {
                break;
            }
        }

        let (p_final, _) = legendre_poly_and_deriv(order, x);
        let w = 2.0 / ((order * (order + 1)) as f64 * p_final * p_final);

        nodes[k] = x;
        nodes[order - k] = -x;
        weights[k] = w;
        weights[order - k] = w;
    }

    (nodes, weights)
}

/// Computes the `(p + 1) x (p + 1)` row-major LGL nodal differentiation matrix `D`.
///
/// Uses the negative-sum trick for diagonal entries so `D * 1 == 0` holds to machine precision.
/// # Panics
/// Panics if fewer than two nodes are supplied; nodes are expected to be distinct LGL nodes.
pub fn lgl_differentiation_matrix(nodes: &[f64]) -> Vec<f64> {
    let n = nodes.len();
    assert!(n >= 2);
    let order = n - 1;
    let p_vals: Vec<f64> = nodes
        .iter()
        .map(|&x| legendre_poly_and_deriv(order, x).0)
        .collect();

    let mut d = vec![0.0; n * n];
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i != j {
                let val = p_vals[i] / (p_vals[j] * (nodes[i] - nodes[j]));
                d[i * n + j] = val;
                row_sum += val;
            }
        }
        d[i * n + i] = -row_sum;
    }

    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lgl_quadrature_exactness() {
        for order in [1, 4, 8, 16, 32] {
            let (nodes, weights) = lgl_nodes_and_weights(order);
            let sum_w: f64 = weights.iter().sum();
            assert!(
                (sum_w - 2.0).abs() < 1e-14,
                "order {order} weight sum={sum_w}"
            );

            // LGL quadrature integrates polynomials up to degree 2*order - 1 exactly.
            let max_deg = 2 * order - 1;
            for deg in 0..=max_deg {
                let num: f64 = nodes
                    .iter()
                    .zip(weights.iter())
                    .map(|(&x, &w)| w * x.powi(deg as i32))
                    .sum();
                let exact = if deg % 2 == 0 {
                    2.0 / ((deg + 1) as f64)
                } else {
                    0.0
                };
                assert!(
                    (num - exact).abs() < 1e-13,
                    "order {order}, deg {deg}: num={num}, exact={exact}"
                );
            }
        }
    }

    #[test]
    fn test_differentiation_matrix_exactness_and_sbp() {
        for order in [2, 5, 8, 16, 24] {
            let (nodes, weights) = lgl_nodes_and_weights(order);
            let d = lgl_differentiation_matrix(&nodes);
            let n = order + 1;

            // 1. Exactness for x^k (k = 0..=order)
            for deg in 0..=order {
                let f: Vec<f64> = nodes.iter().map(|&x| x.powi(deg as i32)).collect();
                for i in 0..n {
                    let df_num: f64 = (0..n).map(|j| d[i * n + j] * f[j]).sum();
                    let df_exact = if deg == 0 {
                        0.0
                    } else {
                        (deg as f64) * nodes[i].powi((deg - 1) as i32)
                    };
                    assert!(
                        (df_num - df_exact).abs() < 1e-11,
                        "order {order}, deg {deg}, node {i}: diff={}",
                        (df_num - df_exact).abs()
                    );
                }
            }

            // 2. Summation-By-Parts (SBP) identity: W*D + D^T*W = diag(-1, 0, ..., 0, +1)
            for i in 0..n {
                for j in 0..n {
                    let sbp_ij = weights[i] * d[i * n + j] + weights[j] * d[j * n + i];
                    let expected = if i == j && i == 0 {
                        -1.0
                    } else if i == j && i == n - 1 {
                        1.0
                    } else {
                        0.0
                    };
                    assert!(
                        (sbp_ij - expected).abs() < 1e-13,
                        "SBP mismatch at ({i},{j}) for order {order}: got {sbp_ij}, expected {expected}"
                    );
                }
            }
        }
    }
}
