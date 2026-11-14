use serde::{Deserialize, Serialize};

/// The 6 faces of the Staircase Cubed-Sphere layout (`P1`..`P6`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CubeFace {
    Front = 0,  // P1: +X
    Right = 1,  // P2: +Y
    Top = 2,    // P3: +Z
    Back = 3,   // P4: -X
    Left = 4,   // P5: -Y
    Bottom = 5, // P6: -Z
}

impl CubeFace {
    pub const ALL: [CubeFace; 6] = [
        CubeFace::Front,
        CubeFace::Right,
        CubeFace::Top,
        CubeFace::Back,
        CubeFace::Left,
        CubeFace::Bottom,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Self {
        match idx {
            0 => CubeFace::Front,
            1 => CubeFace::Right,
            2 => CubeFace::Top,
            3 => CubeFace::Back,
            4 => CubeFace::Left,
            5 => CubeFace::Bottom,
            _ => panic!("CubeFace index out of bounds (expected 0..6)"),
        }
    }
}

/// The 4 edges of a local panel or element in `(\alpha, \beta)` coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EdgeSide {
    West = 0,  // alpha = alpha_min (i = 0)
    East = 1,  // alpha = alpha_max (i = p)
    South = 2, // beta  = beta_min  (j = 0)
    North = 3, // beta  = beta_max  (j = p)
}

impl EdgeSide {
    pub const ALL: [EdgeSide; 4] = [
        EdgeSide::West,
        EdgeSide::East,
        EdgeSide::South,
        EdgeSide::North,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Self {
        match idx {
            0 => EdgeSide::West,
            1 => EdgeSide::East,
            2 => EdgeSide::South,
            3 => EdgeSide::North,
            _ => panic!("EdgeSide index out of bounds (expected 0..4)"),
        }
    }
}

/// Inter-panel connectivity entry for the Staircase Cubed-Sphere architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PanelConnection {
    pub nbr_face: CubeFace,
    pub nbr_side: EdgeSide,
    pub reverse: bool,
}

/// Complete Staircase topology connectivity table for all 6 panels x 4 edges.
pub const PANEL_CONNECTIVITY: [[PanelConnection; 4]; 6] = [
    // P1 (Front)
    [
        PanelConnection { nbr_face: CubeFace::Left, nbr_side: EdgeSide::North, reverse: true },
        PanelConnection { nbr_face: CubeFace::Right, nbr_side: EdgeSide::West, reverse: false },
        PanelConnection { nbr_face: CubeFace::Bottom, nbr_side: EdgeSide::North, reverse: false },
        PanelConnection { nbr_face: CubeFace::Top, nbr_side: EdgeSide::West, reverse: true },
    ],
    // P2 (Right)
    [
        PanelConnection { nbr_face: CubeFace::Front, nbr_side: EdgeSide::East, reverse: false },
        PanelConnection { nbr_face: CubeFace::Back, nbr_side: EdgeSide::South, reverse: true },
        PanelConnection { nbr_face: CubeFace::Bottom, nbr_side: EdgeSide::East, reverse: true },
        PanelConnection { nbr_face: CubeFace::Top, nbr_side: EdgeSide::South, reverse: false },
    ],
    // P3 (Top)
    [
        PanelConnection { nbr_face: CubeFace::Front, nbr_side: EdgeSide::North, reverse: true },
        PanelConnection { nbr_face: CubeFace::Back, nbr_side: EdgeSide::West, reverse: false },
        PanelConnection { nbr_face: CubeFace::Right, nbr_side: EdgeSide::North, reverse: false },
        PanelConnection { nbr_face: CubeFace::Left, nbr_side: EdgeSide::West, reverse: true },
    ],
    // P4 (Back)
    [
        PanelConnection { nbr_face: CubeFace::Top, nbr_side: EdgeSide::East, reverse: false },
        PanelConnection { nbr_face: CubeFace::Bottom, nbr_side: EdgeSide::South, reverse: true },
        PanelConnection { nbr_face: CubeFace::Right, nbr_side: EdgeSide::East, reverse: true },
        PanelConnection { nbr_face: CubeFace::Left, nbr_side: EdgeSide::South, reverse: false },
    ],
    // P5 (Left)
    [
        PanelConnection { nbr_face: CubeFace::Top, nbr_side: EdgeSide::North, reverse: true },
        PanelConnection { nbr_face: CubeFace::Bottom, nbr_side: EdgeSide::West, reverse: false },
        PanelConnection { nbr_face: CubeFace::Back, nbr_side: EdgeSide::North, reverse: false },
        PanelConnection { nbr_face: CubeFace::Front, nbr_side: EdgeSide::West, reverse: true },
    ],
    // P6 (Bottom)
    [
        PanelConnection { nbr_face: CubeFace::Left, nbr_side: EdgeSide::East, reverse: false },
        PanelConnection { nbr_face: CubeFace::Right, nbr_side: EdgeSide::South, reverse: true },
        PanelConnection { nbr_face: CubeFace::Back, nbr_side: EdgeSide::East, reverse: true },
        PanelConnection { nbr_face: CubeFace::Front, nbr_side: EdgeSide::South, reverse: false },
    ],
];

/// Metric quantities computed at a grid node in `(\alpha, \beta)` coordinates.
#[derive(Debug, Clone, Copy)]
pub struct PointMetric {
    pub xyz: [f64; 3],
    pub lon: f64,
    pub lat: f64,
    /// Area expansion factor `J = sqrt(det(G)) = ||e_alpha x e_beta||`.
    pub sqrt_g: f64,
    /// Contravariant basis vector `e^alpha` in 3D Cartesian space (`u^alpha = u_3d . e^alpha`).
    pub contra_alpha: [f64; 3],
    /// Contravariant basis vector `e^beta` in 3D Cartesian space (`u^beta = u_3d . e^beta`).
    pub contra_beta: [f64; 3],
}

/// Abstract spherical cube-face projection mapping `(\alpha, \beta) \in [-\pi/4, \pi/4]^2`
/// onto the sphere of radius `R`.
pub trait Projection: Send + Sync {
    fn name(&self) -> &'static str;

    /// Returns `(xyz, e_alpha, e_beta)` where `e_alpha = dr/dalpha`, `e_beta = dr/dbeta`.
    fn point_and_tangents(
        &self,
        face: CubeFace,
        alpha: f64,
        beta: f64,
        radius: f64,
    ) -> ([f64; 3], [f64; 3], [f64; 3]);

    /// Computes spherical coordinates and singularity-free contravariant basis vectors.
    fn point_metric(&self, face: CubeFace, alpha: f64, beta: f64, radius: f64) -> PointMetric {
        let (xyz, ea, eb) = self.point_and_tangents(face, alpha, beta, radius);
        let (lon, lat) = xyz_to_lon_lat(xyz);

        let g11 = dot3(ea, ea);
        let g12 = dot3(ea, eb);
        let g22 = dot3(eb, eb);
        let det_g = g11 * g22 - g12 * g12;
        debug_assert!(det_g > 0.0, "degenerate metric determinant");

        let sqrt_g = det_g.sqrt();
        let inv_det = 1.0 / det_g;
        let g_inv_11 = g22 * inv_det;
        let g_inv_12 = -g12 * inv_det;
        let g_inv_22 = g11 * inv_det;

        let contra_alpha = [
            g_inv_11 * ea[0] + g_inv_12 * eb[0],
            g_inv_11 * ea[1] + g_inv_12 * eb[1],
            g_inv_11 * ea[2] + g_inv_12 * eb[2],
        ];
        let contra_beta = [
            g_inv_12 * ea[0] + g_inv_22 * eb[0],
            g_inv_12 * ea[1] + g_inv_22 * eb[1],
            g_inv_12 * ea[2] + g_inv_22 * eb[2],
        ];

        PointMetric {
            xyz,
            lon,
            lat,
            sqrt_g,
            contra_alpha,
            contra_beta,
        }
    }
}

/// Standard Equiangular Cubed-Sphere gnomonic projection (`x = tan(alpha)`, `y = tan(beta)`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EquiangularProjection;

impl EquiangularProjection {
    /// Analytical Jacobian formula for Equiangular Cubed-Sphere:
    /// `J(alpha, beta) = R^2 / (rho^3 * cos^2(alpha) * cos^2(beta))`.
    #[inline]
    pub fn analytical_jacobian(alpha: f64, beta: f64, radius: f64) -> f64 {
        let ta = alpha.tan();
        let tb = beta.tan();
        let rho = (1.0 + ta * ta + tb * tb).sqrt();
        let ca = alpha.cos();
        let cb = beta.cos();
        (radius * radius) / (rho * rho * rho * ca * ca * cb * cb)
    }
}

impl Projection for EquiangularProjection {
    fn name(&self) -> &'static str {
        "equiangular"
    }

    fn point_and_tangents(
        &self,
        face: CubeFace,
        alpha: f64,
        beta: f64,
        radius: f64,
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let ta = alpha.tan();
        let tb = beta.tan();
        let ca = alpha.cos();
        let cb = beta.cos();
        let sec2_a = 1.0 / (ca * ca);
        let sec2_b = 1.0 / (cb * cb);

        gnomonic_project_and_tangents(face, ta, tb, sec2_a, sec2_b, radius)
    }
}

/// Equidistant Cubed-Sphere gnomonic projection (`x = 4*alpha/pi`, `y = 4*beta/pi` in `[-1, 1]`).
#[derive(Debug, Clone, Copy, Default)]
pub struct EquidistantProjection;

impl EquidistantProjection {
    /// Analytical Jacobian formula for Equidistant Cubed-Sphere:
    /// `J(alpha, beta) = R^2 * (4/pi)^2 / (1 + (4*alpha/pi)^2 + (4*beta/pi)^2)^(3/2)`.
    /// The gnomonic area factor `R^2/rho^3` is multiplied by `dx/dalpha * dy/dbeta = (4/pi)^2`.
    #[inline]
    pub fn analytical_jacobian(alpha: f64, beta: f64, radius: f64) -> f64 {
        let scale = 4.0 / std::f64::consts::PI;
        let x = alpha * scale;
        let y = beta * scale;
        let rho = (1.0 + x * x + y * y).sqrt();
        (radius * radius * scale * scale) / (rho * rho * rho)
    }
}

impl Projection for EquidistantProjection {
    fn name(&self) -> &'static str {
        "equidistant"
    }

    fn point_and_tangents(
        &self,
        face: CubeFace,
        alpha: f64,
        beta: f64,
        radius: f64,
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let scale = 4.0 / std::f64::consts::PI;
        let x = alpha * scale;
        let y = beta * scale;
        gnomonic_project_and_tangents(face, x, y, scale, scale, radius)
    }
}

#[inline]
fn gnomonic_project_and_tangents(
    face: CubeFace,
    x: f64,
    y: f64,
    dx_da: f64,
    dy_db: f64,
    radius: f64,
) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let (v, dv_dx, dv_dy) = match face {
        CubeFace::Front => ([1.0, x, y], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        CubeFace::Right => ([-x, 1.0, y], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        CubeFace::Top => ([-x, -y, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]),
        CubeFace::Back => ([-1.0, -y, -x], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]),
        CubeFace::Left => ([y, -1.0, -x], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]),
        CubeFace::Bottom => ([y, x, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
    };

    let rho2 = 1.0 + x * x + y * y;
    let rho = rho2.sqrt();
    let inv_rho = 1.0 / rho;
    let inv_rho3 = 1.0 / (rho2 * rho);

    let xyz = [
        radius * v[0] * inv_rho,
        radius * v[1] * inv_rho,
        radius * v[2] * inv_rho,
    ];

    // d(R * v / rho)/dalpha = R/rho^3 * (rho^2 * dv_dx - (v . dv_dx) * v) * dx_da
    let v_dot_dvx = x; // since v.dv_dx == x on all 6 faces
    let v_dot_dvy = y; // since v.dv_dy == y on all 6 faces

    let ea = [
        radius * inv_rho3 * (rho2 * dv_dx[0] - v_dot_dvx * v[0]) * dx_da,
        radius * inv_rho3 * (rho2 * dv_dx[1] - v_dot_dvx * v[1]) * dx_da,
        radius * inv_rho3 * (rho2 * dv_dx[2] - v_dot_dvx * v[2]) * dx_da,
    ];
    let eb = [
        radius * inv_rho3 * (rho2 * dv_dy[0] - v_dot_dvy * v[0]) * dy_db,
        radius * inv_rho3 * (rho2 * dv_dy[1] - v_dot_dvy * v[1]) * dy_db,
        radius * inv_rho3 * (rho2 * dv_dy[2] - v_dot_dvy * v[2]) * dy_db,
    ];

    (xyz, ea, eb)
}

/// Converts 3D Cartesian coordinates on the sphere to `(longitude, latitude)` in radians.
#[inline]
pub fn xyz_to_lon_lat(xyz: [f64; 3]) -> (f64, f64) {
    let r = (xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]).sqrt();
    let lon = xyz[1].atan2(xyz[0]);
    let lat = (xyz[2] / r).clamp(-1.0, 1.0).asin();
    (lon, lat)
}

/// Converts spherical `(u_lon, v_lat)` physical velocity components into 3D Cartesian vector.
///
/// Note: At the exact poles (`|cos(lat)| < 1e-14`), `(u_lon, v_lat)` frame is degenerate;
/// if `u_lon == 0 && v_lat == 0` it cleanly returns `[0, 0, 0]`. For non-zero polar flows,
/// prefer evaluating wind directly in 3D Cartesian coordinates.
#[inline]
pub fn lon_lat_wind_to_3d(lon: f64, lat: f64, u_lon: f64, v_lat: f64) -> [f64; 3] {
    let (sl, cl) = lon.sin_cos();
    let (st, ct) = lat.sin_cos();
    [
        -sl * u_lon - st * cl * v_lat,
        cl * u_lon - st * sl * v_lat,
        ct * v_lat,
    ]
}

/// Great-circle distance between two points `(lon1, lat1)` and `(lon2, lat2)` on a sphere of radius `R`.
#[inline]
pub fn great_circle_distance(lon1: f64, lat1: f64, lon2: f64, lat2: f64, radius: f64) -> f64 {
    let dlat = 0.5 * (lat2 - lat1);
    let dlon = 0.5 * (lon2 - lon1);
    let a = dlat.sin().powi(2) + lat1.cos() * lat2.cos() * dlon.sin().powi(2);
    let c = 2.0 * a.clamp(0.0, 1.0).sqrt().asin();
    radius * c
}

#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_right_handedness_and_analytical_jacobian() {
        let proj = EquiangularProjection;
        let radius = 2.5;
        let samples = [-0.6, -0.2, 0.0, 0.35, 0.71];

        for face in CubeFace::ALL {
            for &a in &samples {
                for &b in &samples {
                    let (xyz, ea, eb) = proj.point_and_tangents(face, a, b, radius);
                    let r_norm = dot3(xyz, xyz).sqrt();
                    assert!((r_norm - radius).abs() < 1e-14);

                    // Orthogonality to radial vector
                    assert!(dot3(xyz, ea).abs() < 1e-13);
                    assert!(dot3(xyz, eb).abs() < 1e-13);

                    // Right-handed: (ea x eb) . xyz > 0
                    let normal = cross3(ea, eb);
                    assert!(dot3(normal, xyz) > 0.0);

                    // Analytical Jacobian matches metric determinant sqrt_g
                    let m = proj.point_metric(face, a, b, radius);
                    let j_ana = EquiangularProjection::analytical_jacobian(a, b, radius);
                    assert!(
                        (m.sqrt_g - j_ana).abs() < 1e-13,
                        "face {:?} ({a},{b}): sqrt_g={} vs j_ana={}",
                        face,
                        m.sqrt_g,
                        j_ana
                    );

                    // Biorthogonality of contravariant basis: e^i . e_j = delta_{ij}
                    assert!((dot3(m.contra_alpha, ea) - 1.0).abs() < 1e-13);
                    assert!(dot3(m.contra_alpha, eb).abs() < 1e-13);
                    assert!(dot3(m.contra_beta, ea).abs() < 1e-13);
                    assert!((dot3(m.contra_beta, eb) - 1.0).abs() < 1e-13);
                }
            }
        }
    }

    #[test]
    fn test_equidistant_analytical_jacobian_and_biorthogonality() {
        let proj = EquidistantProjection;
        let radius = 2.5;
        let pi4 = std::f64::consts::FRAC_PI_4;
        let samples = [-pi4, -0.6, -0.2, 0.0, 0.35, 0.71, pi4];

        for face in CubeFace::ALL {
            for &a in &samples {
                for &b in &samples {
                    let (xyz, ea, eb) = proj.point_and_tangents(face, a, b, radius);
                    let r_norm = dot3(xyz, xyz).sqrt();
                    assert!((r_norm - radius).abs() < 1e-13);

                    // Orthogonality to radial vector
                    assert!(dot3(xyz, ea).abs() < 1e-13);
                    assert!(dot3(xyz, eb).abs() < 1e-13);

                    // Right-handed: (ea x eb) . xyz > 0
                    let normal = cross3(ea, eb);
                    assert!(dot3(normal, xyz) > 0.0);

                    // Analytical Jacobian matches metric determinant sqrt_g
                    let m = proj.point_metric(face, a, b, radius);
                    let j_ana = EquidistantProjection::analytical_jacobian(a, b, radius);
                    assert!(
                        (m.sqrt_g - j_ana).abs() < 1e-13,
                        "face {:?} ({a},{b}): sqrt_g={} vs j_ana={}",
                        face,
                        m.sqrt_g,
                        j_ana
                    );

                    // Biorthogonality of contravariant basis: e^i . e_j = delta_{ij}
                    assert!((dot3(m.contra_alpha, ea) - 1.0).abs() < 1e-13);
                    assert!(dot3(m.contra_alpha, eb).abs() < 1e-13);
                    assert!(dot3(m.contra_beta, ea).abs() < 1e-13);
                    assert!((dot3(m.contra_beta, eb) - 1.0).abs() < 1e-13);
                }
            }
        }
    }

    #[test]
    fn test_panel_connectivity_edges_match_in_3d() {
        let proj = EquiangularProjection;
        let radius = 1.0;
        let pi4 = std::f64::consts::FRAC_PI_4;
        let ts = [-0.7, -0.3, 0.0, 0.4, 0.75];

        let eval_edge = |face: CubeFace, side: EdgeSide, t: f64| -> [f64; 3] {
            let (a, b) = match side {
                EdgeSide::West => (-pi4, t),
                EdgeSide::East => (pi4, t),
                EdgeSide::South => (t, -pi4),
                EdgeSide::North => (t, pi4),
            };
            proj.point_and_tangents(face, a, b, radius).0
        };

        for face in CubeFace::ALL {
            for side in EdgeSide::ALL {
                let conn = PANEL_CONNECTIVITY[face.index()][side.index()];
                for &t in &ts {
                    let p1 = eval_edge(face, side, t);
                    let t_nbr = if conn.reverse { -t } else { t };
                    let p2 = eval_edge(conn.nbr_face, conn.nbr_side, t_nbr);
                    let dist = ((p1[0] - p2[0]).powi(2)
                        + (p1[1] - p2[1]).powi(2)
                        + (p1[2] - p2[2]).powi(2))
                    .sqrt();
                    assert!(
                        dist < 1e-14,
                        "mismatch on {:?} {:?} -> {:?} {:?} rev={}: dist={dist}",
                        face,
                        side,
                        conn.nbr_face,
                        conn.nbr_side,
                        conn.reverse
                    );
                }
            }
        }
    }
}
