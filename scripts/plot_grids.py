"""Render 3D cubed-sphere element grids side by side (equiangular vs equidistant).

Usage examples:
    python scripts/plot_grids.py --k 4 --order 4 -o out/samples/grids.png
    python scripts/plot_grids.py --k 6 --order 0 --elev 25 --azim 40 -o grid_no_lgl.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAMPLES_PER_LINE = 60
FACE_SPECS = [
    (lambda x, y, o: (o, x, y), "#1f77b4"),     # Front
    (lambda x, y, o: (-x, o, y), "#ff7f0e"),    # Right
    (lambda x, y, o: (-x, -y, o), "#2ca02c"),   # Top
    (lambda x, y, o: (-o, -y, -x), "#d62728"),  # Back
    (lambda x, y, o: (y, -o, -x), "#9467bd"),   # Left
    (lambda x, y, o: (y, x, -o), "#8c564b"),    # Bottom
]


def project_face(
    face_idx: int, alpha: np.ndarray, beta: np.ndarray, projection: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map face-local (alpha, beta) arrays onto the unit sphere using the solver's exact face layout."""
    if projection == "equiangular":
        x, y = np.tan(alpha), np.tan(beta)
    elif projection == "equidistant":
        x, y = 4.0 * alpha / np.pi, 4.0 * beta / np.pi
    else:
        raise ValueError(f"Unknown projection: {projection}")

    vx, vy, vz = FACE_SPECS[face_idx][0](x, y, np.ones_like(x))
    inv_r = 1.0 / np.sqrt(1.0 + x * x + y * y)
    return vx * inv_r, vy * inv_r, vz * inv_r


def interior_lgl_coords(k: int, order: int) -> np.ndarray:
    """Return interior LGL node coordinates across [-pi/4, pi/4] for k elements of polynomial order p."""
    if order < 2:
        return np.empty(0, dtype=float)
    xi_int = np.polynomial.legendre.Legendre.basis(order).deriv().roots()
    half_w = (np.pi / 4.0) / k
    return np.concatenate([-np.pi / 4.0 + (2 * e + 1) * half_w + half_w * xi_int for e in range(k)])


def plot_split_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    view: np.ndarray,
    color: str,
    lw: float,
    alpha_front: float,
    alpha_back: float,
    z_front: float,
) -> None:
    """Draw a 3D curve split across the view horizon so front lines sit above the globe core."""
    dot = x * view[0] + y * view[1] + z * view[2]
    for mask, width, alpha, zorder in (
        (dot <= 0.02, lw * 0.85, alpha_back, 1),
        (dot >= -0.02, lw, alpha_front, z_front),
    ):
        if np.any(mask):
            ax.plot(
                np.where(mask, x, np.nan),
                np.where(mask, y, np.nan),
                np.where(mask, z, np.nan),
                color=color,
                linewidth=width,
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )


def draw_projection(ax: plt.Axes, projection: str, k: int, order: int, elev: float, azim: float) -> None:
    """Draw all 6 cubed-sphere faces with element boundaries and optional LGL lines."""
    el_rad, az_rad = np.deg2rad(elev), np.deg2rad(azim)
    view = np.array([np.cos(el_rad) * np.cos(az_rad), np.cos(el_rad) * np.sin(az_rad), np.sin(el_rad)])

    ps, ts = np.meshgrid(np.linspace(0.0, np.pi, 36), np.linspace(0.0, 2.0 * np.pi, 72))
    r_inner = 0.99
    ax.plot_surface(
        r_inner * np.sin(ps) * np.cos(ts),
        r_inner * np.sin(ps) * np.sin(ts),
        r_inner * np.cos(ps),
        color="#f8f9fa", alpha=0.82, linewidth=0, shade=False, zorder=2,
    )

    boundaries = np.linspace(-np.pi / 4.0, np.pi / 4.0, k + 1)
    lgl_int = interior_lgl_coords(k, order)
    sweep = np.linspace(-np.pi / 4.0, np.pi / 4.0, SAMPLES_PER_LINE)

    for face_idx, (_, color) in enumerate(FACE_SPECS):
        for coords, lw, af, ab, zf in ((lgl_int, 0.55, 0.42, 0.16, 3), (boundaries, 1.45, 1.0, 0.30, 4)):
            for val in coords:
                const = np.full_like(sweep, val)
                for a, b in ((const, sweep), (sweep, const)):
                    x, y, z = project_face(face_idx, a, b, projection)
                    plot_split_curve(ax, x, y, z, view, color, lw, af, ab, zf)

    ax.set_xlim(-0.82, 0.82)
    ax.set_ylim(-0.82, 0.82)
    ax.set_zlim(-0.82, 0.82)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(projection.capitalize(), fontsize=13, y=0.95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw 3D cubed-sphere element grids for both projections.")
    parser.add_argument("--k", type=int, default=4, help="Elements per face edge (default: 4).")
    parser.add_argument("--order", type=int, default=4, help="Polynomial order for LGL lines (0 = none, default: 4).")
    parser.add_argument("-o", "--output", required=True, help="Output PNG file path.")
    parser.add_argument("--elev", type=float, default=20.0, help="3D view elevation angle (default: 20).")
    parser.add_argument("--azim", type=float, default=35.0, help="3D view azimuth angle (default: 35).")
    args = parser.parse_args()

    if args.k < 1:
        raise ValueError(f"--k must be >= 1, got {args.k}")
    if args.order < 0:
        raise ValueError(f"--order must be >= 0, got {args.order}")

    fig = plt.figure(figsize=(10.4, 5.2))
    for i, proj in enumerate(("equiangular", "equidistant"), start=1):
        ax = fig.add_subplot(1, 2, i, projection="3d", computed_zorder=False)
        draw_projection(ax, proj, args.k, args.order, args.elev, args.azim)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.02)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
