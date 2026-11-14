"""Convert a ParaView .pvd/.vtu time series into an equirectangular animated GIF.

Usage examples:
    python scripts/pvd_to_gif.py out/samples/deform_anim/frame.pvd -o out/samples/deform.gif
    python scripts/pvd_to_gif.py out/samples/deform_anim/frame.pvd --fps 8 --every 2 -o out/deform_fast.gif
"""

import argparse
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image


def load_vtu_frame(vtu_path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ASCII Points and the named PointData DataArray from a .vtu file."""
    root = ET.parse(vtu_path).getroot()

    pts_el = root.find(".//Points/DataArray")
    if pts_el is None or pts_el.text is None:
        raise ValueError(f"Missing <Points><DataArray> in {vtu_path}")
    if pts_el.attrib.get("NumberOfComponents") != "3":
        raise ValueError(f"Expected NumberOfComponents='3' in {vtu_path}")
    if pts_el.attrib.get("format", "ascii").lower() != "ascii":
        raise ValueError(f"Expected ASCII Points DataArray in {vtu_path}")

    pts = np.array(pts_el.text.split(), dtype=float).reshape(-1, 3)

    field_el = None
    for arr in root.findall(".//PointData/DataArray"):
        if arr.attrib.get("Name") == field:
            field_el = arr
            break
    if field_el is None or field_el.text is None:
        raise ValueError(f"Field '{field}' not found in {vtu_path}")
    if field_el.attrib.get("format", "ascii").lower() != "ascii":
        raise ValueError(f"Expected ASCII PointData '{field}' in {vtu_path}")

    values = np.array(field_el.text.split(), dtype=float)
    if len(values) != len(pts):
        raise ValueError(f"Field length {len(values)} != points count {len(pts)} in {vtu_path}")

    r = np.linalg.norm(pts, axis=1)
    lon = np.arctan2(pts[:, 1], pts[:, 0])
    lat = np.arcsin(np.clip(pts[:, 2] / r, -1.0, 1.0))
    return lon, lat, values


def load_pvd_frames(pvd_path: Path, field: str, every: int) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
    """Load timesteps and VTU frame arrays referenced by a .pvd collection."""
    root = ET.parse(pvd_path).getroot()
    datasets = root.findall(".//Collection/DataSet")
    if not datasets:
        raise ValueError(f"No <DataSet> entries found in {pvd_path}")

    frames = []
    for ds in datasets[::every]:
        timestep = float(ds.attrib["timestep"])
        vtu_path = pvd_path.parent / ds.attrib["file"]
        lon, lat, values = load_vtu_frame(vtu_path, field)
        frames.append((timestep, lon, lat, values))
    return frames


def periodic_pad(
    lon_deg: np.ndarray, lat_deg: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add ghost copies across the poles and the +-180 seam so the triangulation covers the full map.

    A point near a pole is mirrored to (lon +- 180, +-180 - lat), and a point near the seam is
    copied to lon -+ 360; the axes limits then crop the map back to [-180, 180] x [-90, 90].
    """
    polar = np.abs(lat_deg) > 70.0
    lon = np.concatenate([lon_deg, np.where(lon_deg[polar] > 0.0, lon_deg[polar] - 180.0, lon_deg[polar] + 180.0)])
    lat = np.concatenate([lat_deg, np.sign(lat_deg[polar]) * 180.0 - lat_deg[polar]])
    val = np.concatenate([values, values[polar]])

    east, west = lon > 150.0, lon < -150.0
    return (
        np.concatenate([lon, lon[east] - 360.0, lon[west] + 360.0]),
        np.concatenate([lat, lat[east], lat[west]]),
        np.concatenate([val, val[east], val[west]]),
    )


def render_frame(
    timestep: float,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    field: str,
    levels: np.ndarray | int,
    vmin: float,
    vmax: float,
) -> Image.Image:
    """Render one equirectangular tricontourf frame to an in-memory PIL RGB image."""
    lon_deg, lat_deg, values = periodic_pad(np.degrees(lon), np.degrees(lat), values)
    tri = mtri.Triangulation(lon_deg, lat_deg)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    cf = ax.tricontourf(tri, values, levels=levels, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(cf, ax=ax, shrink=0.88, pad=0.03)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"{field}  t = {timestep:.3f}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as img:
        return img.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a .pvd/.vtu series to an equirectangular animated GIF.")
    parser.add_argument("pvd", help="Path to input .pvd file.")
    parser.add_argument("-o", "--output", required=True, help="Output GIF file path.")
    parser.add_argument("--field", default="phi", help="PointData scalar field name (default: phi).")
    parser.add_argument("--fps", type=float, default=6.0, help="Frames per second (default: 6).")
    parser.add_argument("--every", type=int, default=1, help="Use every n-th frame (default: 1).")
    args = parser.parse_args()

    if args.every < 1:
        raise ValueError(f"--every must be >= 1, got {args.every}")
    if args.fps <= 0.0:
        raise ValueError(f"--fps must be > 0, got {args.fps}")

    frames = load_pvd_frames(Path(args.pvd), args.field, args.every)
    vmin = min(float(vals.min()) for _, _, _, vals in frames)
    vmax = max(float(vals.max()) for _, _, _, vals in frames)
    levels: np.ndarray | int = 30 if vmin == vmax else np.linspace(vmin, vmax, 31)

    images = [
        render_frame(t, lon, lat, vals, args.field, levels, vmin, vmax)
        for t, lon, lat, vals in frames
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000.0 / args.fps)))
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


if __name__ == "__main__":
    main()
