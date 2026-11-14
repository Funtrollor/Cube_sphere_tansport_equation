"""Plot h-, p-, or t-refinement convergence curves from cs-cli CSV outputs.

Usage examples:
    python scripts/plot_convergence.py out/samples/h_conv_p3.csv -o out/samples/h_conv_p3.png
    python scripts/plot_convergence.py run1.csv run2.csv --labels Equiangular Equidistant -o cmp.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

EXPECTED_HEADER = [
    "k", "poly_order", "total_dofs", "dt", "steps", "elapsed_secs",
    "l1_rel", "l2_abs", "l2_rel", "order_l2", "l_inf", "order_linf", "mass_rel_err",
]


def load_csv(path: Path) -> dict:
    """Read a convergence CSV file and detect its refinement mode."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_HEADER:
            raise ValueError(f"Unexpected CSV header in {path}: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")

    k = np.array([int(r["k"]) for r in rows], dtype=int)
    p = np.array([int(r["poly_order"]) for r in rows], dtype=int)
    dt = np.array([float(r["dt"]) for r in rows], dtype=float)
    l2_rel = np.array([float(r["l2_rel"]) for r in rows], dtype=float)
    l_inf = np.array([float(r["l_inf"]) for r in rows], dtype=float)

    if len(np.unique(k)) > 1:
        mode = "h"
    elif len(np.unique(p)) > 1:
        mode = "p"
    elif len(np.unique(dt)) > 1:
        mode = "t"
    else:
        raise ValueError(f"Could not detect refinement mode in {path} (k, poly_order, dt all constant)")
    return {"mode": mode, "k": k, "p": p, "dt": dt, "l2_rel": l2_rel, "l_inf": l_inf}


def plot_convergence(datasets: list[dict], labels: list[str], title: str | None) -> plt.Figure:
    """Plot L2_rel and L_inf convergence curves with theoretical reference lines."""
    modes = {d["mode"] for d in datasets}
    if len(modes) != 1:
        raise ValueError(f"Cannot mix refinement modes in one figure: {sorted(modes)}")
    mode = datasets[0]["mode"]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    single = len(datasets) == 1
    seen_p_refs: set[int] = set()
    t_ref_drawn = False
    plot_fn = ax.semilogy if mode == "p" else ax.loglog
    x_key = {"h": "k", "p": "p", "t": "dt"}[mode]

    for idx, (data, label) in enumerate(zip(datasets, labels)):
        c_l2 = palette[0] if single else palette[idx % len(palette)]
        c_linf = palette[1] if single else c_l2
        ls_linf = "-" if single else ":"
        lbl_l2 = r"$L_2$ rel" if single else rf"{label} ($L_2$ rel)"
        lbl_linf = r"$L_\infty$" if single else rf"{label} ($L_\infty$)"
        x = data[x_key]

        plot_fn(x, data["l2_rel"], "o-", color=c_l2, label=lbl_l2)
        plot_fn(x, data["l_inf"], f"s{ls_linf}", color=c_linf, label=lbl_linf)

        if mode == "h":
            p_val = int(data["p"][0])
            if p_val not in seen_p_refs:
                seen_p_refs.add(p_val)
                y_ref = data["l2_rel"][0] * (x / x[0]) ** (-(p_val + 1))
                ax.loglog(x, y_ref, "--", color="0.35", lw=1.4, label=rf"$O(h^{{{p_val + 1}}})$")
        elif mode == "t" and not t_ref_drawn:
            t_ref_drawn = True
            y_ref = data["l2_rel"][0] * (x / x[0]) ** 4
            ax.loglog(x, y_ref, "--", color="0.35", lw=1.4, label=r"$O(dt^4)$")

    if mode == "h":
        uniq_k = np.unique(np.concatenate([d["k"] for d in datasets]))
        if len(uniq_k) <= 10:
            ax.set_xticks(uniq_k)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel("Elements per face edge K")
        default_title = f"h-Refinement Convergence (p = {datasets[0]['p'][0]})" if len(seen_p_refs) == 1 else "h-Refinement Convergence"
    elif mode == "p":
        uniq_p = np.unique(np.concatenate([d["p"] for d in datasets]))
        if len(uniq_p) <= 12:
            ax.set_xticks(uniq_p)
        ax.set_xlabel("Polynomial order p")
        uniq_k = {int(d["k"][0]) for d in datasets}
        default_title = f"p-Refinement Convergence (K = {next(iter(uniq_k))})" if len(uniq_k) == 1 else "p-Refinement Convergence"
    else:
        ax.set_xlabel("Time step dt")
        default_title = "Temporal Convergence"

    ax.set_ylabel("Error")
    ax.set_title(title if title is not None else default_title)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cubed-sphere convergence curves from CSV.")
    parser.add_argument("csv", nargs="+", help="One or more CSV files produced by cs-cli converge.")
    parser.add_argument("-o", "--output", required=True, help="Output PNG file path.")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels (one per CSV).")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.csv):
        raise ValueError(f"--labels count ({len(args.labels)}) must match CSV count ({len(args.csv)})")

    labels = args.labels if args.labels is not None else [Path(p).stem for p in args.csv]
    datasets = [load_csv(Path(p)) for p in args.csv]

    fig = plot_convergence(datasets, labels, args.title)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
