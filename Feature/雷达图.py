from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(r"C:\Code\SVDD\Feature")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def close_loop(values: list[float]) -> list[float]:
    return values + values[:1]


def rmap(x: Sequence[float]) -> np.ndarray:
    # Square-root radial mapping: preserves order and compresses large values.
    return np.sqrt(np.asarray(x, dtype=float))


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["A09", "A10", "A11", "A12", "A13", "EER(A09-A13)"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles = angles + angles[:1]

    # Table 1 values (EER, %)
    data = {
        "B01 (LFCC)": [5.35, 2.92, 5.84, 29.47, 3.65, 11.37],
        "B02 (AASIST)": [6.72, 0.96, 3.59, 26.83, 0.95, 10.39],
        "I2R-ASTAR": [0.65, 0.51, 2.49, 4.57, 0.64, 2.22],
        "NBU_MISL": [0.13, 0.11, 0.94, 5.17, 0.10, 2.00],
        "Fosafer Speech": [0.23, 0.37, 0.06, 4.19, 0.07, 1.65],
        "OPERA-Net (Ours)": [0.20, 0.25, 0.09, 3.85, 0.06, 1.54],
    }

    # Visual hierarchy: weak baselines, highlighted winner and ours
    styles = {
        "B01 (LFCC)": dict(color="#9aa1ac", lw=1.2, ls="--", alpha=0.70, z=2),
        "B02 (AASIST)": dict(color="#7f8ea3", lw=1.2, ls="--", alpha=0.72, z=2),
        "I2R-ASTAR": dict(color="#86a97a", lw=1.4, ls="-.", alpha=0.80, z=3),
        "NBU_MISL": dict(color="#5d8aa8", lw=1.5, ls="-.", alpha=0.85, z=3),
        "Fosafer Speech": dict(color="#e29d2f", lw=1.9, ls="-", alpha=0.95, z=4),
        "OPERA-Net (Ours)": dict(color="#c53434", lw=2.1, ls="-", alpha=1.00, z=5),
    }

    fig, ax = plt.subplots(figsize=(6.2, 6.4), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, float(np.sqrt(30.0)))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")

    yticks_raw = [0.5, 1, 2, 5, 10, 20, 30]
    yticks = rmap(yticks_raw)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks_raw], color="#444444", fontsize=8.7)
    ax.set_rlabel_position(20)

    ax.yaxis.grid(True, color="#cfd4dc", lw=0.9)
    ax.xaxis.grid(True, color="#d8dde6", lw=0.9)
    ax.spines["polar"].set_color("#8e97a6")
    ax.spines["polar"].set_linewidth(1.0)

    for method, values in data.items():
        closed = rmap(close_loop(values))
        st = styles[method]
        ax.plot(
            angles,
            closed,
            color=st["color"],
            linewidth=st["lw"],
            linestyle=st["ls"],
            alpha=st["alpha"],
            zorder=st["z"],
            label=method,
        )
        if method in {"Fosafer Speech", "OPERA-Net (Ours)"}:
            ax.fill(angles, closed, color=st["color"], alpha=0.10 if "Ours" not in method else 0.14, zorder=st["z"] - 1)

    ax.set_title(
        "CtrSVDD Eval Set: EER Breakdown by Attack Type (sqrt radial mapping)\n(lower is better)",
        pad=22,
        fontweight="bold",
    )

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.7,
    )
    legend.get_frame().set_edgecolor("#b8bec8")
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(0.95)

    fig.text(
        0.5,
        0.028,
        "Data source: Table 1 (A09-A13 EER, %).",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4a4a4a",
    )
    fig.subplots_adjust(top=0.86, bottom=0.28, left=0.06, right=0.94)

    out_base = OUT_DIR / "ctrsvdd_radar_eer"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(out_base.with_suffix(".png"))
    print(out_base.with_suffix(".pdf"))
    print(out_base.with_suffix(".svg"))


if __name__ == "__main__":
    main()
