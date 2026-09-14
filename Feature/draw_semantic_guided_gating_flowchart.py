from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(r"C:\Code\SVDD\Feature")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_box(ax, x, y, w, h, text, fc, ec="#2f2f2f", lw=1.2, fontsize=10, rounded=0.10):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.015,rounding_size={rounded}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return box


def add_arrow(ax, start, end, color="#3a3a3a", lw=1.4, style="-|>", mutation_scale=12, connectionstyle="arc3"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    return arrow


def add_label(ax, x, y, text, color="#4a4a4a", fontsize=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)


def draw_flowchart() -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    colors = {
        "input": "#e9f2ff",
        "acoustic": "#ffe9d6",
        "semantic": "#e7f7e8",
        "gate": "#fff2b8",
        "fusion": "#f1e6ff",
        "output": "#f3f4f6",
        "accent": "#8b0000",
    }

    add_box(ax, 0.5, 2.75, 1.7, 1.1, "Input Audio\nWaveform", colors["input"])

    add_box(ax, 3.0, 4.2, 2.2, 1.1, "PC-CQT\nFront-End", colors["acoustic"])
    add_box(ax, 3.0, 1.55, 2.2, 1.1, "WavLM\nEncoder", colors["semantic"])

    add_box(ax, 6.0, 4.2, 2.2, 1.1, "Acoustic Feature\nMap  A", colors["acoustic"])
    add_box(ax, 6.0, 1.55, 2.2, 1.1, "Semantic Embedding\nS", colors["semantic"])

    add_box(ax, 9.0, 1.55, 2.4, 1.1, "Linear Projection\n+ Sigmoid", colors["gate"])
    add_box(ax, 9.0, 4.2, 2.4, 1.1, "Channel-wise Gate\nG = sigmoid(Ws)", colors["gate"])

    add_box(ax, 12.15, 4.2, 1.8, 1.1, "Element-wise\nGating", colors["fusion"])
    add_box(ax, 12.15, 2.45, 1.8, 1.1, "Residual /\nFusion", colors["fusion"])

    add_box(ax, 14.55, 2.45, 1.2, 1.1, "Classifier", colors["output"])

    add_arrow(ax, (2.2, 3.3), (3.0, 4.75))
    add_arrow(ax, (2.2, 3.3), (3.0, 2.1))

    add_arrow(ax, (5.2, 4.75), (6.0, 4.75))
    add_arrow(ax, (5.2, 2.1), (6.0, 2.1))

    add_arrow(ax, (8.2, 2.1), (9.0, 2.1))
    add_arrow(ax, (11.4, 2.1), (11.4, 4.2), connectionstyle="arc3,rad=0.0")
    add_arrow(ax, (8.2, 4.75), (9.0, 4.75))

    add_arrow(ax, (11.4, 4.75), (12.15, 4.75))
    add_arrow(ax, (13.95, 4.75), (13.95, 3.0), connectionstyle="arc3,rad=0.0")
    add_arrow(ax, (8.2, 4.75), (12.15, 4.75), color=colors["accent"], lw=1.7)

    add_arrow(ax, (13.95, 3.0), (12.15, 3.0), color=colors["accent"], lw=1.7)
    add_arrow(ax, (13.95, 3.0), (14.55, 3.0))

    add_label(ax, 10.25, 5.55, "semantic guidance")
    add_label(ax, 10.95, 3.4, "gate generation")
    add_label(ax, 12.95, 5.55, "A x G", color=colors["accent"], fontsize=10)
    add_label(ax, 13.05, 2.0, "fused representation", fontsize=9)

    ax.text(
        8.0,
        6.55,
        "Semantic-Guided Gating Fusion Module",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    ax.text(
        8.0,
        0.5,
        "The semantic branch predicts channel-wise gates to suppress non-vocal interference and refine phase-aware acoustic cues.",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        out_path = OUTPUT_DIR / f"semantic_guided_gating_flowchart.{suffix}"
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        print(out_path)
    plt.close(fig)


if __name__ == "__main__":
    draw_flowchart()
