from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
BASE = OUT_DIR / "mlp_27d_hkappa_nature_final"


def lighten(color, amount=0.66):
    rgba = np.array(color)
    rgba[:3] = (1 - amount) * rgba[:3] + amount * np.ones(3)
    return rgba


def add_arrow(ax, start, end, color="#111111", lw=1.45, scale=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=6,
        )
    )


def draw():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    ink = "#111111"
    blue = "#4556F0"
    blue_fill = "#C9CEFF"
    grey = "#8B8B8B"
    light = "#C6C6C6"

    fig = plt.figure(figsize=(7.08, 2.48), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1770)
    ax.set_ylim(585, 0)
    ax.axis("off")

    # Single, true 3x3 stencil. No offset copies or ghost layers.
    x0, y0 = 95, 104
    cell = 116
    size = 3 * cell
    vals = np.array([[0.45, 0.74, 1.22], [-0.53, -0.16, 0.45], [-1.49, -0.96, -0.17]])
    cmap = mpl.colormaps["coolwarm"]
    vmax = 1.5
    for r in range(3):
        for c in range(3):
            v = vals[r, c]
            ax.add_patch(
                Rectangle(
                    (x0 + c * cell, y0 + r * cell),
                    cell,
                    cell,
                    facecolor=lighten(cmap((v + vmax) / (2 * vmax)), 0.68),
                    edgecolor="white",
                    lw=0.9,
                    zorder=1,
                )
            )
            ax.text(
                x0 + (c + 0.5) * cell,
                y0 + (r + 0.5) * cell + 2,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=9.0,
                color=ink,
                zorder=3,
            )
    ax.add_patch(Rectangle((x0, y0), size, size, fill=False, ec=ink, lw=0.85, zorder=4))

    curve = MplPath(
        [(x0, y0 + 90), (x0 + 108, y0 + 98), (x0 + 248, y0 + 168), (x0 + size, y0 + size)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(curve, fill=False, lw=1.65, ec=ink, capstyle="round", zorder=5))

    ax.text(
        x0 + size / 2,
        y0 + size + 45,
        r"$\mathbf{x}=\mathrm{vec}\left[(\phi/h,n_x,n_y)_{3\times3}\right]\in\mathbb{R}^{27}$",
        ha="center",
        va="center",
        fontsize=8.2,
        color=ink,
    )

    # Input layer: every visible input node is treated identically.
    in_x = 610
    in_y = [128, 194, 260, 326, 392, 458]
    in_r = 15
    add_arrow(ax, (x0 + size + 56, y0 + size / 2), (in_x - 36, y0 + size / 2), lw=1.3, scale=11)
    for y in in_y:
        ax.add_patch(Circle((in_x, y), in_r, fc="#F2F2F2", ec=grey, lw=1.15, zorder=4))

    # Hidden MLP: four 140-neuron ReLU layers represented by repeated visible nodes.
    hidden_x = [815, 962, 1109, 1256]
    hidden_y = [124, 189, 254, 319, 384, 449]
    r = 21

    for iy in in_y:
        for hy in hidden_y:
            ax.plot([in_x + in_r, hidden_x[0] - r], [iy, hy], color=light, lw=0.42, alpha=0.72, zorder=0)

    for x_a, x_b in zip(hidden_x[:-1], hidden_x[1:]):
        for ya in hidden_y:
            for yb in hidden_y:
                ax.plot([x_a + r, x_b - r], [ya, yb], color=light, lw=0.42, alpha=0.72, zorder=0)

    for x in hidden_x:
        for y in hidden_y:
            ax.add_patch(Circle((x, y), r, fc=blue_fill, ec=blue, lw=1.45, zorder=4))

    ax.text(
        np.mean([hidden_x[0], hidden_x[-1]]),
        61,
        r"$4\times(140,\ \mathrm{ReLU})$",
        ha="center",
        va="center",
        fontsize=10.6,
        color=blue,
    )

    out_x, out_y = 1410, 287
    out_r = 18
    for hy in hidden_y:
        ax.plot([hidden_x[-1] + r, out_x - out_r], [hy, out_y], color=grey, lw=0.55, alpha=0.72, zorder=0)
    ax.add_patch(Circle((out_x, out_y), out_r, fc="white", ec=ink, lw=1.35, zorder=4))
    add_arrow(ax, (out_x + out_r + 7, out_y), (1518, out_y), lw=1.35, scale=11)
    ax.text(1551, out_y + 2, r"$h\kappa$", ha="left", va="center", fontsize=13.8, color=ink)

    fig.savefig(BASE.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(BASE.with_suffix(".svg"), facecolor="white")
    fig.savefig(BASE.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
    print(BASE.with_suffix(".png"))
    print(BASE.with_suffix(".svg"))
    print(BASE.with_suffix(".pdf"))
