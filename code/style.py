"""Shared figure style: fixed colour + marker + line style per algorithm (colour follows the entity)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {
    "PSO": "#2a78d6", "GPC": "#eb6834", "DE": "#1baf7a", "GA": "#eda100", "GPC-A": "#e87ba4",
    "ICA": "#008300", "CMA-ES": "#4a3aa7", "FA": "#e34948", "RS": "#7a7975",
    "k-means": "#52514e", "k-medians": "#0b0b0b",
}
MK = {"PSO": "o", "GPC": "s", "DE": "^", "GA": "v", "GPC-A": "D", "ICA": "P", "CMA-ES": "X", "FA": "*", "RS": ".",
      "k-means": "<", "k-medians": ">"}
LS = {"PSO": "-", "GPC": "-", "DE": "--", "GA": ":", "GPC-A": "-.", "ICA": (0, (5, 1)), "CMA-ES": "-",
      "FA": (0, (3, 1, 1, 1)), "RS": ":", "k-means": "--", "k-medians": "-"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300,
        "figure.dpi": 150,
    })
