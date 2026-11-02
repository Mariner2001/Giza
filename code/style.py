"""Common figure style (serif fonts, colour-blind-safe palette, one colour and marker per algorithm)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"PSO": "#2a78d6", "GPC": "#eb6834", "DE": "#1baf7a", "GA": "#eda100", "GPC-A": "#e87ba4",
          "ICA": "#008300", "CMA-ES": "#4a3aa7", "FA": "#e34948", "RS": "#7a7975", "k-means": "#52514e",
          "k-medians": "#0b0b0b", "L-SHADE": "#8c564b", "jSO": "#17becf"}
MARKERS = {"PSO": "o", "GPC": "s", "DE": "^", "GA": "v", "GPC-A": "D", "ICA": "P", "CMA-ES": "X", "FA": "*",
           "RS": ".", "k-means": "<", "k-medians": ">", "L-SHADE": "h", "jSO": "p"}
WIDTH = 6.3


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8, "axes.titlesize": 8,
        "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3.5, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })
