"""Analysis of E3 (structural bias): table + figure."""
import json
import collections
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import style
from io_utils import load_records
from style import COL
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

R = load_records("../results/E3_bias.jsonl.gz")
d = collections.defaultdict(list)
for r in R:
    d[(r["box"], r["alg"])].append(r["xn"])
ALGS = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "RS"]
origin = {"centre": 0.5, "corner": 0.0}
out = {}
rows = []
for a in ALGS:
    cells = []
    for box in ("centre", "corner"):
        X = np.array(d[(box, a)])  # runs x D
        first = X[:, 0]
        ks = stats.kstest(first, "uniform")
        # pooled KS over all coordinates (dependent within runs; descriptive only)
        ksall = stats.kstest(X.ravel(), "uniform")
        near0 = float(np.mean(np.abs(X - origin[box]) <= 0.01))
        atb = float(np.mean((X <= 1e-12) | (X >= 1 - 1e-12)))
        out[(box, a)] = dict(ks_D=float(ks.statistic), ks_p=float(ks.pvalue), ks_all_D=float(ksall.statistic),
                             near_origin=near0, at_bounds=atb)
        cells += ["%.2f" % ks.statistic, ("$<10^{-6}$" if ks.pvalue < 1e-6 else "%.2g" % ks.pvalue),
                  "%.0f" % (100 * near0), "%.0f" % (100 * atb)]
    rows.append(a + " & " + " & ".join(cells) + "\\\\")
json.dump({"%s|%s" % k: v for k, v in out.items()}, open("../results/E3_summary.json", "w"), indent=1)
for k, v in out.items():
    print(k, {kk: round(vv, 4) for kk, vv in v.items()})

tex = r"""\begin{table*}[!tbp]
\caption{Structural-bias test: final best solution of 100 runs ($D=30$, $10^4$ evaluations) on an objective that returns independent uniform random numbers. For an unbiased algorithm the normalised coordinates are uniform on $[0,1]$. KS: Kolmogorov--Smirnov statistic and $p$-value of the first coordinate against the uniform distribution; ``near $\mathbf{0}$'': percentage of coordinates within 1\% of the box width from the origin (expected: 2\% in the centred box, 1\% in the other); ``bound'': percentage of coordinates exactly on the boundary.}\label{tab:bias}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrr@{}}
\toprule
 & \multicolumn{4}{c}{box $[-5,5]^{30}$ (origin at the centre)} & \multicolumn{4}{c}{box $[0,10]^{30}$ (origin at a corner)}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Algorithm & KS & $p$ & near $\mathbf{0}$ (\%) & bound (\%) & KS & $p$ & near $\mathbf{0}$ (\%) & bound (\%)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
open("../paper/tables/tab_bias.tex", "w").write(tex)

style.setup()
show = ["GPC", "GPC-A", "PSO", "DE", "CMA-ES"]
fig, axs = plt.subplots(2, len(show), figsize=(6.3, 3.0), sharex=True)
bins = np.linspace(0, 1, 26)
for i, box in enumerate(("centre", "corner")):
    for j, a in enumerate(show):
        ax = axs[i, j]
        X = np.array(d[(box, a)]).ravel()
        h, e = np.histogram(X, bins=bins)
        frac = h / X.size
        ax.bar(0.5 * (e[1:] + e[:-1]), frac, width=(e[1] - e[0]) * 0.92, color=COL[a], edgecolor="none")
        ax.axhline(1 / 25, color=style.INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(5e-4, 1.2)
        ax.axvline(origin[box], color=style.INK, lw=0.6, ls="--")
        if i == 0:
            ax.set_title(a, loc="left")
        if j == 0:
            ax.set_ylabel(("origin at centre" if box == "centre" else "origin at corner") + "\nfraction", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
fig.supxlabel("normalised coordinate of the final solution", fontsize=8.5, y=0.045)
fig.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0.05, 1, 1))
fig.savefig("../paper/figs/fig_bias.pdf")
fig.savefig("../figs/fig_bias.png", dpi=200)
print("ok")
