"""Analysis of E5 (COCO/BBOB): empirical runtime distributions (ECDF over 51 targets) + table."""
import glob
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import style
from io_utils import load_records
from style import COL, MK, LS
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

res = []
for f in sorted(glob.glob("../results/E5_bbob_*.jsonl.gz")):
    res += load_records(f)
ALGS = ["GPC", "GPC-A", "PSO", "DE", "CMA-ES", "RS"]
DIMS = [2, 5, 10, 20]
targets = 10.0 ** np.arange(2, -8.01, -0.2)
by = collections.defaultdict(list)
for r in res:
    by[(r["alg"], r["D"])].append(r)


def runtimes(r):
    out = np.full(len(targets), np.inf)
    tf, tn = r["tf"], r["tn"]
    for k, t in enumerate(targets):
        idx = np.nonzero(tf <= t)[0]
        if idx.size:
            out[k] = tn[idx[0]]
    return out


summ = {}
style.setup()
fig, axs = plt.subplots(2, 2, figsize=(6.3, 3.9), sharey=True)
axs = axs.ravel()
for ax, D in zip(axs, DIMS):
    grid = np.logspace(0, np.log10(1e4 * D), 200)
    for a in ALGS:
        runs = by[(a, D)]
        if not runs:
            continue
        RT = np.array([runtimes(r) for r in runs])  # problems x targets
        ecdf = np.array([(RT <= b).mean() for b in grid])
        ax.plot(grid / D, ecdf, color=COL[a], ls=LS[a] if a != "GPC" else "-", lw=1.5 if a == "GPC" else 1.2,
                marker=MK[a], markevery=25, ms=3, label=a)
        solved = int(np.sum(RT[:, -1] < np.inf))
        per_fun = collections.defaultdict(list)
        for r, row in zip(runs, RT):
            per_fun[r["fun"]].append(np.mean(row < np.inf))
        summ["%s|%d" % (a, D)] = dict(final=float(ecdf[-1]), solved_1e8=solved, n=len(runs),
                                      per_fun={int(k): float(np.mean(v)) for k, v in per_fun.items()},
                                      median_xnorm=float(np.median([r["xnorm"] for r in runs])))
    ax.set_xscale("log")
    ax.set_title("$D=%d$" % D, loc="left")
    ax.set_xlabel("evaluations / $D$")
    ax.set_ylim(0, 1)
axs[0].set_ylabel("fraction of targets")
axs[2].set_ylabel("fraction of targets")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), fontsize=7)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig("../paper/figs/fig_bbob.pdf")
fig.savefig("../figs/fig_bbob.png", dpi=200)
json.dump(summ, open("../results/E5_summary.json", "w"), indent=1)

rows = []
for a in ALGS:
    cells = []
    for D in DIMS:
        s = summ.get("%s|%d" % (a, D))
        cells += ["%.2f" % s["final"], "%d" % s["solved_1e8"]] if s else ["--", "--"]
    rows.append(a + " & " + " & ".join(cells) + "\\\\")
tex = r"""\begin{table*}[!tbp]
\caption{COCO/BBOB noiseless suite (24 functions $\times$ 5 instances $=$ 120 problems per dimension; budget $10^4D$ evaluations; one run per problem). For each dimension: fraction of the 51 targets $f_{\mathrm{opt}}+10^{\{2,1.8,\dots,-8\}}$ reached at the end of the budget, aggregated over all problems (the final value of the curves in Fig.~\ref{fig:bbob}), and number of problems solved to precision $10^{-8}$.}\label{tab:bbob}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrrrr@{}}
\toprule
 & \multicolumn{2}{c}{$D=2$} & \multicolumn{2}{c}{$D=5$} & \multicolumn{2}{c}{$D=10$} & \multicolumn{2}{c}{$D=20$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Algorithm & targets & solved & targets & solved & targets & solved & targets & solved\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
open("../paper/tables/tab_bbob.tex", "w").write(tex)
for k, v in summ.items():
    print(k, round(v["final"], 3), v["solved_1e8"], v["n"])
