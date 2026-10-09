"""LaTeX tables and figures for E1/E2a/E2b."""
import json
import collections
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import bench as B
import paperdata as P
import style
from style import COL, MK
from common import fstar
from analysis_bench import load, vals, errs, welch_vs_paper
from io_utils import load_records
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

S = json.load(open("../results/bench_summary.json"))
TAB = "../paper/tables/"
ALGS = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "RS"]


def m(x, sig=3):
    """math-mode compact number"""
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "$0$"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        s = "%.*g" % (sig, x)
        if "e" in s:
            s = "%.0f" % x
    else:
        a, e = ("%.*e" % (min(sig, 3) - 1, x)).split("e")
        s = "%s\\mathrm{e}{%d}" % (a, int(e))
    return "$" + s.replace("-", "{-}") + "$"


# ------------------------------------------------------------------ reproduction table
E1 = load("E1")
rows = []
verdicts = {}
for fid in range(1, 31):
    info = B.get(fid)
    fs = fstar(fid, info["D"])
    pm, ps = P.T3[fid][5]
    perr = max(pm - fs, 0.0)
    cells = []
    vd = []
    for a, tag in (("GPC", "M"), ("GPC14", None), ("GPC-A", "A")):
        v = vals(E1, fid, a)
        mu, sd = v.mean(), v.std(ddof=1)
        p = welch_vs_paper(v, pm, ps)
        oerr = max(mu - fs, 0.0)
        same_digits = abs(mu - pm) <= 1e-3 * max(1e-12, abs(pm))
        order_ok = abs(np.log10((oerr + 1e-12) / (perr + 1e-12))) <= 1
        ok = (p >= 0.01 or same_digits) and order_ok
        if tag and ok:
            vd.append(tag)
        cells.append("%s $\\pm$ %s" % (m(mu, 5), m(sd, 2)))
    verdicts[fid] = vd
    rows.append("%d & %s $\\pm$ %s & %s & %s & %s\\\\" % (
        fid, m(pm, 5), m(ps, 2), cells[0], cells[2], ",".join(vd) if vd else "--"))
tex = r"""\begin{table*}[!tbp]
\caption{Reproduction of the GPC column of Table~3 of \cite{harifi2021gpc} (function values, mean $\pm$ standard deviation over 50 runs; same dimensions and budgets; $\theta=10^\circ$, the results for $\theta=14^\circ$ being similar). ``M'' (``A''): the published values are consistent with the multiplicative (additive) update, i.e.\ the means differ by less than 0.1\% or not significantly (Welch test, $p\ge0.01$), and the mean errors agree within one order of magnitude.}\label{tab:repro}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}rlllc@{}}
\toprule
$f$ & Published (Table 3 of \cite{harifi2021gpc}) & GPC (ours) & GPC-A (ours) & Consistent with\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
open(TAB + "tab_repro.tex", "w").write(tex)
S["verdicts"] = verdicts
print("M:", [f for f, v in verdicts.items() if "M" in v])
print("A:", [f for f, v in verdicts.items() if "A" in v])
print("none:", [f for f, v in verdicts.items() if not v])

# ------------------------------------------------------------------ summary table (ranks, W/T/L, invariance)
fr = {e: S[e]["friedman"] for e in ("E1", "E2a", "E2b")}
holm_line = []
for e in ("E1", "E2a", "E2b"):
    h = fr[e]["holm"]
    nb = sum(1 for k, v in h.items() if v["rejected"] and v["better"])
    nw = sum(1 for k, v in h.items() if v["rejected"] and not v["better"])
    holm_line.append((nb, nw))
lines = []
for a in ALGS:
    r1 = S["E1"]["friedman"]["ranks"][a]
    r2 = S["E2a"]["friedman"]["ranks"][a]
    r3 = S["E2b"]["friedman"]["ranks"][a]
    cells = ["%.2f" % r1, "%.2f" % r2, "%.2f" % r3]
    for e in ("E1", "E2a", "E2b"):
        if a == "GPC":
            cells.append("--")
        else:
            w = S[e]["wtl"][a]
            cells.append("%d/%d/%d" % (w.get("+", 0), w.get("=", 0), w.get("-", 0)))
    cells.append("%d" % S["invariance"][a]["n_sig"])
    lines.append(a + " & " + " & ".join(cells) + "\\\\")
tex = r"""\begin{table*}[!tbp]
\caption{Summary of experiments E1 (original problems, 30 functions), E2a (translated problems, 30 functions) and E2b (shifted optimum, 14 functions). Left: mean Friedman ranks of the mean errors (1 = best). Middle: wins/ties/losses of GPC against each algorithm (Mann--Whitney test per function, $\alpha=0.05$). Right: number of functions on which the error distribution of E2a differs significantly from that of E1 ($\alpha=0.05$); for a translation-invariant algorithm about $1.5$ such functions are expected by chance.}\label{tab:summary}
\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrcccc@{}}
\toprule
 & \multicolumn{3}{c}{Mean Friedman rank} & \multicolumn{3}{c}{GPC wins/ties/losses} & E1 vs E2a\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}
Algorithm & E1 & E2a & E2b & E1 & E2a & E2b & changed\\
\midrule
""" + "\n".join(lines) + r"""
\midrule
Iman--Davenport $F_F$ & %.1f & %.1f & %.1f & \multicolumn{4}{l}{($p<10^{-12}$ in all three experiments)}\\
\multicolumn{4}{@{}l}{Holm test, GPC as control: significantly better/worse than} & %d/%d & %d/%d & %d/%d & \\
\bottomrule
\end{tabular*}
\end{table*}
""" % (fr["E1"]["FF"], fr["E2a"]["FF"], fr["E2b"]["FF"], *holm_line[0], *holm_line[1], *holm_line[2])
open(TAB + "tab_summary.tex", "w").write(tex)
print("pFF", fr["E1"]["pFF"], fr["E2a"]["pFF"], fr["E2b"]["pFF"])
print("Holm (better, worse) significant:", holm_line)
for e in ("E1", "E2a", "E2b"):
    print(e, {k: (round(v["z"], 2), "%.2g" % v["p"], v["rejected"]) for k, v in fr[e]["holm"].items()})

# ------------------------------------------------------------------ appendix tables: mean (sd) errors
def full_table(exp, label, caption):
    d = load(exp)
    fids = sorted(d.keys())
    out = []
    for fid in fids:
        means = {a: errs(d, fid, a).mean() for a in ALGS}
        best = min(means.values())
        cells = []
        for a in ALGS:
            x = errs(d, fid, a)
            s = m(x.mean(), 2)
            if np.isclose(x.mean(), best, rtol=1e-9, atol=1e-300):
                s = "\\textbf{" + s + "}"
            cells.append(s)
        out.append("%d & " % fid + " & ".join(cells) + "\\\\")
    return (r"""\begin{table*}[!htbp]
\caption{%s}\label{%s}
\scriptsize\setlength{\tabcolsep}{2pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}r%s@{}}
\toprule
$f$ & %s\\
\midrule
""" % (caption, label, "l" * len(ALGS), " & ".join(ALGS))) + "\n".join(out) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""


tex = full_table("E1", "tab:E1full", "E1, original problems: mean error $f(\\mathbf{x}_{\\text{best}})-f^*$ over 50 runs (best mean in bold). $D$ and budgets as in Table~3 of \\cite{harifi2021gpc}.")
tex += full_table("E2a", "tab:E2afull", "E2a, translated problems: mean error over 50 runs (best mean in bold).")
tex += full_table("E2b", "tab:E2bfull", "E2b, functions with optimum at the origin after shifting the optimum inside the box: mean error over 50 runs (best mean in bold).")
open(TAB + "tab_full.tex", "w").write(tex)

json.dump(S, open("../results/bench_summary.json", "w"), indent=1, default=float)

# ------------------------------------------------------------------ figure: E1 -> E2a dumbbell
style.setup()
E2a = load("E2a")
floor = 1e-25
fig, axs = plt.subplots(1, 3, figsize=(6.3, 5.3), sharey=True)
order = list(range(1, 31))
ylab = ["$f_{%d}$ %s" % (f, B.get(f)["name"]) for f in order]
for ax, a in zip(axs, ["GPC", "PSO", "CMA-ES"]):
    for k, fid in enumerate(order):
        y = len(order) - 1 - k
        x1 = max(np.median(errs(E1, fid, a)), floor)
        x2 = max(np.median(errs(E2a, fid, a)), floor)
        ax.plot([x1, x2], [y, y], color=COL[a], lw=1.0, alpha=0.6, zorder=1)
        ax.scatter([x1], [y], s=16, facecolors="white", edgecolors=COL[a], linewidths=1.0, zorder=2)
        ax.scatter([x2], [y], s=16, color=COL[a], zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(floor / 10, 1e7)
    ax.set_xticks([1e-24, 1e-16, 1e-8, 1])
    ax.set_title(a, loc="left", color=style.INK)
    ax.set_xlabel("median error")
axs[0].set_yticks(range(len(order)))
axs[0].set_yticklabels(ylab[::-1], fontsize=6.2)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", mfc="white", mec=style.INK2, label="E1: original problem"),
     Line2D([], [], marker="o", ls="", color=style.INK2, label="E2a: translated problem")]
fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.6, -0.025))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("../paper/figs/fig_translation.pdf")
fig.savefig("../figs/fig_translation.png", dpi=200)

# ------------------------------------------------------------------ figure: convergence curves
R1 = load_records("../results/E1.jsonl.gz")
R2 = load_records("../results/E2a.jsonl.gz")
cur = collections.defaultdict(list)
cps = {}
for tag, R in (("E1", R1), ("E2a", R2)):
    for r in R:
        fs = fstar(r["fid"], B.get(r["fid"])["D"])
        cur[(tag, r["fid"], r["alg"])].append(np.maximum(r["curve"] - fs, 0.0))
        cps[r["fid"]] = r["cps"]
fig, axs = plt.subplots(2, 2, figsize=(6.3, 3.9))
axs = axs.ravel()
for ax, fid in zip(axs, [27, 22, 23, 1]):
    for tag, a, ls, lab in (("E1", "GPC", "--", "GPC, original"), ("E2a", "GPC", "-", "GPC, translated"),
                            ("E2a", "PSO", "-", "PSO"), ("E2a", "DE", "-", "DE"), ("E2a", "CMA-ES", "-", "CMA-ES")):
        Y = np.median(np.array(cur[(tag, fid, a)]), axis=0)
        ax.plot(cps[fid], np.maximum(Y, 1e-25), ls=ls, color=COL[a], marker=MK[a], ms=2.5, markevery=4, lw=1.2, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("$f_{%d}$ %s" % (fid, B.get(fid)["name"]), loc="left")
    ax.set_xlabel("evaluations")
axs[0].set_ylabel("median error")
axs[2].set_ylabel("median error")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=7)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("../paper/figs/fig_convergence.pdf")
fig.savefig("../figs/fig_convergence.png", dpi=200)
print("figures saved")
