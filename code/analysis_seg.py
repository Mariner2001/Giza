"""Analysis of E6 (segmentation on BSDS500) -> table, figure, summary."""
import json
import collections
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import style
from style import COL
import exp_seg as S
from io_utils import load_records
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

R = load_records("../results/E6_seg.jsonl.gz")
d = collections.defaultdict(list)
for r in R:
    d[(r["img"], r["K"], r["alg"])].append(r)
imgs = sorted({r["img"] for r in R})
ALGS = ["k-means", "k-medians", "GPC", "PSO", "DE"]
KS = (3, 4, 5)
summ = {}
rows = []
for K in KS:
    best = {im: min(r["obj"] for a in ALGS for r in d[(im, K, a)]) for im in imgs}
    # per image means
    M = {a: np.array([np.mean([r["obj"] for r in d[(im, K, a)]]) for im in imgs]) for a in ALGS}
    rel = {a: 100 * (M[a] / np.array([best[im] for im in imgs]) - 1) for a in ALGS}
    pri = {a: np.array([np.mean([r["PRI"] for r in d[(im, K, a)]]) for im in imgs]) for a in ALGS}
    voi = {a: np.array([np.mean([r["VoI"] for r in d[(im, K, a)]]) for im in imgs]) for a in ALGS}
    cpu = {a: np.mean([r["cpu"] for im in imgs for r in d[(im, K, a)]]) for a in ALGS}
    Rk = np.array([stats.rankdata([M[a][i] for a in ALGS]) for i in range(len(imgs))]).mean(0)
    # GPC vs others: Wilcoxon signed-rank over images on mean objective
    wil = {a: float(stats.wilcoxon(M["GPC"], M[a]).pvalue) for a in ALGS if a != "GPC"}
    nworse = {a: int(np.sum(M["GPC"] > M[a])) for a in ALGS if a != "GPC"}
    summ[K] = dict(rel={a: float(np.mean(rel[a])) for a in ALGS}, rel_med={a: float(np.median(rel[a])) for a in ALGS},
                   pri={a: float(pri[a].mean()) for a in ALGS}, voi={a: float(voi[a].mean()) for a in ALGS},
                   cpu={a: float(cpu[a]) for a in ALGS}, rank=dict(zip(ALGS, Rk.tolist())), wilcoxon=wil, gpc_worse_on=nworse)
    for a in ALGS:
        rows.append("%s & %s & %.2f & %.1f & %.1f & %.3f & %.2f & %.2f\\\\" % (
            ("$K=%d$" % K) if a == ALGS[0] else "", a, Rk[ALGS.index(a)], np.mean(rel[a]), np.median(rel[a]),
            pri[a].mean(), voi[a].mean(), cpu[a]))
    if K != KS[-1]:
        rows.append("\\midrule")
json.dump({str(k): v for k, v in summ.items()}, open("../results/E6_summary.json", "w"), indent=1)
for K in KS:
    print(K, json.dumps(summ[K], indent=0)[:1500])

tex = r"""\begin{table*}[!tbp]
\caption{Colour segmentation of 20 BSDS500 test images (centroid search with the objective of Eq.~(10) of \cite{harifi2021gpc}; 10 runs per image; 5000 evaluations for the metaheuristics). Rank: mean Friedman rank of the mean objective over the images. Excess: percentage by which the mean objective exceeds the best value found for the image by any method (mean and median over images). PRI (higher is better) and VoI (lower is better) compare the segmentation with the human ground truth. CPU: mean time per run.}\label{tab:seg}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrrr@{}}
\toprule
 & Method & Rank & Excess, mean (\%) & Excess, median (\%) & PRI & VoI & CPU (s)\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
open("../paper/tables/tab_seg.tex", "w").write(tex)

# ---------------------------------------------------------------- example figure
style.setup()
S.init()
K = 4
# pick the image where GPC's excess over k-medians is the median one
ex = []
for im in imgs:
    g = np.mean([r["obj"] for r in d[(im, K, "GPC")]])
    km = np.mean([r["obj"] for r in d[(im, K, "k-medians")]])
    ex.append((g / km, im))
ex.sort()
im = ex[len(ex) // 2][1]
info = S.IMGS[im]
H, W = info["shape"]
obj = S.SegObjective(info["Z"], K)
fig, axs = plt.subplots(1, 4, figsize=(6.3, 1.65))
axs[0].imshow(info["Z"].reshape(H, W, 3).astype(np.uint8))
axs[0].set_title("image " + im.replace("bsds-", ""), loc="left", fontsize=6.2)
for ax, a in zip(axs[1:], ["GPC", "PSO", "k-medians"]):
    rr = min(d[(im, K, a)], key=lambda r: abs(r["obj"] - np.median([q["obj"] for q in d[(im, K, a)]])))
    c = rr["cent"].reshape(K, 3)
    lab = obj.labels(rr["cent"])
    ax.imshow(np.clip(c[lab], 0, 255).reshape(H, W, 3).astype(np.uint8))
    mant, ex = ("%.2e" % rr["obj"]).split("e")
    ax.set_title(r"%s: $%s\times10^{%d}$, PRI %.2f" % (a, mant, int(ex), rr["PRI"]), loc="left", fontsize=6.2)
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout(w_pad=0.3)
fig.savefig("../paper/figs/fig_seg.pdf")
fig.savefig("../figs/fig_seg.png", dpi=200)
print("example image", im)
