"""Analysis of E4 (high-dimensional check) -> table + summary json."""
import json
import collections
import numpy as np
import bench as B
import paperdata as P
from io_utils import load_records
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

A = json.load(open("../results/audit.json"))
R = load_records("../results/E4_hd.jsonl.gz")
d = collections.defaultdict(list)
for r in R:
    d[(r["kind"], r["fid"], r["D"], r["alg"])].append(r)
FIDS = [1, 10, 14, 16, 17, 20, 21, 22, 23, 27, 28, 29, 30]


def m(x, sig=5, zero_tol=0.0):
    if x is None or not np.isfinite(x):
        return "--"
    if abs(x) <= zero_tol:
        return "$0$"
    ax = abs(x)
    if 1e-3 <= ax < 1e5:
        s = "%.*g" % (sig, x)
        if "e" in s:
            s = "%.0f" % x
    else:
        a, e = ("%.2e" % x).split("e")
        s = "%s\\mathrm{e}{%d}" % (a, int(e))
    return "$" + s.replace("-", "{-}") + "$"


f0 = {}
for fid in FIDS:
    for D in (5000, 10000):
        f0[(fid, D)] = float(B.get(fid)["f"](np.zeros((1, D)))[0])

rows = []
summ = {}
for D in (5000, 10000):
    rows.append("\\multicolumn{7}{@{}l}{$D=%d$}\\\\" % D)
    for fid in FIDS:
        pub = P.T7[fid]["GPC"][0 if D == 5000 else 1]
        def mv(kind, alg):
            v = np.array([r["best_f"] for r in d[(kind, fid, D, alg)]])
            return v.mean()
        g_o, g_t = mv("orig", "GPC"), mv("trans", "GPC")
        p_o, d_o = mv("orig", "PSO"), mv("orig", "DE")
        p_t, d_t = mv("trans", "PSO"), mv("trans", "DE")
        inbox = all(r["in_box"] for k in ("orig", "trans") for a in ("GPC", "PSO", "DE") for r in d[(k, fid, D, a)])
        summ["%d|%d" % (fid, D)] = dict(f0=f0[(fid, D)], pub=None if pub is None else pub[0], gpc_o=g_o, gpc_t=g_t,
                                        pso_o=p_o, de_o=d_o, pso_t=p_t, de_t=d_t, inbox=inbox,
                                        gpc_rms_o=float(np.median([r["rms_x"] for r in d[("orig", fid, D, "GPC")]])))
        rows.append("$f_{%d}$ %s & %s & %s & %s & %s & %s & %s\\\\" % (
            fid, B.get(fid)["name"], m(f0[(fid, D)], zero_tol=1e-12), "infeasible" if pub is None else m(pub[0]), m(g_o), m(g_t), m(p_o), m(d_o)))
json.dump(summ, open("../results/E4_summary.json", "w"), indent=1)

tex = r"""\begin{table*}[!tbp]
\caption{High-dimensional problems (mean objective value over 10 runs; budgets as in Table~7 of \cite{harifi2021gpc}). ``Published'': GPC in Table~7 of \cite{harifi2021gpc}. GPC (orig.) and GPC (trans.): our runs on the original and on the translated problems. The results of PSO and DE (original problems) are shown for reference; on the translated problems they differ significantly in only 3 of 52 cases (Sect.~\ref{sec:res:hd}). All final solutions of our runs lie inside the box.}\label{tab:hd}
\footnotesize
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lllllll@{}}
\toprule
Function & $f(\mathbf{0})$ & Published & GPC (orig.) & GPC (trans.) & PSO & DE\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
open("../paper/tables/tab_hd.tex", "w").write(tex)

# dummy-variable check (paired runs, see exp_dummy.py)
R2 = load_records("../results/E4b_dummy.jsonl.gz")
dum = {}
for fid in (2, 4, 15, 26):
    v = [r for r in R2 if r["fid"] == fid]
    a = np.array([r["best_f_2"] for r in v])
    dum[fid] = dict(pairs=len(v), identical=int(sum(r["identical"] for r in v)), mean=float(a.mean()),
                    sd=float(a.std(ddof=1)), median=float(np.median(a)), pub3=P.T3[fid][2], pub7=P.T7[fid]["PSO"][0])
    print("dummy f%d: identical %d/%d; PSO mean %.4g sd %.2g | published T3 %.4g+-%.2g, T7 %.4g+-%.2g" % (
        fid, dum[fid]["identical"], len(v), a.mean(), a.std(ddof=1), P.T3[fid][2][0], P.T3[fid][2][1],
        P.T7[fid]["PSO"][0][0], P.T7[fid]["PSO"][0][1]))
json.dump({str(k): v for k, v in dum.items()}, open("../results/E4b_dummy_summary.json", "w"), indent=1, default=float)
for k, v in summ.items():
    print(k, {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in v.items()})
