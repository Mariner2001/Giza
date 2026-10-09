"""LaTeX tables for the audit section (from results/audit.json and paperdata)."""
import json
import numpy as np
from scipy import stats
import bench as B
import paperdata as P
from common import FSTAR, fstar
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

A = json.load(open("../results/audit.json"))
TAB = "../paper/tables/"


def num(x, sig=4):
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e5:
        s = "%.*g" % (sig, x)
        if "e" in s:
            s = "%.0f" % x
    else:
        m, e = ("%.*e" % (sig - 2, x)).split("e")
        s = "%s\\mathrm{e}{%d}" % (m, int(e))
    return "$" + s.replace("-", "{-}") + "$" if True else s


# ------------------------------------------------------------------ Table: origin baseline
rows = []
f0 = {int(k): v for k, v in A["f_origin_D30"].items()}
for fid in range(1, 31):
    info = B.get(fid)
    means = np.array([m for (m, s) in P.T3[fid]])
    best5 = means[:5].min()
    gpc = means[5]
    r7 = stats.rankdata(np.r_[means, f0[fid]])[-1]
    origin_opt = "\\checkmark" if fid in B.ORIGIN_OPT else ""
    fs = fstar(fid, info["D"])
    star = ""
    if fid in (5, 9, 20, 25):
        star = "$^{*}$"
    rows.append("%d & %s & %d & %s%s & %s & %s & %s & %s & %s \\\\" % (
        fid, info["name"], info["D"], num(fs), star, origin_opt, num(f0[fid] if abs(f0[fid]) > 1e-15 else 0.0),
        num(gpc), num(best5), ("%.1f" % r7).replace(".0", "")))
tex = r"""\begin{table*}[!tbp]
\caption{The thirty functions of \cite{harifi2021gpc} (dimension $D$ as used in its Table~3), their optimal value $f^*$ on the stated domain (corrected values marked with $^{*}$, see Table~\ref{tab:errata}), whether the minimiser is the origin, the value $f(\mathbf{0})$ of the objective at the origin, the mean reported for GPC and the best mean reported for the five other algorithms in Table~3 of \cite{harifi2021gpc}, and the rank that the constant solution $\mathbf{x}=\mathbf{0}$ would obtain among the seven entries (1 = best; ties averaged).}\label{tab:origin}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular*}{\textwidth}{@{\extracolsep\fill}rlrlcllll@{}}
\toprule
$f$ & Function & $D$ & $f^*$ & $\mathbf{x}^*=\mathbf{0}$ & $f(\mathbf{0})$ & GPC & best other & rank of $\mathbf{0}$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
"""
open(TAB + "tab_origin.tex", "w").write(tex)

# ------------------------------------------------------------------ Table: statistics published vs recomputed
pub = A["ranks_as_published_reversed_f20"]["friedman"]
cor = A["ranks_corrected_minimisation"]["friedman"]
hp = {r["alg"]: r for r in A["ranks_as_published_reversed_f20"]["holm"]}
hc = {r["alg"]: r for r in A["ranks_corrected_minimisation"]["holm"]}
algs = P.ALGS
lines = []
lines.append("Mean Friedman rank (Table~4) & " + " & ".join("%.2f" % P.T4_RANKS[a] for a in algs) + " \\\\")
lines.append("\\quad recomputed, $f_{20}$ ranked as maximisation & " + " & ".join("%.2f" % v for v in pub["Rj"]) + " \\\\")
lines.append("\\quad recomputed, all functions minimised & " + " & ".join("%.2f" % v for v in cor["Rj"]) + " \\\\")
lines.append("Holm $z$ vs GPC (Table~6) & " + " & ".join(("%.3f" % P.T6[a][0]) if a in P.T6 else "--" for a in algs) + " \\\\")
lines.append("\\quad recomputed, all functions minimised & " + " & ".join(("%.3f" % hc[a]["z"]) if a in hc else "--" for a in algs) + " \\\\")
tex = r"""\begin{table}[t]
\caption{Statistical analysis of \cite{harifi2021gpc} (Tables~4--6 there) and its recomputation from the means in Table~3 of \cite{harifi2021gpc}. The published ranks are reproduced to all printed digits only if $f_{20}$ (Michalewicz) is ranked as a maximisation problem.}\label{tab:stats}
\footnotesize
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lrrrrrr@{}}
\toprule
 & GA & ICA & PSO & DE & FA & GPC\\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular*}

\medskip
\begin{tabular*}{\columnwidth}{@{\extracolsep\fill}lrrr@{}}
\toprule
Statistic & Published & Recomputed ($f_{20}$ reversed) & Recomputed (correct)\\
\midrule
Friedman $\chi^2_F$ & 80.08 & %.2f (tie-corr.\ %.2f) & %.2f (tie-corr.\ %.2f)\\
\quad $p$-value & $8.06\mathrm{e}{-16}$ & $%s$ (tie-corr.) & $%s$\\
Iman--Davenport $F_F$ & 31.69 & %.2f (tie-corr.\ %.2f) & %.2f (tie-corr.\ %.2f)\\
\quad degrees of freedom & 5 & (5, 145) & (5, 145)\\
\quad $p$-value & $2.20\mathrm{e}{-16}$ & $%s$ & $%s$\\
\bottomrule
\end{tabular*}
\end{table}
""" % (pub["chi2"], pub["chi2_tie"], cor["chi2"], cor["chi2_tie"],
       ("%.2e" % pub["p_chi2_tie"]).replace("e-", "\\mathrm{e}{-") + "}", ("%.2e" % cor["p_chi2"]).replace("e-", "\\mathrm{e}{-") + "}",
       pub["FF"], pub["FF_tie"], cor["FF"], cor["FF_tie"],
       ("%.2e" % pub["p_FF"]).replace("e-", "\\mathrm{e}{-") + "}", ("%.2e" % cor["p_FF"]).replace("e-", "\\mathrm{e}{-") + "}")
open(TAB + "tab_stats.tex", "w").write(tex)
print(open(TAB + "tab_stats.tex").read())
print(open(TAB + "tab_origin.tex").read()[:3000])
