"""LaTeX bodies of the benchmark tables: Table 5 (reproduction), Table 7 (summary of E1/E2a/E2b), Table 8
(rotation and change of units), Table 15 (E7), and Appendix Tables A1-A3 and B1.  Also numbers_bench.json."""
import json
import os
import numpy as np
from scipy import stats
import bench as B
import paperdata as P
from common import fstar
from analysis_bench import load, summary, changed, mean_matrix, repro_table, write, full_table, ORIG6, ALL11, \
    FIDS30, FIDS14, RES
from stats_utils import m, mw, wtl_u

ROT = ["E1", "E2a", "E2b", "E2c", "E2d", "E2e_lo", "E2e_hi"]


def wil_cell(w):
    p_raw, p_adj, rej = w
    return p_adj


def summary_table():
    S = {e: summary(e, ALL11, FIDS30 if e != "E2b" else FIDS14) for e in ("E1", "E2a", "E2b")}
    ch = {a: changed("E1", "E2a", a, FIDS30) for a in ALL11}
    lines = []
    num = {}
    for a in ALL11:
        cells = ["%.2f" % S[e]["ranks"][a] for e in ("E1", "E2a", "E2b")]
        if a == "GPC":
            cells += ["--"] * 6
        else:
            cells += ["%d/%d/%d" % S[e]["wtl"][a] for e in ("E1", "E2a", "E2b")]
            for e in ("E1", "E2a", "E2b"):
                p_raw, p_adj, rej = S[e]["wilcoxon"][a]
                Mx = S[e]["M"]
                g, j = ALL11.index("GPC"), ALL11.index(a)
                d = Mx[:, g] - Mx[:, j]
                nz = d != 0
                r = stats.rankdata(np.abs(d[nz]))
                better = r[d[nz] < 0].sum() > r[d[nz] > 0].sum()
                sym = ("$+$" if better else "$-$") if rej else "$=$"
                cells.append("%s %s" % (sym, ("(%s)" % ("%.3f" % p_adj if p_adj >= 0.001 else "$<$0.001"))))
        cells.append("%d" % ch[a])
        lines.append("%s & %s\\\\" % (a, " & ".join(cells)))
    ff = " & ".join("%.1f" % S[e]["fr"]["FF"] for e in ("E1", "E2a", "E2b"))
    hb = []
    for e in ("E1", "E2a", "E2b"):
        h = S[e]["holm"]
        hb.append("%d/%d" % (sum(v["reject"] and v["control_better"] for v in h.values()),
                             sum(v["reject"] and not v["control_better"] for v in h.values())))
    num["FF"] = {e: S[e]["fr"]["FF"] for e in ("E1", "E2a", "E2b")}
    num["pFF"] = {e: S[e]["fr"]["pFF"] for e in ("E1", "E2a", "E2b")}
    num["holm"] = dict(zip(("E1", "E2a", "E2b"), hb))
    num["ranks"] = {e: S[e]["ranks"] for e in ("E1", "E2a", "E2b")}
    num["wtl"] = {e: S[e]["wtl"] for e in ("E1", "E2a", "E2b")}
    num["wilcoxon"] = {e: {a: S[e]["wilcoxon"][a] for a in S[e]["wilcoxon"]} for e in ("E1", "E2a", "E2b")}
    num["changed"] = ch
    return lines, ff, hb, num


def six_ranks():
    out = {}
    for e in ("E1", "E2a", "E2b"):
        errs, _ = load(e)
        f = FIDS30 if e != "E2b" else FIDS14
        M = mean_matrix(errs, f, ORIG6)
        R = np.mean([stats.rankdata(r) for r in M], axis=0)
        out[e] = dict(zip(ORIG6, R))
        g = ORIG6.index("GPC")
        out[e + "_best"] = int(sum(stats.rankdata(r)[g] == 1 for r in M))
        out[e + "_lowest"] = int(sum(r[g] < np.delete(r, g).min() for r in M))
        out[e + "_highest"] = int(sum(r[g] > np.delete(r, g).max() for r in M))
    return out


def rot_table():
    """Mean Friedman ranks (11 algorithms, 14 functions) per experiment; functions with a significant change
    of the error distribution with respect to E1 for E2c, E2e_lo, E2e_hi."""
    S = {}
    for e in ROT:
        try:
            S[e] = summary(e, ALL11, FIDS14)
        except KeyError:
            return None, None
    ch = {e: {a: changed("E1", e, a, FIDS14) for a in ALL11} for e in ("E2a", "E2c", "E2e_lo", "E2e_hi")}
    lines = []
    for a in ALL11:
        cells = ["%.2f" % S[e]["ranks"][a] for e in ROT] + ["%d" % ch[e][a] for e in ("E2a", "E2c", "E2e_lo", "E2e_hi")]
        lines.append("%s & %s\\\\" % (a, " & ".join(cells)))
    num = dict(ranks={e: S[e]["ranks"] for e in ROT}, changed=ch,
               wtl={e: S[e]["wtl"] for e in ROT}, FF={e: S[e]["fr"]["FF"] for e in ROT},
               holm={e: {a: (v["reject"], v["control_better"]) for a, v in S[e]["holm"].items()} for e in ROT})
    # GPC medians on selected functions
    med = {}
    for e in ROT:
        errs, _ = load(e)
        med[e] = {f: float(np.median(errs[f]["GPC"])) for f in FIDS14}
    num["gpc_median"] = med
    return lines, num


def repro_lines():
    errs, E = load("E1")
    rows, flags, nums = repro_table()
    lines = []
    for f in FIDS30:
        pm, ps = P.T3[f][5]
        g = np.array([r["best_f"] for r in E[f]["GPC"]])
        ga = np.array([r["best_f"] for r in E[f]["GPC-A"]])
        nfe = B.get(f)["nfe"]
        lines.append("%d & $10^{%d}$ & %s $\\pm$ %s & %s $\\pm$ %s & %s $\\pm$ %s & %s\\\\" % (
            f, int(round(np.log10(nfe))), m(pm, 5), m(ps, 2), m(g.mean(), 5), m(g.std(ddof=1), 2),
            m(ga.mean(), 5), m(ga.std(ddof=1), 2), flags[f]))
    return lines, flags, nums


def comparators_vs_published():
    """Within one order of magnitude of the published mean errors (five comparators)."""
    errs, _ = load("E1")
    within, pub_better, ours_better = 0, [], []
    for f in FIDS30:
        D = B.get(f)["D"]
        for k, a in enumerate(ORIG6[:5]):
            e_pub = max(P.T3[f][k][0] - fstar(f, D), 0.0)
            e_our = errs[f][a].mean()
            r = np.log10((e_our + 1e-12) / (e_pub + 1e-12))
            if abs(r) <= 1:
                within += 1
            elif r > 1:
                pub_better.append((a, f))
            else:
                ours_better.append((a, f))
    return dict(within=within, pub_better=pub_better, ours_better=ours_better)


def theta14_lines():
    errs, E1 = load("E1")
    _, E2 = load("E2a")
    try:
        from io_utils import load_records
        e7 = load_records(os.path.join(RES, "E2a_E7.jsonl.gz"))
    except Exception:
        e7 = []
    t14 = {}
    for r in e7:
        if r["alg"] == "GPC14":
            t14.setdefault(r["fid"], []).append(max(r["err"], 0))
    lines = []
    sig = 0
    for f in FIDS30:
        pm, ps = P.T3[f][5]
        g10 = np.array([r["best_f"] for r in E1[f]["GPC"]])
        g14 = np.array([r["best_f"] for r in E1[f]["GPC14"]])
        sig += mw(g10, g14)[0] < 0.05
        e10 = np.median([max(r["err"], 0) for r in E2[f]["GPC"]])
        e14 = np.median(t14[f]) if f in t14 else np.nan
        lines.append("%d & %s $\\pm$ %s & %s $\\pm$ %s & %s $\\pm$ %s & %s & %s\\\\" % (
            f, m(pm, 5), m(ps, 2), m(g10.mean(), 5), m(g10.std(ddof=1), 2), m(g14.mean(), 5), m(g14.std(ddof=1), 2),
            m(e10, 3), m(e14, 3) if f in t14 else "--"))
    return lines, sig


COMPLEXITY = {
    "GA": ("$O(nD+n\\log n)$", "$O(nD)$"), "ICA": ("$O(nD)$", "$O(nD)$"), "PSO": ("$O(nD)$", "$O(nD)$"),
    "DE": ("$O(nD)$", "$O(nD)$"), "FA": ("$O(n^2D)$", "$O(nD)$"), "GPC": ("$O(nD+n\\log n)$", "$O(nD)$"),
    "GPC-A": ("$O(nD+n\\log n)$", "$O(nD)$"), "CMA-ES": ("$O(\\lambda D^2+D^3)$", "$O(D^2)$"),
    "L-SHADE": ("$O(ND+N\\log N)$", "$O(ND)$"), "jSO": ("$O(ND+N\\log N)$", "$O(ND)$"), "RS": ("$O(D)$ per point", "$O(D)$"),
}


def cost_table():
    """Median CPU time per run (E1, the seven 30-dimensional functions with 10^4 evaluations)."""
    _, E = load("E1")
    fids = [f for f in FIDS30 if B.get(f)["D"] == 30 and B.get(f)["nfe"] == 10000]
    lines, num = [], {}
    for a in ALL11:
        t = np.median([r["cpu"] for f in fids for r in E[f][a]])
        num[a] = float(t)
        lines.append("%s & %s & %s & %.0f\\\\" % (a, COMPLEXITY[a][0], COMPLEXITY[a][1], 1000 * t))
    return lines, num, fids


if __name__ == "__main__":
    num = {}
    l, flags, rn = repro_lines()
    write("tab_repro.tex", l)
    num["repro_flags"] = flags
    l, ff, hb, sn = summary_table()
    l.append("\\midrule")
    pmax = max(sn["pFF"].values())
    l.append("Iman--Davenport $F_F$ & %s & \\multicolumn{7}{l}{($p<10^{%d}$ in all three experiments)}\\\\" % (
        ff, int(np.floor(np.log10(pmax))) + 1))
    l.append("Holm: GPC better/worse & & & & %s & & & & \\\\" % " & ".join(hb))
    write("tab_summary.tex", l)
    cl, cnum, cf = cost_table()
    write("tab_cost.tex", cl)
    num["cost"] = cnum
    num["summary"] = sn
    num["summary_holm_row"] = hb
    num["six"] = six_ranks()
    num["comparators"] = comparators_vs_published()
    for e in ("E1", "E2a"):
        write("tab_%sfull.tex" % e, full_table(e, ALL11, FIDS30))
    write("tab_E2bfull.tex", full_table("E2b", ALL11, FIDS14))
    rl, rnum = rot_table()
    if rl:
        write("tab_rot.tex", rl)
        num["rot"] = rnum
        for e in ("E2c", "E2d", "E2e_lo", "E2e_hi"):
            write("tab_%sfull.tex" % e, full_table(e, ALL11, FIDS14))
    tl, sig = theta14_lines()
    write("tab_theta14.tex", tl)
    num["theta14_sig"] = int(sig)
    json.dump(num, open(os.path.join(RES, "numbers_bench.json"), "w"), indent=1, default=lambda x: x if not isinstance(x, np.generic) else x.item())
    print(json.dumps({k: v for k, v in num.items() if k in ("six", "comparators", "summary_holm_row", "theta14_sig")}, indent=1, default=str))
