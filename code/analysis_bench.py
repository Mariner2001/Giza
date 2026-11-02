"""Analysis of the benchmark experiments E1, E2a-E2e and E7 (tables of Sects. 6.1, 6.2, 6.7 and Appendix A).

All quantities are computed from the per-run records in ../results; nothing is typed in by hand.
Writes LaTeX table bodies to ../results/tables/ and the numbers quoted in the text to ../results/numbers_bench.json.
"""
import json
import os
import numpy as np
import bench as B
import paperdata as P
from common import fstar
from io_utils import load_records
from stats_utils import m, pfmt, friedman, holm, holm_control, mw, wtl_u, wtl_median, wilcoxon_paired, sign_test, \
    welch_vs_summary

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TAB = os.path.join(RES, "tables")
ORIG6 = ["GA", "ICA", "PSO", "DE", "FA", "GPC"]
ORIG9 = ORIG6 + ["GPC-A", "CMA-ES", "RS"]
ALL11 = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "L-SHADE", "jSO", "RS"]
FIDS30 = list(range(1, 31))
FIDS14 = sorted(B.ORIGIN_OPT)
_cache = {}


def load(exp, tags=("", "shade")):
    """errors[fid][alg] -> array over runs (sorted by run), clipped at 0; also the raw records."""
    key = (exp, tags)
    if key in _cache:
        return _cache[key]
    recs = []
    for t in tags:
        p = os.path.join(RES, "%s%s.jsonl.gz" % (exp, "_" + t if t else ""))
        if os.path.exists(p):
            recs += load_records(p)
    E = {}
    for r in sorted(recs, key=lambda r: (r["fid"], r["alg"], r["run"])):
        E.setdefault(r["fid"], {}).setdefault(r["alg"], []).append(r)
    errs = {f: {a: np.maximum(np.array([x["err"] for x in v], float), 0.0) for a, v in d.items()} for f, d in E.items()}
    _cache[key] = (errs, E)
    return errs, E


def mean_matrix(errs, fids, algs):
    return np.array([[errs[f][a].mean() for a in algs] for f in fids])


def summary(exp, algs, fids, rule="u", control="GPC"):
    errs, _ = load(exp)
    M = mean_matrix(errs, fids, algs)
    fr = friedman(M)
    hc = holm_control(list(fr["ranks"]), algs, control, len(fids))
    wtl = {}
    wil = {}
    for a in algs:
        if a == control:
            continue
        res = [(wtl_u if rule == "u" else wtl_median)(errs[f][control], errs[f][a]) for f in fids]
        wtl[a] = (res.count("w"), res.count("t"), res.count("l"))
        wil[a] = wilcoxon_paired(M[:, algs.index(control)], M[:, algs.index(a)])
    others = [a for a in algs if a != control]
    adj, rej = holm([wil[a] for a in others])
    wil_holm = {a: (wil[a], adj[i], bool(rej[i])) for i, a in enumerate(others)}
    st = {a: sign_test(M[:, algs.index(control)], M[:, algs.index(a)]) for a in others}
    return dict(ranks=dict(zip(algs, fr["ranks"])), fr=fr, holm=hc, wtl=wtl, wilcoxon=wil_holm, sign=st, M=M)


def changed(expA, expB, alg, fids, alpha=0.05):
    a, _ = load(expA)
    b, _ = load(expB)
    return sum(mw(a[f][alg], b[f][alg])[0] < alpha for f in fids)


def repro_table():
    """Table 5: GPC column of Table 3 of the original study vs our GPC and GPC-A (E1)."""
    errs, E = load("E1")
    rows, flags, numbers = [], {}, {}
    for f in FIDS30:
        D = B.get(f)["D"]
        fs = fstar(f, D)
        pm, ps = P.T3[f][5]
        g = np.array([r["best_f"] for r in E[f]["GPC"]])
        ga = np.array([r["best_f"] for r in E[f]["GPC-A"]])

        def ok(x):
            p = welch_vs_summary(x, pm, ps, 50)
            close = abs(x.mean() - pm) <= 1e-3 * abs(pm) if pm != 0 else x.mean() == 0
            e_ours = max(x.mean() - fs, 0.0)
            e_pub = max(pm - fs, 0.0)
            mag = abs(np.log10((e_ours + 1e-12) / (e_pub + 1e-12))) <= 1
            return (p >= 0.01 or close) and mag, p
        okm, pmult = ok(g)
        oka, padd = ok(ga)
        fl = "M" if okm else ("A" if oka else "--")
        flags[f] = fl
        numbers[f] = dict(p_mult=pmult, p_add=padd)
        nfe = B.get(f)["nfe"]
        rows.append("%d & %s & %s & %s $\\pm$ %s & %s $\\pm$ %s & %s $\\pm$ %s & %s\\\\" % (
            f, "$10^{%d}$" % int(round(np.log10(nfe))), "$%d$" % 50 if False else "",
            m(pm, 5), m(ps, 2), m(g.mean(), 5), m(g.std(ddof=1), 2), m(ga.mean(), 5), m(ga.std(ddof=1), 2), fl))
    return rows, flags, numbers


def write(name, lines):
    os.makedirs(TAB, exist_ok=True)
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def full_table(exp, algs, fids, sig=2):
    errs, _ = load(exp)
    lines = []
    for f in fids:
        means = [errs[f][a].mean() for a in algs]
        best = min(means)
        cells = []
        for v in means:
            s = m(v, sig)
            cells.append("{\\boldmath%s}" % s if v == best else s)
        lines.append("%d & %s\\\\" % (f, " & ".join(cells)))
    return lines


if __name__ == "__main__":
    out = {}
    # ---------------------------------------------------------------- consistency with the submitted version
    for exp, fids in (("E1", FIDS30), ("E2a", FIDS30), ("E2b", FIDS14)):
        s9 = summary(exp, ORIG9, fids, rule="median")
        print(exp, "ranks(9):", {a: round(v, 2) for a, v in s9["ranks"].items()}, "FF %.1f" % s9["fr"]["FF"])
        print("   wtl(median rule):", s9["wtl"])
        s9u = summary(exp, ORIG9, fids, rule="u")
        print("   wtl(U rule):     ", s9u["wtl"])
        print("   Holm better/worse:", sum(h["reject"] and h["control_better"] for h in s9["holm"].values()),
              sum(h["reject"] and not h["control_better"] for h in s9["holm"].values()))
    print("changed E1 vs E2a:", {a: changed("E1", "E2a", a, FIDS30) for a in ORIG9})
    rows, flags, nums = repro_table()
    print("M:", sorted(f for f, v in flags.items() if v == "M"))
    print("A:", sorted(f for f, v in flags.items() if v == "A"))
    print("--:", sorted(f for f, v in flags.items() if v == "--"))
