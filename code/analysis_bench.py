"""Analysis of E1/E2a/E2b: tables (LaTeX), statistics and JSON summaries."""
import json
import collections
import numpy as np
from scipy import stats
import bench as B
import paperdata as P
from common import fstar
from io_utils import load_records
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

RES = "../results/"
TAB = "../paper/tables/"
ALGS_MAIN = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "RS"]


def load(exp):
    R = load_records(RES + exp + ".jsonl.gz")
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        d[r["fid"]][r["alg"]].append(r)
    for fid in d:
        for a in d[fid]:
            d[fid][a].sort(key=lambda r: r["run"])
    return d


def errs(d, fid, alg):
    return np.array([max(r["err"], 0.0) for r in d[fid][alg]])


def vals(d, fid, alg):
    return np.array([r["best_f"] for r in d[fid][alg]])


def sci(x, digits=2):
    """compact LaTeX number"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    ax = abs(x)
    if 1e-3 <= ax < 1e4:
        s = "%.*g" % (digits + 1, x)
        return s.replace("-", "\\text{-}") if False else ("$%s$" % s if s.startswith("-") else s)
    m, e = ("%.*e" % (digits - 1, x)).split("e")
    return "$%s\\mathrm{e}{%d}$" % (m, int(e))


def friedman_holm(M, names, control):
    """M: (N problems, k algs) matrix of mean errors (lower better)."""
    R = np.array([stats.rankdata(row) for row in M])
    N, k = R.shape
    Rj = R.mean(0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(Rj ** 2) - k * (k + 1) ** 2 / 4)
    FF = (N - 1) * chi2 / (N * (k - 1) - chi2)
    pF = stats.f.sf(FF, k - 1, (k - 1) * (N - 1))
    se = np.sqrt(k * (k + 1) / (6 * N))
    c = names.index(control)
    rows = []
    for j, nm in enumerate(names):
        if j == c:
            continue
        z = (Rj[j] - Rj[c]) / se
        rows.append([nm, z, 2 * stats.norm.sf(abs(z))])
    rows.sort(key=lambda r: r[2])
    m = len(rows)
    stop = False
    holm = {}
    for i, (nm, z, p) in enumerate(rows):
        thr = 0.05 / (m - i)
        rej = (p <= thr) and not stop
        stop = stop or not rej
        holm[nm] = dict(z=float(z), p=float(p), thr=thr, rejected=bool(rej), better=bool(z > 0))
    return dict(ranks=dict(zip(names, Rj.tolist())), chi2=float(chi2), FF=float(FF), pFF=float(pF),
                df=(k - 1, (k - 1) * (N - 1)), holm=holm)


def wtl(d, fid, a, b, alpha=0.05):
    x, y = errs(d, fid, a), errs(d, fid, b)
    if np.allclose(x, y) or (np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]):
        return "="
    p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
    if p >= alpha:
        return "="
    return "+" if np.median(x) < np.median(y) or (np.median(x) == np.median(y) and x.mean() < y.mean()) else "-"


def welch_vs_paper(x, m, s, n=50):
    """p-value of Welch t-test of sample x against a published mean m, sd s with n runs."""
    mx, sx = np.mean(x), np.std(x, ddof=1)
    if sx == 0 and s == 0:
        return 1.0 if np.isclose(mx, m) else 0.0
    se = np.sqrt(sx ** 2 / len(x) + s ** 2 / n)
    t = (mx - m) / se
    dfw = se ** 4 / ((sx ** 2 / len(x)) ** 2 / (len(x) - 1) + (s ** 2 / n) ** 2 / (n - 1))
    return float(2 * stats.t.sf(abs(t), dfw))


if __name__ == "__main__":
    import os
    os.makedirs(TAB, exist_ok=True)
    summary = {}
    E = {e: load(e) for e in ("E1", "E2a", "E2b")}

    # ------------------------------------------------------------ reproduction of the GPC column (E1)
    rep = []
    for fid in range(1, 31):
        pm, ps = P.T3[fid][5]
        row = dict(fid=fid, paper=(pm, ps))
        for a in ("GPC", "GPC14", "GPC-A"):
            v = vals(E["E1"], fid, a)
            # the paper reports raw function values
            row[a] = (float(v.mean()), float(v.std(ddof=1)), welch_vs_paper(v, pm, ps))
        rep.append(row)
    summary["reproduction"] = rep
    ok_mult = [r["fid"] for r in rep if r["GPC"][2] >= 0.01 or abs(r["GPC"][0] - r["paper"][0]) <= 1e-4 * max(1, abs(r["paper"][0]))]
    ok_add = [r["fid"] for r in rep if r["GPC-A"][2] >= 0.01]
    summary["consistent_mult"] = ok_mult
    summary["consistent_add"] = ok_add
    print("consistent with multiplicative GPC:", len(ok_mult), ok_mult)
    print("consistent with additive GPC:", len(ok_add), ok_add)
    print("consistent with neither:", [f for f in range(1, 31) if f not in ok_mult and f not in ok_add])

    # comparators vs paper (sanity)
    comp = {}
    for fid in range(1, 31):
        comp[fid] = {}
        for j, a in enumerate(P.ALGS[:5]):
            v = vals(E["E1"], fid, a)
            comp[fid][a] = (float(v.mean()), float(v.std(ddof=1)), P.T3[fid][j][0])
    summary["comparators_vs_paper"] = comp

    # ------------------------------------------------------------ main tables for E1, E2a, E2b
    for exp in ("E1", "E2a", "E2b"):
        d = E[exp]
        fids = sorted(d.keys())
        names = ALGS_MAIN
        M = np.array([[errs(d, fid, a).mean() for a in names] for fid in fids])
        fr = friedman_holm(M, names, "GPC")
        fr_orig6 = friedman_holm(M[:, :6], names[:6], "GPC")
        wt = {a: collections.Counter(wtl(d, fid, "GPC", a) for fid in fids) for a in names if a != "GPC"}
        summary[exp] = dict(friedman=fr, friedman_orig6=fr_orig6, wtl={a: dict(c) for a, c in wt.items()},
                            mean_err={fid: dict(zip(names, M[i].tolist())) for i, fid in enumerate(fids)},
                            median_err={fid: {a: float(np.median(errs(d, fid, a))) for a in names} for fid in fids},
                            sd_err={fid: {a: float(errs(d, fid, a).std(ddof=1)) for a in names} for fid in fids})
        print("\n==", exp, "mean ranks:", {k: round(v, 2) for k, v in fr["ranks"].items()})
        print("   FF=%.2f p=%.2e" % (fr["FF"], fr["pFF"]))
        print("   original six:", {k: round(v, 2) for k, v in fr_orig6["ranks"].items()})
        print("   W/T/L of GPC:", {a: dict(c) for a, c in wt.items()})

    # ------------------------------------------------------------ invariance test E1 vs E2a
    inv = {}
    for a in ALGS_MAIN:
        n_sig = 0
        ratios = []
        for fid in range(1, 31):
            x, y = errs(E["E1"], fid, a), errs(E["E2a"], fid, a)
            if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
                p = 1.0
            else:
                p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
            n_sig += p < 0.05
            ratios.append((np.median(y) + 1e-300) / (np.median(x) + 1e-300))
        inv[a] = dict(n_sig=int(n_sig), median_log10_ratio=float(np.median(np.log10(ratios))))
    summary["invariance"] = inv
    print("\ninvariance (E1 vs E2a): functions with significant change:", {a: v["n_sig"] for a, v in inv.items()})

    # distance to origin vs optimum for GPC in E2b
    dd = {}
    for fid in sorted(B.ORIGIN_OPT):
        rr = E["E2b"][fid]["GPC"]
        dd[fid] = dict(d_origin=float(np.median([r["d_origin"] for r in rr])), d_opt=float(np.median([r["d_opt"] for r in rr])),
                       o_norm=float(np.median([r["o_norm"] for r in rr])))
    summary["E2b_gpc_distances"] = dd
    print("E2b GPC final distances (rms per coordinate):", dd)

    json.dump(summary, open(RES + "bench_summary.json", "w"), indent=1, default=float)
    print("saved")
