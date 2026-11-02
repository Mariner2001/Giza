"""Audit of the published material (Sect. 3): every number is recomputed from the tables of the original study
(transcribed in paperdata.py) and the definitions of the test functions (bench.py).  No optimiser is run.
Writes ../results/numbers_audit.json and the bodies of Tables 2-3 to ../results/tables/."""
import itertools
import json
import math
import os
import numpy as np
from scipy import optimize, stats
import bench as B
import paperdata as P
from common import FSTAR, fstar
from stats_utils import friedman, m

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ALG6 = P.ALGS
T4 = np.array([P.T4_RANKS[a] for a in ALG6])


def published_means(drop=()):
    fids = [f for f in range(1, 31) if f not in drop]
    return fids, np.array([[P.T3[f][k][0] for k in range(6)] for f in fids])


def ranks_with(reverse=(), drop=()):
    fids, M = published_means(drop)
    M = M.copy()
    for i, f in enumerate(fids):
        if f in reverse:
            M[i] = -M[i]
    return friedman(M)


def reproduces(fr):
    return bool(np.all(np.round(fr["ranks"], 2) == T4))


def enumerate_changes():
    """Which single-function changes reproduce the six published mean ranks to the printed digits?"""
    single_rev = [f for f in range(1, 31) if reproduces(ranks_with(reverse=(f,)))]
    single_drop = [f for f in range(1, 31) if reproduces(ranks_with(drop=(f,)))]
    pair_rev = [p for p in itertools.combinations(range(1, 31), 2) if reproduces(ranks_with(reverse=p))]
    return dict(none=reproduces(ranks_with()), single_reversal=single_rev, single_omission=single_drop,
                pair_reversal=[list(p) for p in pair_rev])


def stats_table():
    pub = ranks_with(reverse=(20,))
    cor = ranks_with()
    N, k = 30, 6
    se = math.sqrt(k * (k + 1) / (6.0 * N))
    z_pub = {a: (P.T4_RANKS[a] - P.T4_RANKS["GPC"]) / se for a in ALG6 if a != "GPC"}
    g = ALG6.index("GPC")
    z_cor = {a: (cor["ranks"][i] - cor["ranks"][g]) / se for i, a in enumerate(ALG6) if a != "GPC"}
    p_cor = {a: 2 * stats.norm.sf(abs(z)) for a, z in z_cor.items()}
    # Iman-Davenport from the tie-corrected chi2 as well
    FF_t = lambda fr: (N - 1) * fr["chi2_t"] / (N * (k - 1) - fr["chi2_t"])
    return dict(ranks_reversed=list(pub["ranks"]), ranks_correct=list(cor["ranks"]),
                chi2_reversed=pub["chi2"], chi2_t_reversed=pub["chi2_t"], p_chi2_t_reversed=pub["p_chi2_t"],
                FF_reversed=pub["FF"], FF_t_reversed=FF_t(pub), pFF_reversed=pub["pFF"],
                chi2_correct=cor["chi2"], chi2_t_correct=cor["chi2_t"], p_chi2_correct=cor["p_chi2"],
                FF_correct=cor["FF"], FF_t_correct=FF_t(cor), pFF_correct=cor["pFF"],
                z_published_from_T4=z_pub, z_correct=z_cor, p_correct=p_cor)


def f_at_origin(fid):
    info = B.get(fid)
    return float(info["f"](np.zeros((1, info["D"])))[0])


def origin_table():
    rows = []
    M7 = []
    for f in range(1, 31):
        info = B.get(f)
        D = info["D"]
        f0 = f_at_origin(f)
        pub = [P.T3[f][k][0] for k in range(6)]
        vals = pub + [f0]
        r = stats.rankdata(vals, method="average")
        M7.append(vals)
        rows.append(dict(fid=f, name=info["name"], D=D, fstar=fstar(f, D), corrected=f in (5, 9, 20, 25),
                         origin=f in B.ORIGIN_OPT, f0=f0, gpc=pub[5], best_other=min(pub[:5]), rank0=float(r[6])))
    fr7 = friedman(np.array(M7))
    names7 = ALG6 + ["x=0"]
    strictly_better = [r["fid"] for r in rows if r["f0"] < min(r["gpc"], r["best_other"])]
    equal_gpc = [r["fid"] for r in rows if r["f0"] == r["gpc"] and r["f0"] <= r["best_other"]]
    second = [r["fid"] for r in rows if r["gpc"] < r["f0"] < r["best_other"]]
    return rows, dict(ranks7=dict(zip(names7, fr7["ranks"])), strictly_better=strictly_better,
                      equal_gpc=equal_gpc, second_only_to_gpc=second,
                      beats_all_five=sorted(set(strictly_better + equal_gpc + second)))


def claims_s4():
    best, tie, notbest = [], [], []
    for f in range(1, 31):
        v = [P.T3[f][k][0] for k in range(6)]
        if v[5] < min(v[:5]):
            best.append(f)
        elif v[5] == min(v[:5]):
            tie.append(f)
        else:
            notbest.append((f, int(stats.rankdata(v, method="min")[5])))
    return dict(lowest=len(best), ties=tie, not_best=notbest)


def bhatia_davis():
    lo = fstar(8, 2)       # minimum of cross-in-tray
    hi = -1e-4             # f <= -1e-4 everywhere
    mu_hi = -2.0625 + 5e-5  # printed mean -2.0625 (largest value that rounds to it)
    return math.sqrt((hi - mu_hi) * (mu_hi - lo))


def optima_checks():
    out = {}
    # Michalewicz, D = 30: separable, sum of 1-D minima
    tot = 0.0
    for i in range(1, 31):
        g = lambda x, i=i: -math.sin(x) * math.sin(i * x * x / math.pi) ** 20
        xs = np.linspace(0, math.pi, 20001)
        vals = -np.sin(xs) * np.sin(i * xs ** 2 / math.pi) ** 20
        j = int(np.argmin(vals))
        r = optimize.minimize_scalar(g, bounds=(xs[max(j - 1, 0)], xs[min(j + 1, len(xs) - 1)]), method="bounded",
                                     options={"xatol": 1e-12})
        tot += min(r.fun, vals[j])
    out["michalewicz_D30"] = tot
    out["michalewicz_D2"] = FSTAR_D2 = None
    # De Jong N.5: local search from the 25 foxholes
    best = np.inf
    for a1, a2 in zip(B._A1, B._A2):
        r = optimize.minimize(lambda x: B.dejong5(x[None, :])[0], [a1, a2], method="Nelder-Mead",
                              options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 5000})
        best = min(best, r.fun)
    out["dejong5"] = best
    # Bukin N.6 on [-3,3]^2: minimum 0.01*|x1 + 10| at x1 = -3 on the curve x2 = 0.01 x1^2
    out["bukin_box"] = float(B.bukin6(np.array([[-3.0, 0.09]]))[0])
    # Schaffer N.4 as printed: infimum on [-100,100]^2 (grid of 4001^2 points)
    g = np.linspace(-100, 100, 4001)
    best = np.inf
    for k in range(0, 4001, 401):
        X1, X2 = np.meshgrid(g[k:k + 401], g, indexing="ij")
        v = B.schaffer4_printed(np.c_[X1.ravel(), X2.ravel()])
        best = min(best, float(v.min()))
    out["schaffer4_printed_inf"] = best
    out["schaffer4_standard"] = FSTAR[25]
    return out


def seg_audit():
    """Statistics of Tables 8 and 9 of the original study quoted in Sect. 3.4."""
    won, tied, lost, rel = 0, 0, [], []
    for img, d in P.T8.items():
        for K, rows in d.items():
            means = [mm for mm, s in rows]
            g = means[5]
            others = means[:5]
            if g < min(others):
                won += 1
                rel.append((min(others) - g) / g * 100)
            elif g == min(others):
                tied += 1
            else:
                lost.append((img, K, int(stats.rankdata(means, method="min")[5])))
    fastest, spread, gpc_gain, avg = 0, [], [], {}
    for img, d in P.T9.items():
        for K, t in d.items():
            t = np.array(t)
            fastest += int(t[5] < t[:5].min())
            spread.append((t.max() - t.min()) / t.min() * 100)
            gpc_gain.append((t[:5].min() - t[5]) / t[:5].min() * 100)
            avg.setdefault(K, []).extend(t.tolist())
    return dict(gpc_lowest=won, ties=tied, outperformed=lost, median_rel_diff_won=float(np.median(rel)),
                max_rel_diff_won=float(np.max(rel)), gpc_fastest=fastest, median_spread=float(np.median(spread)),
                median_gpc_vs_fastest_other=float(np.median(gpc_gain)),
                mean_time={K: float(np.mean(v)) for K, v in avg.items()})


def write_origin_table(rows):
    lines = []
    for r in rows:
        fs = m(r["fstar"], 4) + ("$^{*}$" if r["corrected"] else "")
        lines.append("%d & %s & %d & %s & %s & %s & %s & %s & %s \\\\" % (
            r["fid"], r["name"], r["D"], fs, "\\checkmark" if r["origin"] else "", m(r["f0"], 4), m(r["gpc"], 4),
            m(r["best_other"], 4), ("%g" % r["rank0"])))
    os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    open(os.path.join(RES, "tables", "tab_origin.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    out = {}
    out["changes"] = enumerate_changes()
    out["stats"] = stats_table()
    rows, o = origin_table()
    write_origin_table(rows)
    out["origin"] = o
    out["claims_s4"] = claims_s4()
    out["bhatia_davis_max_sd"] = bhatia_davis()
    out["optima"] = optima_checks()
    out["seg"] = seg_audit()
    json.dump(out, open(os.path.join(RES, "numbers_audit.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))
