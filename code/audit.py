"""Numerical audit of Harifi et al. (2021).  Produces results/audit.json and prints a log."""
import json
import numpy as np
from scipy import optimize, stats
import bench as B
import paperdata as P
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

out = {}
np.set_printoptions(precision=6, suppress=False)

# ------------------------------------------------------------------ 1. f(0) values
f0 = {}
for fid in range(1, 31):
    info = B.get(fid)
    x0 = np.zeros((1, info["D"]))
    f0[fid] = float(info["f"](x0)[0])
out["f_origin_D30"] = f0
print("f(0):", {k: round(v, 6) for k, v in f0.items()})

# Levy / Dixon-Price / Rosenbrock / others at the origin in high dimension
hd = {}
for fid in [1, 10, 14, 16, 17, 21, 22, 23, 27, 28, 29, 30]:
    info = B.get(fid)
    hd[fid] = {}
    for D in (30, 500, 1000, 5000, 10000):
        hd[fid][D] = float(info["f"](np.zeros((1, D)))[0])
out["f_origin_highD"] = hd
print("f(0) high-D:", json.dumps(hd, indent=0)[:2000])

# ------------------------------------------------------------------ 2. true optima
# Michalewicz (m=10) is separable: min_x -sin(x) sin^20(i x^2/pi) per coordinate
def mich_1d_min(i):
    g = np.linspace(0, np.pi, 400001)
    v = -np.sin(g) * np.sin(i * g ** 2 / np.pi) ** 20
    k = np.argmin(v)
    res = optimize.minimize_scalar(lambda x: -np.sin(x) * np.sin(i * x ** 2 / np.pi) ** 20,
                                   bounds=(max(0, g[k] - 1e-3), min(np.pi, g[k] + 1e-3)), method="bounded",
                                   options={"xatol": 1e-14})
    return min(v[k], res.fun), res.x

mich = {}
for D in (2, 5, 10, 30):
    mich[D] = float(sum(mich_1d_min(i)[0] for i in range(1, D + 1)))
out["michalewicz_opt"] = mich
print("Michalewicz optimum:", mich)

# De Jong N.5 (Shekel foxholes): multi-start local search
best = np.inf
for a1 in (-32, -16, 0, 16, 32):
    for a2 in (-32, -16, 0, 16, 32):
        r = optimize.minimize(lambda x: B.dejong5(x[None, :])[0], x0=np.array([a1, a2], float), method="Nelder-Mead",
                              options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
        best = min(best, r.fun)
out["dejong5_opt"] = float(best)
print("De Jong N.5 optimum:", best)

# Bukin N.6 on [-3,3]^2 (Table 1 box): minimum along the ridge x2 = 0.01 x1^2 at x1 = -3
bk = float(B.bukin6(np.array([[-3.0, 0.09]]))[0])
g = np.linspace(-3, 3, 3001)
G1, G2 = np.meshgrid(g, g)
grid = B.bukin6(np.c_[G1.ravel(), G2.ravel()])
out["bukin_box_opt"] = bk
out["bukin_origin"] = float(B.bukin6(np.zeros((1, 2)))[0])
print("Bukin min on [-3,3]^2:", bk, " grid min:", grid.min(), " f(0):", out["bukin_origin"])

# Schaffer N.4: standard (cos^2) optimum and printed-version infimum on the box
s4 = optimize.minimize(lambda x: B.schaffer4(x[None, :])[0], x0=np.array([0.0, 1.25]), method="Nelder-Mead",
                       options={"xatol": 1e-12, "fatol": 1e-14})
g = np.linspace(-100, 100, 4001)
G1, G2 = np.meshgrid(g, g)
pv = B.schaffer4_printed(np.c_[G1.ravel(), G2.ravel()])
out["schaffer4_std_opt"] = float(s4.fun)
out["schaffer4_printed_inf_box"] = float(pv.min())
out["schaffer4_printed_inf_where"] = [float(G1.ravel()[pv.argmin()]), float(G2.ravel()[pv.argmin()])]
print("Schaffer N.4 std opt:", s4.fun, s4.x, " printed-version min on grid:", pv.min(), out["schaffer4_printed_inf_where"])

# Cross-in-tray / Holder / Eggholder / Shubert / camel checks (Table 1 values)
chk = {}
chk["crossintray"] = float(B.crossintray(np.array([[1.34941, 1.34941]]))[0])
chk["holder"] = float(B.holdertable(np.array([[8.05502, 9.66459]]))[0])
chk["eggholder"] = float(B.eggholder(np.array([[512.0, 404.2319]]))[0])
chk["shubert_grid_min"] = float(B.shubert(np.c_[np.meshgrid(np.linspace(-10, 10, 4001), np.linspace(-10, 10, 4001))[0].ravel(),
                                                 np.meshgrid(np.linspace(-10, 10, 4001), np.linspace(-10, 10, 4001))[1].ravel()]).min())
chk["camel6"] = float(B.camel6(np.array([[0.0898, -0.7126]]))[0])
out["table1_checks"] = chk
print("Table 1 checks:", chk)

# ------------------------------------------------------------------ 3. maxima on the box (for Bhatia-Davis)
# Var <= (mu - m)(M - mu) for any distribution supported on [m, M].
fmax = {}
rng = np.random.default_rng(0)
for fid in range(1, 31):
    info = B.get(fid)
    if info["D"] == 2:
        g = np.linspace(info["lb"], info["ub"], 2001)
        G1, G2 = np.meshgrid(g, g)
        fmax[fid] = float(np.nanmax(info["f"](np.c_[G1.ravel(), G2.ravel()])))
out["fmax_2d"] = fmax


def lower_bound(fid):
    lb = {5: out["bukin_box_opt"], 9: out["dejong5_opt"], 20: mich[30], 25: out["schaffer4_std_opt"]}
    return lb.get(fid, P.T3 and B.get(fid)["fstar_table1"])


bd_flags = []
for fid in range(1, 31):
    for a, (m, s) in zip(P.ALGS, P.T3[fid]):
        lo = lower_bound(fid)
        hi = fmax.get(fid, np.inf)
        bound = np.sqrt(max(0.0, (m - lo) * (hi - m))) if np.isfinite(hi) else np.inf
        # allow for rounding of the printed mean (half unit of the last printed digit ~ 5e-5)
        m_lo = m - 5e-5
        bound2 = np.sqrt(max(0.0, (m + 5e-5 - lo) * (hi - m_lo))) if np.isfinite(hi) else np.inf
        if s > bound2 * 1.0001:
            bd_flags.append(dict(fid=fid, alg=a, mean=m, sd=s, sd_max=bound2, lo=lo, hi=hi))
out["bhatia_davis_violations"] = bd_flags
print("Bhatia-Davis violations:", bd_flags)

# ------------------------------------------------------------------ 4. ranks
def ranks_from_means(means):
    # rank with ties on the printed values (1 = best)
    return stats.rankdata(means, method="average")


def table_ranks(reverse_f20=False):
    R = []
    for fid in range(1, 31):
        means = np.array([m for (m, s) in P.T3[fid]])
        if reverse_f20 and fid == 20:
            means = -means
        R.append(ranks_from_means(means))
    return np.array(R)


def friedman(R):
    N, k = R.shape
    Rj = R.mean(axis=0)
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(Rj ** 2) - k * (k + 1) ** 2 / 4)
    # tie-corrected version
    T = 0.0
    for row in R:
        _, c = np.unique(row, return_counts=True)
        T += np.sum(c ** 3 - c)
    chi2_t = chi2 / (1 - T / (N * k * (k * k - 1)))
    FF = (N - 1) * chi2 / (N * (k - 1) - chi2)
    FF_t = (N - 1) * chi2_t / (N * (k - 1) - chi2_t)
    p_chi = stats.chi2.sf(chi2, k - 1)
    p_chi_t = stats.chi2.sf(chi2_t, k - 1)
    p_FF = stats.f.sf(FF, k - 1, (k - 1) * (N - 1))
    p_FF_t = stats.f.sf(FF_t, k - 1, (k - 1) * (N - 1))
    return dict(Rj=Rj.tolist(), chi2=chi2, chi2_tie=chi2_t, p_chi2=p_chi, p_chi2_tie=p_chi_t, FF=FF, FF_tie=FF_t,
                p_FF=p_FF, p_FF_tie=p_FF_t, df_FF=(k - 1, (k - 1) * (N - 1)))


def holm(Rj, control, N, names, alpha=0.05):
    k = len(Rj)
    se = np.sqrt(k * (k + 1) / (6 * N))
    rows = []
    for j, nm in enumerate(names):
        if j == control:
            continue
        z = (Rj[j] - Rj[control]) / se
        p = 2 * stats.norm.sf(abs(z))
        rows.append([nm, z, p])
    rows.sort(key=lambda r: r[2])
    m = len(rows)
    res = []
    stop = False
    for i, (nm, z, p) in enumerate(rows):
        thr = alpha / (m - i)
        rej = (p <= thr) and not stop
        if not rej:
            stop = True
        res.append(dict(alg=nm, z=z, p=p, alpha_adj=thr, rejected=bool(rej)))
    return res


for label, rev in (("as_published_reversed_f20", True), ("corrected_minimisation", False)):
    R = table_ranks(reverse_f20=rev)
    fr = friedman(R)
    hm = holm(np.array(fr["Rj"]), P.ALGS.index("GPC"), 30, P.ALGS)
    out["ranks_" + label] = dict(friedman=fr, holm=hm, rank_f20=R[19].tolist())
    print("\n==", label)
    print("mean ranks:", dict(zip(P.ALGS, np.round(fr["Rj"], 4))))
    print("chi2=%.4f (tie-corr %.4f), FF=%.4f (tie-corr %.4f), p_chi2=%.3e, p_FF=%.3e, df_FF=%s" % (
        fr["chi2"], fr["chi2_tie"], fr["FF"], fr["FF_tie"], fr["p_chi2"], fr["p_FF"], fr["df_FF"]))
    for r in hm:
        print("  Holm", r)

# Iman-Davenport implied chi2 from published FF = 31.694
FF = 31.6940
chi2_impl = FF * 150 / (29 + FF)
out["ID_implied_chi2"] = chi2_impl
print("\nchi2 implied by published Iman-Davenport statistic:", chi2_impl)

# ------------------------------------------------------------------ 5. wins as printed
wins = {}
for fid in range(1, 31):
    means = np.array([m for (m, s) in P.T3[fid]])
    best = means.min()
    winners = [a for a, m in zip(P.ALGS, means) if m == best]
    wins[fid] = winners
out["winners_printed_means"] = wins
gpc_sole = [f for f, w in wins.items() if w == ["GPC"]]
gpc_tie = [f for f, w in wins.items() if "GPC" in w and len(w) > 1]
gpc_not = [f for f, w in wins.items() if "GPC" not in w]
out["gpc_sole_best"] = gpc_sole
out["gpc_tied_best"] = gpc_tie
out["gpc_not_best"] = gpc_not
print("\nGPC sole best on", len(gpc_sole), gpc_sole)
print("GPC tied best on", len(gpc_tie), gpc_tie)
print("GPC not best on", len(gpc_not), gpc_not, [wins[f] for f in gpc_not])

# rank of GPC on f20 and f18
for fid in (18, 20):
    means = np.array([m for (m, s) in P.T3[fid]])
    print("f%d ranks:" % fid, dict(zip(P.ALGS, stats.rankdata(means))))

# ------------------------------------------------------------------ 6. origin baseline vs Table 3
ob = {}
for fid in range(1, 31):
    means = np.array([m for (m, s) in P.T3[fid]])
    v = f0[fid]
    ob[fid] = dict(f0=v, better_or_equal_than_all=bool(v <= means.min() + 1e-12),
                   strictly_better_than_all=bool(v < means.min() - 1e-12),
                   better_or_equal_than_all_but_gpc=bool(v <= means[:5].min()),
                   rank_among_7=float(stats.rankdata(np.r_[means, v])[-1]))
out["origin_baseline"] = ob
n_all = sum(1 for d in ob.values() if d["better_or_equal_than_all"])
n_strict = sum(1 for d in ob.values() if d["strictly_better_than_all"])
n_nogpc = sum(1 for d in ob.values() if d["better_or_equal_than_all_but_gpc"])
print("\nOrigin baseline: <= all six on %d functions (strictly better on %d); <= all five comparators on %d" % (
    n_all, n_strict, n_nogpc))
print("  functions:", [f for f, d in ob.items() if d["better_or_equal_than_all"]])
print("  vs comparators only:", [f for f, d in ob.items() if d["better_or_equal_than_all_but_gpc"]])

# Friedman including the origin baseline as a 7th 'algorithm'
R7 = []
for fid in range(1, 31):
    means = np.array([m for (m, s) in P.T3[fid]] + [f0[fid]])
    R7.append(stats.rankdata(means))
R7 = np.array(R7)
out["ranks_with_origin_baseline"] = dict(zip(P.ALGS + ["x=0"], R7.mean(axis=0).tolist()))
print("Mean ranks with x=0 baseline:", out["ranks_with_origin_baseline"])

# ------------------------------------------------------------------ 7. Table 8 checks
t8 = {}
nbest = 0
nties = 0
for img, dd in P.T8.items():
    for K, vals in dd.items():
        means = np.array([m for (m, s) in vals])
        r = stats.rankdata(means)
        best = [a for a, m in zip(P.ALGS, means) if m == means.min()]
        bold = P.T8_BOLD.get((img, K), "GPC")
        t8["%s-K%d" % (img, K)] = dict(best=best, gpc_rank=float(r[-1]), bold=bold,
                                       rel_gap_gpc_vs_best_other=float((means[-1] - means[:5].min()) / means[:5].min()))
        if best == ["GPC"]:
            nbest += 1
        elif "GPC" in best:
            nties += 1
out["table8"] = t8
print("\nTable 8: GPC sole best in %d/24, tied in %d" % (nbest, nties))
for k, v in t8.items():
    if v["best"] != ["GPC"] or v["bold"] not in v["best"]:
        print("  ", k, v)

# ------------------------------------------------------------------ 8. Table 9 checks
t9 = {}
for img, dd in P.T9.items():
    for K, vals in dd.items():
        v = np.array(vals)
        t9["%s-K%d" % (img, K)] = dict(min_alg=P.ALGS[int(np.argmin(v))], spread_pct=float((v.max() - v.min()) / v.min() * 100),
                                       gpc_vs_min_other_pct=float((v[-1] - v[:5].min()) / v[:5].min() * 100))
out["table9"] = t9
sp = [d["spread_pct"] for d in t9.values()]
gp = [d["gpc_vs_min_other_pct"] for d in t9.values()]
print("Table 9 spread across algorithms: median %.2f%%, max %.2f%%; GPC vs fastest other: median %.2f%%, min %.2f%%" % (
    np.median(sp), np.max(sp), np.median(gp), np.min(gp)))
# mean CPU time per K
for K in (3, 4, 5):
    print("  K=%d mean time %.2f s" % (K, np.mean([np.mean(P.T9[i][K]) for i in P.T9])))

json.dump(out, open("../results/audit.json", "w"), indent=1, default=float)
print("\nsaved results/audit.json")
