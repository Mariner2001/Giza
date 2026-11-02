"""Cross-check of our L-SHADE and jSO against the independent C++ implementations in minionpy 1.9.1
on CEC2017 functions (D = 10, budget 1e5).  Writes ../results/validation_shade.json."""
import sys, json, time
import numpy as np
sys.path.insert(0, "../code")
import minionpy as mpy
from algos_shade import lshade, jso
from scipy.stats import mannwhitneyu

D = 10
NFE = 100000
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
FUNS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 4, 5, 7, 10, 22]
out = {}
for fn in FUNS:
    prob = mpy.CEC2017Functions(fn, D)
    fopt = prob.f_opt
    F = lambda X, prob=prob: np.asarray(prob(np.asarray(X).tolist()), float)
    for name, mine, cls, opts in [
        ("L-SHADE", lshade, mpy.LSHADE, {"population_size": 18 * D, "memory_size": 6, "archive_size_ratio": 2.6,
                                          "minimum_population_size": 4, "x_tol": -1.0, "f_tol": -1.0}),
        ("jSO", jso, mpy.jSO, {"population_size": int(round(25 * np.log(D) * np.sqrt(D))), "memory_size": 5,
                               "archive_size_ratio": 1.0, "minimum_population_size": 4, "x_tol": -1.0, "f_tol": -1.0})]:
        a, b, nf = [], [], []
        t0 = time.time()
        for r in range(RUNS):
            res = mine(F, np.full(D, -100.0), np.full(D, 100.0), NFE, np.random.default_rng([12, fn, r]))
            a.append(res["best_f"] - fopt)
            m = cls(lambda X, F=F: F(X).tolist(), [(-100.0, 100.0)] * D, maxevals=NFE, seed=1000 + r, options=opts)
            mr = m.optimize()
            b.append(mr.fun - fopt)
            nf.append(mr.nfev)
        a, b = np.maximum(a, 0), np.maximum(b, 0)
        p = mannwhitneyu(a, b).pvalue if np.ptp(np.r_[a, b]) > 0 else 1.0
        out["%s|F%d" % (name, fn)] = dict(ours_mean=float(a.mean()), ours_med=float(np.median(a)),
                                         minion_mean=float(b.mean()), minion_med=float(np.median(b)), p=float(p),
                                         minion_nfev=[int(min(nf)), int(max(nf))])
        print("%-8s F%-2d ours %.3g (med %.3g)  minion %.3g (med %.3g)  MW p=%.2f  nfev %s  %.0fs" % (
            name, fn, a.mean(), np.median(a), b.mean(), np.median(b), p, out["%s|F%d" % (name, fn)]["minion_nfev"],
            time.time() - t0), flush=True)
json.dump(out, open("../results/validation_shade_%s.json" % "_".join(map(str, FUNS)), "w"), indent=1)
