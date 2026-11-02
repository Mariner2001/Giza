"""E3: structural-bias test (Table 8 and Fig. 4).

For every algorithm and box: Kolmogorov-Smirnov tests of uniformity of the final solutions (100 runs x 30
normalised coordinates): (i) first coordinate only (as in the submitted version), (ii) each of the 30 coordinates
separately, Holm-corrected (number of rejected coordinates), (iii) all 3000 values pooled ("all D"; descriptive,
since coordinates of the same run need not be independent); percentage of coordinates within 1 % of the box width
from the origin and exactly on the bound."""
import json
import os
import numpy as np
from scipy import stats
from io_utils import load_records
from stats_utils import holm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ALGS = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "RS", "L-SHADE", "jSO"]


def load():
    recs = load_records(os.path.join(RES, "E3_bias.jsonl.gz"))
    p = os.path.join(RES, "E3_bias_shade.jsonl.gz")
    if os.path.exists(p):
        recs += load_records(p)
    X = {}
    for r in sorted(recs, key=lambda r: (r["box"], r["alg"], r["run"])):
        X.setdefault((r["box"], r["alg"]), []).append(r["xn"])
    return {k: np.array(v) for k, v in X.items()}


def analyse():
    X = load()
    out = {}
    for (box, alg), A in X.items():
        origin = 0.5 if box == "centre" else 0.0
        ks1 = stats.kstest(A[:, 0], "uniform")
        pc = [stats.kstest(A[:, j], "uniform").pvalue for j in range(A.shape[1])]
        adj, rej = holm(pc)
        ksall = stats.kstest(A.ravel(), "uniform")
        near = np.mean(np.abs(A - origin) <= 0.01) * 100
        bound = np.mean((A == 0) | (A == 1)) * 100
        out["%s|%s" % (box, alg)] = dict(ks1=float(ks1.statistic), p1=float(ks1.pvalue), n_rej=int(rej.sum()),
                                         ks_all=float(ksall.statistic), p_all=float(ksall.pvalue),
                                         near0=float(near), bound=float(bound), runs=int(A.shape[0]))
    return out


if __name__ == "__main__":
    res = analyse()
    for k in sorted(res, key=lambda k: (k.split("|")[0], ALGS.index(k.split("|")[1]))):
        v = res[k]
        print("%-14s KS1 %.2f p %.2g | rej %2d/30 | KSall %.3f p %.2g | near0 %.0f%% bound %.0f%%" % (
            k, v["ks1"], v["p1"], v["n_rej"], v["ks_all"], v["p_all"], v["near0"], v["bound"]))
    json.dump(res, open(os.path.join(RES, "numbers_bias.json"), "w"), indent=1)
