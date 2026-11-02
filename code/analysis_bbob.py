"""E5: COCO/BBOB noiseless suite (Table 11, Table 12 and Fig. 5).

Targets f_opt + 10^k, k in {2, 1.8, ..., -8} (51 targets); a target counts as reached when the best precision
f - f_opt is at most 10^k.  Fractions are aggregated over the 24 functions x 5 instances of a dimension."""
import json
import os
import numpy as np
from io_utils import load_records

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ALGS = ["GPC", "GPC-A", "PSO", "DE", "CMA-ES", "L-SHADE", "jSO", "RS"]
DIMS = [2, 5, 10, 20]
EXPO = np.round(np.arange(2, -8.0001, -0.2), 10)
TARGETS = 10.0 ** EXPO
GROUPS = {"separable": range(1, 6), "low/moderate conditioning": range(6, 10),
          "high conditioning, unimodal": range(10, 15), "multimodal, adequate structure": range(15, 20),
          "multimodal, weak structure": range(20, 25)}


def load():
    recs = []
    if not os.path.exists(os.path.join(RES, "E5_bbob_all.jsonl.gz")) and os.path.isdir(os.path.join(RES, "E5_parts")):
        import glob
        for p in sorted(glob.glob(os.path.join(RES, "E5_parts", "*.jsonl.gz"))):
            recs += load_records(p)
        return recs
    for t in ("all",) if os.path.exists(os.path.join(RES, "E5_bbob_all.jsonl.gz")) else ("main", "RS", "LSHADE_jSO"):
        p = os.path.join(RES, "E5_bbob_%s.jsonl.gz" % t)
        if os.path.exists(p):
            recs += load_records(p)
    return recs


def hit_times(r):
    """Number of evaluations at which each target is first reached (inf if never)."""
    tn, tf = np.asarray(r["tn"], float), np.asarray(r["tf"], float)
    out = np.full(len(TARGETS), np.inf)
    for k, t in enumerate(TARGETS):
        idx = np.nonzero(tf <= t)[0]
        if len(idx):
            out[k] = tn[idx[0]]
    return out


def analyse(recs):
    R = {}
    for r in recs:
        R.setdefault((r["alg"], r["D"]), []).append(r)
    out = {}
    for (alg, D), rs in R.items():
        H = np.array([hit_times(r) for r in rs])          # problems x targets
        fun = np.array([r["fun"] for r in rs])
        final = float(np.mean(np.isfinite(H)))
        solved = int(sum(np.min(r["tf"]) < 1e-8 for r in rs))
        at = {b: float(np.mean(H <= b * D)) for b in (100, 1000, 10000)}
        groups = {g: float(np.mean(np.isfinite(H[np.isin(fun, list(fs))]))) for g, fs in GROUPS.items()}
        per_fun = {int(f): float(np.mean(np.isfinite(H[fun == f]))) for f in range(1, 25)}
        budgets = np.unique(np.round(np.logspace(0, 4, 81) * D))
        ecdf = [float(np.mean(H <= b)) for b in budgets]
        xn = [r["xnorm"] for r in rs]
        out["%s|%d" % (alg, D)] = dict(final=final, solved=solved, at=at, groups=groups, per_fun=per_fun,
                                       ecdf_budgets=(budgets / D).tolist(), ecdf=ecdf, n=len(rs),
                                       median_xrms=float(np.median(xn)))
    return out


if __name__ == "__main__":
    res = analyse(load())
    json.dump(res, open(os.path.join(RES, "numbers_bbob.json"), "w"), indent=1)
    for D in DIMS:
        for a in ALGS:
            k = "%s|%d" % (a, D)
            if k in res:
                v = res[k]
                print("D=%2d %-8s final %.2f solved %3d | at 1e2D %.2f 1e3D %.2f 1e4D %.2f | groups %s | xrms %.2f" % (
                    D, a, v["final"], v["solved"], v["at"][100], v["at"][1000], v["at"][10000],
                    " ".join("%.2f" % x for x in v["groups"].values()), v["median_xrms"]))
