"""Benchmark experiments on the thirty functions of the original study.

  E1      original problems                                  (30 functions)
  E2a     translated problems: f(x - o) over [l + o, u + o]  (30 functions)
  E2b     shifted optimum: f(x - o) over [l, u]              (14 functions minimised at the origin)
  E2c     rotation about the origin: f(R x) over [l, u]      (14 functions minimised at the origin)
  E2d     rotation and translation: f(R(x - o)) over [l + o, u + o]   (same 14 functions)
  E2e_lo  change of units: f(x / s) over [s l, s u], s = 1e-2         (same 14 functions)
  E2e_hi  the same with s = 1e2                                       (same 14 functions)

usage: python3 exp_bench.py EXP [runs] [comma-separated algorithms] [output tag]
Without an algorithm list, E1/E2a/E2b run the nine algorithms of the submitted version (plus GPC14 in E1)
and the other experiments run all eleven algorithms.  Algorithm names: see ALGS below; "E7" stands for all
GPC variants of the ablation / sensitivity / initialisation study.
Seeds: problem instance  default_rng([EXP_ID, fid, run])  (identical for all algorithms);
       algorithm         default_rng([EXP_ID, fid, ALG_ID, run, 7]).
"""
import sys
import time
import numpy as np
from multiprocessing import Pool
import algos as A
import algos_shade as S
from algos_gpcvar import gpc_var
import bench as B
from common import make_problem
from io_utils import save_records

KINDS = {"E1": ("orig", 1), "E2a": ("trans", 2), "E2b": ("shift", 3), "E2c": ("rot", 4), "E2d": ("rottrans", 5),
         "E2e_lo": ("scale_lo", 6), "E2e_hi": ("scale_hi", 7)}

# name: (function, keyword arguments, algorithm id used in the seed)
ALGS = {
    "GA": (A.ga, {}, 0), "ICA": (A.ica, {}, 1), "PSO": (A.pso, {}, 2), "DE": (A.de, {}, 3), "FA": (A.fa, {}, 4),
    "GPC": (A.gpc, {"mode": "mult", "theta": 10.0}, 5),
    "GPC-A": (A.gpc, {"mode": "add", "theta": 10.0}, 6),
    "CMA-ES": (A.cmaes, {}, 7), "RS": (A.rs, {}, 8),
    "GPC14": (A.gpc, {"mode": "mult", "theta": 14.0}, 9),
    "L-SHADE": (S.lshade, {}, 10), "jSO": (S.jso, {}, 11),
}
# E7: GPC variants (ablation, parameter sensitivity, initialisation); ids >= 20
E7 = {
    "GPC_theta5": {"theta": 5.0}, "GPC_theta30": {"theta": 30.0}, "GPC_theta60": {"theta": 60.0},
    "GPC_mu0.1-1": {"mu_min": 0.1, "mu_max": 1.0}, "GPC_mu10-100": {"mu_min": 10.0, "mu_max": 100.0},
    "GPC_n10": {"n": 10}, "GPC_n50": {"n": 50}, "GPC_n100": {"n": 100},
    "GPC_pss0.1": {"pss": 0.1}, "GPC_pss0.3": {"pss": 0.3}, "GPC_pss0.7": {"pss": 0.7}, "GPC_pss0.9": {"pss": 0.9},
    "GPC_pss0": {"pss": 0.0}, "GPC_pss1": {"pss": 1.0}, "GPC_nod": {"use_d": False},
    "GPC_reinit": {"bounds": "reinit"}, "GPC_nonelit": {"elitist": False},
    "GPC_lhs": {"init": "lhs"}, "GPC_corner": {"init": "corner"},
}
for _k, (_name, _kw) in enumerate(E7.items()):
    ALGS[_name] = (gpc_var, dict({"mode": "mult", "theta": 10.0}, **_kw), 20 + _k)

ORIG9 = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "RS"]
ALL11 = ORIG9 + ["L-SHADE", "jSO"]

EXP = sys.argv[1] if len(sys.argv) > 1 else "E1"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
KIND, EXP_ID = KINDS[EXP]
if len(sys.argv) > 3 and sys.argv[3]:
    SEL = []
    for a in sys.argv[3].split(","):
        SEL += list(E7) if a == "E7" else [a]
else:
    SEL = ORIG9 + ["GPC14"] if EXP == "E1" else (ORIG9 if EXP in ("E2a", "E2b") else ALL11)
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
FIDS = list(range(1, 31)) if EXP in ("E1", "E2a") else sorted(B.ORIGIN_OPT)


def checkpoints(nfe):
    return np.unique(np.round(np.logspace(np.log10(20), np.log10(nfe), 30)).astype(int))


def job(args):
    fid, alg, r = args
    prob = make_problem(fid, KIND, np.random.default_rng([EXP_ID, fid, r]))
    fn, kw, aid = ALGS[alg]
    rng = np.random.default_rng([EXP_ID, fid, aid, r, 7])
    t = time.process_time()
    res = fn(prob["f"], prob["lb"], prob["ub"], prob["nfe"], rng, **kw)
    cpu = time.process_time() - t
    cps = checkpoints(prob["nfe"])
    idx = np.searchsorted(res["trace_n"], cps, side="right") - 1
    curve = np.where(idx >= 0, res["trace_f"][np.maximum(idx, 0)], np.inf)
    xb = res["best_x"]
    # distance of the best point to the optimum location (if known: origin-optimum functions) and to 0
    d_origin = float(np.linalg.norm(xb)) / np.sqrt(len(xb))
    d_opt = float(np.linalg.norm(xb - prob["o"])) / np.sqrt(len(xb)) if fid in B.ORIGIN_OPT else np.nan
    return dict(fid=fid, alg=alg, run=r, best_f=res["best_f"], err=res["best_f"] - prob["fstar"],
                nfe=res["nfe"], cpu=cpu, curve=curve, cps=cps, d_origin=d_origin, d_opt=d_opt,
                o_norm=float(np.linalg.norm(prob["o"]) / np.sqrt(len(xb))))


if __name__ == "__main__":
    jobs = [(fid, alg, r) for fid in FIDS for alg in SEL for r in range(RUNS)]
    # heavy jobs first for better load balance
    jobs.sort(key=lambda j: -(B.get(j[0])["nfe"] * (20 if j[1] in ("CMA-ES", "L-SHADE", "jSO") else 1)))
    t0 = time.time()
    out = []
    with Pool(2) as pool:
        for k, res in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            out.append(res)
            if (k + 1) % 2000 == 0:
                print(f"{EXP}: {k + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    save_records(f"../results/{EXP}{'_' + TAG if TAG else ''}.jsonl.gz", out)
    print(f"{EXP}: finished {len(out)} runs in {time.time() - t0:.0f}s")
