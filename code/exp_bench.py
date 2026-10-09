"""E1 (original problems), E2a (translated problems), E2b (shifted optimum).

usage: python3 exp_bench.py E1|E2a|E2b [runs]
"""
import sys
import time
import numpy as np
from multiprocessing import Pool
import algos as A
import bench as B
from common import make_problem
from io_utils import save_records

EXP = sys.argv[1]
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
KIND = {"E1": "orig", "E2a": "trans", "E2b": "shift"}[EXP]
EXP_ID = {"E1": 1, "E2a": 2, "E2b": 3}[EXP]

ALGS = {
    "GA": (A.ga, {}), "ICA": (A.ica, {}), "PSO": (A.pso, {}), "DE": (A.de, {}), "FA": (A.fa, {}),
    "GPC": (A.gpc, {"mode": "mult", "theta": 10.0}),
    "GPC-A": (A.gpc, {"mode": "add", "theta": 10.0}),
    "CMA-ES": (A.cmaes, {}), "RS": (A.rs, {}),
}
if EXP == "E1":
    ALGS["GPC14"] = (A.gpc, {"mode": "mult", "theta": 14.0})
ALG_IDS = {k: i for i, k in enumerate(["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "RS", "GPC14"])}

FIDS = sorted(B.ORIGIN_OPT) if EXP == "E2b" else list(range(1, 31))


def checkpoints(nfe):
    return np.unique(np.round(np.logspace(np.log10(20), np.log10(nfe), 30)).astype(int))


def job(args):
    fid, alg, r = args
    prob = make_problem(fid, KIND, np.random.default_rng([EXP_ID, fid, r]))
    fn, kw = ALGS[alg]
    rng = np.random.default_rng([EXP_ID, fid, ALG_IDS[alg], r, 7])
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
    jobs = [(fid, alg, r) for fid in FIDS for alg in ALGS for r in range(RUNS)]
    # heavy jobs first for better load balance
    jobs.sort(key=lambda j: -(B.get(j[0])["nfe"] * (20 if j[1] == "CMA-ES" else 1)))
    t0 = time.time()
    out = []
    with Pool(2) as pool:
        for k, res in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            out.append(res)
            if (k + 1) % 500 == 0:
                print(f"{EXP}: {k + 1}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    save_records(f"../results/{EXP}.jsonl.gz", out)
    print(f"{EXP}: finished {len(out)} runs in {time.time() - t0:.0f}s")
