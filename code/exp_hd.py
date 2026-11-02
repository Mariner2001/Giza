"""E4: high-dimensional check (Table 7 of the original paper) with original and translated problems.
(The dummy-variable check for PSO is in exp_dummy.py.)"""
import time
import numpy as np
from multiprocessing import Pool
import algos as A
import bench as B
from common import fstar
from io_utils import save_records

RUNS = 10
FIDS = [1, 10, 14, 16, 17, 20, 21, 22, 23, 27, 28, 29, 30]
DIMS = [5000, 10000]
ALGS = {"GPC": (A.gpc, {"mode": "mult", "theta": 10.0}), "PSO": (A.pso, {}), "DE": (A.de, {})}
AID = {"GPC": 0, "PSO": 1, "DE": 2}


def job(a):
    kind, fid, D, alg, r = a
    info = B.get(fid)
    lb = np.full(D, info["lb"])
    ub = np.full(D, info["ub"])
    f = info["f"]
    if kind == "trans":
        o = np.random.default_rng([41, fid, D, r]).uniform(-(ub - lb) / 4, (ub - lb) / 4)
        lb, ub = lb + o, ub + o
        g = lambda X: f(X - o)
    else:
        g = f
    fn, kw = ALGS[alg]
    t = time.process_time()
    res = fn(g, lb, ub, info["nfe"], np.random.default_rng([40, fid, D, AID[alg], r, 1 if kind == "orig" else 2]), **kw)
    return dict(kind=kind, fid=fid, D=D, alg=alg, run=r, best_f=res["best_f"],
                err=res["best_f"] - (fstar(fid, 30) if fid != 20 else np.nan),
                rms_x=float(np.sqrt(np.mean((res["best_x"] - (o if kind == "trans" else 0.0)) ** 2))),
                in_box=bool(np.all(res["best_x"] >= lb) and np.all(res["best_x"] <= ub)),
                cpu=time.process_time() - t)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(k, fid, D, alg, r) for k in ("orig", "trans") for fid in FIDS for D in DIMS for alg in ALGS for r in range(RUNS)]
    jobs.sort(key=lambda j: -j[2] * B.get(j[1])["nfe"])
    with Pool(2) as p:
        out = p.map(job, jobs, chunksize=2)
    save_records("../results/E4_hd.jsonl.gz", out)
    print("E4 done in %.0fs" % (time.time() - t0))
