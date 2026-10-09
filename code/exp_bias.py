"""E3: structural-bias test (Kononova et al. 2015): optimise a fitness that is pure noise, f0(x) ~ U(0,1),
and record where the final best solution lies.  An unbiased algorithm returns positions that are uniformly
distributed in the box.  Two boxes are used: [-5,5]^30 (origin at the centre) and [0,10]^30 (origin at a corner).
"""
import time
import numpy as np
from multiprocessing import Pool
import algos as A
from io_utils import save_records

D = 30
NFE = 10000
RUNS = 100
BOXES = {"centre": (-5.0, 5.0), "corner": (0.0, 10.0)}
ALGS = {"GA": (A.ga, {}), "ICA": (A.ica, {}), "PSO": (A.pso, {}), "DE": (A.de, {}), "FA": (A.fa, {}),
        "GPC": (A.gpc, {"mode": "mult"}), "GPC-A": (A.gpc, {"mode": "add"}), "CMA-ES": (A.cmaes, {}), "RS": (A.rs, {})}
AID = {k: i for i, k in enumerate(ALGS)}


def job(a):
    box, alg, r = a
    lo, hi = BOXES[box]
    noise = np.random.default_rng([99, AID[alg], r, 1 if box == "centre" else 2])
    f = lambda X: noise.random(X.shape[0])
    fn, kw = ALGS[alg]
    res = fn(f, np.full(D, lo), np.full(D, hi), NFE, np.random.default_rng([98, AID[alg], r]), **kw)
    return dict(box=box, alg=alg, run=r, xn=(res["best_x"] - lo) / (hi - lo))


if __name__ == "__main__":
    jobs = [(b, a, r) for b in BOXES for a in ALGS for r in range(RUNS)]
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(job, jobs, chunksize=10)
    save_records("../results/E3_bias.jsonl.gz", out)
    print("E3 done in %.0fs" % (time.time() - t0))
