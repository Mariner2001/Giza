"""E4b: dummy-variable check.  PSO (settings of Table 2 of the original study) on four two-dimensional functions,
run with D = 2 and with D = 5000, i.e. with 4998 variables that do not enter the objective.  PSO updates every
coordinate independently, so the two active coordinates must follow exactly the same dynamics in both cases.
We verify this directly: in each pair of runs the two active coordinates receive the same random numbers
(the extra coordinates use an independent stream), and the best values found are compared for equality.
1000 evaluations per run, as in Tables 3 and 7 of the original study; 50 pairs of runs per function."""
import time
import numpy as np
from multiprocessing import Pool
import algos as A
import bench as B
from io_utils import save_records

PAIRS = 50
FIDS = (2, 4, 15, 26)
D_BIG = 5000


class SplitRNG:
    """Random-number source whose first two columns reproduce the stream of a two-dimensional run."""

    def __init__(self, seed):
        self.a = np.random.default_rng(seed)
        self.b = np.random.default_rng(list(seed) + [999])

    def random(self, shape):
        n, D = shape
        return np.hstack([self.a.random((n, 2)), self.b.random((n, D - 2))])


def job(a):
    fid, r = a
    info = B.get(fid)
    seed = [44, fid, r]
    r2 = A.pso(info["f"], np.full(2, info["lb"]), np.full(2, info["ub"]), 1000, np.random.default_rng(seed))
    rb = A.pso(info["f"], np.full(D_BIG, info["lb"]), np.full(D_BIG, info["ub"]), 1000, SplitRNG(seed))
    return dict(fid=fid, run=r, best_f_2=float(r2["best_f"]), best_f_big=float(rb["best_f"]),
                identical=bool(r2["best_f"] == rb["best_f"]), nfe_2=int(r2["nfe"]), nfe_big=int(rb["nfe"]))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(fid, r) for fid in FIDS for r in range(PAIRS)]
    with Pool(2) as p:
        out = p.map(job, jobs, chunksize=5)
    save_records("../results/E4b_dummy.jsonl.gz", out)
    print("E4b done in %.0fs; identical pairs: %d of %d" % (time.time() - t0, sum(o["identical"] for o in out), len(out)))
