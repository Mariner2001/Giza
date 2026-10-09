"""E5: COCO/BBOB noiseless suite (24 functions, instances 1-5), D in {2,5,10,20}, budget 1e4*D evaluations.
Optima are placed at random in [-4,4]^D, i.e. not at the origin. Instances 1-5.  Runs stop when f - f_opt < 1e-8.
Stores, for every (algorithm, D, function, instance), the best-so-far precision trace."""
import sys
import time
import numpy as np
from multiprocessing import Pool
import json
import cocoex
import algos as A
from io_utils import save_records

FOPT = json.load(open('../results/bbob_fopt.json'))  # Fopt read from the COCO logger headers

DIMS = [2, 5, 10, 20]
BUDGET = 10000
ALGS = {"GPC": (A.gpc, {"mode": "mult", "theta": 10.0}), "GPC-A": (A.gpc, {"mode": "add", "theta": 10.0}),
        "PSO": (A.pso, {}), "DE": (A.de, {}), "CMA-ES": (A.cmaes, {}), "RS": (A.rs, {"batch": 20})}
AID = {k: i for i, k in enumerate(ALGS)}


class Stop(Exception):
    pass


def run_alg_dim(arg):
    alg, D = arg
    suite = cocoex.Suite("bbob", "", "dimensions:%d instance_indices:1-5" % D)
    out = []
    fn, kw = ALGS[alg]
    t0 = time.time()
    for prob in suite:
        fopt = FOPT['%d|%d|%d' % (D, prob.id_function, prob.id_instance)]
        state = {"hit": False}

        def f(X, prob=prob):
            y = np.fromiter((prob(x) for x in X), dtype=float, count=X.shape[0])
            return y

        # wrap evaluator to stop at the final target
        ev_cls = A.Evaluator

        class EvStop(ev_cls):
            def __call__(self, X):
                y = super().__call__(X)
                if self.best - fopt < 1e-8:
                    self.nfe = self.used
                return y

        A.Evaluator = EvStop
        try:
            res = fn(f, prob.lower_bounds, prob.upper_bounds, BUDGET * D,
                     np.random.default_rng([55, AID[alg], D, prob.id_function, prob.id_instance]), **kw)
        finally:
            A.Evaluator = ev_cls
        tn, tf = res["trace_n"], res["trace_f"] - fopt
        keep = np.r_[True, tf[1:] < tf[:-1]]
        keep[-1] = True
        out.append(dict(alg=alg, D=D, fun=prob.id_function, inst=prob.id_instance,
                        tn=tn[keep].astype(np.int32), tf=tf[keep].astype(np.float64),
                        nfe=res["nfe"], xnorm=float(np.linalg.norm(res["best_x"]) / np.sqrt(D))))
    print("BBOB %s D=%d done in %.0fs" % (alg, D, time.time() - t0), flush=True)
    return out


if __name__ == "__main__":
    algs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ALGS)
    jobs = [(a, D) for D in sorted(DIMS, reverse=True) for a in algs]
    t0 = time.time()
    res = []
    with Pool(2) as p:
        for r in p.imap_unordered(run_alg_dim, jobs):
            res.extend(r)
    tag = "main" if set(algs) == {"GPC", "GPC-A", "PSO", "DE", "CMA-ES"} else "_".join(algs).replace("-", "")
    save_records("../results/E5_bbob_%s.jsonl.gz" % tag, res)
    print("E5 finished in %.0fs" % (time.time() - t0))
