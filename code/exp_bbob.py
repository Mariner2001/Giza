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
import algos_shade as S
from io_utils import save_records

FOPT = json.load(open('../results/bbob_fopt.json'))  # Fopt read from the COCO logger headers

DIMS = [2, 5, 10, 20]
BUDGET = 10000
ALGS = {"GPC": (A.gpc, {"mode": "mult", "theta": 10.0}), "GPC-A": (A.gpc, {"mode": "add", "theta": 10.0}),
        "PSO": (A.pso, {}), "DE": (A.de, {}), "CMA-ES": (A.cmaes, {}), "RS": (A.rs, {"batch": 20}),
        "L-SHADE": (S.lshade, {}), "jSO": (S.jso, {})}
AID = {k: i for i, k in enumerate(ALGS)}   # GPC 0, GPC-A 1, PSO 2, DE 3, CMA-ES 4, RS 5, L-SHADE 6, jSO 7


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
    # usage: python3 exp_bbob.py ALG1,ALG2,... [output tag]
    # Each (algorithm, dimension) job is stored in ../results/E5_parts/ as soon as it is finished, so that an
    # interrupted run can be resumed; the parts are then combined into ../results/E5_bbob_<tag>.jsonl.gz.
    import os
    from io_utils import load_records
    algs = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ALGS)
    tag = sys.argv[2] if len(sys.argv) > 2 else (
        "main" if set(algs) == {"GPC", "GPC-A", "PSO", "DE", "CMA-ES"} else "_".join(algs).replace("-", ""))
    parts = "../results/E5_parts"
    os.makedirs(parts, exist_ok=True)
    part = lambda a, D: os.path.join(parts, "%s_D%d.jsonl.gz" % (a.replace("-", ""), D))
    jobs = [(a, D) for D in sorted(DIMS, reverse=True) for a in algs if not os.path.exists(part(a, D))]
    # longest jobs first
    jobs.sort(key=lambda j: -(j[1] * (8 if j[0] == "CMA-ES" else 1)))
    t0 = time.time()
    with Pool(2) as p:
        for r in p.imap_unordered(run_alg_dim, jobs):
            save_records(part(r[0]["alg"], r[0]["D"]), r)
    res = []
    for D in DIMS:
        for a in algs:
            res.extend(load_records(part(a, D)))
    save_records("../results/E5_bbob_%s.jsonl.gz" % tag, res)
    print("E5 finished in %.0fs" % (time.time() - t0))
