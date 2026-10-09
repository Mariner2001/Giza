"""Equivalence test: the authors' MATLAB code (run in GNU Octave) and our Python GPC, fed with the same random numbers.

For every case, one set of random numbers is drawn with numpy and written as three streams in the order in which
GPC.m consumes them (rand, randi, unifrnd).  The script, unchanged apart from the problem definition and the
parameters (make_variants.py), is run in Octave with rand, randi and unifrnd replaced by functions that read
these streams (shadow_tape/).  algos.gpc is run with a generator that returns the same numbers.  The final
populations and the best value after every iteration are compared.

Usage:  python3 check_equivalence.py   (needs octave-cli; writes ../../results/matlab_equivalence.json)
"""
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import algos  # noqa: E402
import bench as B  # noqa: E402
from make_variants import read_official, make_variant, SHA256_USED  # noqa: E402

WORK = os.path.join(HERE, "work")
N = 20  # nPop of GPC.m

PYF = {"Sphere": B.sphere, "Rastrigin": B.rastrigin, "Rosenbrock": B.rosenbrock, "Michalewicz": B.michalewicz,
       "Beale": B.beale}

# The same functions evaluated with the floating-point operations of the MATLAB/Octave files in funcs/:
# left-to-right summation (numpy sums pairwise) and the C library's cos, sin and pow (numpy may use its own
# vectorised versions).  Objective values then agree to the last bit, so that ties between candidates are
# broken identically and the comparison isolates the algorithm.
_cos, _sin, _pow = (np.vectorize(fn, otypes=[float]) for fn in (math.cos, math.sin, math.pow))


def _seqsum(T):
    s = T[:, 0].copy()
    for j in range(1, T.shape[1]):
        s = s + T[:, j]
    return s


PYF_EXACT = {
    "Sphere": lambda X: _seqsum(X * X),
    "Rastrigin": lambda X: 10.0 * X.shape[1] + _seqsum(X * X - 10.0 * _cos((2 * math.pi) * X)),
    "Rosenbrock": lambda X: _seqsum(100.0 * (X[:, 1:] - X[:, :-1] * X[:, :-1]) ** 2 + (X[:, :-1] - 1.0) ** 2),
    "Michalewicz": lambda X: -_seqsum(_sin(X) * _pow(_sin(np.arange(1, X.shape[1] + 1) * (X * X) / math.pi), 20.0)),
    # scalar powers a^b are computed by Octave with pow(a, b) (element-wise x.^2 with x.*x)
    "Beale": lambda X: (_pow(1.5 - X[:, 0] + X[:, 0] * X[:, 1], 2.0)
                        + _pow(2.25 - X[:, 0] + X[:, 0] * _pow(X[:, 1], 2.0), 2.0)
                        + _pow(2.625 - X[:, 0] + X[:, 0] * _pow(X[:, 1], 3.0), 2.0)),
}

# name, function, D, lb, ub, iterations, theta, update, translated
CASES = [
    ("as_released", "Sphere", 30, -5.12, 5.12, 1000, 14, "mult", False),   # GPC.m exactly as distributed
    ("sphere_translated", "Sphere", 30, -5.12, 5.12, 1000, 14, "mult", True),
    ("rastrigin_theta10", "Rastrigin", 30, -5.12, 5.12, 499, 10, "mult", False),
    ("rosenbrock_theta10", "Rosenbrock", 30, -5.0, 10.0, 499, 10, "mult", False),
    ("michalewicz", "Michalewicz", 30, 0.0, "pi", 49, 14, "mult", False),
    ("beale_theta10", "Beale", 2, -4.5, 4.5, 49, 10, "mult", False),
    ("rastrigin_additive", "Rastrigin", 30, -5.12, 5.12, 499, 10, "add", False),
    ("beale_additive", "Beale", 2, -4.5, 4.5, 49, 14, "add", False),
]
SEEDS = [1, 2, 3]


class TapeRNG:
    """Stands in for numpy.random.Generator in algos.gpc and returns the pre-drawn numbers in the order used there."""

    def __init__(self, U0, V0, MU, EPS, S, K0):
        self.U0, self.V0, self.MU, self.EPS, self.S, self.K0 = U0, V0, MU, EPS, S, K0
        self.init_done = False
        self.t = 0
        self.k = 0  # position within a generation: 0 v0, 1 mu, 2 eps, 3 substitution

    def random(self, size=None):
        if not self.init_done:
            self.init_done = True
            assert tuple(np.atleast_1d(size)) == self.U0.shape
            return self.U0.copy()
        arr = (self.V0, self.MU, self.EPS, self.S)[self.k][self.t]
        assert tuple(np.atleast_1d(size)) == arr.shape, (size, arr.shape)
        self.k += 1
        return arr.copy()

    def integers(self, low, high=None, size=None):
        assert self.k == 4 and low == 0
        r = self.K0[self.t].copy()
        self.t += 1
        self.k = 0
        return r


def draw(seed, D, T):
    g = np.random.default_rng(seed)
    return dict(U0=g.random((N, D)), V0=g.random((T, N)), MU=g.random((T, N)), EPS=g.random((T, N, D)),
                S=g.random((T, N, D)), K0=g.integers(0, D, (T, N)))


def write_tapes(R, folder):
    """Streams in the order of consumption in GPC.m: loops it = 1..T, i = 1..nPop, k = 1..nVar."""
    os.makedirs(folder, exist_ok=True)
    T, n, D = R["EPS"].shape
    uni = np.concatenate([R["U0"].ravel(), R["EPS"].ravel()])
    rnd = []
    for t in range(T):
        for i in range(n):
            k0 = R["K0"][t, i]
            rnd.append([R["V0"][t, i], R["MU"][t, i]])
            rnd.append(np.delete(R["S"][t, i], k0))  # no rand call for k == k0 (short-circuit ||)
    rnd = np.concatenate([np.atleast_1d(np.asarray(a, float)) for a in rnd])
    uni.astype("<f8").tofile(os.path.join(folder, "unifrnd.bin"))
    rnd.astype("<f8").tofile(os.path.join(folder, "rand.bin"))
    (R["K0"].ravel() + 1).astype("<f8").tofile(os.path.join(folder, "randi.bin"))
    return dict(unifrnd=uni.size, rand=rnd.size, randi=R["K0"].size)


def run_case(case, seed, text):
    name, fun, D, lb, ub, T, theta, mode, translated = case
    tag = "%s_s%d" % (name, seed)
    shift = None
    if translated:  # translation as in E2a: o ~ U(-w/4, w/4), function and box moved together
        w = 10.24
        shift = np.random.default_rng(1000 + seed).uniform(-w / 4, w / 4, D)
    if name == "as_released":
        script_text = text  # unchanged file
    else:
        script_text, _ = make_variant(text, fun=fun, D=D, lb=lb, ub=ub, iters=T, theta=theta, mode=mode,
                                      shift=shift)
    script = os.path.join(WORK, "GPC_%s.m" % tag)
    with open(script, "w", newline="") as fh:
        fh.write(script_text)
    R = draw(seed, D, T)
    tapes = os.path.join(WORK, "tapes", tag)
    sizes = write_tapes(R, tapes)
    out = os.path.join(WORK, "out_%s.mat" % tag)
    env = dict(os.environ, GPC_TAPE_DIR=tapes, GPC_SCRIPT=script, GPC_OUT=out)
    t0 = time.time()
    p = subprocess.run(["octave-cli", "--no-gui", "-q", "--no-window-system", "run_tape.m"], cwd=HERE, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t_oct = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    M = loadmat(out)
    P_m, C_m, best_m = M["P"], M["C"].ravel(), M["BestCost"].ravel()
    st = M["st"]

    def maxrel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        den = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)
        return float(np.max(np.abs(a - b) / den))

    # ---- Python (algos.gpc unchanged), with the objective evaluated in the same order as in Octave and with numpy
    rec = dict(case=name, seed=seed, fun=fun, D=D, iterations=T, theta=theta, update=mode, translated=translated,
               streams_fully_used=bool(np.all(st[:, 0] == st[:, 1])), stream_sizes=sizes,
               best_final_octave=float(best_m[-1]), seconds_octave=round(t_oct, 1))
    for label, table in (("exact", PYF_EXACT), ("numpy", PYF)):
        lbv = np.full(D, math.pi if lb == "pi" else float(lb))
        ubv = np.full(D, math.pi if ub == "pi" else float(ub))
        f = table[fun]
        if shift is not None:
            lbv, ubv = lbv + shift, ubv + shift
            f = (lambda X, f=f, o=shift: f(X - o))
        res = algos.gpc(f, lbv, ubv, N + N * T, TapeRNG(**R), n=N, theta=float(theta), mode=mode, return_pop=True)
        P_py, best_py = res["pop"], res["trace_f"][1:]
        rec[label] = dict(populations_identical=bool(np.array_equal(P_py, P_m)),
                          best_values_identical=bool(np.array_equal(best_py, best_m)),
                          max_rel_diff_positions=maxrel(P_py, P_m),
                          max_rel_diff_costs=maxrel(f(P_py), C_m),
                          max_rel_diff_best_per_iteration=maxrel(best_py, best_m),
                          best_final_python=float(best_py[-1]))
    return rec


def main():
    text, sha = read_official()
    if sha != SHA256_USED:
        print("WARNING: GPC.m differs from the file used in the paper (sha256 %s)" % sha)
    os.makedirs(WORK, exist_ok=True)
    jobs = [(c, s) for c in CASES for s in SEEDS]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        out = list(ex.map(lambda cs: run_case(cs[0], cs[1], text), jobs))
    for r in out:
        e, q = r["exact"], r["numpy"]
        print("%-20s seed %d  streams ok=%-5s | same-order objective: identical pop=%-5s best=%-5s | numpy objective: "
              "identical pop=%-5s  max rel. diff. best %.1e | final %.4e / %.4e (%.0f s)" % (
                  r["case"], r["seed"], r["streams_fully_used"], e["populations_identical"], e["best_values_identical"],
                  q["populations_identical"], q["max_rel_diff_best_per_iteration"], r["best_final_octave"],
                  e["best_final_python"], r["seconds_octave"]))
    ver = subprocess.run(["octave-cli", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    summary = dict(gpc_m_sha256=sha, octave=ver, numpy=np.__version__, runs=len(out),
                   all_streams_fully_used=all(r["streams_fully_used"] for r in out),
                   same_order_objective=dict(
                       populations_identical=sum(r["exact"]["populations_identical"] for r in out),
                       best_values_identical=sum(r["exact"]["best_values_identical"] for r in out)),
                   numpy_objective=dict(
                       populations_identical=sum(r["numpy"]["populations_identical"] for r in out),
                       max_rel_diff_final_best=max(abs(r["numpy"]["best_final_python"] - r["best_final_octave"])
                                                   / max(abs(r["best_final_octave"]), 1e-300) for r in out)),
                   details=out)
    os.makedirs(os.path.join(HERE, "..", "..", "results"), exist_ok=True)
    json.dump(summary, open(os.path.join(HERE, "..", "..", "results", "matlab_equivalence.json"), "w"), indent=1)
    print({k: v for k, v in summary.items() if k != "details"})


if __name__ == "__main__":
    main()
