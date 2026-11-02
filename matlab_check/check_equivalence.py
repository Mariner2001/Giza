"""Equivalence of our GPC (algos.gpc) with the authors' MATLAB script GPC.m, run in GNU Octave.

Both implementations consume the same random numbers: Python writes them to tapes in the order in which GPC.m
draws them (unifrnd: initial positions, then epsilon for every iteration and worker; rand: V0, Mu and the
substitution draws for k != k0; randi: k0), Octave replays them through shadowed rand/randi/unifrnd, and
algos.gpc receives the same numbers through TapeRNG.  The objectives are evaluated by the same floating-point
operations in both languages (left-to-right sums).  For each case the best cost after every iteration and the
final population are compared.  Writes ../results/matlab_equivalence.json."""
import json
import math
import os
import subprocess
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import algos as A  # noqa: E402
from common import make_problem  # noqa: E402
from make_variants import make_variant, vec  # noqa: E402

N_POP = 20


def _sum(it):
    s = 0.0
    for t in it:
        s += t
    return s


# objectives with the operation order of funcs/*.m
PYF_EXACT = {
    "Sphere": lambda r: _sum(v * v for v in r),
    "Rastrigin": lambda r: 10 * len(r) + _sum(v * v - 10 * math.cos(2 * math.pi * v) for v in r),
    "Rosenbrock": lambda r: _sum(100 * (b - a ** 2) ** 2 + (a - 1) ** 2 for a, b in zip(r[:-1], r[1:])),
    "Michalewicz": lambda r: -_sum(math.sin(v) * math.pow(math.sin(i * (v ** 2) / math.pi), 20)
                                   for i, v in enumerate(r, 1)),
    "Beale": lambda r: ((1.5 - r[0] + r[0] * r[1]) ** 2 + (2.25 - r[0] + r[0] * r[1] ** 2) ** 2
                        + (2.625 - r[0] + r[0] * r[1] ** 3) ** 2),
}

# name: (function, D, lower, upper, iterations, theta, additive, translated)
CASES = {
    "as_released": ("Sphere", 30, -5.12, 5.12, 1000, 14, False, False),
    "sphere_translated": ("Sphere", 30, -5.12, 5.12, 1000, 14, False, True),
    "rastrigin_theta10": ("Rastrigin", 30, -5.12, 5.12, 499, 10, False, False),
    "rosenbrock_theta10": ("Rosenbrock", 30, -5.0, 10.0, 499, 10, False, False),
    "michalewicz": ("Michalewicz", 30, 0.0, math.pi, 49, 14, False, False),
    "beale_theta10": ("Beale", 2, -4.5, 4.5, 49, 10, False, False),
    "rastrigin_additive": ("Rastrigin", 30, -5.12, 5.12, 499, 10, True, False),
    "beale_additive": ("Beale", 2, -4.5, 4.5, 49, 14, True, False),
}


class TapeRNG:
    """Serves algos.gpc the numbers that GPC.m reads from the tapes."""

    def __init__(self, U0, V0, MU, EPS, K0, SUB):
        self.U0, self.V0, self.MU, self.EPS, self.K0, self.SUB = U0, V0, MU, EPS, K0, SUB
        self.first = True
        self.it = 0
        self.step = 0

    def random(self, shape):
        if self.first:
            self.first = False
            return self.U0.copy()
        it, s = self.it, self.step
        self.step += 1
        if s == 0:
            return self.V0[it].copy()
        if s == 1:
            return self.MU[it].copy()
        if s == 2:
            return self.EPS[it].copy()
        if s == 3:  # substitution draws; position k0 is forced and receives no draw in GPC.m
            n, D = shape
            M = np.zeros((n, D))
            for i in range(n):
                k0 = int(self.K0[it, i]) - 1
                M[i, [k for k in range(D) if k != k0]] = self.SUB[it, i]
            return M
        raise RuntimeError("unexpected call")

    def integers(self, lo, hi, size=None):
        it = self.it
        self.it += 1
        self.step = 0
        return self.K0[it].astype(np.int64) - 1


def tapes(rng, D, iters):
    U0 = rng.random((N_POP, D))
    V0 = rng.random((iters, N_POP))
    MU = rng.random((iters, N_POP))
    EPS = rng.random((iters, N_POP, D))
    K0 = rng.integers(1, D + 1, size=(iters, N_POP)).astype(float)
    SUB = rng.random((iters, N_POP, D - 1))
    unif = [U0.ravel()] + [EPS[t, i] for t in range(iters) for i in range(N_POP)]
    rnd = [np.r_[V0[t, i], MU[t, i], SUB[t, i]] for t in range(iters) for i in range(N_POP)]
    return (U0, V0, MU, EPS, K0, SUB), np.concatenate(unif), np.concatenate(rnd), K0.ravel()


def run_case(name, seed, workdir):
    fname, D, lo, hi, iters, theta, additive, translated = CASES[name]
    lb, ub = np.full(D, lo), np.full(D, hi)
    o = np.zeros(D)
    if translated:
        o = make_problem(27, "trans", np.random.default_rng([2, 27, seed]))["o"]
        lb, ub = lb + o, ub + o
    if translated:
        cost = "%s(x-%s)" % (fname, vec(o))
        vmin, vmax = "%.17g+%s" % (lo, vec(o)), "%.17g+%s" % (hi, vec(o))
    else:
        cost = "%s(x)" % fname
        vmin, vmax = ("pi" if lo == math.pi else "%.17g" % lo), ("pi" if hi == math.pi else "%.17g" % hi)
    var = make_variant("v_%s_%d" % (name, seed), cost, D, vmin, vmax, iters, theta, additive, outdir=workdir)
    T, unif, rnd, ri = tapes(np.random.default_rng([77, list(CASES).index(name), seed]), D, iters)
    tdir = os.path.join(workdir, "tape_%s_%d" % (name, seed))
    os.makedirs(tdir, exist_ok=True)
    unif.astype("<f8").tofile(os.path.join(tdir, "unifrnd.bin"))
    rnd.astype("<f8").tofile(os.path.join(tdir, "rand.bin"))
    ri.astype("<f8").tofile(os.path.join(tdir, "randi.bin"))
    out = os.path.join(tdir, "out.bin")
    env = dict(os.environ, GPC_VARIANT=var, GPC_TAPE_DIR=tdir, GPC_OUT=out)
    subprocess.run(["octave", "--no-gui", "--quiet", "--no-window-system", "run_tape.m"], cwd=HERE, env=env,
                   check=True, stdout=subprocess.DEVNULL)
    v = np.fromfile(out, dtype="<f8")
    nb = int(v[0])
    best_m = v[1:1 + nb]
    r, c = int(v[1 + nb]), int(v[2 + nb])
    P_m = v[3 + nb:3 + nb + r * c].reshape(r, c)
    # Python
    pyf = PYF_EXACT[fname]
    f = (lambda X: np.array([pyf(list(row)) for row in X])) if not translated else \
        (lambda X: np.array([pyf(list(row)) for row in (X - o)]))
    res = A.gpc(f, lb, ub, N_POP * (iters + 1), TapeRNG(*T), theta=float(theta),
                mode="add" if additive else "mult", return_pop=True)
    best_p = res["trace_f"][1:]
    same_best = bool(np.array_equal(best_m, best_p))
    same_pop = bool(np.array_equal(P_m, res["pop"]))
    maxdiff = float(np.max(np.abs(P_m - res["pop"]))) if not same_pop else 0.0
    ulps = float(np.max(np.abs(P_m - res["pop"]) / np.maximum(np.spacing(np.abs(res["pop"])), 1e-300))) if not same_pop else 0.0
    return dict(case=name, seed=seed, iterations=iters, same_best_every_iteration=same_best,
                identical_final_population=same_pop, max_abs_diff=maxdiff, max_ulps=ulps,
                final_best_octave=float(best_m[-1]), final_best_python=float(best_p[-1]))


if __name__ == "__main__":
    workdir = tempfile.mkdtemp(prefix="gpc_eq_", dir=os.path.join(HERE))
    out = []
    for name in CASES:
        for seed in (1, 2, 3):
            r = run_case(name, seed, workdir)
            out.append(r)
            print(r, flush=True)
    summary = dict(runs=len(out), same_best_every_iteration=sum(r["same_best_every_iteration"] for r in out),
                   identical_final_population=sum(r["identical_final_population"] for r in out),
                   max_ulps=max(r["max_ulps"] for r in out))
    print(summary)
    json.dump(dict(summary=summary, runs=out), open(os.path.join(HERE, "..", "results", "matlab_equivalence.json"), "w"), indent=1)
