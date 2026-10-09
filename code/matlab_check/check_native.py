"""The authors' GPC.m run with Octave's own random numbers, on the original and on the translated Sphere function.

A  GPC.m exactly as distributed (Sphere, D = 30, [-5.12, 5.12]^30, 1000 iterations = 20 020 evaluations, theta = 14)
   versus algos.gpc with the same settings.
B  the same, with the function and the box translated as in experiment E2a (o ~ U(-w/4, w/4)^30, w = 10.24).
C  budget of the original study for Sphere (1000 evaluations = 49 iterations) and theta = 10 (Table 2), original and
   translated with the translations of runs 0..49 of E2a, versus our stored results of E1 and E2a.

Usage:  python3 check_native.py   (needs octave-cli; writes ../../results/matlab_native.json)
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.io import loadmat
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import algos  # noqa: E402
import bench as B  # noqa: E402
from common import make_problem  # noqa: E402
from io_utils import load_records  # noqa: E402
from make_variants import read_official, make_variant  # noqa: E402

WORK = os.path.join(HERE, "work", "native")
RES = os.path.join(HERE, "..", "..", "results")
N = 20


def e2a_shift(r):
    """Translation of run r of experiment E2a for Sphere (fid 27)."""
    return make_problem(27, "trans", np.random.default_rng([2, 27, r]))["o"]


def octave_run(tag, script_text, seed):
    script = os.path.join(WORK, "GPC_%s.m" % tag)
    with open(script, "w", newline="") as fh:
        fh.write(script_text)
    out = os.path.join(WORK, "out_%s.mat" % tag)
    env = dict(os.environ, GPC_SCRIPT=script, GPC_OUT=out, GPC_SEED=str(seed))
    p = subprocess.run(["octave-cli", "--no-gui", "-q", "run_native.m"], cwd=HERE, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return float(loadmat(out)["BestCost"].ravel()[-1])


def summary(v):
    v = np.asarray(v, float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(n=int(v.size), median=float(med), q1=float(q1), q3=float(q3), min=float(v.min()), max=float(v.max()))


def compare(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def main(runs_ab=30, runs_c=50):
    os.makedirs(WORK, exist_ok=True)
    text, sha = read_official()
    jobs = []
    for r in range(runs_ab):
        jobs.append(("A", r, text, 1000 + r))
        jobs.append(("B", r, make_variant(text, iters=1000, theta=14, shift=e2a_shift(r))[0], 2000 + r))
    for r in range(runs_c):
        jobs.append(("C_orig", r, make_variant(text, iters=49, theta=10)[0], 3000 + r))
        jobs.append(("C_trans", r, make_variant(text, iters=49, theta=10, shift=e2a_shift(r))[0], 4000 + r))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as ex:
        vals = list(ex.map(lambda j: (j[0], octave_run("%s_r%d" % (j[0], j[1]), j[2], j[3])), jobs))
    oct_ = {k: [v for s, v in vals if s == k] for k in ("A", "B", "C_orig", "C_trans")}
    # Python counterparts
    lb, ub = np.full(30, -5.12), np.full(30, 5.12)
    pyA = [algos.gpc(B.sphere, lb, ub, N + N * 1000, np.random.default_rng([90, r]), theta=14.0)["best_f"]
           for r in range(runs_ab)]
    pyB = []
    for r in range(runs_ab):
        o = e2a_shift(r)
        pyB.append(algos.gpc(lambda X, o=o: B.sphere(X - o), lb + o, ub + o, N + N * 1000,
                             np.random.default_rng([91, r]), theta=14.0)["best_f"])
    e1 = [x["err"] for x in load_records(os.path.join(RES, "E1.jsonl.gz")) if x["fid"] == 27 and x["alg"] == "GPC"]
    e2a = [x["err"] for x in load_records(os.path.join(RES, "E2a.jsonl.gz")) if x["fid"] == 27 and x["alg"] == "GPC"]
    out = dict(
        gpc_m_sha256=sha,
        A_as_released=dict(octave=summary(oct_["A"]), python=summary(pyA), mannwhitney_p=compare(oct_["A"], pyA)),
        B_translated=dict(octave=summary(oct_["B"]), python=summary(pyB), mannwhitney_p=compare(oct_["B"], pyB)),
        C_paper_budget_original=dict(octave=summary(oct_["C_orig"]), python_E1=summary(e1),
                                     mannwhitney_p=compare(oct_["C_orig"], e1)),
        C_paper_budget_translated=dict(octave=summary(oct_["C_trans"]), python_E2a=summary(e2a),
                                       mannwhitney_p=compare(oct_["C_trans"], e2a)),
    )
    json.dump(out, open(os.path.join(RES, "matlab_native.json"), "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, {kk: (("median %.3g [%.3g, %.3g]" % (vv["median"], vv["q1"], vv["q3"])) if isinstance(vv, dict)
                           else "%.3g" % vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
