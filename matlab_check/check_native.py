"""GPC.m run in GNU Octave with Octave's own random numbers, compared with our implementation.

A  the script as released (Sphere, D = 30, theta = 14 deg, 1000 iterations = 20020 evaluations), 30 runs
   (Octave seeds 1000 + r), against algos.gpc with the same settings (30 runs);
B  the same script on the translated Sphere function of E2a (translation of run r of E2a, function 27);
C  the budget and ramp angle of E1 (49 iterations = 1000 evaluations, theta = 10 deg), original and translated
   (50 runs each), against our E1 and E2a results for GPC on Sphere.
Writes ../results/matlab_native.json."""
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import algos as A  # noqa: E402
import bench as B  # noqa: E402
from common import make_problem  # noqa: E402
from io_utils import load_records  # noqa: E402
from make_variants import make_variant, vec  # noqa: E402

RES = os.path.join(HERE, "..", "results")


def octave_run(var, seed, workdir):
    out = os.path.join(workdir, "out_%d.bin" % seed)
    env = dict(os.environ, GPC_VARIANT=var, GPC_SEED=str(seed), GPC_OUT=out)
    subprocess.run(["octave", "--no-gui", "--quiet", "--no-window-system", "run_native.m"], cwd=HERE, env=env,
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    v = np.fromfile(out, dtype="<f8")
    return float(v[int(v[0])])


def shift(r):
    return make_problem(27, "trans", np.random.default_rng([2, 27, r]))["o"]


def variant(name, o, iters, theta, workdir):
    if o is None:
        return make_variant(name, "Sphere(x)", 30, "-5.12", "5.12", iters, theta, outdir=workdir)
    return make_variant(name, "Sphere(x-%s)" % vec(o), 30, "-5.12+%s" % vec(o), "5.12+%s" % vec(o), iters, theta,
                        outdir=workdir)


if __name__ == "__main__":
    w = tempfile.mkdtemp(prefix="gpc_native_", dir=HERE)
    out = {}
    # A: as released
    v = variant("v_released", None, 1000, 14, w)
    oct_a = [octave_run(v, 1000 + r, w) for r in range(30)]
    sph = B.get(27)["f"]
    py_a = [A.gpc(sph, np.full(30, -5.12), np.full(30, 5.12), 20020, np.random.default_rng([1000 + r]), theta=14.0)["best_f"]
            for r in range(30)]
    out["A"] = dict(octave_median=float(np.median(oct_a)), python_median=float(np.median(py_a)),
                    p=float(stats.mannwhitneyu(oct_a, py_a).pvalue), octave=oct_a, python=py_a)
    print("A", {k: v for k, v in out["A"].items() if "median" in k or k == "p"}, flush=True)
    # B: as released, translated
    oct_b = []
    for r in range(30):
        vb = variant("v_released_tr%d" % r, shift(r), 1000, 14, w)
        oct_b.append(octave_run(vb, 2000 + r, w))
    out["B"] = dict(octave_median=float(np.median(oct_b)), octave=oct_b)
    print("B", out["B"]["octave_median"], flush=True)
    # C: budget and angle of E1, original and translated, against E1/E2a
    vc = variant("v_e1", None, 49, 10, w)
    oct_c1 = [octave_run(vc, 3000 + r, w) for r in range(50)]
    oct_c2 = []
    for r in range(50):
        vt = variant("v_e2a_%d" % r, shift(r), 49, 10, w)
        oct_c2.append(octave_run(vt, 4000 + r, w))
    e1 = [x["err"] for x in load_records(os.path.join(RES, "E1.jsonl.gz")) if x["fid"] == 27 and x["alg"] == "GPC"]
    e2 = [x["err"] for x in load_records(os.path.join(RES, "E2a.jsonl.gz")) if x["fid"] == 27 and x["alg"] == "GPC"]
    out["C"] = dict(octave_orig_median=float(np.median(oct_c1)), ours_orig_median=float(np.median(e1)),
                    p_orig=float(stats.mannwhitneyu(oct_c1, e1).pvalue),
                    octave_trans_median=float(np.median(oct_c2)), ours_trans_median=float(np.median(e2)),
                    p_trans=float(stats.mannwhitneyu(oct_c2, e2).pvalue))
    print("C", out["C"], flush=True)
    json.dump(out, open(os.path.join(RES, "matlab_native.json"), "w"), indent=1)
