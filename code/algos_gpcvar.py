"""GPC with switchable components, for the ablation, parameter-sensitivity and initialisation study (E7).

gpc_var(...) with the default arguments is identical to algos.gpc (same random-number stream, same result);
this is checked by check_gpcvar() below.  Options:
  theta, mu_min, mu_max, n, pss   parameters of Table 2 of the original paper
  use_d=False                     drop the stone displacement d (alpha = 0)
  bounds="reinit"                 components outside the box are redrawn uniformly instead of clipped
  elitist=False                   every child replaces its parent (no (mu+lambda) truncation)
  init="lhs" | "corner"           Latin-hypercube initial population, or uniform initial population in the
                                  upper quarter [l + 0.75 (u - l), u] of every coordinate range
"""
import math
import numpy as np
from algos import Evaluator, _clip


def gpc_var(f, lb, ub, nfe, rng, n=20, g=9.8, theta=10.0, mu_min=1.0, mu_max=10.0, pss=0.5, mode="mult",
            use_d=True, bounds="clip", elitist=True, init="uniform"):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    th = math.radians(theta)
    sin_t, cos_t = math.sin(th), math.cos(th)
    half = (ub - lb) / 2.0
    if init == "uniform":
        P = lb + (ub - lb) * rng.random((n, D))
    elif init == "lhs":
        U = (np.argsort(rng.random((n, D)), axis=0) + rng.random((n, D))) / n
        P = lb + (ub - lb) * U
    elif init == "corner":
        P = lb + (ub - lb) * (0.75 + 0.25 * rng.random((n, D)))
    else:
        raise ValueError(init)
    C = ev(P)
    while not ev.done:
        v0 = rng.random(n)
        mu = mu_min + (mu_max - mu_min) * rng.random(n)
        d = v0 ** 2 / (2 * g * (sin_t + mu * cos_t))
        if not use_d:
            d = np.zeros(n)
        x = v0 ** 2 / (2 * g * sin_t)
        eps = -half + 2 * half * rng.random((n, D))
        if mode == "mult":
            Y = (P + d[:, None]) * (x[:, None] * eps)
        else:
            Y = (P + d[:, None]) + x[:, None] * eps
        if bounds == "clip":
            Y = _clip(Y, lb, ub)
        elif bounds == "reinit":
            out = (Y < lb) | (Y > ub)
            Y = np.where(out, lb + (ub - lb) * rng.random((n, D)), Y)
        else:
            raise ValueError(bounds)
        mask = rng.random((n, D)) <= pss
        k0 = rng.integers(0, D, size=n)
        mask[np.arange(n), k0] = True
        Z = np.where(mask, Y, P)
        CZ = ev(Z)
        if elitist:
            allP = np.vstack([P, Z])
            allC = np.concatenate([C, CZ])
            order = np.argsort(allC, kind="stable")[:n]
            P, C = allP[order], allC[order]
        else:
            P, C = Z, CZ
    return ev.result()


def check_gpcvar():
    """gpc_var with default options reproduces algos.gpc exactly."""
    import algos as A
    import bench as B
    for fid in (1, 2, 22, 27):
        info = B.get(fid)
        lb, ub = np.full(info["D"], info["lb"]), np.full(info["D"], info["ub"])
        for mode in ("mult", "add"):
            a = A.gpc(info["f"], lb, ub, 2000, np.random.default_rng([5, fid]), mode=mode)
            b = gpc_var(info["f"], lb, ub, 2000, np.random.default_rng([5, fid]), mode=mode)
            assert a["best_f"] == b["best_f"] and np.array_equal(a["best_x"], b["best_x"]), (fid, mode)
    return True


if __name__ == "__main__":
    print("gpc_var == algos.gpc:", check_gpcvar())
