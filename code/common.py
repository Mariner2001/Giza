"""Shared helpers: optimum values, problem construction (original / translated / shifted)."""
import numpy as np
import bench as B

# Optimal values on the boxes of Table 1 (corrected where Table 1 is wrong; see audit.py)
FSTAR = {fid: 0.0 for fid in range(1, 31)}
FSTAR.update({
    5: 0.07,                      # Bukin N.6 restricted to [-3,3]^2 (global optimum (-10,1) lies outside)
    6: -1.0316284534898774,
    8: -2.0626118708227397,
    9: 0.9980038377944498,        # De Jong N.5 (Shekel's foxholes); Table 1 says 0
    11: -1.0,
    12: -1.0,
    13: -959.6406627106155,
    15: -19.208502567767606,
    20: -29.63088385032169,       # Michalewicz, D = 30, m = 10; Table 1 gives the D = 2 value -1.8013
    25: 0.29257863203598045,      # Schaffer N.4 (standard cos^2 form)
    26: -186.73090883102378,
})
FSTAR_BY_D = {20: {2: -1.8013034100985492, 30: -29.63088385032169}}

ALG_ORDER = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "RS"]


def haar_rotation(D, rng):
    """Random orthogonal matrix with determinant +1, uniformly distributed (QR of a Gaussian matrix)."""
    Q, Rm = np.linalg.qr(rng.standard_normal((D, D)))
    Q = Q * np.sign(np.diag(Rm))[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def fstar(fid, D):
    if fid in FSTAR_BY_D and D in FSTAR_BY_D[fid]:
        return FSTAR_BY_D[fid][D]
    return FSTAR[fid]


def make_problem(fid, kind, rng_problem, D=None):
    """kind: 'orig' | 'trans' (function and box translated) | 'shift' (optimum moved inside same box)."""
    info = B.get(fid) if D is None else dict(B.get(fid), D=(D if B.get(fid)["scalable"] else 2))
    Dd = info["D"]
    lb = np.full(Dd, info["lb"], dtype=float)
    ub = np.full(Dd, info["ub"], dtype=float)
    f = info["f"]
    o = np.zeros(Dd)
    if kind == "orig":
        g = f
    elif kind == "trans":
        w = ub - lb
        o = rng_problem.uniform(-w / 4, w / 4)
        lb, ub = lb + o, ub + o
        g = (lambda X, f=f, o=o: f(X - o))
    elif kind == "shift":
        # optimum (originally at 0) moved to o, drawn uniformly in 80% of the box (CEC-style)
        o = rng_problem.uniform(0.8 * lb, 0.8 * ub)
        g = (lambda X, f=f, o=o: f(X - o))
    elif kind == "rot":
        # rotation about the origin (Haar-distributed orthogonal matrix); box and optimum (0) unchanged
        R = haar_rotation(Dd, rng_problem)
        g = (lambda X, f=f, R=R: f(X @ R.T))
    elif kind == "rottrans":
        # rotation followed by the translation of E2a: f(R(x - o)) over [l + o, u + o]
        w = ub - lb
        o = rng_problem.uniform(-w / 4, w / 4)
        R = haar_rotation(Dd, rng_problem)
        lb, ub = lb + o, ub + o
        g = (lambda X, f=f, o=o, R=R: f((X - o) @ R.T))
    elif kind in ("scale_lo", "scale_hi"):
        # change of units about the origin: f(x / s) over [s l, s u]; optimum (0) and f* unchanged
        s = 1e-2 if kind == "scale_lo" else 1e2
        lb, ub = s * lb, s * ub
        g = (lambda X, f=f, s=s: f(X / s))
    else:
        raise ValueError(kind)
    return dict(f=g, lb=lb, ub=ub, D=Dd, o=o, nfe=info["nfe"], name=info["name"], fid=fid,
                fstar=fstar(fid, Dd))
