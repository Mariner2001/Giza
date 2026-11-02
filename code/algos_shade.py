"""L-SHADE and jSO, the two success-history adaptive differential-evolution variants added in the revision
as state-of-the-art comparators.  Same interface as algos.py:  run(f, lb, ub, nfe, rng) -> dict.

L-SHADE  R. Tanabe and A. S. Fukunaga, "Improving the search performance of SHADE using linear population
         size reduction", IEEE CEC 2014, pp. 1658-1665.
         N_init = round(18 D), N_min = 4, memory H = 6 (M_F = M_CR = 0.5), archive rate 2.6, p = 0.11,
         current-to-pbest/1/bin, weighted Lehmer means, terminal CR value, linear population-size reduction.
jSO      J. Brest, M. S. Maucec and B. Boskovic, "Single objective real-parameter optimization: algorithm jSO",
         IEEE CEC 2017, pp. 1311-1318.
         N_init = round(25 ln(D) sqrt(D)), N_min = 4, H = 5 (M_F = 0.3, M_CR = 0.8, last entry fixed at 0.9),
         archive rate 1, p = p_min + (p_max - p_min) nfes/max_nfes with p_max = 0.25, p_min = p_max/2 (Eq. 1 of
         the paper), CR >= 0.7 (first quarter of the budget) / 0.6 (first half), F <= 0.7 (first 60 %),
         current-to-pBest-w/1 with Fw = 0.7F / 0.8F / 1.2F (nfes < 0.2 / < 0.4 / otherwise of max_nfes),
         memory entries averaged with their previous value (as in iL-SHADE).
Both: a component that leaves the box is set half-way between the parent component and the violated bound
(the rule of the L-SHADE reference code); generation-based schedules are expressed as fractions of the
evaluation budget; the budget is counted exactly (the last generation is truncated)."""
import math
import numpy as np
import algos


def _wlehmer(s, w):
    den = np.sum(w * s)
    return float(np.sum(w * s * s) / den) if den > 0 else 0.0


def _shade(f, lb, ub, nfe, rng, variant):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = algos.Evaluator(f, nfe)      # looked up at call time (exp_bbob.py substitutes a stopping evaluator)
    jso = variant == "jso"
    if jso:
        N0 = int(round(25.0 * math.log(D) * math.sqrt(D)))
        H = 5
        MF = np.full(H, 0.3)
        MCR = np.full(H, 0.8)
        MF[-1] = MCR[-1] = 0.9
        arc_rate = 1.0
        p_max, p_min = 0.25, 0.125
        n_upd = H - 1                  # the last entry is never updated
    else:
        N0 = int(round(18.0 * D))
        H = 6
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.5)
        arc_rate = 2.6
        n_upd = H
    N_min = 4
    N0 = max(N0, N_min)
    N = N0
    X = lb + (ub - lb) * rng.random((N, D))
    C = ev(X)
    A = np.empty((0, D))
    k = 0
    while not ev.done:
        frac = ev.used / nfe
        idx = np.arange(N)
        r = rng.integers(0, H, N)
        # --- control parameters
        mcr = MCR[r]
        CR = np.clip(rng.normal(np.nan_to_num(mcr), 0.1), 0.0, 1.0)
        CR[np.isnan(mcr)] = 0.0            # terminal value
        mf = MF[r]
        F = mf + 0.1 * rng.standard_cauchy(N)
        bad = F <= 0
        while bad.any():
            F[bad] = mf[bad] + 0.1 * rng.standard_cauchy(int(bad.sum()))
            bad = F <= 0
        F = np.minimum(F, 1.0)
        if jso:
            if frac < 0.25:
                CR = np.maximum(CR, 0.7)
            elif frac < 0.5:
                CR = np.maximum(CR, 0.6)
            if frac < 0.6:
                F = np.minimum(F, 0.7)
            Fw = F * (0.7 if frac < 0.2 else (0.8 if frac < 0.4 else 1.2))
            p = p_min + (p_max - p_min) * frac
        else:
            Fw = F
            p = 0.11
        # --- current-to-pbest(-w)/1 with archive
        order = np.argsort(C, kind="stable")
        n_best = max(2, int(round(p * N)))
        pb = order[rng.integers(0, n_best, N)]
        r1 = rng.integers(0, N - 1, N)
        r1 += r1 >= idx                       # r1 != i
        PA = np.vstack([X, A]) if len(A) else X
        r2 = rng.integers(0, PA.shape[0], N)
        bad = (r2 == idx) | (r2 == r1)
        while bad.any():
            r2[bad] = rng.integers(0, PA.shape[0], int(bad.sum()))
            bad = (r2 == idx) | (r2 == r1)
        V = X + Fw[:, None] * (X[pb] - X) + F[:, None] * (X[r1] - PA[r2])
        lo = V < lb
        hi = V > ub
        V = np.where(lo, (lb + X) / 2.0, V)
        V = np.where(hi, (ub + X) / 2.0, V)
        # --- binomial crossover
        mask = rng.random((N, D)) <= CR[:, None]
        mask[idx, rng.integers(0, D, N)] = True
        U = np.where(mask, V, X)
        CU = ev(U)
        # --- selection, archive, success history
        better = CU < C
        keep = CU <= C
        if better.any():
            dF = C[better] - CU[better]
            A = np.vstack([A, X[better]])
            SF, SCR = F[better], CR[better]
            fin = np.isfinite(dF)
            w = dF / dF.sum() if (fin.all() and dF.sum() > 0) else np.full(dF.size, 1.0 / dF.size)
            mF = _wlehmer(SF, w)
            if np.isnan(MCR[k]) or SCR.max() == 0:
                MCR[k] = np.nan
            else:
                mCR = _wlehmer(SCR, w)
                MCR[k] = 0.5 * (mCR + MCR[k]) if jso else mCR
            MF[k] = 0.5 * (mF + MF[k]) if jso else mF
            k = (k + 1) % n_upd
        X = np.where(keep[:, None], U, X)
        C = np.where(keep, CU, C)
        # --- linear population-size reduction and archive size
        Nn = max(N_min, int(round((N_min - N0) / nfe * ev.used + N0)))
        if Nn < N:
            sel = np.argsort(C, kind="stable")[:Nn]
            X, C = X[sel], C[sel]
            N = Nn
        amax = int(round(arc_rate * N))
        if A.shape[0] > amax:
            A = A[rng.choice(A.shape[0], amax, replace=False)]
    return ev.result()


def lshade(f, lb, ub, nfe, rng):
    return _shade(f, lb, ub, nfe, rng, "lshade")


def jso(f, lb, ub, nfe, rng):
    return _shade(f, lb, ub, nfe, rng, "jso")
