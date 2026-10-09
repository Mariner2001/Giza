"""Algorithms used in the re-evaluation.

All optimisers share the interface  run(f, lb, ub, nfe, rng, **kw) -> dict
  f   : callable, X (n, D) -> (n,)  (minimisation)
  lb, ub : arrays (D,)
  nfe : evaluation budget (counted exactly; the last batch is truncated)
  rng : numpy Generator

GPC follows the authors' public MATLAB code (main loop reproduced in the paper, Sect. 2):
  d = v0^2 / (2 g (sin(th) + mu cos(th))),  x = v0^2 / (2 g sin(th)),
  eps ~ U(-(ub-lb)/2, (ub-lb)/2)^D,
  y = (p + d) * (x eps)      (multiplicative, default)   or   y = (p + d) + x eps   (additive)
  clipping to the box, substitution (uniform crossover with one forced component, pSS=0.5),
  (mu + lambda) truncation of parents and children.
GA, ICA, PSO, DE and FA follow the canonical implementations whose parameter names and values
appear in Table 2 of the original paper (Yarpiz-style), with exact NFE accounting.
"""
import math
import numpy as np


class Budget(Exception):
    pass


class Evaluator:
    def __init__(self, f, nfe, keep_trace=True):
        self.f = f
        self.nfe = int(nfe)
        self.used = 0
        self.best = np.inf
        self.bestx = None
        self.keep = keep_trace
        self.tn = []
        self.tf = []

    @property
    def done(self):
        return self.used >= self.nfe

    def __call__(self, X):
        X = np.atleast_2d(X)
        r = self.nfe - self.used
        y = np.full(X.shape[0], np.inf)
        if r <= 0:
            return y
        m = min(r, X.shape[0])
        v = np.asarray(self.f(X[:m]), dtype=float)
        v = np.where(np.isfinite(v), v, np.inf)
        y[:m] = v
        self.used += m
        k = int(np.argmin(v))
        if v[k] < self.best:
            self.best = float(v[k])
            self.bestx = X[k].copy()
        if self.keep:
            self.tn.append(self.used)
            self.tf.append(self.best)
        return y

    def result(self):
        return dict(best_f=self.best, best_x=self.bestx, nfe=self.used,
                    trace_n=np.array(self.tn, dtype=np.int64), trace_f=np.array(self.tf))


def _clip(X, lb, ub):
    return np.minimum(np.maximum(X, lb), ub)


# --------------------------------------------------------------------------------- GPC
def gpc(f, lb, ub, nfe, rng, n=20, g=9.8, theta=10.0, mu_min=1.0, mu_max=10.0, pss=0.5,
        mode="mult", return_pop=False, eps_dist="uniform"):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    th = math.radians(theta)
    sin_t, cos_t = math.sin(th), math.cos(th)
    half = (ub - lb) / 2.0
    P = lb + (ub - lb) * rng.random((n, D))
    C = ev(P)
    while not ev.done:
        v0 = rng.random(n)
        mu = mu_min + (mu_max - mu_min) * rng.random(n)
        d = v0 ** 2 / (2 * g * (sin_t + mu * cos_t))
        x = v0 ** 2 / (2 * g * sin_t)
        if eps_dist == "uniform":
            eps = -half + 2 * half * rng.random((n, D))
        else:
            raise ValueError
        if mode == "mult":
            Y = (P + d[:, None]) * (x[:, None] * eps)
        elif mode == "add":
            Y = (P + d[:, None]) + x[:, None] * eps
        else:
            raise ValueError(mode)
        Y = _clip(Y, lb, ub)
        mask = rng.random((n, D)) <= pss
        k0 = rng.integers(0, D, size=n)
        mask[np.arange(n), k0] = True
        Z = np.where(mask, Y, P)
        CZ = ev(Z)
        allP = np.vstack([P, Z])
        allC = np.concatenate([C, CZ])
        order = np.argsort(allC, kind="stable")[:n]
        P, C = allP[order], allC[order]
    res = ev.result()
    if return_pop:
        res["pop"] = P
    return res


# --------------------------------------------------------------------------------- PSO
def pso(f, lb, ub, nfe, rng, n=20, w=1.0, wdamp=0.99, c1=2.0, c2=2.0, vfrac=0.1):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    vmax = vfrac * (ub - lb)
    X = lb + (ub - lb) * rng.random((n, D))
    V = np.zeros((n, D))
    C = ev(X)
    PB, PC = X.copy(), C.copy()
    g = int(np.argmin(PC))
    while not ev.done:
        r1, r2 = rng.random((n, D)), rng.random((n, D))
        V = w * V + c1 * r1 * (PB - X) + c2 * r2 * (PB[g] - X)
        V = np.clip(V, -vmax, vmax)
        X = X + V
        out = (X < lb) | (X > ub)
        V[out] = -V[out]
        X = _clip(X, lb, ub)
        C = ev(X)
        imp = C < PC
        PB[imp], PC[imp] = X[imp], C[imp]
        g = int(np.argmin(PC))
        w *= wdamp
    return ev.result()


# --------------------------------------------------------------------------------- DE
def de(f, lb, ub, nfe, rng, n=20, beta_min=0.2, beta_max=0.8, pcr=0.2):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    X = lb + (ub - lb) * rng.random((n, D))
    C = ev(X)
    idx = np.arange(n)
    while not ev.done:
        # distinct a, b, c != i
        A = np.empty((n, 3), dtype=int)
        for i in range(n):
            A[i] = rng.choice(np.delete(idx, i), 3, replace=False)
        beta = beta_min + (beta_max - beta_min) * rng.random((n, D))
        Y = X[A[:, 0]] + beta * (X[A[:, 1]] - X[A[:, 2]])
        Y = _clip(Y, lb, ub)
        mask = rng.random((n, D)) <= pcr
        mask[idx, rng.integers(0, D, size=n)] = True
        Z = np.where(mask, Y, X)
        CZ = ev(Z)
        imp = CZ < C
        X[imp], C[imp] = Z[imp], CZ[imp]
    return ev.result()


# --------------------------------------------------------------------------------- GA
def ga(f, lb, ub, nfe, rng, n=20, pc=0.8, gamma=0.05, pm=0.3, mu=0.02, beta=8.0):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    nc = 2 * int(round(pc * n / 2))
    nm = int(round(pm * n))
    nmu = int(math.ceil(mu * D))
    sigma = 0.1 * (ub - lb)
    X = lb + (ub - lb) * rng.random((n, D))
    C = ev(X)
    o = np.argsort(C, kind="stable")
    X, C = X[o], C[o]
    while not ev.done:
        # roulette-wheel selection with selection pressure beta on normalised costs
        fin = np.isfinite(C)
        cmin, cmax = C[fin].min(), C[fin].max()
        cn = (C - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(C)
        cn = np.where(fin, cn, 1e9)
        Pw = np.exp(-beta * cn)
        Pw /= Pw.sum()
        i1 = rng.choice(n, nc // 2, p=Pw)
        i2 = rng.choice(n, nc // 2, p=Pw)
        al = rng.uniform(-gamma, 1 + gamma, (nc // 2, D))
        Y1 = al * X[i1] + (1 - al) * X[i2]
        Y2 = al * X[i2] + (1 - al) * X[i1]
        Yc = _clip(np.vstack([Y1, Y2]), lb, ub)
        im = rng.integers(0, n, nm)
        Ym = X[im].copy()
        for r in range(nm):
            j = rng.choice(D, nmu, replace=False)
            Ym[r, j] = Ym[r, j] + sigma[j] * rng.standard_normal(nmu)
        Ym = _clip(Ym, lb, ub)
        Y = np.vstack([Yc, Ym])
        CY = ev(Y)
        allX = np.vstack([X, Y])
        allC = np.concatenate([C, CY])
        o = np.argsort(allC, kind="stable")[:n]
        X, C = allX[o], allC[o]
    return ev.result()


# --------------------------------------------------------------------------------- FA
def fa(f, lb, ub, nfe, rng, n=20, gamma=1.0, beta0=2.0, alpha=0.2, alpha_damp=0.98, dfrac=0.05, m=2):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    delta = dfrac * (ub - lb)
    dmax = np.linalg.norm(ub - lb)
    X = lb + (ub - lb) * rng.random((n, D))
    C = ev(X)
    while not ev.done:
        pairs = [(i, j) for i in range(n) for j in range(n) if C[j] < C[i]]
        if not pairs:
            pairs = [(i, i) for i in range(n)]
        I = np.array([p[0] for p in pairs])
        J = np.array([p[1] for p in pairs])
        rij = np.linalg.norm(X[I] - X[J], axis=1) / dmax
        b = beta0 * np.exp(-gamma * rij ** m)
        e = delta * rng.uniform(-1, 1, (len(pairs), D))
        Y = X[I] + b[:, None] * rng.random((len(pairs), D)) * (X[J] - X[I]) + alpha * e
        Y = _clip(Y, lb, ub)
        CY = ev(Y)
        # best move per firefly i
        NX = np.empty((n, D))
        NC = np.full(n, np.inf)
        for k in range(len(pairs)):
            i = I[k]
            if CY[k] <= NC[i]:
                NC[i] = CY[k]
                NX[i] = Y[k]
        keep = np.isfinite(NC)
        allX = np.vstack([X, NX[keep]])
        allC = np.concatenate([C, NC[keep]])
        o = np.argsort(allC, kind="stable")[:n]
        X, C = allX[o], allC[o]
        alpha *= alpha_damp
    return ev.result()


# --------------------------------------------------------------------------------- ICA
def ica(f, lb, ub, nfe, rng, n=20, n_emp=10, alpha=1.0, beta=1.5, p_rev=0.05, mu=0.1, zeta=0.2):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    sigma = 0.1 * (ub - lb)
    nmu = int(math.ceil(mu * D))

    def roulette(P):
        return int(rng.choice(len(P), p=P))

    def norm_probs(costs):
        costs = np.asarray(costs, float)
        cmin, cmax = costs.min(), costs.max()
        cn = (costs - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(costs)
        P = np.exp(-alpha * cn)
        return P / P.sum()

    X = lb + (ub - lb) * rng.random((n, D))
    C = ev(X)
    o = np.argsort(C, kind="stable")
    X, C = X[o], C[o]
    n_col = n - n_emp
    P = norm_probs(C[:n_emp])
    raw = P * n_col
    cnt = np.floor(raw).astype(int)
    rem = n_col - cnt.sum()
    if rem > 0:
        cnt[np.argsort(-(raw - cnt))[:rem]] += 1
    perm = rng.permutation(np.arange(n_emp, n))
    emps = []
    s = 0
    for k in range(n_emp):
        cols = perm[s:s + cnt[k]]
        s += cnt[k]
        emps.append(dict(ix=X[k].copy(), ic=C[k], cx=X[cols].copy(), cc=C[cols].copy()))

    def mutate(x):
        newp = x + sigma * rng.standard_normal(D)
        jj = rng.choice(D, nmu, replace=False)
        y = x.copy()
        y[jj] = newp[jj]
        return _clip(y, lb, ub)

    while not ev.done:
        # assimilation
        for e in emps:
            if len(e["cc"]) == 0 or ev.done:
                continue
            e["cx"] = _clip(e["cx"] + beta * rng.random(e["cx"].shape) * (e["ix"] - e["cx"]), lb, ub)
            e["cc"] = ev(e["cx"])
        # revolution
        for e in emps:
            if ev.done:
                break
            y = mutate(e["ix"])
            cy = ev(y)[0]
            if cy < e["ic"]:
                e["ix"], e["ic"] = y, cy
            for c in range(len(e["cc"])):
                if rng.random() <= p_rev and not ev.done:
                    e["cx"][c] = mutate(e["cx"][c])
                    e["cc"][c] = ev(e["cx"][c])[0]
        # intra-empire competition
        for e in emps:
            if len(e["cc"]) == 0:
                continue
            b = int(np.argmin(e["cc"]))
            if e["cc"][b] < e["ic"]:
                ix, ic = e["ix"].copy(), e["ic"]
                e["ix"], e["ic"] = e["cx"][b].copy(), e["cc"][b]
                e["cx"][b], e["cc"][b] = ix, ic
        # total costs and inter-empire competition
        if len(emps) > 1:
            TC = np.array([e["ic"] + (zeta * np.mean(e["cc"]) if len(e["cc"]) else 0.0) for e in emps])
            TC = np.where(np.isfinite(TC), TC, np.nanmax(np.where(np.isfinite(TC), TC, -np.inf)) + 1)
            weak = int(np.argmax(TC))
            Pp = norm_probs(TC)
            Pp[weak] = 0.0
            if Pp.sum() <= 0:
                Pp = np.ones(len(emps))
                Pp[weak] = 0
            Pp = Pp / Pp.sum()
            ew = emps[weak]
            if len(ew["cc"]) > 0:
                wc = int(np.argmax(ew["cc"]))
                win = roulette(Pp)
                emps[win]["cx"] = np.vstack([emps[win]["cx"], ew["cx"][wc][None, :]])
                emps[win]["cc"] = np.append(emps[win]["cc"], ew["cc"][wc])
                ew["cx"] = np.delete(ew["cx"], wc, axis=0)
                ew["cc"] = np.delete(ew["cc"], wc)
            if len(ew["cc"]) == 0:
                win = roulette(Pp)
                emps[win]["cx"] = np.vstack([emps[win]["cx"], ew["ix"][None, :]])
                emps[win]["cc"] = np.append(emps[win]["cc"], ew["ic"])
                emps.pop(weak)
    return ev.result()


# --------------------------------------------------------------------------------- RS
def rs(f, lb, ub, nfe, rng, batch=100):
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    ev = Evaluator(f, nfe)
    while not ev.done:
        ev(lb + (ub - lb) * rng.random((batch, lb.size)))
    return ev.result()


# --------------------------------------------------------------------------------- CMA-ES
def cmaes(f, lb, ub, nfe, rng, sigma_frac=0.3, popsize=None):
    import cma
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    D = lb.size
    ev = Evaluator(f, nfe)
    x0 = lb + (ub - lb) * rng.random(D)
    opts = {"bounds": [lb.tolist(), ub.tolist()], "verbose": -9, "seed": int(rng.integers(1, 2 ** 31 - 1)),
            "maxfevals": 10 ** 12, "tolfun": 0, "tolfunhist": 0, "tolx": 0, "tolstagnation": 10 ** 9,
            "CMA_stds": None}
    if popsize:
        opts["popsize"] = popsize
    if D == 1:
        raise ValueError("D must be >= 2")
    while not ev.done:  # restarts with fresh random x0 if the run terminates early
        es = cma.CMAEvolutionStrategy(x0, sigma_frac * float(np.mean(ub - lb)), opts)
        while not es.stop() and not ev.done:
            sols = es.ask()
            S = np.array(sols)
            y = ev(S)
            if ev.done:
                break
            es.tell(sols, y.tolist())
        x0 = lb + (ub - lb) * rng.random(D)
    return ev.result()


ALGOS = {"GA": ga, "ICA": ica, "PSO": pso, "DE": de, "FA": fa, "GPC": gpc, "RS": rs, "CMA-ES": cmaes}
