"""Benchmark functions used by Harifi et al. (2021), Table 1, implemented in vectorised form.

Every function takes an array X of shape (n, D) and returns an array of shape (n,).
Definitions follow the Virtual Library of Simulation Experiments (Surjanovic & Bingham),
which is the source cited in the original paper.  Where the printed Table 1 deviates from
that source the deviation is recorded in NOTES (it does not change the experiment unless
stated).
"""
import numpy as np

PI = np.pi


def ackley(X):
    D = X.shape[1]
    s1 = np.sqrt(np.sum(X * X, axis=1) / D)
    s2 = np.sum(np.cos(2 * PI * X), axis=1) / D
    return -20.0 * np.exp(-0.2 * s1) - np.exp(s2) + 20.0 + np.e


def beale(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (2.625 - x1 + x1 * x2 ** 3) ** 2


def bohachevsky(X):
    x1, x2 = X[:, 0], X[:, 1]
    return x1 ** 2 + 2 * x2 ** 2 - 0.3 * np.cos(3 * PI * x1) - 0.4 * np.cos(4 * PI * x2) + 0.7


def booth(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2


def bukin6(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 100.0 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2)) + 0.01 * np.abs(x1 + 10.0)


def camel6(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2


def camel3(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6 + x1 * x2 + x2 ** 2


def crossintray(X):
    x1, x2 = X[:, 0], X[:, 1]
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    with np.errstate(over="ignore"):
        e = np.exp(np.abs(100.0 - r / PI))
    return -0.0001 * (np.abs(np.sin(x1) * np.sin(x2) * e) + 1.0) ** 0.1


_A1 = np.tile(np.array([-32.0, -16.0, 0.0, 16.0, 32.0]), 5)
_A2 = np.repeat(np.array([-32.0, -16.0, 0.0, 16.0, 32.0]), 5)
_I25 = np.arange(1, 26, dtype=float)


def dejong5(X):
    x1, x2 = X[:, 0:1], X[:, 1:2]
    s = np.sum(1.0 / (_I25 + (x1 - _A1) ** 6 + (x2 - _A2) ** 6), axis=1)
    return 1.0 / (0.002 + s)


def dixonprice(X):
    D = X.shape[1]
    i = np.arange(2, D + 1, dtype=float)
    return (X[:, 0] - 1) ** 2 + np.sum(i * (2 * X[:, 1:] ** 2 - X[:, :-1]) ** 2, axis=1)


def dropwave(X):
    x1, x2 = X[:, 0], X[:, 1]
    r2 = x1 ** 2 + x2 ** 2
    return -(1 + np.cos(12 * np.sqrt(r2))) / (0.5 * r2 + 2)


def easom(X):
    x1, x2 = X[:, 0], X[:, 1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - PI) ** 2 - (x2 - PI) ** 2)


def eggholder(X):
    x1, x2 = X[:, 0], X[:, 1]
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))


def griewank(X):
    D = X.shape[1]
    i = np.sqrt(np.arange(1, D + 1, dtype=float))
    return np.sum(X ** 2, axis=1) / 4000.0 - np.prod(np.cos(X / i), axis=1) + 1.0


def holdertable(X):
    x1, x2 = X[:, 0], X[:, 1]
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    return -np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - r / PI)))


def rothyp(X):
    # rotated hyper-ellipsoid: sum_i sum_{j<=i} x_j^2 = sum_j (D-j+1) x_j^2
    D = X.shape[1]
    w = np.arange(D, 0, -1, dtype=float)
    return np.sum(w * X ** 2, axis=1)


def levy(X):
    W = 1 + (X - 1) / 4.0
    w1, wd = W[:, 0], W[:, -1]
    t1 = np.sin(PI * w1) ** 2
    wi = W[:, :-1]
    t2 = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(PI * wi + 1) ** 2), axis=1)
    t3 = (wd - 1) ** 2 * (1 + np.sin(2 * PI * wd) ** 2)
    return t1 + t2 + t3


def levy13(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (np.sin(3 * PI * x1) ** 2 + (x1 - 1) ** 2 * (1 + np.sin(3 * PI * x2) ** 2)
            + (x2 - 1) ** 2 * (1 + np.sin(2 * PI * x2) ** 2))


def matyas(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2


def michalewicz(X, m=10):
    D = X.shape[1]
    i = np.arange(1, D + 1, dtype=float)
    return -np.sum(np.sin(X) * np.sin(i * X ** 2 / PI) ** (2 * m), axis=1)


def powell(X):
    # sum over floor(D/4) blocks (MATLAB 1:(d/4) semantics used by the cited library);
    # standard signs of the cited library (Table 1 prints '+' in three places)
    D = X.shape[1]
    k = D // 4
    Z = X[:, : 4 * k].reshape(X.shape[0], k, 4)
    a, b, c, d = Z[:, :, 0], Z[:, :, 1], Z[:, :, 2], Z[:, :, 3]
    return np.sum((a + 10 * b) ** 2 + 5 * (c - d) ** 2 + (b - 2 * c) ** 4 + 10 * (a - d) ** 4, axis=1)


def rastrigin(X):
    D = X.shape[1]
    return 10.0 * D + np.sum(X ** 2 - 10.0 * np.cos(2 * PI * X), axis=1)


def rosenbrock(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (X[:, :-1] - 1) ** 2, axis=1)


def schaffer2(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 0.5 + (np.sin(x1 ** 2 - x2 ** 2) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2


def schaffer4(X):
    # standard definition with cos^2 (Table 1 prints cos without the square)
    x1, x2 = X[:, 0], X[:, 1]
    return 0.5 + (np.cos(np.sin(np.abs(x1 ** 2 - x2 ** 2))) ** 2 - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2


def schaffer4_printed(X):
    x1, x2 = X[:, 0], X[:, 1]
    return 0.5 + (np.cos(np.sin(np.abs(x1 ** 2 - x2 ** 2))) - 0.5) / (1 + 0.001 * (x1 ** 2 + x2 ** 2)) ** 2


_I5 = np.arange(1, 6, dtype=float)


def shubert(X):
    x1, x2 = X[:, 0:1], X[:, 1:2]
    s1 = np.sum(_I5 * np.cos((_I5 + 1) * x1 + _I5), axis=1)
    s2 = np.sum(_I5 * np.cos((_I5 + 1) * x2 + _I5), axis=1)
    return s1 * s2


def sphere(X):
    return np.sum(X * X, axis=1)


def sumpow(X):
    D = X.shape[1]
    e = np.arange(2, D + 2, dtype=float)
    return np.sum(np.abs(X) ** e, axis=1)


def sumsqu(X):
    D = X.shape[1]
    i = np.arange(1, D + 1, dtype=float)
    return np.sum(i * X ** 2, axis=1)


def zakharov(X):
    D = X.shape[1]
    i = np.arange(1, D + 1, dtype=float)
    s = np.sum(0.5 * i * X, axis=1)
    return np.sum(X ** 2, axis=1) + s ** 2 + s ** 4


# ----------------------------------------------------------------------------------------
# Metadata.  'dim' = 2 for the functions that are defined only in two variables (as stated
# in Sect. 4 of the original paper); 'n' for scalable ones (D = 30 in the original Table 3).
# 'fstar' = optimal value on the box used in the experiments (corrected where Table 1 is wrong)
# 'xstar' = a global minimiser (None if several / boundary)
# 'nfe'   = budget used in the original Table 3.
# ----------------------------------------------------------------------------------------
FUNCS = [
    # id, name, f, lb, ub, dim, fstar(Table1), nfe
    (1, "Ackley", ackley, -32.768, 32.768, "n", 0.0, 1000),
    (2, "Beale", beale, -4.5, 4.5, 2, 0.0, 1000),
    (3, "Bohachevsky", bohachevsky, -100.0, 100.0, 2, 0.0, 1000),
    (4, "Booth", booth, -10.0, 10.0, 2, 0.0, 1000),
    (5, "Bukin N.6", bukin6, -3.0, 3.0, 2, 0.0, 1000),
    (6, "Camel six-hump", camel6, -3.0, 3.0, 2, -1.0316, 1000),
    (7, "Camel three-hump", camel3, -5.0, 5.0, 2, 0.0, 1000),
    (8, "Cross-in-tray", crossintray, -10.0, 10.0, 2, -2.06261, 1000),
    (9, "De Jong N.5", dejong5, -65.536, 65.536, 2, 0.0, 1000),
    (10, "Dixon-Price", dixonprice, -10.0, 10.0, "n", 0.0, 10000),
    (11, "Drop-wave", dropwave, -5.12, 5.12, 2, -1.0, 1000),
    (12, "Easom", easom, -100.0, 100.0, 2, -1.0, 1000),
    (13, "Eggholder", eggholder, -512.0, 512.0, 2, -959.6407, 1000),
    (14, "Griewank", griewank, -600.0, 600.0, "n", 0.0, 1000),
    (15, "Holder table", holdertable, -10.0, 10.0, 2, -19.2085, 1000),
    (16, "Hyper-ellipsoid", rothyp, -65.536, 65.536, "n", 0.0, 10000),
    (17, "Levy", levy, -10.0, 10.0, "n", 0.0, 1000),
    (18, "Levy N.13", levy13, -10.0, 10.0, 2, 0.0, 1000),
    (19, "Matyas", matyas, -10.0, 10.0, 2, 0.0, 1000),
    (20, "Michalewicz", michalewicz, 0.0, PI, "n", -1.8013, 1000),
    (21, "Powell", powell, -4.0, 5.0, "n", 0.0, 10000),
    (22, "Rastrigin", rastrigin, -5.12, 5.12, "n", 0.0, 10000),
    (23, "Rosenbrock", rosenbrock, -5.0, 10.0, "n", 0.0, 10000),
    (24, "Schaffer N.2", schaffer2, -100.0, 100.0, 2, 0.0, 1000),
    (25, "Schaffer N.4", schaffer4, -100.0, 100.0, 2, 0.292579, 1000),
    (26, "Shubert", shubert, -10.0, 10.0, 2, -186.7309, 1000),
    (27, "Sphere", sphere, -5.12, 5.12, "n", 0.0, 1000),
    (28, "Sum of powers", sumpow, -1.0, 1.0, "n", 0.0, 1000),
    (29, "Sum of squares", sumsqu, -10.0, 10.0, "n", 0.0, 10000),
    (30, "Zakharov", zakharov, -5.0, 10.0, "n", 0.0, 10000),
]


def get(fid, D_scalable=30):
    """Return dict with function info for id fid (1..30)."""
    t = FUNCS[fid - 1]
    assert t[0] == fid
    D = D_scalable if t[5] == "n" else 2
    return dict(id=fid, name=t[1], f=t[2], lb=t[3], ub=t[4], D=D, fstar_table1=t[6], nfe=t[7],
                scalable=(t[5] == "n"))


ORIGIN_OPT = {1, 3, 7, 11, 14, 16, 19, 21, 22, 24, 27, 28, 29, 30}  # x* = 0 (unique)
