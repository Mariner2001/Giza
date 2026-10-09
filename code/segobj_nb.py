"""Compiled objective for the segmentation experiment: sum of Euclidean distances to the nearest centroid."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def seg_obj(Z, C, K):
    N = Z.shape[0]
    P = C.shape[0]
    out = np.zeros(P)
    for p in range(P):
        s = 0.0
        for i in range(N):
            z0 = Z[i, 0]
            z1 = Z[i, 1]
            z2 = Z[i, 2]
            best = 1e300
            for k in range(K):
                a = z0 - C[p, 3 * k]
                b = z1 - C[p, 3 * k + 1]
                c = z2 - C[p, 3 * k + 2]
                d = a * a + b * b + c * c
                if d < best:
                    best = d
            s += np.sqrt(best)
        out[p] = s
    return out
