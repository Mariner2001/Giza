"""E6: colour-image segmentation by centroid search (Eq. 10 of the original paper: sum of Euclidean
distances of pixels to their nearest centroid), K = 3, 4, 5, 5000 evaluations, 10 runs.

Data: 20 BSDS500 test images (random selection, seed 2026), subsampled by 2 in each direction,
with the human ground-truth segmentations (PRI and VoI are reported).
Baselines: k-means++ (Lloyd, one k-means++ initialisation per run) and k-medians (Weiszfeld refinement
of that k-means solution, which minimises Eq. 10 locally)."""
import glob
import zlib
import os
import sys
import time
import numpy as np
from multiprocessing import Pool
from PIL import Image
from scipy.io import loadmat
import algos as A
from io_utils import save_records

# BSDS500 data directory (containing images/test and groundTruth/test); see ../data/get_bsds500.sh
BSDS = os.environ.get("BSDS500_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "BSDS500", "data"))
# The 20 test images used in the paper (drawn at random from the 200 test images with seed 2026)
IMAGE_IDS = ["118031", "123057", "16068", "198087", "235098", "279005", "285022", "290035", "317043", "326085",
             "335094", "35028", "365072", "385022", "41029", "77062", "8068", "87015", "92014", "94095"]
NFE = 5000
RUNS = 10
KS = (3, 4, 5)
META = {"GPC": (A.gpc, {"mode": "mult", "theta": 10.0}), "PSO": (A.pso, {}), "DE": (A.de, {})}
AID = {"GPC": 0, "PSO": 1, "DE": 2, "k-means": 3, "k-medians": 4}


def load_images():
    imgs = {}
    for b in sorted(IMAGE_IDS):
        p = os.path.join(BSDS, "images", "test", b + ".jpg")
        im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)[::2, ::2]
        gt = loadmat(os.path.join(BSDS, "groundTruth", "test", b + ".mat"))["groundTruth"][0]
        gts = [np.asarray(g["Segmentation"][0][0], dtype=np.int64)[::2, ::2] for g in gt]
        imgs["bsds-" + b] = dict(Z=im.reshape(-1, 3), shape=im.shape[:2], gts=gts)
    return imgs


class SegObjective:
    def __init__(self, Z, K):
        self.Zd = np.ascontiguousarray(Z, dtype=np.float64)
        self.Z = np.ascontiguousarray(Z, dtype=np.float32)
        self.zz = np.sum(self.Z * self.Z, axis=1)
        self.K = K

    def __call__(self, X):
        from segobj_nb import seg_obj
        return seg_obj(self.Zd, np.ascontiguousarray(X, dtype=np.float64), self.K)

    def labels(self, c):
        C = c.reshape(self.K, 3).astype(np.float32)
        d2 = self.zz[:, None] - 2.0 * (self.Z @ C.T) + np.sum(C * C, axis=1)[None, :]
        return np.argmin(d2, axis=1)


def weiszfeld_kmedians(Z, C, iters=200, tol=1e-9):
    Z = Z.astype(np.float64)
    C = C.astype(np.float64).copy()
    K = C.shape[0]
    prev = np.inf
    for _ in range(iters):
        d = np.sqrt(np.maximum(((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1), 0))
        lab = d.argmin(1)
        obj = d[np.arange(len(Z)), lab].sum()
        if np.isfinite(prev) and prev - obj <= tol * prev:
            break
        prev = obj
        for k in range(K):
            P = Z[lab == k]
            if len(P) == 0:
                continue
            m = C[k]
            for _ in range(5):
                w = 1.0 / np.maximum(np.sqrt(((P - m) ** 2).sum(1)), 1e-9)
                m = (P * w[:, None]).sum(0) / w.sum()
            C[k] = m
    return C


def rand_index_voi(a, b):
    a = np.unique(a.ravel(), return_inverse=True)[1]
    b = np.unique(b.ravel(), return_inverse=True)[1]
    n = a.size
    nb = b.max() + 1
    M = np.bincount(a * nb + b, minlength=(a.max() + 1) * nb).reshape(-1, nb).astype(np.float64)
    ai, bj = M.sum(1), M.sum(0)
    tot = n * (n - 1) / 2.0
    s_ij = (M * (M - 1) / 2).sum()
    s_a = (ai * (ai - 1) / 2).sum()
    s_b = (bj * (bj - 1) / 2).sum()
    ri = (tot + 2 * s_ij - s_a - s_b) / tot
    P = M / n
    pa, pb = ai / n, bj / n
    nz = P > 0
    I = (P[nz] * np.log(P[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    Ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    Hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return ri, Ha + Hb - 2 * I


IMGS = None


def init():
    global IMGS
    IMGS = load_images()


def job(a):
    name, K, alg, r = a
    im = IMGS[name]
    obj = SegObjective(im["Z"], K)
    D = 3 * K
    t = time.process_time()
    if alg in META:
        fn, kw = META[alg]
        res = fn(obj, np.zeros(D), np.full(D, 255.0), NFE, np.random.default_rng([66, AID[alg], K, r, zlib.crc32(name.encode()) % 100000]), **kw)
        c = res["best_x"]
        val = res["best_f"]
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=1, init="k-means++",
                    random_state=int(r) + 1000 * K + zlib.crc32(name.encode()) % 1000).fit(im["Z"].astype(np.float64))
        C = km.cluster_centers_
        if alg == "k-medians":
            C = weiszfeld_kmedians(im["Z"], C)
        c = C.ravel()
        val = float(obj(c[None, :])[0])
    cpu = time.process_time() - t
    out = dict(img=name, K=K, alg=alg, run=r, obj=float(val), cpu=cpu, cent=np.asarray(c, float))
    if im["gts"] is not None:
        lab = obj.labels(c)
        ris, vois = zip(*[rand_index_voi(lab, g.ravel()) for g in im["gts"]])
        out["PRI"] = float(np.mean(ris))
        out["VoI"] = float(np.mean(vois))
    return out


if __name__ == "__main__":
    init()
    names = sorted(IMGS)
    algs = list(META) + ["k-means", "k-medians"]
    jobs = [(nm, K, alg, r) for nm in names for K in KS for alg in algs for r in range(RUNS)]
    t0 = time.time()
    out = []
    with Pool(2, initializer=init) as p:
        for k, res in enumerate(p.imap_unordered(job, jobs, chunksize=3)):
            out.append(res)
            if (k + 1) % 200 == 0:
                print("seg %d/%d %.0fs" % (k + 1, len(jobs), time.time() - t0), flush=True)
    save_records("../results/E6_seg.jsonl.gz", out)
    print("E6 done in %.0fs" % (time.time() - t0))
