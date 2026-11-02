"""Segmentation experiments added in the revision.

E6b  the two colour test images of the original study that are publicly available in colour (Baboon and
     Peppers; Lena is not used), resized to 128 x 128 pixels (PIL, bicubic), in RGB and in HSV (PIL, all
     channels in [0, 255]) -- the representations that give objective values of the order of Table 8 of
     the original study (audit_seg.py); K = 3, 4, 5; 5000 evaluations, as stated in the original study;
     GPC, PSO, DE, k-means++ and k-medians, 10 runs.
E6c  the 20 BSDS500 images of E6 with mean-centred colour coordinates: z - mean(z) over the translated
     colour cube [-mean, 255 - mean]^3, i.e. a translation of the E6 problem that puts the origin at the
     mean colour of the image; GPC, PSO and DE, 10 runs (k-means and k-medians are translation invariant,
     so their E6 results apply unchanged).
usage: python3 exp_seg2.py E6b|E6c
"""
import os
import sys
import time
import zlib
import numpy as np
from multiprocessing import Pool
from PIL import Image
import algos as A
import exp_seg as S6
from io_utils import save_records

STD = os.environ.get("STD_IMAGES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                                    "standard_test_images"))
STD_IMAGES = {"Baboon": "baboon.png", "Peppers": "peppers.png"}
NFE = 5000
RUNS = 10
KS = (3, 4, 5)
META = {"GPC": (A.gpc, {"mode": "mult", "theta": 10.0}), "PSO": (A.pso, {}), "DE": (A.de, {})}
AID = {"GPC": 0, "PSO": 1, "DE": 2, "k-means": 3, "k-medians": 4}
EXP = sys.argv[1] if len(sys.argv) > 1 else "E6b"
IMGS = None


def load_std():
    imgs = {}
    for nm, fn in STD_IMAGES.items():
        im = Image.open(os.path.join(STD, fn)).convert("RGB").resize((128, 128), Image.BICUBIC)
        for rep in ("RGB", "HSV"):
            a = np.asarray(im.convert("HSV") if rep == "HSV" else im, dtype=np.float32).reshape(-1, 3)
            imgs["%s-%s" % (nm, rep)] = dict(Z=a, lb=np.zeros(3), ub=np.full(3, 255.0), gts=None)
    return imgs


def load_centred():
    imgs = S6.load_images()
    out = {}
    for nm, d in imgs.items():
        m = d["Z"].astype(np.float64).mean(0)
        out[nm] = dict(Z=(d["Z"] - m).astype(np.float32), lb=-m, ub=255.0 - m, gts=d["gts"], mean=m)
    return out


def init():
    global IMGS
    IMGS = load_std() if EXP == "E6b" else load_centred()


def job(a):
    name, K, alg, r = a
    im = IMGS[name]
    obj = S6.SegObjective(im["Z"], K)
    lb, ub = np.tile(im["lb"], K), np.tile(im["ub"], K)
    tag = 67 if EXP == "E6b" else 68
    t = time.process_time()
    if alg in META:
        fn, kw = META[alg]
        res = fn(obj, lb, ub, NFE, np.random.default_rng([tag, AID[alg], K, r, zlib.crc32(name.encode()) % 100000]), **kw)
        c, val = res["best_x"], res["best_f"]
    else:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=1, init="k-means++",
                    random_state=int(r) + 1000 * K + zlib.crc32(name.encode()) % 1000).fit(im["Z"].astype(np.float64))
        C = km.cluster_centers_
        if alg == "k-medians":
            C = S6.weiszfeld_kmedians(im["Z"], C)
        c = C.ravel()
        val = float(obj(c[None, :])[0])
    cpu = time.process_time() - t
    out = dict(img=name, K=K, alg=alg, run=r, obj=float(val), cpu=cpu, cent=np.asarray(c, float))
    if im["gts"] is not None:
        lab = obj.labels(c)
        ris, vois = zip(*[S6.rand_index_voi(lab, g.ravel()) for g in im["gts"]])
        out["PRI"] = float(np.mean(ris))
        out["VoI"] = float(np.mean(vois))
    return out


if __name__ == "__main__":
    init()
    algs = list(META) + (["k-means", "k-medians"] if EXP == "E6b" else [])
    jobs = [(nm, K, alg, r) for nm in sorted(IMGS) for K in KS for alg in algs for r in range(RUNS)]
    t0 = time.time()
    out = []
    with Pool(2, initializer=init) as p:
        for k, res in enumerate(p.imap_unordered(job, jobs, chunksize=3)):
            out.append(res)
            if (k + 1) % 200 == 0:
                print("%s %d/%d %.0fs" % (EXP, k + 1, len(jobs), time.time() - t0), flush=True)
    save_records("../results/%s_seg.jsonl.gz" % EXP, out)
    print("%s done in %.0fs" % (EXP, time.time() - t0))
