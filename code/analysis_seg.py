"""E6, E6b, E6c: colour segmentation (Tables 13-14, Fig. 6).

For every run the label map is recomputed from the stored centroids (nearest centroid) and compared with each
human segmentation by the probabilistic Rand index (PRI), the variation of information (VoI) and the segmentation
covering (Cov; Arbelaez et al. 2011: for each human region, the best intersection-over-union with a region of the
machine segmentation, averaged with weights proportional to the region sizes), averaged over the human
segmentations of the image."""
import json
import os
import numpy as np
from scipy import stats
import exp_seg as S6
import paperdata as P
from io_utils import load_records

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
METHODS = ["k-means", "k-medians", "GPC", "PSO", "DE"]


def covering(seg, gt):
    a = np.unique(gt.ravel(), return_inverse=True)[1]
    b = np.unique(seg.ravel(), return_inverse=True)[1]
    nb = b.max() + 1
    M = np.bincount(a * nb + b, minlength=(a.max() + 1) * nb).reshape(-1, nb).astype(float)
    ga, sb = M.sum(1), M.sum(0)
    iou = M / (ga[:, None] + sb[None, :] - M)
    return float(np.sum(ga * iou.max(1)) / a.size)


def label_stats(recs, imgs, centred=False):
    out = []
    for r in recs:
        im = imgs[r["img"]]
        obj = S6.SegObjective(im["Z"], r["K"])
        lab = obj.labels(np.asarray(r["cent"]))
        covs = [covering(lab, g.ravel()) for g in im["gts"]]
        counts = np.bincount(lab, minlength=r["K"]) / lab.size
        H = -np.sum(counts[counts > 0] * np.log(counts[counts > 0]))
        c = np.asarray(r["cent"]).reshape(r["K"], 3) + (im["mean"] if centred else 0.0)
        out.append(dict(r, Cov=float(np.mean(covs)), n_used=int(np.sum(counts > 0.01)), H=float(H),
                        cent_abs=c.ravel()))
    return out


def seg_table(recs, methods, centred_recs=None):
    """Per K and method: Friedman rank of the mean objective over images, excess (%) over the best value found
    for the image by any method (mean, median over images), PRI, VoI, Cov, CPU."""
    by = {}
    for r in recs:
        by.setdefault((r["K"], r["alg"], r["img"]), []).append(r)
    imgs = sorted({r["img"] for r in recs})
    res = {}
    for K in (3, 4, 5):
        mean_obj = np.array([[np.mean([x["obj"] for x in by[(K, a, i)]]) for a in methods] for i in imgs])
        best = np.array([min(x["obj"] for a in methods for x in by[(K, a, i)]) for i in imgs])
        ranks = np.mean([stats.rankdata(row) for row in mean_obj], axis=0)
        for j, a in enumerate(methods):
            ex = (mean_obj[:, j] / best - 1) * 100
            allr = [x for i in imgs for x in by[(K, a, i)]]
            res["%d|%s" % (K, a)] = dict(rank=float(ranks[j]), ex_mean=float(ex.mean()), ex_med=float(np.median(ex)),
                                         PRI=float(np.mean([x["PRI"] for x in allr])),
                                         VoI=float(np.mean([x["VoI"] for x in allr])),
                                         Cov=float(np.mean([x["Cov"] for x in allr])),
                                         cpu=float(np.mean([x["cpu"] for x in allr])),
                                         H=float(np.mean([x["H"] for x in allr])),
                                         n_used=float(np.mean([x["n_used"] for x in allr])),
                                         bound=float(np.mean([np.mean((x["cent_abs"] <= 0.5) | (x["cent_abs"] >= 254.5)) for x in allr])),
                                         exact_bound=float(np.mean([np.mean((x["cent_abs"] <= 0) | (x["cent_abs"] >= 255)) for x in allr])))
        # Wilcoxon signed-rank of GPC against each other method over images
        g = methods.index("GPC")
        for j, a in enumerate(methods):
            if a != "GPC":
                res["%d|%s" % (K, a)]["wilcoxon_vs_gpc"] = float(stats.wilcoxon(mean_obj[:, g], mean_obj[:, j]).pvalue)
                res["%d|%s" % (K, a)]["gpc_worse_on"] = int(np.sum(mean_obj[:, g] > mean_obj[:, j]))
    return res


if __name__ == "__main__":
    out = {}
    imgs = S6.load_images()
    for nm, d in imgs.items():
        d["mean"] = d["Z"].astype(np.float64).mean(0)
    e6 = label_stats(load_records(os.path.join(RES, "E6_seg.jsonl.gz")), imgs)
    out["E6"] = seg_table(e6, METHODS)
    p = os.path.join(RES, "E6c_seg.jsonl.gz")
    if os.path.exists(p):
        cimgs = {nm: dict(Z=(d["Z"] - d["mean"]).astype(np.float32), gts=d["gts"], mean=d["mean"]) for nm, d in imgs.items()}
        e6c = label_stats(load_records(p), cimgs, centred=True)
        # E6c metaheuristics compared with the E6 baselines (k-means, k-medians are translation invariant)
        base = [r for r in e6 if r["alg"] in ("k-means", "k-medians")]
        out["E6c"] = seg_table(base + e6c, METHODS)
        # translation invariance of PSO and DE: Mann-Whitney per image and K between E6 and E6c objective values.
        # Runs that converge to the same optimum give values that differ only by floating-point rounding (the
        # centred coordinates change the arithmetic), so values equal up to a relative tolerance of 1e-6 are
        # treated as ties; the count without tolerance is also stored.
        ch, ch_raw = {}, {}
        for a in ("GPC", "PSO", "DE"):
            n = n_raw = 0
            for K in (3, 4, 5):
                for i in sorted(imgs):
                    x = np.array([r["obj"] for r in e6 if r["alg"] == a and r["K"] == K and r["img"] == i])
                    y = np.array([r["obj"] for r in e6c if r["alg"] == a and r["K"] == K and r["img"] == i])
                    n_raw += stats.mannwhitneyu(x, y).pvalue < 0.05
                    s0 = np.median(x)
                    xr, yr = np.round(x / s0, 6), np.round(y / s0, 6)
                    if not (np.all(xr == xr[0]) and np.all(yr == xr[0])):
                        n += stats.mannwhitneyu(xr, yr).pvalue < 0.05
            ch[a], ch_raw[a] = int(n), int(n_raw)
        out["E6c_changed"] = ch
        out["E6c_changed_no_tolerance"] = ch_raw
        # GPC: change of the mean objective per image and K between E6 and E6c
        ratio = {}
        for K in (3, 4, 5):
            rr = []
            for i in sorted(imgs):
                x = np.mean([r["obj"] for r in e6 if r["alg"] == "GPC" and r["K"] == K and r["img"] == i])
                y = np.mean([r["obj"] for r in e6c if r["alg"] == "GPC" and r["K"] == K and r["img"] == i])
                rr.append(y / x)
            ratio[str(K)] = dict(median=float(np.median(rr)), max=float(np.max(rr)), improved=int(np.sum(np.array(rr) < 1)))
        out["E6c_gpc_ratio"] = ratio
        # distance of the centroids from the mean colour of the image (median over all centroids of all runs) and
        # fraction of centroids within 10 intensity levels of it, in E6 and in E6c
        out["E6c_centroid_dist"] = {}
        out["E6_centroid_dist"] = {}
        out["E6c_centroid_near"] = {}
        out["E6_centroid_near"] = {}
        for a in ("GPC", "PSO", "DE", "k-medians"):
            dd = [np.linalg.norm(np.asarray(r["cent"]).reshape(r["K"], 3) - imgs[r["img"]]["mean"], axis=1)
                  for r in e6 if r["alg"] == a]
            out["E6_centroid_dist"][a] = float(np.median(np.concatenate(dd)))
            out["E6_centroid_near"][a] = float(np.mean(np.concatenate(dd) < 10))
            if a == "k-medians":
                dd0 = [np.linalg.norm(np.asarray(r["cent"]).reshape(r["K"], 3), axis=1) for r in e6 if r["alg"] == a]
                out["E6_kmedians_dist_black"] = float(np.median(np.concatenate(dd0)))
                continue
            dd = [np.linalg.norm(np.asarray(r["cent"]).reshape(r["K"], 3), axis=1) for r in e6c if r["alg"] == a]
            out["E6c_centroid_dist"][a] = float(np.median(np.concatenate(dd)))
            out["E6c_centroid_near"][a] = float(np.mean(np.concatenate(dd) < 10))
    p = os.path.join(RES, "E6b_seg.jsonl.gz")
    if os.path.exists(p):
        e6b = load_records(p)
        tab = {}
        for r in e6b:
            tab.setdefault((r["img"], r["K"], r["alg"]), []).append(r["obj"])
        out["E6b"] = {"%s|%d|%s" % k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in tab.items()}
        for img in ("Baboon", "Peppers"):
            for K in (3, 4, 5):
                out["E6b"]["%s|%d|published" % (img, K)] = dict(mean=float(P.T8[img][K][5][0]), best=float(min(m for m, s in P.T8[img][K])))
    json.dump(out, open(os.path.join(RES, "numbers_seg.json"), "w"), indent=1)
    for k, v in out["E6"].items():
        print("E6", k, {a: round(b, 3) for a, b in v.items()})
