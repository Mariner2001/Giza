"""Sect. 3.4: are the objective values of Table 8 of the original study compatible with the stated setting?

For the two test images of the original study that are publicly available in colour (Baboon, Peppers; see
../data/get_standard_images.sh), Eq. (10) of the original study (sum of Euclidean distances of the pixels to the
nearest centroid) is evaluated at the k-means++ solution (scikit-learn, 10 initialisations) and at its k-medians
refinement, for several image representations, and divided by the best mean reported in Table 8 for the image
and K.  Ratios well above 1 mean that the published values are unattainable in that representation.
Writes ../results/numbers_audit_seg.json and ../results/tables/tab_segaudit.tex."""
import json
import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import paperdata as P
from exp_seg import weiszfeld_kmedians
from segobj_nb import seg_obj

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
STD = os.environ.get("STD_IMAGES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                                    "standard_test_images"))
IMAGES = {"Baboon": "baboon.png", "Peppers": "peppers.png"}
# (label, size, colour space)
REPS = [("RGB, $512\\times512$ (original size)", 512, "RGB"), ("RGB, $256\\times256$ (stated setting)", 256, "RGB"),
        ("RGB, $128\\times128$", 128, "RGB"), ("HSV, $256\\times256$", 256, "HSV"), ("HSV, $128\\times128$", 128, "HSV"),
        ("CIELAB, $256\\times256$", 256, "LAB")]


def representation(img, size, space):
    im = img.resize((size, size), Image.BICUBIC) if size != img.size[0] else img
    if space == "HSV":
        return np.asarray(im.convert("HSV"), dtype=np.float64).reshape(-1, 3)
    if space == "LAB":
        from skimage.color import rgb2lab
        return rgb2lab(np.asarray(im, dtype=np.float64) / 255.0).reshape(-1, 3)
    return np.asarray(im, dtype=np.float64).reshape(-1, 3)


if __name__ == "__main__":
    out = {}
    for nm, fn in IMAGES.items():
        img = Image.open(os.path.join(STD, fn)).convert("RGB")
        for label, size, space in REPS:
            Z = representation(img, size, space)
            for K in (3, 4, 5):
                best_pub = min(mm for mm, s in P.T8[nm][K])
                km = KMeans(n_clusters=K, n_init=10, init="k-means++", random_state=0).fit(Z)
                v_km = float(seg_obj(Z, km.cluster_centers_.ravel()[None, :], K)[0])
                Cm = weiszfeld_kmedians(Z, km.cluster_centers_)
                v_med = float(seg_obj(Z, Cm.ravel()[None, :], K)[0])
                out["%s|%s|%d|%s|%d" % (nm, space, size, label, K)] = dict(kmeans=v_km, kmedians=v_med, published=best_pub,
                                                                           ratio=v_km / best_pub, ratio_med=v_med / best_pub)
                print("%-8s %-40s K=%d  k-means/pub %.3f  k-medians/pub %.3f" % (nm, label, K, v_km / best_pub, v_med / best_pub),
                      flush=True)
    json.dump(out, open(os.path.join(RES, "numbers_audit_seg.json"), "w"), indent=1)
    lines = []
    for label, size, space in REPS:
        cells = []
        for nm in IMAGES:
            for K in (3, 4, 5):
                cells.append("%.2f" % out["%s|%s|%d|%s|%d" % (nm, space, size, label, K)]["ratio"])
        lines.append("%s & %s\\\\" % (label, " & ".join(cells)))
    os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    open(os.path.join(RES, "tables", "tab_segaudit.tex"), "w").write("\n".join(lines) + "\n")
