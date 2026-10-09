"""Order-of-magnitude check of Table 8 of the original study (Sect. 3.4 of the paper).

Eq. (10) (sum of Euclidean distances of the pixels to their nearest centroid) is evaluated at the k-means++
solution for the Baboon (Mandrill) and Peppers images resized to 256 x 256 pixels, as stated in the original study,
using several image representations.  The images are the colour versions of the standard test images available at
https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing (baboon.png, peppers.png);
set STD_IMAGES_DIR to the folder that contains them."""
import os
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
from segobj_nb import seg_obj

DIR = os.environ.get("STD_IMAGES_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                                                    "standard_test_images"))
PUBLISHED = {"baboon.png": {3: 888033, 4: 724431, 5: 642019}, "peppers.png": {3: 762524, 4: 654789, 5: 569241}}


def objective(Z, K, seed=0):
    km = KMeans(n_clusters=K, n_init=5, random_state=seed).fit(Z)
    if Z.shape[1] == 1:
        return float(np.abs(Z - km.cluster_centers_.ravel()[None, :]).min(1).sum())
    return float(seg_obj(np.ascontiguousarray(Z), km.cluster_centers_.ravel()[None, :].copy(), K)[0])


if __name__ == "__main__":
    out = {}
    for fn, pub in PUBLISHED.items():
        im = Image.open(os.path.join(DIR, fn)).convert("RGB")
        for size in (256, 128):
            A = np.asarray(im.resize((size, size), Image.BICUBIC), dtype=np.float64)
            reps = {"RGB": A.reshape(-1, 3),
                    "grey": (0.2989 * A[..., 0] + 0.5870 * A[..., 1] + 0.1140 * A[..., 2]).reshape(-1, 1),
                    "CIELAB": color.rgb2lab(A / 255.0).reshape(-1, 3),
                    "HSVx255": color.rgb2hsv(A / 255.0).reshape(-1, 3) * 255}
            for name, Z in reps.items():
                for K in (3, 4, 5):
                    v = objective(Z, K)
                    out["%s|%d|%s|K%d" % (fn, size, name, K)] = dict(kmeans_value=v, published=pub[K], ratio=v / pub[K])
                    print("%-12s %3dpx %-8s K=%d  k-means++: %10.0f  published: %8d  ratio %.2f" % (fn, size, name, K, v, pub[K], v / pub[K]))
    json.dump(out, open("../results/audit_seg.json", "w"), indent=1)


def kmedians_gain():
    """Relative decrease of Eq. (10) obtained by refining the k-means++ solution with k-medians (Weiszfeld)."""
    from exp_seg import weiszfeld_kmedians
    res = {}
    for fn in PUBLISHED:
        im = Image.open(os.path.join(DIR, fn)).convert("RGB")
        Z = np.asarray(im.resize((256, 256), Image.BICUBIC), dtype=np.float64).reshape(-1, 3)
        for K in (3, 4, 5):
            km = KMeans(n_clusters=K, n_init=5, random_state=0).fit(Z)
            v0 = seg_obj(Z, km.cluster_centers_.ravel()[None, :].copy(), K)[0]
            C = weiszfeld_kmedians(Z, km.cluster_centers_)
            v1 = seg_obj(Z, C.ravel()[None, :].copy(), K)[0]
            res["%s|K%d" % (fn, K)] = dict(kmeans=float(v0), kmedians=float(v1), rel_gain=float(1 - v1 / v0))
            print("%-12s K=%d  k-means++ %.0f  k-medians %.0f  gain %.2f%%  published %d" % (fn, K, v0, v1, 100 * (1 - v1 / v0), PUBLISHED[fn][K]))
    json.dump(res, open("../results/audit_seg_kmedians.json", "w"), indent=1)


if __name__ == "__main__":
    kmedians_gain()
