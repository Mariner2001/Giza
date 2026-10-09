"""Flat CSV copies of the per-run results (for spreadsheets and other languages).
Arrays (convergence curves, final solutions, centroids) are kept only in the JSONL files."""
import csv
import os
import numpy as np
from io_utils import load_records

OUT = "../results/csv"
os.makedirs(OUT, exist_ok=True)

SPEC = {
    "E1": ["fid", "alg", "run", "best_f", "err", "nfe", "cpu", "d_origin", "d_opt", "o_norm"],
    "E2a": ["fid", "alg", "run", "best_f", "err", "nfe", "cpu", "d_origin", "d_opt", "o_norm"],
    "E2b": ["fid", "alg", "run", "best_f", "err", "nfe", "cpu", "d_origin", "d_opt", "o_norm"],
    "E4_hd": ["kind", "fid", "D", "alg", "run", "best_f", "err", "rms_x", "in_box", "cpu"],
    "E4b_dummy": ["fid", "run", "best_f_2", "best_f_big", "identical", "nfe_2", "nfe_big"],
    "E6_seg": ["img", "K", "alg", "run", "obj", "PRI", "VoI", "cpu"],
}

if __name__ == "__main__":
    for name, cols in SPEC.items():
        R = load_records("../results/%s.jsonl.gz" % name)
        with open(os.path.join(OUT, name + ".csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in R:
                w.writerow([r.get(c, "") for c in cols])
        print(name, len(R))
    # E3: summary of the normalised final solutions per run (mean distance to the origin, fraction near it)
    R = load_records("../results/E3_bias.jsonl.gz")
    with open(os.path.join(OUT, "E3_bias.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["box", "alg", "run"] + ["x%d" % (j + 1) for j in range(30)])
        for r in R:
            w.writerow([r["box"], r["alg"], r["run"]] + ["%.6g" % v for v in np.asarray(r["xn"])])
    print("E3_bias", len(R))
    # E5: final precision of each run
    R = load_records("../results/E5_bbob_main.jsonl.gz") + load_records("../results/E5_bbob_RS.jsonl.gz")
    with open(os.path.join(OUT, "E5_bbob.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alg", "D", "function", "instance", "nfe", "final_precision", "rms_coordinate"])
        for r in R:
            w.writerow([r["alg"], r["D"], r["fun"], r["inst"], r["nfe"], "%.6g" % float(np.asarray(r["tf"])[-1]), "%.6g" % r["xnorm"]])
    print("E5_bbob", len(R))
