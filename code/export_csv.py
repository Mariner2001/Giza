"""Exports the per-run results to CSV files in ../results/csv/ (one row per run; arrays are omitted except the
final solution of E3, which is written as 30 columns)."""
import csv
import glob
import os
import numpy as np
from io_utils import load_records

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RES, "csv")


def export(src, dst, fields, expand=None):
    recs = load_records(src)
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh)
        head = list(fields)
        if expand:
            k, n = expand
            head += ["%s_%d" % (k, i + 1) for i in range(n)]
        w.writerow(head)
        for r in recs:
            row = [r.get(f, "") for f in fields]
            if expand:
                row += list(np.asarray(r[expand[0]], float))
            w.writerow(row)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    bench = ["fid", "alg", "run", "best_f", "err", "nfe", "cpu", "d_origin", "d_opt", "o_norm"]
    for p in sorted(glob.glob(os.path.join(RES, "E[12]*.jsonl.gz"))):
        name = os.path.basename(p).replace(".jsonl.gz", "")
        export(p, os.path.join(OUT, name + ".csv"), bench)
    for p in sorted(glob.glob(os.path.join(RES, "E3_bias*.jsonl.gz"))):
        name = os.path.basename(p).replace(".jsonl.gz", "")
        export(p, os.path.join(OUT, name + ".csv"), ["box", "alg", "run"], expand=("xn", 30))
    if os.path.exists(os.path.join(RES, "E4_hd.jsonl.gz")):
        export(os.path.join(RES, "E4_hd.jsonl.gz"), os.path.join(OUT, "E4_hd.csv"),
               ["kind", "fid", "D", "alg", "run", "best_f", "err", "rms_x", "in_box", "cpu"])
    export(os.path.join(RES, "E4b_dummy.jsonl.gz"), os.path.join(OUT, "E4b_dummy.csv"),
           ["fid", "run", "best_f_2", "best_f_big", "identical", "nfe_2", "nfe_big"])
    for p in sorted(glob.glob(os.path.join(RES, "E5_bbob_*.jsonl.gz"))):
        recs = load_records(p)
        with open(os.path.join(OUT, os.path.basename(p).replace(".jsonl.gz", ".csv")), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["alg", "D", "fun", "inst", "nfe", "final_precision", "xnorm"])
            for r in recs:
                w.writerow([r["alg"], r["D"], r["fun"], r["inst"], r["nfe"], float(np.min(r["tf"])), r["xnorm"]])
    for p in sorted(glob.glob(os.path.join(RES, "E6*_seg.jsonl.gz"))):
        name = os.path.basename(p).replace(".jsonl.gz", "")
        export(p, os.path.join(OUT, name + ".csv"), ["img", "K", "alg", "run", "obj", "cpu", "PRI", "VoI"])
    print("CSV files in", OUT)
