"""E4 and E4b: high-dimensional problems (Table 10) and the dummy-variable check."""
import json
import os
import numpy as np
import bench as B
import paperdata as P
from io_utils import load_records
from stats_utils import m, mw

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FIDS = [1, 10, 14, 16, 17, 20, 21, 22, 23, 27, 28, 29, 30]


def f0(fid, D):
    info = B.get(fid, D_scalable=D)
    return float(info["f"](np.zeros((1, D)))[0])


if __name__ == "__main__":
    recs = load_records(os.path.join(RES, "E4_hd.jsonl.gz"))
    by = {}
    for r in recs:
        by.setdefault((r["kind"], r["fid"], r["D"], r["alg"]), []).append(r)
    out = {"rows": {}}
    lines = []
    changed = {"PSO": 0, "DE": 0, "GPC": 0}
    inbox = all(r["in_box"] for r in recs)
    rms = {}
    for D in (5000, 10000):
        lines.append("\\multicolumn{7}{@{}l}{$D=%d$}\\\\" % D if D == 5000 else "\\multicolumn{7}{@{}l}{$D=10000$}\\\\")
        for fid in FIDS:
            mean = lambda k, a: float(np.mean([r["best_f"] for r in by[(k, fid, D, a)]]))
            pub = P.T7[fid]["GPC"][0 if D == 5000 else 1]
            pubs = "infeasible" if pub is None else m(pub[0], 5)
            z = f0(fid, D)
            z = 0.0 if abs(z) < 1e-12 else z
            lines.append("$f_{%d}$ %s & %s & %s & %s & %s & %s & %s\\\\" % (
                fid, B.get(fid)["name"], m(z, 5), pubs, m(mean("orig", "GPC"), 5), m(mean("trans", "GPC"), 5),
                m(mean("orig", "PSO"), 5), m(mean("orig", "DE"), 5)))
            for a in changed:
                x = [r["best_f"] for r in by[("orig", fid, D, a)]]
                y = [r["best_f"] for r in by[("trans", fid, D, a)]]
                changed[a] += mw(x, y)[0] < 0.05
            rms["%d|%d" % (fid, D)] = float(np.median([r["rms_x"] for r in by[("orig", fid, D, "GPC")]]))
            out["rows"]["%d|%d" % (fid, D)] = dict(f0=z, gpc_orig=mean("orig", "GPC"), gpc_trans=mean("trans", "GPC"),
                                                  pso=mean("orig", "PSO"), de=mean("orig", "DE"))
    out["changed"] = changed
    out["all_in_box"] = inbox
    out["median_rms_gpc"] = rms
    os.makedirs(os.path.join(RES, "tables"), exist_ok=True)
    open(os.path.join(RES, "tables", "tab_hd.tex"), "w").write("\n".join(lines) + "\n")
    d = load_records(os.path.join(RES, "E4b_dummy.jsonl.gz"))
    out["E4b"] = dict(identical=int(sum(r["identical"] for r in d)), pairs=len(d),
                      booth=float(np.mean([r["best_f_2"] for r in d if r["fid"] == 4])),
                      shubert=float(np.mean([r["best_f_2"] for r in d if r["fid"] == 26])))
    json.dump(out, open(os.path.join(RES, "numbers_hd.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
