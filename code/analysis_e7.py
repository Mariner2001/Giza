"""E7: ablation, parameter sensitivity and initialisation of GPC (Table 15).

For each variant: number of the fourteen functions minimised at the origin whose median error is below 1e-8 in
E1 (original), E2a (translated) and E2b (shifted optimum); wins/ties/losses against the default GPC (U-statistic
rule, alpha = 0.05) on the 30 functions of E1 and E2a; and the number of the 30 functions whose error
distribution differs significantly between E1 and E2a (Mann-Whitney, alpha = 0.05)."""
import json
import os
import numpy as np
import bench as B
from io_utils import load_records
from stats_utils import mw, wtl_u

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
VARIANTS = [("GPC", "GPC (default)"), ("GPC-A", "additive update (GPC-A)"), ("GPC_nod", "no stone displacement ($d=0$)"),
            ("GPC_pss0", "one coordinate per child ($p_{ss}=0$)"), ("GPC_pss1", "no substitution ($p_{ss}=1$)"),
            ("GPC_reinit", "random re-initialisation instead of clipping"), ("GPC_nonelit", "no elitism (children replace parents)"),
            ("GPC_theta5", "$\\theta=5^\\circ$"), ("GPC14", "$\\theta=14^\\circ$"), ("GPC_theta30", "$\\theta=30^\\circ$"),
            ("GPC_theta60", "$\\theta=60^\\circ$"), ("GPC_mu0.1-1", "$\\mu_k\\in[0.1,1]$"), ("GPC_mu10-100", "$\\mu_k\\in[10,100]$"),
            ("GPC_n10", "$n=10$"), ("GPC_n50", "$n=50$"), ("GPC_n100", "$n=100$"),
            ("GPC_pss0.1", "$p_{ss}=0.1$"), ("GPC_pss0.3", "$p_{ss}=0.3$"), ("GPC_pss0.7", "$p_{ss}=0.7$"),
            ("GPC_pss0.9", "$p_{ss}=0.9$"), ("GPC_lhs", "Latin-hypercube initialisation"),
            ("GPC_corner", "initialisation in the upper quarter of the box")]
O14 = sorted(B.ORIGIN_OPT)


def load(exp):
    recs = load_records(os.path.join(RES, "%s.jsonl.gz" % exp))
    p = os.path.join(RES, "%s_E7.jsonl.gz" % exp)
    if os.path.exists(p):
        recs += load_records(p)
    E = {}
    for r in sorted(recs, key=lambda r: (r["fid"], r["alg"], r["run"])):
        E.setdefault(r["fid"], {}).setdefault(r["alg"], []).append(max(r["err"], 0.0))
    return {f: {a: np.array(v) for a, v in d.items()} for f, d in E.items()}


if __name__ == "__main__":
    E = {e: load(e) for e in ("E1", "E2a", "E2b")}
    out = {}
    for key, label in VARIANTS:
        row = dict(label=label)
        for e in ("E1", "E2a", "E2b"):
            fids = [f for f in O14 if key in E[e].get(f, {})]
            row["solved_" + e] = int(sum(np.median(E[e][f][key]) < 1e-8 for f in fids))
            row["n_" + e] = len(fids)
        for e in ("E1", "E2a"):
            if key != "GPC" and all(key in E[e][f] for f in range(1, 31)):
                res = [wtl_u(E[e][f][key], E[e][f]["GPC"]) for f in range(1, 31)]
                row["wtl_" + e] = (res.count("w"), res.count("t"), res.count("l"))
        if all(key in E["E1"][f] and key in E["E2a"][f] for f in range(1, 31)):
            row["changed"] = int(sum(mw(E["E1"][f][key], E["E2a"][f][key])[0] < 0.05 for f in range(1, 31)))
            # median over the 14 origin functions of log10(median error E2a / median error E1)
            row["dlog"] = float(np.median([np.log10((np.median(E["E2a"][f][key]) + 1e-30) / (np.median(E["E1"][f][key]) + 1e-30))
                                           for f in O14]))
        out[key] = row
        print("%-14s" % key, {k: v for k, v in row.items() if k != "label"})
    json.dump(out, open(os.path.join(RES, "numbers_e7.json"), "w"), indent=1)
