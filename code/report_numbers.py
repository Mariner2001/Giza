"""Prints the numbers quoted in the text of the paper (for checking the manuscript against the results)."""
import json
import os
import numpy as np
import bench as B
from analysis_bench import load, ALL11, ORIG6, FIDS30, FIDS14
from stats_utils import mw

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def J(n):
    p = os.path.join(RES, n)
    return json.load(open(p)) if os.path.exists(p) else {}


if __name__ == "__main__":
    nb = J("numbers_bench.json")
    if nb:
        s = nb["summary"]
        print("== ranks (11):")
        for e in ("E1", "E2a", "E2b"):
            r = s["ranks"][e]
            print("  ", e, ", ".join("%s %.2f" % (a, r[a]) for a in sorted(r, key=r.get)))
        print("== FF", s["FF"], "pFF", s["pFF"], "holm row", nb["summary_holm_row"])
        print("== Wilcoxon (raw, adj, rej):")
        for e in ("E1", "E2a", "E2b"):
            print("  ", e, {a: (round(v[0], 4), round(v[1], 4), v[2]) for a, v in s["wilcoxon"][e].items()})
        print("== W/T/L:", s["wtl"])
        print("== changed E1->E2a:", s["changed"])
        print("== six-algorithm ranks:", nb["six"])
        print("== comparators vs published:", {k: (v if isinstance(v, int) else len(v)) for k, v in nb["comparators"].items()})
        print("== cost (s per run):", nb.get("cost"))
        if "rot" in nb:
            r = nb["rot"]
            print("== rot ranks:")
            for e, d in r["ranks"].items():
                print("  ", e, ", ".join("%s %.2f" % (a, d[a]) for a in sorted(d, key=d.get)))
            print("== rot changed:", r["changed"])
            print("== rot GPC W/T/L:", {e: r["wtl"][e] for e in r["wtl"]})
            print("== rot GPC medians:", {e: {f: "%.2g" % v for f, v in d.items()} for e, d in r["gpc_median"].items()})
    # E2a details for GPC
    e1, _ = load("E1")
    e2, _ = load("E2a")
    big = [f for f in FIDS30 if np.median(e1[f]["GPC"]) < 1e-12]
    inc = [f for f in big if np.log10(max(np.median(e2[f]["GPC"]), 1e-300)) - np.log10(max(np.median(e1[f]["GPC"]), 1e-300)) > 10]
    print("== E1 GPC median < 1e-12 on", len(big), "functions; increase > 10 orders on", len(inc),
          "; E2a medians range %.2g - %.2g" % (min(np.median(e2[f]["GPC"]) for f in inc), max(np.median(e2[f]["GPC"]) for f in inc)))
    unaff = [f for f in FIDS30 if mw(e1[f]["GPC"], e2[f]["GPC"])[0] >= 0.05]
    print("== GPC unaffected by translation:", [(f, B.get(f)["name"]) for f in unaff])
    for a in ALL11:
        if a == "GPC":
            continue
        d = [abs(np.log10((np.median(e2[f][a]) + 1e-30) / (np.median(e1[f][a]) + 1e-30))) for f in FIDS30]
        print("   median |log10 change of median error| %s: %.3f" % (a, np.median(d)))
    # GPC vs RS and others in E2a
    for e in ("E2a",):
        worst = sum(np.argmax([e2[f][a].mean() for a in ORIG6]) == ORIG6.index("GPC") for f in FIDS30)
        print("== E2a: GPC worst of the six on", worst)
    for k in ("numbers_bias.json", "numbers_hd.json", "numbers_e7.json", "matlab_native.json"):
        d = J(k)
        if d:
            print("==", k, json.dumps(d if k != "numbers_hd.json" else {x: d[x] for x in d if x != "rows"}, default=str)[:3000])
    bb = J("numbers_bbob.json")
    for D in (2, 5, 10, 20):
        print("== BBOB D=%d" % D, {a: (round(bb["%s|%d" % (a, D)]["final"], 3), bb["%s|%d" % (a, D)]["solved"],
                                      round(bb["%s|%d" % (a, D)]["median_xrms"], 2))
                                   for a in ("GPC", "GPC-A", "PSO", "DE", "CMA-ES", "L-SHADE", "jSO", "RS") if "%s|%d" % (a, D) in bb})
    if bb:
        for a in ("GPC", "RS", "L-SHADE", "jSO", "CMA-ES"):
            if "%s|2" % a in bb:
                print("   f5 D=2", a, bb["%s|2" % a]["per_fun"]["5"], " f1 D=20", bb["%s|20" % a]["per_fun"]["1"] if "%s|20" % a in bb else None)
        print("   total solved:", {a: sum(bb["%s|%d" % (a, D)]["solved"] for D in (2, 5, 10, 20) if "%s|%d" % (a, D) in bb)
                                   for a in ("GPC", "GPC-A", "PSO", "DE", "CMA-ES", "L-SHADE", "jSO", "RS")})
    sg = J("numbers_seg.json")
    if sg:
        for e in ("E6", "E6c"):
            if e in sg:
                for k, v in sg[e].items():
                    if "GPC" in k or e == "E6":
                        print("==", e, k, {x: round(y, 3) for x, y in v.items()})
        print("== E6c changed:", sg.get("E6c_changed"), "centroid dist:", sg.get("E6c_centroid_dist"))
        if "E6b" in sg:
            for k, v in sorted(sg["E6b"].items()):
                print("   E6b", k, v)
