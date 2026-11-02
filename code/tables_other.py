"""LaTeX bodies of Tables 9 (structural bias), 11-12 (BBOB), 13-14 (segmentation), 15 (E7) and 16 (cost),
from the numbers_*.json files written by the analysis scripts."""
import json
import os
from stats_utils import m

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TAB = os.path.join(RES, "tables")
ALG11 = ["GA", "ICA", "PSO", "DE", "FA", "GPC", "GPC-A", "CMA-ES", "L-SHADE", "jSO", "RS"]
BB = ["GPC", "GPC-A", "PSO", "DE", "CMA-ES", "L-SHADE", "jSO", "RS"]


def J(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, lines):
    os.makedirs(TAB, exist_ok=True)
    open(os.path.join(TAB, name), "w").write("\n".join(lines) + "\n")


def p_cell(p):
    if p < 1e-6:
        return "$<10^{-6}$"
    return "%.2g" % p if p >= 0.001 else "$%s$" % m(p, 1).strip("$")


def bias_table():
    b = J("numbers_bias.json")
    lines = []
    for a in ALG11:
        cells = []
        for box in ("centre", "corner"):
            k = "%s|%s" % (box, a)
            if k not in b:
                cells += ["--"] * 5
                continue
            v = b[k]
            cells += ["%.2f" % v["ks1"], p_cell(v["p1"]), "%d" % v["n_rej"], "%.2f" % v["ks_all"],
                      "%.0f / %.0f" % (v["near0"], v["bound"])]
        lines.append("%s & %s\\\\" % (a, " & ".join(cells)))
    write("tab_bias.tex", lines)


def bbob_tables():
    b = J("numbers_bbob.json")
    if b is None:
        return
    lines = []
    for a in BB:
        cells = []
        for D in (2, 5, 10, 20):
            v = b.get("%s|%d" % (a, D))
            cells += ["%.2f" % v["final"], "%d" % v["solved"]] if v else ["--", "--"]
        lines.append("%s & %s\\\\" % (a, " & ".join(cells)))
    write("tab_bbob.tex", lines)
    # detail: function groups (averaged over D) and speed (fraction of targets after 10^2 D, 10^3 D, 10^4 D)
    G = ["separable", "low/moderate conditioning", "high conditioning, unimodal", "multimodal, adequate structure",
         "multimodal, weak structure"]
    lines = []
    for a in BB:
        cells = []
        for g in G:
            vals = [b["%s|%d" % (a, D)]["groups"][g] for D in (2, 5, 10, 20) if "%s|%d" % (a, D) in b]
            cells.append("%.2f" % (sum(vals) / len(vals)) if vals else "--")
        for D in (5, 20):
            v = b.get("%s|%d" % (a, D))
            cells += ["%.2f" % v["at"][str(x)] for x in (100, 1000, 10000)] if v else ["--"] * 3
        lines.append("%s & %s\\\\" % (a, " & ".join(cells)))
    write("tab_bbob_detail.tex", lines)


def seg_tables():
    s = J("numbers_seg.json")
    if s is None:
        return
    lines = []
    for K in (3, 4, 5):
        first = True
        rows = [("k-means", "E6"), ("k-medians", "E6"), ("GPC", "E6"), ("PSO", "E6"), ("DE", "E6")]
        if "E6c" in s:
            rows.append(("GPC", "E6c"))
        for a, e in rows:
            v = s[e]["%d|%s" % (K, a)]
            name = a if e == "E6" else "GPC, centred colours"
            lines.append("%s & %s & %.2f & %.1f & %.1f & %.3f & %.2f & %.3f & %.2f\\\\" % (
                "$K=%d$" % K if first else "", name, v["rank"], v["ex_mean"], v["ex_med"], v["PRI"], v["VoI"],
                v["Cov"], v["cpu"]))
            first = False
        if K < 5:
            lines.append("\\midrule")
    write("tab_seg.tex", lines)
    if "E6b" in s:
        e = s["E6b"]
        lines = []
        for img in ("Baboon", "Peppers"):
            for rep in ("RGB", "HSV"):
                for K in (3, 4, 5):
                    pub = e["%s|%d|published" % (img, K)]
                    vals = [e["%s-%s|%d|%s" % (img, rep, K, a)]["mean"] for a in ("GPC", "PSO", "DE", "k-means", "k-medians")]
                    best = min(vals)
                    cells = [("{\\boldmath%s}" % m(round(v), 7) if v == best else m(round(v), 7)) for v in vals]
                    assert pub["mean"] == pub["best"]   # GPC has the best published mean for these images
                    lines.append("%s & %s & %d & %s & %s\\\\" % (
                        img if (rep == "RGB" and K == 3) else "", rep if K == 3 else "", K, m(pub["mean"], 6),
                        " & ".join(cells)))
            if img == "Baboon":
                lines.append("\\midrule")
        write("tab_seg_std.tex", lines)


def e7_table():
    e = J("numbers_e7.json")
    if e is None:
        return
    order = ["GPC", "GPC-A", "GPC_nod", "GPC_pss0", "GPC_pss1", "GPC_reinit", "GPC_nonelit", "GPC_theta5", "GPC14",
             "GPC_theta30", "GPC_theta60", "GPC_mu0.1-1", "GPC_mu10-100", "GPC_n10", "GPC_n50", "GPC_n100",
             "GPC_pss0.1", "GPC_pss0.3", "GPC_pss0.7", "GPC_pss0.9", "GPC_lhs", "GPC_corner"]
    groups = {"GPC-A": "Components", "GPC_theta5": "Parameters", "GPC_lhs": "Initialisation"}
    lines = []
    for k in order:
        if k not in e:
            continue
        v = e[k]
        if k in groups:
            lines.append("\\midrule")
            lines.append("\\multicolumn{7}{@{}l}{\\emph{%s}}\\\\" % groups[k])
        w1 = "%d/%d/%d" % tuple(v["wtl_E1"]) if "wtl_E1" in v else "--"
        w2 = "%d/%d/%d" % tuple(v["wtl_E2a"]) if "wtl_E2a" in v else "--"
        lines.append("%s & %d & %d & %d & %s & %s & %s\\\\" % (
            v["label"], v["solved_E1"], v["solved_E2a"], v["solved_E2b"], w1, w2, v.get("changed", "--")))
    write("tab_e7.tex", lines)


if __name__ == "__main__":
    bias_table()
    bbob_tables()
    seg_tables()
    e7_table()
    print("ok")
