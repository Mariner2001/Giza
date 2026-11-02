"""All figures of the paper (written to ../figs/).  usage: python3 figures.py [name ...]"""
import json
import math
import os
import sys
import numpy as np
import style
from style import COLORS, MARKERS, WIDTH

style.setup()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figs")
FLOOR = 1e-25


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------- Fig. 1 (theory)
def fig_theory(N=4_000_000, seed=1):
    rng = np.random.default_rng(seed)
    g, th = 9.8, math.radians(10.0)
    lb, ub = -5.12, 5.12
    R = (ub - lb) / 2

    def cand(p, mode, n):
        v0 = rng.random(n)
        mu = 1 + 9 * rng.random(n)
        d = v0 ** 2 / (2 * g * (math.sin(th) + mu * math.cos(th)))
        s = v0 ** 2 / (2 * g * math.sin(th))
        eps = -R + 2 * R * rng.random(n)
        y = (p + d) * (s * eps) if mode == "mult" else (p + d) + s * eps
        return np.clip(y, lb, ub)

    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    bins = np.linspace(lb, ub, 401)
    for p, c in ((0.5, "#9ecae1"), (2.0, "#2a78d6"), (4.0, "#08306b")):
        y = cand(p, "mult", N)
        h, e = np.histogram(y, bins=bins, density=True)
        ax[0].plot(0.5 * (e[1:] + e[:-1]), h, color=c, lw=0.9, label="parent $p_j=%.1f$" % p)
        ax[0].axvline(p, color=c, ls="--", lw=0.7)
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-3, 30)
    ax[0].set_xlabel("candidate coordinate $y_j$")
    ax[0].set_ylabel("density")
    ax[0].set_title("(a) density of a GPC candidate $y_j$", loc="left")
    ax[0].legend(loc="upper right")
    deltas = np.logspace(-6, -0.5, 23)
    for t, mode, lab, col, mk, ls in ((0.0, "mult", "GPC, target $t=0$ (parent $p=1$)", COLORS["GPC"], "o", "-"),
                                      (1.0, "mult", "GPC, target $t=1$ (parent $p=t$)", COLORS["GPC"], "s", "--"),
                                      (1.0, "add", "additive GPC, target $t=1$ (parent $p=t$)", COLORS["GPC-A"], "D", "-")):
        p = 1.0
        y = cand(p, mode, N)
        pr = np.array([np.mean(np.abs(y - t) <= dl) for dl in deltas])
        ok = pr > 0
        big = dict(mfc="none", ms=6, mew=1.0) if (t == 0.0) else dict(ms=3)
        ax[1].plot(deltas[ok], pr[ok], color=col, marker=mk, ls=ls, label=lab, **big)
    ax[1].plot(deltas, 2 * deltas / 1.0, color="k", ls=":", lw=0.8, label="bound $2\\delta/|t|$ (Prop. 3)")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-7, 1)
    ax[1].set_xlabel("precision $\\delta$")
    ax[1].set_ylabel("$\\Pr(|y_j-t|\\leq\\delta)$")
    ax[1].set_title("(b) $\\Pr(|y_j-t|\\leq\\delta)$", loc="left")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    for a in ax:
        a.grid(alpha=0.3, lw=0.4)
    fig.tight_layout()
    save(fig, "fig_theory")


# ------------------------------------------------------------------------------------- Fig. 2 (translation)
def fig_translation():
    import analysis_bench as AB
    import bench as B
    e1, _ = AB.load("E1")
    e2, _ = AB.load("E2a")
    algs = ["GPC", "PSO", "CMA-ES", "jSO"]
    fig, axes = plt.subplots(1, len(algs), figsize=(WIDTH, 5.3), sharey=True)
    fids = list(range(1, 31))
    ypos = np.arange(30)[::-1]
    for ax, a in zip(axes, algs):
        m1 = np.array([max(np.median(e1[f][a]), FLOOR) for f in fids])
        m2 = np.array([max(np.median(e2[f][a]), FLOOR) for f in fids])
        for y, u, v in zip(ypos, m1, m2):
            ax.plot([u, v], [y, y], color=COLORS[a], lw=0.8)
        ax.scatter(m1, ypos, facecolors="white", edgecolors=COLORS[a], s=14, zorder=3, lw=0.8)
        ax.scatter(m2, ypos, color=COLORS[a], s=14, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(1e-27, 1e7)
        ax.set_xticks([1e-24, 1e-12, 1])
        ax.set_title(a, loc="left")
        ax.set_xlabel("median error")
        ax.grid(axis="x", alpha=0.3, lw=0.4)
        ax.grid(axis="y", alpha=0.15, lw=0.4)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(["$f_{%d}$ %s" % (f, B.get(f)["name"]) for f in fids])
    hl = [Line2D([], [], marker="o", ls="", mfc="white", mec="k", label="E1: original problem"),
          Line2D([], [], marker="o", ls="", color="k", label="E2a: translated problem")]
    fig.legend(handles=hl, loc="lower center", ncol=2, bbox_to_anchor=(0.55, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig_translation")


# ------------------------------------------------------------------------------------- Fig. 3 (convergence)
def fig_convergence():
    import analysis_bench as AB
    _, E1 = AB.load("E1")
    _, E2 = AB.load("E2a")
    panels = [(27, "Sphere"), (22, "Rastrigin"), (23, "Rosenbrock"), (1, "Ackley")]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.9))
    for ax, (f, nm) in zip(axes.ravel(), panels):
        def curve(E, a):
            rs = E[f][a]
            C = np.array([np.maximum(np.asarray(r["curve"], float) - (r["best_f"] - r["err"]), 0) for r in rs])
            return np.asarray(rs[0]["cps"]), np.maximum(np.median(C, axis=0), FLOOR)
        x, y = curve(E1, "GPC")
        ax.plot(x, y, color=COLORS["GPC"], ls="--", marker="s", markevery=3, label="GPC, original")
        for a, E in (("GPC", E2), ("PSO", E2), ("DE", E2), ("CMA-ES", E2), ("jSO", E2)):
            x, y = curve(E, a)
            lab = "GPC, translated" if a == "GPC" else a
            ax.plot(x, y, color=COLORS[a], marker=MARKERS[a], markevery=3, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("$f_{%d}$ %s" % (f, nm), loc="left")
        ax.set_xlabel("evaluations")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0, 0].set_ylabel("median error")
    axes[1, 0].set_ylabel("median error")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_convergence")


# ------------------------------------------------------------------------------------- Fig. 4 (structural bias)
def fig_bias():
    import analysis_bias as AB
    X = AB.load()
    algs = ["GPC", "GPC-A", "PSO", "DE", "CMA-ES", "L-SHADE", "jSO"]
    fig, axes = plt.subplots(2, len(algs), figsize=(WIDTH, 3.0), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 26)
    for i, box in enumerate(("centre", "corner")):
        for j, a in enumerate(algs):
            ax = axes[i, j]
            A = X[(box, a)].ravel()
            h, _ = np.histogram(A, bins=bins)
            ax.bar(0.5 * (bins[1:] + bins[:-1]), h / A.size, width=0.036, color=COLORS[a])
            ax.axvline(0.5 if box == "centre" else 0.0, color="k", ls="--", lw=0.8)
            ax.axhline(1 / 25, color="k", ls=":", lw=0.7)
            ax.set_yscale("log")
            ax.set_ylim(5e-4, 1.2)
            if i == 0:
                ax.set_title(a, loc="left")
    axes[0, 0].set_ylabel("origin at centre\nfraction")
    axes[1, 0].set_ylabel("origin at corner\nfraction")
    fig.supxlabel("normalised coordinate of the final solution", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_bias")


# ------------------------------------------------------------------------------------- Fig. 5 (BBOB)
def fig_bbob():
    res = json.load(open(os.path.join(RES, "numbers_bbob.json")))
    algs = ["GPC", "GPC-A", "PSO", "DE", "CMA-ES", "L-SHADE", "jSO", "RS"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.9))
    for ax, D in zip(axes.ravel(), (2, 5, 10, 20)):
        for a in algs:
            k = "%s|%d" % (a, D)
            if k not in res:
                continue
            x, y = res[k]["ecdf_budgets"], res[k]["ecdf"]
            ls = ":" if a == "RS" else ("-." if a in ("GPC-A", "jSO") else ("--" if a in ("DE",) else "-"))
            ax.plot(x, y, color=COLORS[a], marker=MARKERS[a], markevery=10, ls=ls, label=a)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_title("$D=%d$" % D, loc="left")
        ax.set_xlabel("evaluations / $D$")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0, 0].set_ylabel("fraction of targets")
    axes[1, 0].set_ylabel("fraction of targets")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig_bbob")


# ------------------------------------------------------------------------------------- Fig. 6 (centroids)
def fig_seg():
    from io_utils import load_records
    import exp_seg as S6
    e6 = load_records(os.path.join(RES, "E6_seg.jsonl.gz"))
    p6c = os.path.join(RES, "E6c_seg.jsonl.gz")
    e6c = load_records(p6c) if os.path.exists(p6c) else []
    fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 2.3))
    b1 = np.linspace(0, 255, 52)
    for a in ("GPC", "PSO", "DE", "k-medians"):
        c = np.concatenate([np.asarray(r["cent"]) for r in e6 if r["alg"] == a])
        h, _ = np.histogram(np.clip(c, 0, 255), bins=b1)
        ax[0].step(b1[:-1], h / c.size, where="post", color=COLORS[a], lw=1.0, label=a)
    ax[0].set_xlim(0, 255)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("centroid coordinate (RGB, origin at 0)")
    ax[0].set_ylabel("fraction")
    ax[0].set_title("(a) E6: original colour coordinates", loc="left")
    b2 = np.linspace(-200, 200, 81)
    for a in ("GPC", "PSO", "DE"):
        if not any(r["alg"] == a for r in e6c):
            continue
        c = np.concatenate([np.asarray(r["cent"]) for r in e6c if r["alg"] == a])
        h, _ = np.histogram(np.clip(c, -200, 200), bins=b2)
        ax[1].step(b2[:-1], h / c.size, where="post", color=COLORS[a], lw=1.0, label=a)
    ax[1].axvline(0, color="k", ls="--", lw=0.7)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("centroid coordinate minus mean colour (origin at 0)")
    ax[1].set_title("(b) E6c: centred colour coordinates", loc="left")
    for a in ax:
        a.grid(alpha=0.3, lw=0.4)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig_seg")


ALL = {"theory": fig_theory, "translation": fig_translation, "convergence": fig_convergence, "bias": fig_bias,
       "bbob": fig_bbob, "seg": fig_seg}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
        print("figure", n, "done", flush=True)
