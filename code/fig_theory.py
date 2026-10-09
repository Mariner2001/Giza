"""Figure: distribution of a GPC candidate coordinate (theory illustration)."""
import math
import numpy as np
import matplotlib.pyplot as plt
import style
from style import COL
# ensure output folders exist
import os as _os
for _d in ("../results", "../paper/tables", "../paper/figs", "../figs"):
    _os.makedirs(_d, exist_ok=True)

style.setup()
rng = np.random.default_rng(1)
g, th, mu_min, mu_max = 9.8, math.radians(10.0), 1.0, 10.0
lb, ub = -5.12, 5.12
R = (ub - lb) / 2
M = 4_000_000


def sample(p, mode="mult", clip=True):
    v = rng.random(M)
    mu = mu_min + (mu_max - mu_min) * rng.random(M)
    d = v ** 2 / (2 * g * (math.sin(th) + mu * math.cos(th)))
    x = v ** 2 / (2 * g * math.sin(th))
    eps = rng.uniform(-R, R, M)
    y = (p + d) * (x * eps) if mode == "mult" else (p + d) + x * eps
    return np.clip(y, lb, ub) if clip else y


fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.75))
ax = axs[0]
bins = np.linspace(-5.2, 5.2, 261)
shades = ["#86b6ef", "#2a78d6", "#104281"]
for p, c in zip([0.5, 2.0, 4.0], shades):
    y = sample(p)
    h, e = np.histogram(y, bins=bins, density=True)
    ax.step(0.5 * (e[1:] + e[:-1]), h, where="mid", color=c, lw=1.2, label=r"parent $p_j=%.1f$" % p)
    ax.axvline(p, color=c, lw=0.9, ls="--")
ax.set_yscale("log")
ax.set_xlabel(r"candidate coordinate $y_j$")
ax.set_ylabel("density")
ax.set_xlim(-5.4, 5.4)
ax.set_ylim(1e-3, 1e2)
ax.set_title(r"(a) density of a GPC candidate $y_j$", loc="left")
ax.legend(loc="upper right", handlelength=1.2, fontsize=6.5)

ax = axs[1]
deltas = np.logspace(-6, -0.5, 23)
cases = [("GPC", "mult", 1.0, 0.0, "GPC, target $t=0$ (parent $p=1$)", "-"),
         ("GPC", "mult", 1.0, 1.0, "GPC, target $t=1$ (parent $p=t$)", "--"),
         ("GPC-A", "add", 1.0, 1.0, "additive GPC, target $t=1$ (parent $p=t$)", "-.")]
for alg, mode, p, t, lab, ls in cases:
    y = sample(p, mode=mode, clip=False)
    dist = np.abs(y - t)
    prob = np.array([(dist <= dd).mean() for dd in deltas])
    ok = prob > 0
    ax.loglog(deltas[ok], prob[ok], ls=ls, color=COL[alg], marker=style.MK[alg] if t else "o", ms=3.2,
              markevery=(1 if mode == "add" else 0, 2), lw=1.6 if t == 0 else 1.2, label=lab, zorder=3 if mode == "add" else 2)
ax.loglog(deltas, np.minimum(1, 2 * deltas / 1.0), color=style.INK2, lw=0.9, ls=":", label=r"bound $2\delta/|t|$ (Prop. 3)")
ax.set_xlabel(r"precision $\delta$")
ax.set_ylabel(r"$\Pr(|y_j-t|\leq\delta)$")
ax.set_title(r"(b) $\Pr(|y_j-t|\leq\delta)$", loc="left")
ax.set_ylim(1e-7, 1.5)
fig.tight_layout(w_pad=1.5)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=6.5, handlelength=2.4)
fig.savefig("../paper/figs/fig_theory.pdf")
fig.savefig("../figs/fig_theory.png", dpi=200)
print("ok")
