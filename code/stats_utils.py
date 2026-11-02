"""Statistical procedures and number formatting shared by the analysis scripts."""
import math
import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------------------------- formatting
def m(x, sig=3):
    """Number for LaTeX tables: fixed notation for 1e-3 <= |x| < 10^max(4, sig) (integers when |x| >= 10^(sig-1)),
    otherwise mantissa (min(sig, 3) significant digits) and exponent; minus signs as {-}."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, float) and math.isinf(x):
        return "$\\infty$" if x > 0 else "${-}\\infty$"
    if x == 0:
        return "$0$"
    ax = abs(x)
    if 1e-3 <= ax < 10 ** max(4, sig):
        s = ("%.0f" % x) if ax >= 10 ** (sig - 1) else ("%.*g" % (sig, x))
        if "e" in s:  # %g may still switch to exponent notation for borderline values
            s = "%.0f" % x
        return "$" + s.replace("-", "{-}") + "$"
    d = min(sig, 3)
    s = "%.*e" % (d - 1, x)
    mant, ex = s.split("e")
    return "$" + mant.replace("-", "{-}") + "\\mathrm{e}{" + str(int(ex)).replace("-", "{-}") + "}$"


def pfmt(p):
    """p-value for tables."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 1e-6:
        return "$<10^{-6}$"
    if p < 1e-3:
        mant, ex = ("%.1e" % p).split("e")
        return "$" + mant + "\\cdot10^{" + str(int(ex)) + "}$"
    return "$%.3f$" % p if p < 0.01 else "$%.2f$" % p


# ----------------------------------------------------------------------------------------------- tests
def rank_rows(M):
    """Ranks within each row (minimisation, ties averaged)."""
    return np.vstack([stats.rankdata(r, method="average") for r in M])


def friedman(M):
    """M: N problems x k algorithms (lower is better).  Returns mean ranks, chi2 (no tie correction),
    chi2 with tie correction, Iman-Davenport F and its p-value (from the uncorrected chi2)."""
    M = np.asarray(M, float)
    N, k = M.shape
    R = rank_rows(M)
    Rj = R.mean(0)
    chi2 = 12.0 * N / (k * (k + 1)) * (np.sum(Rj ** 2) - k * (k + 1) ** 2 / 4.0)
    # tie correction
    T = 0.0
    for r in R:
        _, c = np.unique(r, return_counts=True)
        T += np.sum(c ** 3 - c)
    chi2_t = chi2 / (1.0 - T / (N * (k ** 3 - k))) if T > 0 else chi2
    FF = (N - 1) * chi2 / (N * (k - 1) - chi2)
    pF = stats.f.sf(FF, k - 1, (k - 1) * (N - 1))
    return dict(ranks=Rj, chi2=chi2, chi2_t=chi2_t, p_chi2=stats.chi2.sf(chi2, k - 1),
                p_chi2_t=stats.chi2.sf(chi2_t, k - 1), FF=FF, pFF=pF, N=N, k=k)


def holm(pvals, alpha=0.05):
    """Holm step-down: returns adjusted p-values and rejection flags (input order)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m_ = len(p)
    adj = np.empty(m_)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m_ - i) * p[j]))
        adj[j] = run
    return adj, adj <= alpha


def holm_control(ranks, names, control, N, alpha=0.05):
    """Friedman post-hoc with a control method (z statistics from mean ranks), Holm-corrected."""
    k = len(ranks)
    se = math.sqrt(k * (k + 1) / (6.0 * N))
    c = names.index(control)
    out = {}
    others = [i for i in range(k) if i != c]
    z = [(ranks[i] - ranks[c]) / se for i in others]
    p = [2 * stats.norm.sf(abs(v)) for v in z]
    adj, rej = holm(p, alpha)
    for t, i in enumerate(others):
        out[names[i]] = dict(z=z[t], p=p[t], p_holm=adj[t], reject=bool(rej[t]),
                             control_better=bool(ranks[c] < ranks[i]))
    return out


def mw(x, y):
    """Two-sided Mann-Whitney U test; returns (p, U_x) where U_x counts pairs with x > y (ties 1/2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
        return 1.0, len(x) * len(y) / 2.0
    r = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return float(r.pvalue), float(r.statistic)


def wtl_u(x, y, alpha=0.05):
    """Outcome of x against y (minimisation): 'w', 't' or 'l'.  A significant difference is assigned to the
    algorithm whose values tend to be smaller, i.e. x wins if U_x < n_x n_y / 2 (U-statistic rule)."""
    p, U = mw(x, y)
    if p >= alpha:
        return "t"
    return "w" if U < len(x) * len(y) / 2.0 else "l"


def wtl_median(x, y, alpha=0.05):
    """Rule of the submitted version: direction by the median, ties broken by the mean."""
    p, _ = mw(x, y)
    if p >= alpha:
        return "t"
    mx, my = np.median(x), np.median(y)
    if mx != my:
        return "w" if mx < my else "l"
    return "w" if np.mean(x) < np.mean(y) else "l"


def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon signed-rank test on paired per-problem values (zero differences discarded)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)


def sign_test(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w, l = int(np.sum(a < b)), int(np.sum(a > b))
    if w + l == 0:
        return 1.0, w, l
    return float(stats.binomtest(w, w + l, 0.5).pvalue), w, l


def welch_vs_summary(x, m_pub, s_pub, n_pub=50):
    """Welch t test of our sample against a published mean and standard deviation (n_pub runs)."""
    x = np.asarray(x, float)
    n = len(x)
    mx, sx = x.mean(), x.std(ddof=1)
    if sx == 0 and s_pub == 0:
        return 1.0 if mx == m_pub else 0.0
    se2 = sx ** 2 / n + s_pub ** 2 / n_pub
    t = (mx - m_pub) / math.sqrt(se2)
    dof = se2 ** 2 / ((sx ** 2 / n) ** 2 / (n - 1) + (s_pub ** 2 / n_pub) ** 2 / (n_pub - 1)) if se2 > 0 else 1.0
    return float(2 * stats.t.sf(abs(t), dof))
