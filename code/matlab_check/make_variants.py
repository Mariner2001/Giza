"""Create variants of the authors' GPC.m that differ only in the problem definition and the parameters.

The authors' script is not redistributed: place it as GPC.m in this folder (see README.md).
A variant changes only the lines that define the cost function, the dimension, the bounds, the number of
iterations and the ramp angle, and, for the additive update (GPC-A), moves the comment sign from the
additive to the multiplicative update line.  make_variant() checks that every other line of the file,
in particular the whole algorithm section, is left byte-identical.
"""
import hashlib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SHA256_USED = "e4deb9e385c0dba7d576dbc7b443fcb3fed5efdb2ecb1a051eb74eeaeeaec593"  # GPC.m used in the paper

# line patterns (start of the code part) -> name of the parameter they define
TARGETS = {
    "cost": r"^CostFunction=@\(x\) Sphere\(x\);",
    "nvar": r"^nVar=30;",
    "lb": r"^VarMin=-5\.12;",
    "ub": r"^VarMax= 5\.12;",
    "iters": r"^MaxIteration=1000;",
    "theta": r"^Tetha = 14;",
    "mult": r"^        newsol\.Position = \(pop\(i\)\.Position\+d\)\.\*\(x\*epsilon\);",
    "add": r"^      % newsol\.Position = \(pop\(i\)\.Position\+d\)\+\(x\*epsilon\);",
}


def read_official(path=None):
    path = path or os.path.join(HERE, "GPC.m")
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    return raw.decode("ascii"), sha


def _lines(text):
    return text.split("\r\n") if "\r\n" in text else text.split("\n")


def _find(lines, key):
    idx = [k for k, s in enumerate(lines) if re.match(TARGETS[key], s)]
    if len(idx) != 1:
        raise RuntimeError("line for %r found %d times" % (key, len(idx)))
    return idx[0]


def _num(v):
    return "pi" if v == "pi" else repr(float(v))


def make_variant(text, fun="Sphere", D=30, lb=-5.12, ub=5.12, iters=1000, theta=14, mode="mult", shift=None):
    """Return the text of the variant script and the list of changed line numbers (1-based)."""
    eol = "\r\n" if "\r\n" in text else "\n"
    L = _lines(text)
    new = list(L)
    keep_comment = lambda s: s[s.index("%"):] if "%" in s else ""
    o = ""
    if shift is not None:
        o = "o=[" + " ".join("%.17g" % v for v in shift) + "];  "
    i = _find(L, "cost")
    new[i] = o + "CostFunction=@(x) %s(x%s);        " % (fun, "-o" if shift is not None else "") + keep_comment(L[i])
    i = _find(L, "nvar")
    new[i] = "nVar=%d;                  " % D + keep_comment(L[i])
    i = _find(L, "lb")
    new[i] = "VarMin=%s%s;             " % (_num(lb), "+o" if shift is not None else "") + keep_comment(L[i])
    i = _find(L, "ub")
    new[i] = "VarMax=%s%s;             " % (_num(ub), "+o" if shift is not None else "") + keep_comment(L[i])
    i = _find(L, "iters")
    new[i] = "MaxIteration=%d;   " % iters + keep_comment(L[i])
    i = _find(L, "theta")
    new[i] = "Tetha = %s;          " % _num(theta) + keep_comment(L[i])
    allowed = {_find(L, k) for k in ("cost", "nvar", "lb", "ub", "iters", "theta")}
    if mode == "add":
        im, ia = _find(L, "mult"), _find(L, "add")
        new[im] = "      % " + L[im].lstrip()          # comment out the multiplicative update
        new[ia] = L[ia].replace("      % ", "        ", 1)  # activate the additive update
        allowed |= {im, ia}
    elif mode != "mult":
        raise ValueError(mode)
    changed = [k for k in range(len(L)) if new[k] != L[k]]
    assert set(changed) <= allowed, "unexpected change"
    # the algorithm section (from '%% Initialization' to the end) must be untouched, except the update line
    start = next(k for k, s in enumerate(L) if s.startswith("%% Initialization"))
    algo_changed = [k for k in changed if k >= start]
    assert set(algo_changed) <= ({_find(L, "mult"), _find(L, "add")} if mode == "add" else set())
    return eol.join(new), [k + 1 for k in changed]


if __name__ == "__main__":
    text, sha = read_official()
    print("GPC.m sha256", sha, "(as used in the paper)" if sha == SHA256_USED else "(DIFFERENT FILE)")
    v, ch = make_variant(text, fun="Rastrigin", iters=499, theta=10, mode="add")
    print("changed lines:", ch)
    for k in ch:
        print("%4d: %s" % (k, _lines(v)[k - 1]))
