"""Creates variants of the authors' GPC.m that differ from it only in the problem definition (cost function,
number of variables, bounds), the number of iterations, the ramp angle and, for GPC-A, in which of the two update
lines (83/84) is commented out.  Every other line is checked to be identical to the original.

GPC.m is not redistributed (copyright of its authors); set GPC_M to its path (default ../data/GPC.m).
Expected SHA-256: e4deb9e385c0dba7d576dbc7b443fcb3fed5efdb2ecb1a051eb74eeaeeaec593."""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GPC_M = os.environ.get("GPC_M", os.path.join(HERE, "..", "data", "GPC.m"))
SHA256 = "e4deb9e385c0dba7d576dbc7b443fcb3fed5efdb2ecb1a051eb74eeaeeaec593"
MULT = "        newsol.Position = (pop(i).Position+d).*(x*epsilon);                 % Position of Stone and Worker"
ADD = "      % newsol.Position = (pop(i).Position+d)+(x*epsilon);                  % Note: In some cases or some problems use this instead of the previous line to get better results"


def vec(v):
    return "[" + " ".join("%.17g" % x for x in v) + "]"


def original_lines():
    raw = open(GPC_M, "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    if h != SHA256:
        raise SystemExit("GPC.m has SHA-256 %s, expected %s" % (h, SHA256))
    return raw.decode("latin-1").splitlines()


def make_variant(name, cost, nvar, varmin, varmax, maxit, theta, additive=False, outdir=None):
    """cost: MATLAB expression of the cost function of x, e.g. 'Sphere(x)' or 'Sphere(x-[...])';
    varmin/varmax: MATLAB expressions (scalars or row vectors)."""
    L = original_lines()
    edits = {}
    for k, line in enumerate(L):
        s = line.strip()
        if s.startswith("CostFunction=@(x)"):
            edits[k] = "CostFunction=@(x) %s;        %% Cost Function" % cost
        elif s.startswith("nVar="):
            edits[k] = "nVar=%d;                  %% Number of Decision Variables" % nvar
        elif s.startswith("VarMin="):
            edits[k] = "VarMin=%s;             %% Decision Variables Lower Bound" % varmin
        elif s.startswith("VarMax="):
            edits[k] = "VarMax=%s;             %% Decision Variables Upper Bound" % varmax
        elif s.startswith("MaxIteration="):
            edits[k] = "MaxIteration=%d;   %% Maximum Number of Iterations (Days of work)" % maxit
        elif s.startswith("Tetha ="):
            edits[k] = "Tetha = %s;          %% Angle of Ramp" % repr(float(theta)).rstrip("0").rstrip(".")
        elif additive and line == MULT:
            edits[k] = "      % " + MULT.lstrip()
        elif additive and line == ADD:
            edits[k] = "        " + ADD.split("% ", 1)[1]
    assert len(edits) == (8 if additive else 6), sorted(edits)
    out = [edits.get(k, l) for k, l in enumerate(L)]
    # every line outside the edited ones is unchanged
    assert all(out[k] == L[k] for k in range(len(L)) if k not in edits)
    outdir = outdir or os.path.join(HERE, "work")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name + ".m")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    return path
