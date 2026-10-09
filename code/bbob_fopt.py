"""Optimal values f_opt of the BBOB problems used in E5 (instances 1-5, D = 2, 5, 10, 20).

cocoex does not expose f_opt directly, so it is read from the headers that the COCO logger writes
("best noise-free fitness - Fopt (...)").  Writes ../results/bbob_fopt.json, keyed "D|function|instance"."""
import glob
import json
import os
import re
import shutil
import tempfile
import numpy as np
import cocoex

if __name__ == "__main__":
    fopt = {}
    here = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.chdir(tmp)
    try:
        for D in (2, 5, 10, 20):
            suite = cocoex.Suite("bbob", "", "dimensions:%d instance_indices:1-5" % D)
            obs = cocoex.Observer("bbob", "result_folder: fopt_d%d" % D)
            order = []
            for p in suite:
                p.observe_with(obs)
                p(np.zeros(D))
                order.append((p.id_function, p.id_instance))
                p.free()
            for fn in range(1, 25):
                files = glob.glob("exdata/fopt_d%d*/data_f%d/bbobexp_f%d_DIM%d.dat" % (D, fn, fn, D))
                assert len(files) == 1, files
                vals = [float(v) for v in re.findall(r"Fopt \(([-+0-9.eE]+)\)", open(files[0]).read())]
                insts = [i for (f, i) in order if f == fn]
                assert len(vals) == len(insts)
                for i, v in zip(insts, vals):
                    fopt["%d|%d|%d" % (D, fn, i)] = v
    finally:
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(fopt, open("../results/bbob_fopt.json", "w"), indent=0)
    print("wrote %d optimal values" % len(fopt))
