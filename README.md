# Revisiting the Giza Pyramids Construction metaheuristic — code and data

This repository contains the code and the raw results of the paper

> Úbeda-Flores, M. (2026). *Revisiting the Giza Pyramids Construction metaheuristic: errors, origin bias and an
> independent re-evaluation.* Submitted to Evolutionary Intelligence (revised version).

which re-examines Harifi et al., *Giza Pyramids Construction: an ancient-inspired metaheuristic algorithm for
optimization*, Evolutionary Intelligence 14:1743–1761 (2021), doi:10.1007/s12065-020-00451-3.

Every number, table and figure of the paper is produced by the scripts in `code/` from the per-run records in
`results/`; nothing is typed in by hand.

## Archived versions

Every release is archived at Zenodo. Version 1.0.0, which accompanied the submitted version of the paper, is
https://doi.org/10.5281/zenodo.23857080. Version 2.0.0 (this version) accompanies the revised paper; its DOI is
listed among the versions of the same Zenodo record.

## Contents

| Folder | Contents |
|---|---|
| `code/` | Test functions, algorithms, experiments, analyses, tables and figures (Python) |
| `matlab_check/` | Comparison of our GPC with the authors' MATLAB script `GPC.m` in GNU Octave |
| `data/` | Download scripts for the BSDS500 images and the two standard test images (no images are included) |
| `results/` | Raw results of every run (`*.jsonl.gz`, and as CSV in `csv/`), numbers quoted in the paper (`numbers_*.json`), table bodies (`tables/*.tex`), logs |
| `validation/` | Cross-check of our L-SHADE and jSO implementations against the minionpy library |
| `figs/` | The figures of the paper (PDF) |

## Requirements

Python 3.13 with the packages in `requirements.txt` (NumPy 2.5, SciPy 1.18, Matplotlib 3.11, scikit-learn 1.9,
scikit-image 0.26, Pillow 12.3, Numba 0.68, cma 4.5, coco-experiment 2.8); GNU Octave 8.4 for `matlab_check/`;
git ≥ 2.25 for the download scripts. The experiments use two processes.

```sh
pip install -r requirements.txt
sh data/get_bsds500.sh            # 20 BSDS500 test images and their human segmentations (E6, E6c)
sh data/get_standard_images.sh    # Baboon and Peppers (Sect. 3.4, E6b)
sh run_all.sh                     # all experiments, analyses, tables and figures
```

`run_all.sh` takes a little over three hours of wall-clock time on a two-core machine; each step can also be run on its own
(see the script). The MATLAB comparison additionally needs the authors' file `GPC.m`, which is **not**
redistributed (copyright of its authors): download it from the address given in the original paper, put it in
`data/GPC.m` (expected SHA-256 `e4deb9e385c0dba7d576dbc7b443fcb3fed5efdb2ecb1a051eb74eeaeeaec593`) and run
`python3 matlab_check/check_equivalence.py` and `python3 matlab_check/check_native.py`.

## Where each result of the paper comes from

| Paper | Experiment script | Analysis / output |
|---|---|---|
| Tables 1–3 (errata, statistics, origin) | – | `code/audit.py` → `results/numbers_audit.json`, `tables/tab_origin.tex` |
| Table 4 (segmentation audit) | – | `code/audit_seg.py` → `numbers_audit_seg.json`, `tables/tab_segaudit.tex` |
| Fig. 1 (theory) | – | `code/figures.py theory` |
| Table 6 (reproduction), Appendices A and B | `exp_bench.py E1/E2a/E2b` | `code/tables_bench.py` → `tables/tab_repro.tex`, `tab_E1full.tex`, … |
| Table 7 (summary), Figs. 2–3 | `exp_bench.py E1/E2a/E2b` | `code/tables_bench.py`, `code/figures.py translation convergence` |
| Table 8 (rotation, units) | `exp_bench.py E2c/E2d/E2e_lo/E2e_hi` | `code/tables_bench.py` → `tables/tab_rot.tex` |
| Table 9, Fig. 4 (structural bias) | `exp_bias.py` | `code/analysis_bias.py`, `code/tables_other.py`, `code/figures.py bias` |
| Table 10 (high dimensions), E4b | `exp_hd.py`, `exp_dummy.py` | `code/analysis_hd.py` → `tables/tab_hd.tex` |
| Tables 11–12, Fig. 5 (BBOB) | `bbob_fopt.py`, `exp_bbob.py GPC,GPC-A,PSO,DE,CMA-ES,RS,L-SHADE,jSO all` | `code/analysis_bbob.py`, `code/tables_other.py`, `code/figures.py bbob` |
| Tables 13–14, Fig. 6 (segmentation) | `exp_seg.py`, `exp_seg2.py E6b/E6c` | `code/analysis_seg.py`, `code/tables_other.py`, `code/figures.py seg` |
| Table 15 (ablation, sensitivity, initialisation) | `exp_bench.py E1/E2a/E2b 50 E7,GPC14 E7` | `code/analysis_e7.py`, `code/tables_other.py` |
| Table 16 (computational cost) | (E1 records) | `code/tables_bench.py` → `tables/tab_cost.tex` |
| L-SHADE / jSO cross-check (Sect. 5.1) | – | `validation/check_shade.py` → `results/validation_shade_*.json` |
| MATLAB comparison (Sect. 5.1) | – | `matlab_check/check_equivalence.py`, `matlab_check/check_native.py` |

`code/fill_tables.py main.tex` inserts the table bodies into the LaTeX source of the paper, `code/report_numbers.py` prints the numbers quoted in the text, and `code/export_csv.py` writes one CSV file per experiment to `results/csv/`.

## Algorithms

`code/algos.py` (GA, ICA, PSO, DE, FA with the settings of Table 2 of the original study, GPC and GPC-A as
released by its authors, CMA-ES via pycma, random search), `code/algos_shade.py` (L-SHADE and jSO with the
settings of their original publications), `code/algos_gpcvar.py` (GPC with switchable components for E7; with
default arguments it is identical to `algos.gpc`). All share the interface
`run(f, lb, ub, nfe, rng, **options) -> dict(best_f, best_x, nfe, trace_n, trace_f)`, and function evaluations
are counted exactly.

## Random numbers

Every run has its own NumPy `Generator` (PCG64) seeded with a tuple. Benchmark experiments: problem instance
`default_rng([EXP_ID, fid, run])` (identical for all algorithms), algorithm `default_rng([EXP_ID, fid, ALG_ID, run, 7])`,
with `EXP_ID` = 1 (E1), 2 (E2a), 3 (E2b), 4 (E2c), 5 (E2d), 6 (E2e, 10^-2), 7 (E2e, 10^2) and the `ALG_ID`s listed
in `exp_bench.py`. The other experiments document their seeds in their scripts. CMA-ES results are reproducible
only up to the last bits of the eigendecompositions of the linear-algebra library, which can make long runs diverge.

## Format of the result files

One JSON object per run, gzip-compressed (JSON Lines); non-finite numbers are stored as the strings `"inf"`,
`"-inf"`, `"nan"`; `code/io_utils.load_records` reads them.

| File | Fields |
|---|---|
| `E1.jsonl.gz`, `E2a`, `E2b`, `E2c`, `E2d`, `E2e_lo`, `E2e_hi` (+ `_shade`, `_E7`) | `fid`, `alg`, `run`, `best_f`, `err` (= best_f − f*), `nfe`, `cpu` (s), `curve` (best-so-far at the evaluation counts `cps`), `d_origin`, `d_opt` (RMS distance of the best point to the origin / to the optimum), `o_norm` |
| `E3_bias.jsonl.gz` (+ `_shade`) | `box`, `alg`, `run`, `xn` (final best point, normalised to [0, 1]) |
| `E4_hd.jsonl.gz` | `kind` (orig/trans), `fid`, `D`, `alg`, `run`, `best_f`, `err`, `rms_x`, `in_box`, `cpu` |
| `E4b_dummy.jsonl.gz` | `fid`, `run`, `best_f_2`, `best_f_big`, `identical`, `nfe_2`, `nfe_big` |
| `E5_bbob_all.jsonl.gz` (assembled from the per-algorithm and per-dimension files that `exp_bbob.py` writes to `results/E5_parts/`, which allow an interrupted run to be resumed) | `alg`, `D`, `fun`, `inst`, `tn`, `tf` (evaluation counts and best precision f − f_opt where it improved), `nfe`, `xnorm` |
| `E6_seg.jsonl.gz`, `E6b_seg`, `E6c_seg` | `img`, `K`, `alg`, `run`, `obj` (Eq. 10), `cpu`, `cent` (centroids), `PRI`, `VoI` |

## Changes in version 2 (revision)

* L-SHADE and jSO added to E1–E3 and E5; new experiments E2c–E2e (rotation, change of units), E6b (Baboon and
  Peppers at 128 × 128), E6c (mean-centred colours) and E7 (ablation, parameter sensitivity, initialisation).
* Statistics: Wilcoxon signed-rank tests with Holm's correction, win/tie/loss counts with the direction given by
  the U statistic, per-coordinate Kolmogorov–Smirnov tests in the structural-bias test, segmentation covering.
* Audit: enumeration of single-function changes of the published ranking; segmentation audit for six image
  representations.
* `matlab_check/` rebuilt and moved to the top level; the summaries of version 1 (`E3_summary.json`, `E5_summary.json`, `audit_seg.json`) are replaced by `numbers_*.json`; metadata (LICENSE, CITATION.cff) completed.
* The results of version 1 are regenerated exactly by this version, except for the CMA-ES runs (see above).

## Licence

MIT (see `LICENSE`) for the code and the result files. The BSDS500 images, the standard test images and the
MATLAB code of GPC are not part of this repository and are subject to the terms of their owners.
