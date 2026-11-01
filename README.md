# Revisiting the Giza Pyramids Construction metaheuristic — code and data

This repository contains the code and the raw results of the paper

> Úbeda-Flores, M. (2026). *Revisiting the Giza Pyramids Construction metaheuristic: errors, origin bias and an
> independent re-evaluation.* Submitted to *Evolutionary Intelligence*.

Repository: [DOI](https://doi.org/10.5281/zenodo.23857080)

which re-examines

> Harifi S, Mohammadzadeh J, Khalilian M, Ebrahimnejad S (2021). Giza Pyramids Construction: an
> ancient-inspired metaheuristic algorithm for optimization. *Evolutionary Intelligence* 14:1743–1761.
> https://doi.org/10.1007/s12065-020-00451-3

Every number, table and figure of the paper can be regenerated from this repository.

## Contents

```
code/       Python implementation of all algorithms, test functions, experiments and analyses
code/matlab_check/  comparison with the authors' MATLAB code of GPC (needs GNU Octave; GPC.m is not redistributed)
results/    raw results of every run (JSON Lines, gzip) and summaries (JSON); flat CSV copies in results/csv
data/       scripts that download the images used in the segmentation experiments (not redistributed)
run_all.sh  reruns everything (about 2.5 h on two CPU cores)
```

Tables (LaTeX) and figures (PDF) are written to `paper/tables` and `paper/figs`.

## Installation

Python 3.11 or later. The paper used Python 3.13 and the versions pinned in `requirements.txt`:

```
pip install -r requirements.txt
```

## Reproducing the paper

Regenerate all tables and figures from the stored results (a few minutes; the segmentation figure needs the
BSDS500 images, see below):

```
cd code
python3 analysis_bench.py && python3 tables_bench.py && python3 tables_audit.py
python3 analysis_bias.py && python3 analysis_hd.py && python3 analysis_bbob.py && python3 analysis_seg.py
python3 fig_theory.py
```

Rerun all experiments from scratch with `./run_all.sh`. Every run has its own seeded random-number
generator, so the results are identical with the pinned software versions.

### Data needed for the segmentation experiments

* `data/get_bsds500.sh` downloads the 20 BSDS500 test images and their human segmentations used in E6
  (Arbeláez et al., IEEE TPAMI 33:898–916, 2011). Alternatively set `BSDS500_DIR` to an existing copy of
  `BSDS500/data`.
* `data/get_standard_images.sh` downloads the colour Baboon and Peppers images used in Sect. 3.4
  (`audit_seg.py`). Alternatively set `STD_IMAGES_DIR`.

## Where each result comes from

| Paper | Script(s) | Output |
|---|---|---|
| Sect. 3, Tables 1–3 (audit of the published tables) | `audit.py`, `tables_audit.py` | `results/audit.json`, `paper/tables/tab_stats.tex`, `tab_origin.tex` |
| Sect. 3.4 (order of magnitude of the segmentation values) | `audit_seg.py` | `results/audit_seg*.json` |
| Sect. 4, Fig. 1 (analysis of the GPC update) | `theory_checks.py`, `fig_theory.py` | `results/theory_checks.json`, `paper/figs/fig_theory.pdf` |
| E1, E2a, E2b: Tables 5–6, Figs. 2–3, Appendix A | `exp_bench.py`, `analysis_bench.py`, `tables_bench.py` | `results/E1.jsonl.gz`, `E2a`, `E2b`, `bench_summary.json` |
| E3: Table 7, Fig. 4 (structural bias) | `exp_bias.py`, `analysis_bias.py` | `results/E3_bias.jsonl.gz`, `E3_summary.json` |
| E4, E4b: Table 8 (high dimension, dummy variables) | `exp_hd.py`, `exp_dummy.py`, `analysis_hd.py` | `results/E4_hd.jsonl.gz`, `E4b_dummy.jsonl.gz`, `E4_summary.json` |
| E5: Table 9, Fig. 5 (COCO/BBOB) | `bbob_fopt.py`, `exp_bbob.py`, `analysis_bbob.py` | `results/E5_bbob_*.jsonl.gz`, `E5_summary.json` |
| E6: Table 10, Fig. 6 (segmentation) | `exp_seg.py`, `analysis_seg.py` | `results/E6_seg.jsonl.gz`, `E6_summary.json` |
| Sects. 5.1 and 6.2 (check against the authors' MATLAB code) | `matlab_check/check_equivalence.py`, `matlab_check/check_native.py` | `results/matlab_equivalence.json`, `matlab_native.json` |

## Code overview

| File | Content |
|---|---|
| `bench.py` | The thirty test functions of Harifi et al. (2021), vectorised (standard definitions of the Virtual Library of Simulation Experiments) |
| `algos.py` | GPC (multiplicative and additive updates), GA, ICA, PSO, DE, FA, CMA-ES (via pycma) and random search, with exact counting of function evaluations |
| `common.py` | Corrected optimal values; original, translated and shifted problems |
| `paperdata.py` | Numbers transcribed from Tables 3–9 of Harifi et al. (2021) |
| `io_utils.py` | Reading and writing of the result files |
| `segobj_nb.py` | Compiled segmentation objective (Eq. 10 of Harifi et al.) |
| `style.py` | Figure style |

GPC follows the MATLAB implementation made public by its authors (www.harifi.com): uniform ε on
[−(u−l)/2, (u−l)/2], clipping to the box, substitution with one forced component, and (μ+λ) truncation.
`code/matlab_check` verifies that, given the same random numbers, our implementation and the authors' `GPC.m`
(run in GNU Octave) produce the same results. To run this check, you need GNU Octave and a copy of `GPC.m` from
the authors' site (see `code/matlab_check/README.md`).

## Result files

Each `results/*.jsonl.gz` file has one JSON object per run. Non-finite numbers are stored as `"inf"`, `"-inf"`
or `"nan"`. Read them with `io_utils.load_records` or any JSON Lines reader.

| File | Fields |
|---|---|
| `E1`, `E2a`, `E2b` | `fid`, `alg`, `run`, `best_f`, `err` (= best_f − f*), `nfe`, `cpu`, `curve` (best-so-far value at the evaluation counts `cps`), `d_origin`, `d_opt` (RMS distance of the best point to the origin and to the optimum), `o_norm` |
| `E3_bias` | `box`, `alg`, `run`, `xn` (final solution, normalised to [0, 1]) |
| `E4_hd` | `kind` (`orig`/`trans`), `fid`, `D`, `alg`, `run`, `best_f`, `err`, `rms_x`, `in_box`, `cpu` |
| `E4b_dummy` | `fid`, `run`, `best_f_2`, `best_f_big` (D = 2 and D = 5000 with the same random numbers), `identical` |
| `E5_bbob_*` | `alg`, `D`, `fun`, `inst`, `tn`, `tf` (evaluations and best-so-far precision f − f_opt), `nfe`, `xnorm` |
| `E6_seg` | `img`, `K`, `alg`, `run`, `obj` (Eq. 10), `cpu`, `cent` (centroids), `PRI`, `VoI` |

## Licence

Code: MIT (see `LICENSE`). Result files: CC BY 4.0. The BSDS500 images and the standard test images are not
part of this repository; they are downloaded from their original sources.
