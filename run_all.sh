#!/bin/sh
# Reproduces every result, table and figure of the paper.
# About 2.5 hours of wall-clock time on two CPU cores. The experiments overwrite the files in results/.
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/code"

python3 audit.py                              # Sect. 3: audit of the published tables
python3 theory_checks.py                      # Sect. 4: constants quoted in the analysis
python3 exp_bench.py E1 50                    # E1: original problems
python3 exp_bench.py E2a 50                   # E2a: translated problems
python3 exp_bench.py E2b 50                   # E2b: shifted optimum
python3 exp_bias.py                           # E3: structural bias
python3 exp_hd.py                             # E4: 5000 and 10 000 variables
python3 exp_dummy.py                          # E4b: dummy variables
python3 bbob_fopt.py                          # optimal values of the BBOB instances
python3 exp_bbob.py GPC,GPC-A,PSO,DE,CMA-ES   # E5: COCO/BBOB
python3 exp_bbob.py RS                        # E5: random search
python3 exp_seg.py                            # E6: segmentation (needs data/BSDS500, see data/get_bsds500.sh)
python3 audit_seg.py                          # Sect. 3.4 (needs data/standard_test_images)

python3 analysis_bench.py && python3 tables_bench.py && python3 tables_audit.py
python3 analysis_bias.py && python3 analysis_hd.py && python3 analysis_bbob.py && python3 analysis_seg.py
python3 fig_theory.py && python3 export_csv.py

# Comparison with the authors' MATLAB code (needs GNU Octave and code/matlab_check/GPC.m; see code/matlab_check/README.md)
if command -v octave-cli >/dev/null 2>&1 && [ -f matlab_check/GPC.m ]; then
  (cd matlab_check && python3 check_equivalence.py && python3 check_native.py)
else
  echo "Skipping the comparison with the authors' MATLAB code (needs octave-cli and code/matlab_check/GPC.m)"
fi
