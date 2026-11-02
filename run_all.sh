#!/bin/sh
# Runs every experiment, analysis, table and figure of the paper (a little over three hours on a two-core machine).
# Prerequisites: pip install -r requirements.txt; sh data/get_bsds500.sh; sh data/get_standard_images.sh
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
cd "$(dirname "$0")/code"
L=../results/logs; mkdir -p $L ../results/tables ../figs

# --- experiments of the submitted version
python3 bbob_fopt.py > $L/fopt.log 2>&1
python3 exp_bench.py E1 50 > $L/E1.log 2>&1
python3 exp_bench.py E2a 50 > $L/E2a.log 2>&1
python3 exp_bench.py E2b 50 > $L/E2b.log 2>&1
python3 exp_bias.py > $L/E3.log 2>&1
python3 exp_dummy.py > $L/E4b.log 2>&1
python3 exp_seg.py > $L/E6.log 2>&1
python3 exp_hd.py > $L/E4.log 2>&1

# --- experiments added in the revision
python3 exp_bench.py E1 50 L-SHADE,jSO shade > $L/E1_shade.log 2>&1
python3 exp_bench.py E2a 50 L-SHADE,jSO shade > $L/E2a_shade.log 2>&1
python3 exp_bench.py E2b 50 L-SHADE,jSO shade > $L/E2b_shade.log 2>&1
python3 exp_bias.py L-SHADE,jSO shade > $L/E3_shade.log 2>&1
for E in E2c E2d E2e_lo E2e_hi; do python3 exp_bench.py $E 50 > $L/$E.log 2>&1; done
for E in E1 E2a E2b; do python3 exp_bench.py $E 50 E7,GPC14 E7 > $L/${E}_E7.log 2>&1; done
python3 exp_seg2.py E6b > $L/E6b.log 2>&1
python3 exp_bbob.py GPC,GPC-A,PSO,DE,CMA-ES,RS,L-SHADE,jSO all > $L/E5_all.log 2>&1
python3 exp_seg2.py E6c > $L/E6c.log 2>&1
(cd ../validation && python3 check_shade.py 10 1,4,5,7,10,22 > ../results/logs/validation_shade.log 2>&1)

# --- analyses, tables and figures
python3 audit.py > $L/audit.log 2>&1
python3 audit_seg.py > $L/audit_seg.log 2>&1
python3 analysis_bias.py > $L/analysis_bias.log 2>&1
python3 analysis_hd.py > $L/analysis_hd.log 2>&1
python3 analysis_bbob.py > $L/analysis_bbob.log 2>&1
python3 analysis_seg.py > $L/analysis_seg.log 2>&1
python3 analysis_e7.py > $L/analysis_e7.log 2>&1
python3 tables_bench.py > $L/tables_bench.log 2>&1
python3 tables_other.py > $L/tables_other.log 2>&1
python3 figures.py > $L/figures.log 2>&1
python3 export_csv.py > $L/export_csv.log 2>&1
echo "done: results in ../results, figures in ../figs"
