# Check against the authors' MATLAB code

This folder compares our Python implementation of GPC (`../algos.py`) with the MATLAB script `GPC.m` released by
the authors of GPC (www.harifi.com). The script is **not** included because its licence does not allow
redistribution: copy it into this folder as `GPC.m`. The file used for the paper has SHA-256
`e4deb9e385c0dba7d576dbc7b443fcb3fed5efdb2ecb1a051eb74eeaeeaec593`. The scripts warn if a different file is found.

Requirements: GNU Octave (tested with 8.4.0, `octave-cli` on the PATH) and the Python packages of the main
repository. `GPC.m` calls `unifrnd`, which belongs to the MATLAB Statistics Toolbox (or to the Octave
*statistics* package). This folder supplies a version with the same algorithm as the toolbox.

## Scripts

| Script | What it does | Output |
|---|---|---|
| `check_equivalence.py` | Runs `GPC.m` and `algos.gpc` with the same random numbers: 8 settings × 3 random streams | `../../results/matlab_equivalence.json` |
| `check_native.py` | Runs `GPC.m` with Octave's own random numbers, on the original and the translated Sphere function | `../../results/matlab_native.json` |
| `make_variants.py` | Writes copies of `GPC.m` for other problems and parameters (used by both scripts) | `work/` |

The 8 settings of the equivalence test cover:

* the multiplicative and the additive update;
* θ = 10 and 14;
* Sphere, Rastrigin, Rosenbrock, Michalewicz and Beale;
* D = 2 and 30;
* symmetric and asymmetric boxes;
* a translated problem with vector bounds.

## How the comparison works

* **Only the problem and parameters change.** A variant changes only the lines that define the cost function,
  the dimension, the bounds, the number of iterations and the ramp angle. For GPC-A it also moves the comment
  sign from the additive update line to the multiplicative one. `make_variants.py` checks that every other
  line, in particular the whole algorithm section, is byte-identical to `GPC.m`. The file used as distributed
  (Sphere, D = 30, 1000 iterations, θ = 14) is run without any change.
* **Shared random numbers.** `shadow_tape/` contains replacements for `rand`, `randi` and `unifrnd`. They return
  numbers that Python writes to files beforehand, in the order in which `GPC.m` consumes them. `algos.gpc`
  receives the same numbers through a stand-in for the numpy generator. `figure`, `semilogy`, `xlabel` and
  `ylabel` are replaced by no-ops so that the script runs without a display.
* **Same floating-point operations.** The objective is evaluated in Python in two ways: with the same
  floating-point operations as the MATLAB files in `funcs/` (left-to-right sums and the C library's `cos`,
  `sin` and `pow`), and with the vectorised numpy functions of `../bench.py`.

## Results (as reported in the paper)

**Same random numbers, objective evaluated with the same operations**

* In all 24 runs the best value is identical after every iteration.
* The final populations are bit-identical in 23 runs. In the remaining run they differ in the last bit,
  because Octave computes the scalar `V0^2` with `pow()`, which differs from `V0*V0` in the last bit for
  about 0.1% of the values.

**Same random numbers, objective evaluated with numpy**

* The objective values differ in the last bit because numpy uses pairwise summation.
* In 3 of the 24 runs this changes the order of candidates with almost equal values.
* The final best values still agree to a relative difference of 3e-16.

**Own random numbers** (median of the final best value)

| Setting | `GPC.m` (Octave) | `algos.gpc` |
|---|---|---|
| Sphere, D = 30, 20 020 evaluations (the distributed demo; 30 runs) | 1.3e-25 | 1.4e-25 (Mann–Whitney p = 0.68) |
| The same, translated as in E2a (30 runs) | 52.3 | 53.6 (p = 0.63) |
| Sphere with the budget of E1 (1000 evaluations, θ = 10; 50 runs) | 5.9e-14 | 1.3e-13 in E1 (p = 0.34) |
| The same, translated (50 runs) | 52.7 | 52.6 in E2a (p = 0.89) |
