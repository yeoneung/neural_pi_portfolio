# Neural policy iteration for portfolio control

Numerical experiments for a stopped portfolio-control model with mixed-power utility, a stochastic factor, and control-dependent diffusion. The implementation includes neural policy evaluation, direct PINN training, fixed-base correction evaluators, independent finite-policy values, paired payoff simulation, and derivative-error diagnostics.

## Install and check

Use a fresh Python environment. Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/summarize_results.py
```

`environment.json` records the original numerical environment. `validation_environment.json` records a separate environment used to check this code export. GPU training and timed comparisons require CUDA; the unit tests and bundled CSV summaries run on CPU. Floating-point results and timings can vary across hardware and library versions.

## Files

| Location | Contents |
| --- | --- |
| `src/npi/` | Model, neural evaluators, generators, simulation, and error diagnostics |
| `configs/` | Model parameters, seeds, architecture, training budgets, and comparison protocols |
| `scripts/` | Training, policy simulation, independent evaluation, and numerical reporting |
| `tests/` | Numerical identities, boundary behavior, derivative extraction, and workflow checks |
| `data/results/` | Recorded CSV outcomes, including individual seeds and adverse outcomes |
| `data/protocols.json` | Recorded numerical protocols without historical machine paths |
| `data/SHA256_MANIFEST.json` | Integrity hashes of the bundled numerical summaries |
| `EXPERIMENTS.md` | Computation order, input dependencies, and commands |

`data/results/` is read-only recorded output. New computations write to `results/`; numerical report scripts write to `reports/`. Large checkpoints, policy grids, payoff arrays, and finite-reference arrays are not bundled in Git. The CSV summaries can be inspected immediately. Recomputing derivative references and payoff statistics requires either generating the binary inputs in the order below or supplying the corresponding saved experiment directories. Copying CSV files into `results/` does not replace those inputs.

All numerical cases are retained. The original seedwise Monte Carlo intervals condition on trained policies and exclude discretization bias. The 100-seed bundles document their separate sampling analysis. The compared evaluation representations also differ in their approximation spaces and optimization paths. Mesh refinements are sensitivity checks rather than certified continuum error bounds.

## 100-seed revision results

[`seedwise_100.csv`](seedwise_100.csv) contains all 100 paired training seeds used in the revised principal performance comparison. Each construction-specific `*_gain_bp` column is the certainty-equivalent change from 50 to 1000 frozen-policy evaluation steps, in basis points; `correction_minus_*_bp` columns subtract the corresponding control change from the correction change. These are evaluation-length changes, not final-policy rankings or annual returns.

The `cohort` column identifies the three run stages (12, 24 and 64 initializations). All stages and seeds are retained. The sample was expanded after the earlier analyses; stagewise path sharing and the exploratory integrated bootstrap are described in Supplement S15. This file retains the full precision of the archived integrated analysis.

## Versioned reproducibility bundles

The complete experiment code and cached numerical data for the 100-seed study are available in the [CNSNS reproducibility release](https://github.com/yeoneung/neural_pi_portfolio/releases/tag/cnsns-revision-20261116):

- [`repro_code.zip`](https://github.com/yeoneung/neural_pi_portfolio/releases/download/cnsns-revision-20261116/repro_code.zip): experiment scripts, original protocols, tabulated results, scientific figures and integrity manifests.
- [`repro_arrays.zip`](https://github.com/yeoneung/neural_pi_portfolio/releases/download/cnsns-revision-20261116/repro_arrays.zip): cached policy grids, path-payoff arrays, derivative-reference arrays and trained checkpoints.

Extract both archives at the same root for analyses requiring saved arrays. `CODE_SHA256_MANIFEST.json` and `ARRAY_SHA256_MANIFEST.json` verify the corresponding files. The compact package alone reproduces the final 64-seed Student t intervals and the stage summaries with `python experiments/reporting_checks.py`. The pooled 100-seed bootstrap is exploratory and preserves the documented simulation-stream dependence.
