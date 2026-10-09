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

All numerical cases are retained. Monte Carlo intervals condition on trained policies and exclude discretization bias. The compared evaluation representations also differ in their approximation spaces and optimization paths. Mesh refinements are sensitivity checks rather than certified continuum error bounds.
