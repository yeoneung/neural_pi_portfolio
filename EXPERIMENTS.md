# Experiment commands and dependencies

Run commands from the repository root after installation. Output directory names match the paths expected by downstream scripts. Use a disposable checkout for a fresh replication; completed results are generally skipped or protected against overwriting. The expensive stages below are not executed by the unit tests.

## Equal-update neural comparison

```bash
python scripts/run_confirmatory_study.py --manifest configs/confirmatory_study.json --output results/confirmatory_20260925
python scripts/prepare_policy_grids.py --run results/confirmatory_20260925 --output results/confirmatory_policy_grids_20260925
python scripts/crosscheck_reference.py --output results/crosscheck_20260925
python scripts/simulate_policy_grids.py --grids results/confirmatory_policy_grids_20260925 --reference results/crosscheck_20260925/bounded_160.npz --output results/mc_main_240_20260925 --paths 65536 --steps 240 --fine-steps 480
python scripts/simulate_policy_grids.py --grids results/confirmatory_policy_grids_20260925 --reference results/crosscheck_20260925/bounded_160.npz --output results/mc_main_480_20260925 --paths 65536 --steps 480 --fine-steps 480
```

The locked configuration specifies seeds 101–110 for the principal direct-PINN and policy-iteration comparison, 10000 updates, and additional ablations. CUDA is required. `scripts/run_experiment.py --help`, `scripts/crosscheck_reference.py --help`, and `scripts/simulate_policy_grids.py --help` expose the reference-solution and paired-payoff stages; the latter requires a reference NPZ as well as policy grids.

## Exploratory fixed-policy and directional diagnostics

These stages generate the saved-input dependencies used by the quotient and damping controls:

```bash
python scripts/diagnose_npi_iteration.py --output results/npi_iteration_diagnostic_20260926
python scripts/study_fixed_policy_accuracy.py --output results/fixed_policy_accuracy_20260926
python scripts/check_npi_early_update.py --run results/npi_iteration_diagnostic_20260926
python scripts/audit_frozen_directional_error.py --mesh fine --output results/npi_directional_bridge_20260926/fine
python scripts/audit_frozen_directional_error.py --mesh finest --output results/npi_directional_bridge_20260926/finest
python scripts/followup_derivative_curves.py
```

The iteration diagnosis depends on the equal-update results above. The fixed-policy evaluation saves checkpoints every 50 updates. The derivative curves use independent frozen-policy values on the two indicated meshes. Additional `--mesh` choices are `coarse`, `medium`, `time`, `action`, and `alternate`; save each under the corresponding subdirectory to repeat the finite-generator sensitivity analysis.

## Evaluation length and boundary representations

Generate the six-seed warm starts and the single-network trajectories, then the matched correction trajectories:

```bash
python scripts/run_focused_strengthening.py --phase continuous
python scripts/run_focused_strengthening.py --phase simulation
python scripts/run_focused_strengthening.py --phase finite
python scripts/followup_continuous_boundary.py --phase frozen
python scripts/followup_continuous_boundary.py --phase iteration
python scripts/followup_continuous_boundary.py --phase simulation
```

`configs/focused_strengthening.json` specifies seeds 701–706, the baseline and parameter variants, and finite-grid full/relaxed correction comparisons. Its continuous and finite training stages use CUDA. `configs/continuous_boundary_followup.json` specifies CPU full/relaxed correction and repeated-iteration comparisons with shared random streams; it reads the baseline warm starts produced by the first command.

## Matched CPU controls and shared derivative references

After completing the exploratory and six-seed stages above:

```bash
python scripts/evaluation_controls.py all
python scripts/matched_derivative_bridge.py
python scripts/run_followup.py report --tables-dir reports/followup_tables
python scripts/run_followup.py controls
python scripts/run_followup.py matched-derivatives --report-dir reports/matched_derivatives
```

`evaluation_controls.py` requires the baseline warm starts, exploratory derivative arrays/checkpoints, and saved correction payoffs. It provides `train`, `matched`, `decomposition`, and `damping` phases as well as `all`. `matched_derivative_bridge.py` independently solves the same six frozen policies on two meshes and evaluates the saved single/full CPU checkpoints. Reports check stored input hashes. Historical saved inputs require their exact source snapshots; fresh runs create new hashes and snapshots.

## Equal-time comparison

```bash
python scripts/benchmark_inexact_npi.py --output results/npi_walltime_20260926
python scripts/audit_npi_walltime.py --study results/npi_walltime_20260926 --output results/npi_walltime_audit_20260926
```

The timed protocol specifies seeds 201–203, six methods, and 30-second budgets. Run training sequentially on a CUDA GPU with no simultaneous audit jobs. The second command performs independent post-training payoff simulation. Timing and target-crossing observations depend on the execution environment.

## Bundled summaries

```bash
python scripts/summarize_results.py
```

This reads `data/results/npi_evaluation_controls_20260929/matched_comparisons.csv`, checks the complete six-seed comparison groups, and reports median CE changes and adjusted interval counts. `data/results/` also contains all recorded training histories and CSV diagnostics for the stages above. Those records are observational outputs, not replacements for raw-array provenance checks.
