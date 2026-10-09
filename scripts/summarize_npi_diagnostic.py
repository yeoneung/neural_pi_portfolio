"""Summarize and cross-check the iteration audit without choosing checkpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from diagnose_npi_iteration import load_network, probe, rms
from npi.neural import sample
from npi.simulation import paired_ce


def read_csv(path):
    return list(csv.DictReader(path.open(encoding='utf-8')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    out = Path(a.run)
    completion = json.loads((out/'completion.json').read_text())
    assert completion['status'] == 'complete'
    hashes = json.loads((out/'input_sha256.json').read_text())
    changed = [f for f, sha in hashes.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest() != sha]
    assert not changed, changed
    original = read_csv(out/'historical_all_iterations.csv')
    hist = [r for r in original if r['run']=='main' and r['method']=='pi']
    historical_summary = []
    for outer in [1, 2, 5, 10, 20, 50]:
        rr = [r for r in hist if float(r['outer']) == outer]
        historical_summary.append(dict(outer=outer,
            median_evaluation_rms=float(np.median([float(r['validation_eval_rms_scaled']) for r in rr])),
            median_gap_to_eval=float(np.median([float(r['validation_update_gap_rms_scaled'])/float(r['validation_eval_rms_scaled']) for r in rr]))))
    checkpoint_probes = read_csv(out/'historical_checkpoint_probes.csv')
    replay_probes = read_csv(out/'replay_iteration_probes.csv')
    curvature = []
    for label, data in [('historical_pi', [r for r in checkpoint_probes if r['method']=='pi']),
                        ('historical_pinn', [r for r in checkpoint_probes if r['method']=='pinn']),
                        ('replay_pi', replay_probes)]:
        for region in ['full', 'core']:
            rr = [r for r in data if r['region']==region]
            curvature.append(dict(dataset=label, region=region, checkpoints=len(rr),
                nonpositive=sum(int(r['curvature_nonpositive']) for r in rr),
                below_point1=sum(int(r['curvature_below_point1']) for r in rr),
                min_scaled_curvature=min(float(r['curvature_min']) for r in rr)))
    iterations = []
    for outer in [1, 2, 5, 10, 20, 50]:
        for region in ['full', 'core']:
            rr = [r for r in replay_probes if r['region']==region and int(r['outer'])==outer]
            iterations.append(dict(outer=outer, region=region,
                mean_q_change_rms=float(np.mean([float(r['q_change_rms']) for r in rr])),
                mean_low_q_change_rms=float(np.mean([float(r['low_old_q_change_rms']) for r in rr])),
                mean_other_q_change_rms=float(np.mean([float(r['other_q_change_rms']) for r in rr])),
                mean_eval_rms=float(np.mean([float(r['eval_rms']) for r in rr])),
                mean_gap_to_eval=float(np.mean([float(r['gap_to_eval']) for r in rr]))))

    # Reconstruct the original trainer's validation stream independently.
    # This tests the Hamiltonian gap calculation against its AD residuals.
    torch.set_num_threads(4)
    maximum_rms_difference = 0.
    for seed in [101, 102, 103]:
        folder = out/f'pi_seed{seed}'
        hr = read_csv(folder/'history.csv')
        for outer in [1, 2, 10, 20, 50]:
            net = load_network(folder/f'step_{outer*200}.pt')
            x = sample(net.market, 2048, torch.Generator().manual_seed(seed+29000), 'cpu', torch.float32).numpy()
            # Duplicate the probes to satisfy the full/core reporting interface.
            xp = np.column_stack([np.concatenate([x, x]), np.r_[np.zeros(len(x)), np.ones(len(x))]])
            old = load_network(folder/'checkpoint.pt', initial=True) if outer==1 else load_network(folder/f'step_{(outer-1)*200}.pt')
            oldd, _ = probe(old, xp)
            _, rr = probe(net, xp, oldd)
            r = hr[outer-1]
            for key, stored in [('eval_rms', 'validation_eval_rms_scaled'), ('gap_rms', 'validation_update_gap_rms_scaled')]:
                maximum_rms_difference = max(maximum_rms_difference, abs(rr[0][key]-float(r[stored])))
    assert maximum_rms_difference < 2e-6, maximum_rms_difference

    mc = read_csv(out/'mc_vs_myopic.csv')
    contrasts = read_csv(out/'mc_consecutive_updates.csv')
    n = load_network(out/'pi_seed101/checkpoint.pt').market
    mc_difference = 0.
    for r in mc:
        seed, outer, steps = int(r['seed']), int(r['outer']), int(r['time_steps'])
        payoff = np.load(out/f'paths_seed{seed}_outer{outer}_{steps}.npz')['payoff']
        base = np.load(out/f'myopic_paths_{steps}.npz')['payoff']
        ce = paired_ce(n, payoff, base)
        mc_difference = max(mc_difference, abs(-1e4*ce['relative_CE_shortfall_vs_reference']-float(r['ce_gain_vs_myopic_bp'])))
    assert mc_difference < 1e-9
    montecarlo = []
    for steps in [240, 480]:
        for outer in [1, 2, 5, 10, 20, 50]:
            rr = [r for r in mc if int(r['time_steps'])==steps and int(r['outer'])==outer]
            gains = [float(r['ce_gain_vs_myopic_bp']) for r in rr]
            montecarlo.append(dict(time_steps=steps, outer=outer, mean_gain_bp=float(np.mean(gains)),
                min_gain_bp=min(gains), max_gain_bp=max(gains),
                pointwise_positive_count=sum(float(r['ci_low_bp'])>0 for r in rr)))
    report = dict(historical_main_pi_iterations=len(hist),
        historical_main_pi_max_curvature_violation_fraction=max(float(r['curvature_violation_fraction']) for r in hist),
        history=historical_summary, curvature=curvature, iteration_probes=iterations, mc_vs_myopic=montecarlo,
        mc_contrasts_480=[{k:float(v) for k,v in r.items()} for r in contrasts if int(r['time_steps'])==480],
        max_core_policy_interpolation_rms=max(float(r['interpolation_rms_core']) for r in mc),
        late_contrast_scope='Pointwise paired MC intervals; 3 old training seeds replayed; no continuum guarantee or new seed replication',
        verification=dict(input_files_unchanged=len(hashes), validation_rms_max_abs_difference=maximum_rms_difference,
                          raw_mc_ce_max_abs_difference=mc_difference,
                          replay=json.loads((out/'replay_comparison.json').read_text())))
    (out/'diagnostic_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__=='__main__':
    main()
