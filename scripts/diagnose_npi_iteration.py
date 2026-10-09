"""Exploratory iteration audit, preserving the original trainer and all old runs.

Replays the first three prespecified study seeds and saves every outer iterate.
Independent probes and paired simulations are diagnostics, not certificates.
No speed comparison: checkpoint and diagnostic overhead is deliberately added.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import zipfile

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from npi.model import Model
from npi.neural import ValueNetwork, predict, train
from npi.simulation import simulate, paired_ce


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2)))


def load_network(path, device='cuda', initial=False):
    data = torch.load(path, map_location='cpu', weights_only=True)
    meta = data.get('metadata', data)
    m = Model.from_dict(meta['model'])
    s = meta['settings']
    net = ValueNetwork(m, s['width'], s['depth']).to(device)
    net.load_state_dict(data['initial_state_dict' if initial else 'state_dict'])
    return net


def probe(net, points, old=None):
    m = net.market
    d = {k: v.astype(float) for k, v in predict(net, points[:, :3]).items()}
    scale = m.utility_bundle(points[:, 1])[1]
    b = m.phi(points[:, 2]) * d['vy'] + m.rho * m.sigma_z * d['vyz']
    rows = []
    for region, mask in [('full', points[:, 3] == 0), ('core', points[:, 3] == 1)]:
        c = d['curvature'][mask] / scale[mask]
        row = dict(region=region, n=int(mask.sum()), hjb_rms=rms(d['residual'][mask] / scale[mask]),
                   curvature_nonpositive=int(np.sum(c <= 0)), curvature_below_point1=int(np.sum(c < .1)),
                   curvature_min=float(c.min()), curvature_p01=float(np.quantile(c, .01)),
                   curvature_median=float(np.median(c)), vy_nonpositive=int(np.sum(d['vy'][mask] <= 0)),
                   low_q_fraction=float(np.mean(np.abs(d['q'][mask]) < .05)),
                   clipped_fraction=float(np.mean(np.abs(d['q'][mask]) >= m.qmax - 1e-6)))
        if old is not None:
            dq = d['q'] - old['q']
            gap = (b * dq - .5 * d['curvature'] * dq * (d['q'] + old['q'])) / scale
            er = d['residual'] / scale - gap
            low = mask & (np.abs(old['q']) < .05)
            high = mask & ~low
            row.update(eval_rms=rms(er[mask]), gap_rms=rms(gap[mask]),
                       gap_to_eval=rms(gap[mask]) / max(rms(er[mask]), 1e-30),
                       gap_min=float(gap[mask].min()), q_change_rms=rms(dq[mask]),
                       q_change_max=float(np.max(np.abs(dq[mask]))),
                       low_old_q_n=int(low.sum()), low_old_q_change_rms=rms(dq[low]),
                       other_q_change_rms=rms(dq[high]),
                       low_old_q_eval_rms=rms(er[low]), other_eval_rms=rms(er[high]),
                       curvature_change_rms=rms((d['curvature'][mask] - old['curvature'][mask]) / scale[mask]),
                       cross_derivative_change_rms=rms((d['vyz'][mask] - old['vyz'][mask]) / scale[mask]))
        rows.append(row)
    return d, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'results/confirmatory_20260925'
    manifest = json.loads((ROOT / 'configs/confirmatory_study.json').read_text())
    plan = dict(kind='exploratory diagnosis, no algorithm or model change',
                original=str(original), replay_seeds=[101, 102, 103],
                probe_seed=620260926, full_points=8192, core_points=4096,
                core_bounds=[[0, -.7, -.3], [3, .7, .3]],
                low_exposure_threshold=.05, normalized_curvature_threshold=.1,
                mc_outers=[0, 1, 2, 5, 10, 20, 50], mc_paths=16384,
                mc_steps=[240, 480], mc_fine_steps=480, mc_seed=9262026,
                mc_initial_state=[0., 0.], policy_grid=[41, 81, 81],
                intervals='pointwise paired Monte Carlo; no selection or simultaneous guarantee',
                scope='One original stopped model; existing 10-seed findings already seen. '
                      'Three replays are not additional independent training seeds. '
                      'Fixed-budget replay with extra saving, not wall-time evidence.')
    write_json(out / 'protocol.json', plan)
    write_json(out / 'original_config.json', manifest)
    sources = sorted([f for sub in ['src', 'scripts'] for f in (ROOT/sub).rglob('*.py')])
    input_files = sorted(original.glob('*/history.csv')) + sorted(original.glob('*/step_*.pt'))
    input_files += sorted(original.glob('*/checkpoint.pt'))
    write_json(out / 'input_sha256.json', {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                          for f in sources + input_files})
    with zipfile.ZipFile(out / 'sources_before_run.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sources:
            z.write(f, str(f.relative_to(ROOT)))
        z.write(out / 'protocol.json', 'protocol.json')
        z.write(out / 'original_config.json', 'original_config.json')

    torch.set_num_threads(4)
    history = []
    for f in sorted(original.glob('*/history.csv')):
        run, method, seed = f.parent.name.rsplit('_', 2)
        for r in csv.DictReader(f.open()):
            history.append(dict(run=run, method=method, seed=int(seed[4:]), **{k: float(v) for k, v in r.items()}))
    write_csv(out / 'historical_all_iterations.csv', history)
    m = Model.from_dict(manifest['model'])
    rng = np.random.default_rng(plan['probe_seed'])
    full = rng.uniform([0, m.ymin, m.zmin], [m.horizon, m.ymax, m.zmax], (8192, 3))
    core = rng.uniform(*plan['core_bounds'], (4096, 3))
    points = np.column_stack([np.concatenate([full, core]), np.r_[np.zeros(8192), np.ones(4096)]])
    np.save(out / 'independent_probe_points.npy', points)
    oldrows = []
    for method in ['pi', 'pinn']:
        for seed in range(101, 111):
            for step in [2000, 4000, 10000]:
                net = load_network(original/f'main_{method}_seed{seed}'/f'step_{step}.pt')
                _, rs = probe(net, points)
                oldrows.extend(dict(method=method, seed=seed, step=step, **r) for r in rs)
        print('Historical probes done:', method, flush=True)
    write_csv(out / 'historical_checkpoint_probes.csv', oldrows)

    replay_meta = []
    for seed in plan['replay_seeds']:
        settings = {**manifest['training'], 'save_steps': list(range(200, 10001, 200))}
        _, meta = train(m, settings, 'pi', seed, out/f'pi_seed{seed}')
        replay_meta.append(meta)
    replay_checks = []
    rows = []
    for seed in plan['replay_seeds']:
        folder = out/f'pi_seed{seed}'
        net = load_network(folder/'checkpoint.pt', initial=True)
        old, _ = probe(net, points)
        for outer in range(1, 51):
            path = folder/f'step_{200*outer}.pt'
            net = load_network(path)
            d, rs = probe(net, points, old)
            rows.extend(dict(seed=seed, outer=outer, steps=outer*200, **r) for r in rs)
            np.savez_compressed(folder/f'probe_{outer:02d}.npz', **d)
            old = d
            if outer in [10, 20, 50]:
                before = torch.load(original/f'main_pi_seed{seed}'/path.name, weights_only=True, map_location='cpu')['state_dict']
                after = torch.load(path, weights_only=True, map_location='cpu')['state_dict']
                replay_checks.append(dict(seed=seed, step=outer*200,
                                          max_parameter_difference=max(float((before[k]-after[k]).abs().max()) for k in before)))
        print('Replay probes done:', seed, flush=True)
    write_csv(out / 'replay_iteration_probes.csv', rows)
    write_json(out / 'replay_comparison.json', replay_checks)

    axes = [np.linspace(0, m.horizon, 41), np.linspace(m.ymin, m.ymax, 81), np.linspace(m.zmin, m.zmax, 81)]
    gp = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing='ij')], axis=1)
    mcrows = []
    payoffs = {}
    bases = {}
    for nsteps in plan['mc_steps']:
        kw = dict(npaths=plan['mc_paths'], nsteps=nsteps, fine_steps=480, seed=plan['mc_seed'])
        bases[nsteps] = simulate(m, None, myopic=True, **kw)['payoff']
        np.savez_compressed(out/f'myopic_paths_{nsteps}.npz', payoff=bases[nsteps])
    # Verify the saved initial network implements the analytic myopic controller.
    initial = load_network(out/'pi_seed101/checkpoint.pt', initial=True)
    qi = predict(initial, points[:, :3])['q']
    qa = np.clip(m.phi(points[:, 2])/m.relative_risk_aversion(points[:, 1]), m.qmin, m.qmax)
    write_json(out/'initial_policy_check.json', {'max_myopic_difference': float(np.max(np.abs(qi-qa)))})
    for seed in plan['replay_seeds']:
        for outer in plan['mc_outers'][1:]:
            net = load_network(out/f'pi_seed{seed}'/f'step_{outer*200}.pt')
            qgrid = predict(net, gp)['q'].reshape(41, 81, 81)
            policy = RegularGridInterpolator(axes, qgrid)
            # Grid error is recorded separately from MC uncertainty.
            qtrue = predict(net, core)['q']
            interpolation_rms = rms(policy(core)-qtrue)
            for nsteps in plan['mc_steps']:
                result = simulate(m, policy, npaths=plan['mc_paths'], nsteps=nsteps,
                                  fine_steps=480, seed=plan['mc_seed'])
                payoffs[seed, outer, nsteps] = result['payoff']
                np.savez_compressed(out/f'paths_seed{seed}_outer{outer}_{nsteps}.npz', **result)
                ce = paired_ce(m, result['payoff'], bases[nsteps])
                row = dict(seed=seed, outer=outer, steps=outer*200, time_steps=nsteps,
                           ce_gain_vs_myopic_bp=-1e4*ce['relative_CE_shortfall_vs_reference'],
                           ci_low_bp=-1e4*ce['paired_mc_95_high'], ci_high_bp=-1e4*ce['paired_mc_95_low'],
                           interpolation_rms_core=interpolation_rms, exit_fraction=result['exit_fraction'])
                mcrows.append(row)
            print('MC done:', seed, outer, flush=True)
            write_csv(out/'mc_vs_myopic.csv', mcrows)
    contrasts = []
    # All predeclared consecutive checkpoints; late increment 20->50 included.
    for seed in plan['replay_seeds']:
        for nsteps in plan['mc_steps']:
            for earlier, later in zip(plan['mc_outers'][:-1], plan['mc_outers'][1:]):
                before = bases[nsteps] if earlier == 0 else payoffs[seed, earlier, nsteps]
                ce = paired_ce(m, payoffs[seed, later, nsteps], before)
                contrasts.append(dict(seed=seed, time_steps=nsteps, earlier=earlier, later=later,
                                      ce_gain_bp=-1e4*ce['relative_CE_shortfall_vs_reference'],
                                      ci_low_bp=-1e4*ce['paired_mc_95_high'], ci_high_bp=-1e4*ce['paired_mc_95_low']))
    write_csv(out/'mc_consecutive_updates.csv', contrasts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    for seed in range(101, 111):
        rr = [r for r in history if r['run']=='main' and r['method']=='pi' and r['seed']==seed]
        x = [r['steps'] for r in rr]
        axs[0].semilogy(x, [r['validation_eval_rms_scaled'] for r in rr], color='tab:blue', alpha=.3)
        axs[0].semilogy(x, [r['validation_update_gap_rms_scaled'] for r in rr], color='tab:orange', alpha=.3)
    axs[0].set(title='Ten original NPI runs', xlabel='Optimizer updates', ylabel='Held-out scaled RMS')
    axs[0].plot([], [], color='tab:blue', label='Evaluation residual')
    axs[0].plot([], [], color='tab:orange', label='Greedy improvement gap')
    axs[0].legend()
    for seed in plan['replay_seeds']:
        rr = [r for r in rows if r['seed']==seed and r['region']=='core']
        axs[1].semilogy([r['steps'] for r in rr], [r['q_change_rms'] for r in rr], label=f'Seed {seed}')
        mr = [r for r in mcrows if r['seed']==seed and r['time_steps']==480]
        axs[2].plot([0]+[r['steps'] for r in mr], [0]+[r['ce_gain_vs_myopic_bp'] for r in mr], 'o-', label=f'Seed {seed}')
    axs[1].set(title='Independent core probes', xlabel='Optimizer updates', ylabel='RMS policy change per outer iteration')
    axs[2].set(title='Paired MC: 16,384 paths, 480 steps', xlabel='Optimizer updates', ylabel='CE gain over myopic (bp)')
    axs[2].legend()
    fig.savefig(out/'iteration_diagnostic.png', dpi=180)
    fig.savefig(out/'iteration_diagnostic.pdf')
    write_json(out/'completion.json', dict(status='complete', replay_runs=len(replay_meta),
                                         historical_probe_rows=len(oldrows), replay_probe_rows=len(rows),
                                         mc_rows=len(mcrows), comparison_rows=len(contrasts)))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
