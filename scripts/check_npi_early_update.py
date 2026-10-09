"""Fresh-path check of the first NPI refresh after the exploratory audit.

All three replay seeds are retained. This is a targeted follow-up, chosen after
seeing the exploratory results; it is not an independently selected hypothesis.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from diagnose_npi_iteration import load_network, write_csv, write_json
from npi.neural import predict
from npi.simulation import simulate, paired_ce


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    run = Path(a.run)
    out = run/'independent_early_update_check'
    out.mkdir(exist_ok=False)
    config = dict(seeds=[101, 102, 103], outers=[1, 2], paths=65536, steps=[480, 960],
                  fine_steps=960, simulation_seed=9262027, policy_grid=[81, 161, 161],
                  selection='Targeted follow-up after exploratory results: retain all three seeds, first 200->400 update only.',
                  scope='Fresh common paths, doubled time resolution, finer policy interpolation; time-discrete simulation, not continuum certificate.')
    write_json(out/'protocol.json', config)
    inputs = [run/f'pi_seed{s}'/f'step_{200*k}.pt' for s in config['seeds'] for k in config['outers']]
    inputs += [Path(__file__), ROOT/'src/npi/simulation.py', ROOT/'src/npi/neural.py', ROOT/'scripts/diagnose_npi_iteration.py']
    write_json(out/'input_sha256.json', {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs})
    torch.set_num_threads(4)
    rows = []
    for seed in config['seeds']:
        payoffs = {}
        for outer in config['outers']:
            net = load_network(run/f'pi_seed{seed}'/f'step_{outer*200}.pt')
            m = net.market
            axes = [np.linspace(0, m.horizon, 81), np.linspace(m.ymin, m.ymax, 161), np.linspace(m.zmin, m.zmax, 161)]
            points = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing='ij')], axis=1)
            q = predict(net, points)['q'].reshape(*config['policy_grid'])
            policy = RegularGridInterpolator(axes, q)
            for nsteps in config['steps']:
                result = simulate(m, policy, npaths=config['paths'], nsteps=nsteps,
                                  fine_steps=config['fine_steps'], seed=config['simulation_seed'])
                payoffs[outer, nsteps] = result['payoff']
                np.savez_compressed(out/f'paths_seed{seed}_outer{outer}_{nsteps}.npz', **result)
            print('Fresh paths done:', seed, outer, flush=True)
        for nsteps in config['steps']:
            ce = paired_ce(m, payoffs[2, nsteps], payoffs[1, nsteps])
            rows.append(dict(seed=seed, time_steps=nsteps, ce_gain_bp=-1e4*ce['relative_CE_shortfall_vs_reference'],
                             ci_low_bp=-1e4*ce['paired_mc_95_high'], ci_high_bp=-1e4*ce['paired_mc_95_low']))
        write_csv(out/'paired_changes.csv', rows)
    write_json(out/'completion.json', dict(status='complete', rows=rows))
    print(json.dumps(rows, indent=2), flush=True)


if __name__=='__main__':
    main()
