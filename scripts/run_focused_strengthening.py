"""New-seed replication and a matched boundary-representation intervention.

All cases are specified in configs/focused_strengthening.json before outcomes.
Phases have separate outputs, refuse incomplete overwrites, and may be resumed
only by skipping completed cases. No experiment uses reference-based selection.
"""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import sys
import time
import zipfile
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from npi.model import Model
from npi.neural import ValueNetwork, derivatives, greedy, residual, sample, predict
from npi.inexact import frozen_copy, evaluation_loss
from npi.focused_evaluation import EvaluationCorrection, ScalarFeedback, network_values, grid_points, full_evaluation_rms
from npi.boundary_neural import SparseAction, torch_sparse
from npi.cheap_bounds import FiniteSystem
from npi.policy_error_audit import action_rows, fixed_evaluation, compare_step
from npi.simulation import simulate, paired_ce
from npi.discrete_evaluation import certainty_equivalent


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def checkpoint(path, net, **metadata):
    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                    model=net.market.to_dict(), **metadata), path)


def load_net(path, config):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    net = ValueNetwork(Model.from_dict(saved['model']), **config['network']).cuda()
    net.load_state_dict(saved['state_dict'])
    return net


def learning_rate(config, absolute):
    s = config['learning_rate']
    return s['minimum']+.5*(s['initial']-s['minimum'])*(1+np.cos(np.pi*absolute/s['cosine_horizon']))


def initialize(out):
    config_path = ROOT/'configs/focused_strengthening.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out/'protocol.json', config)
        files = [p for directory in ['src', 'scripts', 'configs', 'tests']
                 for p in (ROOT/directory).rglob('*') if p.suffix in ['.py', '.json']]
        write_json(out/'input_sha256.json', {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        with zipfile.ZipFile(out/'sources_before_run.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, p.relative_to(ROOT))
    elif json.loads((out/'protocol.json').read_text()) != config:
        raise ValueError('The locked protocol differs from the current configuration')
    return config


def continuous_case(out, config, scenario, seed):
    dest = out/'continuous'/scenario['name']/f'seed{seed}'
    if (dest/'completion.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT/config['model_source']).read_text())
    model = Model.from_dict({**original['model'], **scenario['changes']})
    settings = config['continuous']
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net = ValueNetwork(model, **config['network']).cuda()
    rng = np.random.default_rng(settings['probe_seed'])
    points = np.concatenate([
        rng.uniform([0, model.ymin, model.zmin], [model.horizon, model.ymax, model.zmax], (settings['probe_full'], 3)),
        rng.uniform([0, -.7, -.3], [model.horizon, .7, .3], (settings['probe_core'], 3))])
    np.save(dest/'probe_points.npy', points)
    xp = torch.as_tensor(points, dtype=torch.float32, device='cuda')
    scale = model.utility_bundle(points[:, 1])[1]
    axes = [np.linspace(0, model.horizon, settings['policy_grid'][0]),
            np.linspace(model.ymin, model.ymax, settings['policy_grid'][1]),
            np.linspace(model.zmin, model.zmax, settings['policy_grid'][2])]
    grid = np.stack([x.ravel() for x in np.meshgrid(*axes, indexing='ij')], axis=1)
    history = []
    started = time.perf_counter()
    for phase, count, offset in [('warmup', config['warmup_steps'], settings['warmup_stream_offset']),
                                  ('evaluation', config['evaluation_steps'], settings['evaluation_stream_offset'])]:
        fixed = frozen_copy(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate']['initial'], fused=True)
        stream = torch.Generator().manual_seed(seed+offset)
        if phase == 'evaluation':
            old_probe = greedy(model, derivatives(fixed, xp)).detach()
        for step in range(1, count+1):
            absolute = step-1+(config['warmup_steps'] if phase == 'evaluation' else 0)
            optimizer.param_groups[0]['lr'] = learning_rate(config, absolute)
            if (step-1) % settings['resample_every'] == 0:
                x = sample(model, settings['batch_size'], stream, 'cuda', torch.float32)
                q = greedy(model, derivatives(fixed, x)).detach()
            optimizer.zero_grad(set_to_none=True)
            loss = evaluation_loss(net, x, q, settings['shape_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError((scenario['name'], seed, phase, step))
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            save_step = 0 if phase == 'warmup' and step == count else step
            save = (phase == 'warmup' and step == count) or (phase == 'evaluation' and step in config['checkpoints'])
            if save:
                checkpoint(dest/f'step_{save_step}.pt', net, seed=seed, step=save_step, scenario=scenario['name'])
                fields = predict(net, grid, chunk=8192)
                np.savez_compressed(dest/f'policy_{save_step}.npz', tau=axes[0], y=axes[1], z=axes[2],
                                    q=fields['q'].reshape(tuple(settings['policy_grid'])))
                if phase == 'evaluation':
                    d = derivatives(net, xp)
                    er = (residual(model, d, old_probe).detach().cpu().numpy()/scale)
                    c = (d['vy']-d['vyy']).detach().cpu().numpy()/scale
                    row = dict(scenario=scenario['name'], seed=seed, step=step,
                               eval_rms_full=float(np.sqrt(np.mean(er[:settings['probe_full']]**2))),
                               eval_rms_core=float(np.sqrt(np.mean(er[settings['probe_full']:]**2))),
                               curvature_min=float(c.min()), curvature_nonpositive=int(np.count_nonzero(c <= 0)),
                               seconds=time.perf_counter()-started)
                    history.append(row); write_csv(dest/'history.csv', history)
                    print('CONTINUOUS', json.dumps(row), flush=True)
    write_json(dest/'completion.json', dict(status='complete', seconds=time.perf_counter()-started))


def finite_audit(system, values, indices, uq):
    n = len(system.ids); idx = np.arange(n); dt = system.tau[1]
    space, _ = grid_points(system)
    scale = system.model.utility_bundle(space[:, 0])[1]
    core = (np.abs(space[:, 0]) <= .7+1e-12) & (np.abs(space[:, 1]) <= .3+1e-12)
    center = np.flatnonzero(np.max(np.abs(space), axis=1) < 1e-12).item()
    up = system.terminal.ravel()[system.ids].copy()
    advantage = np.zeros(n); correction = np.zeros(n)
    error_sums = np.zeros(2); value_sums = np.zeros(2); max_identity = max_solve = 0.
    min_advantage = np.inf
    proposal = np.zeros_like(indices)
    for k in range(1, len(system.tau)):
        h = system.hamiltonians(values[k]); pi = h.argmax(axis=0)
        proposal[k] = pi
        qi = indices[k]
        oq = system.ay[qi*n+idx]; op = system.ay[pi*n+idx]
        up, advantage, correction, fields = compare_step(
            oq, op, system.sources[qi, idx], system.sources[pi, idx], uq[k], values[k],
            up, advantage, correction, dt)
        err = fields['directional_error']/scale
        verr = (values[k]-uq[k])/scale
        error_sums += [np.mean(err**2), np.mean(err[core]**2)]
        value_sums += [np.mean(verr**2), np.mean(verr[core]**2)]
        max_identity = max(max_identity, float(np.max(np.abs(up-uq[k]-advantage+correction))))
        max_solve = max(max_solve, fields['linear_residual'])
        min_advantage = min(min_advantage, float(np.min(fields['candidate_advantage'])))
    nt = len(system.tau)-1
    ce = float(certainty_equivalent(system.model, up[center]))
    ceq = float(certainty_equivalent(system.model, uq[-1, center]))
    row = dict(proposal_value=float(up[center]), frozen_value=float(uq[-1, center]),
               proposal_ce=ce, frozen_ce=ceq, gain_vs_frozen_bp=1e4*(ce/ceq-1),
               candidate_integral=float(advantage[center]), error_integral=float(correction[center]),
               directional_rms_full=float(np.sqrt(error_sums[0]/nt)),
               directional_rms_core=float(np.sqrt(error_sums[1]/nt)),
               value_rms_full=float(np.sqrt(value_sums[0]/nt)), value_rms_core=float(np.sqrt(value_sums[1]/nt)),
               minimum_candidate_advantage=min_advantage, maximum_identity_error=max_identity,
               maximum_linear_residual=max_solve, **full_evaluation_rms(system, values, indices))
    arrays = dict(value=up, candidate_integral=advantage, error_integral=correction, policy_indices=proposal)
    return row, arrays


def finite_case(out, config, scenario, seed, grid_config, system):
    dest = out/'finite'/grid_config['name']/scenario['name']/f'seed{seed}'
    if (dest/'completion.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=False)
    source = out/'continuous'/scenario['name']/f'seed{seed}'
    base = load_net(source/'step_0.pt', config)
    m = system.model; n = len(system.ids); nt = len(system.tau)-1; idx = np.arange(n)
    space, points = grid_points(system)
    original = network_values(base, points).reshape(nt+1, n)
    original[0] = system.terminal.ravel()[system.ids]
    pg = np.load(source/'policy_0.npz')
    controller = ScalarFeedback([pg['tau'], pg['y'], pg['z']], pg['q'])
    q = controller(points).reshape(nt+1, n)
    indices = np.argmin(np.abs(q[:, :, None]-system.actions), axis=2)
    operators = [system.ay[indices[k]*n+idx].tocsr() for k in range(1, nt+1)]
    sources = [system.sources[indices[k], idx] for k in range(1, nt+1)]
    uq, solve_residual = fixed_evaluation(operators, sources, system.terminal.ravel()[system.ids], system.tau[1])
    np.savez_compressed(dest/'frozen.npz', value=uq, initial=original, indices=indices,
                        tau=system.tau, y=system.y, z=system.z, ids=system.ids)
    device='cuda'; dtype=torch.float32
    top = [torch_sparse(a, device, dtype) for a in operators]
    ttranspose = [torch_sparse(a.T, device, dtype) for a in operators]
    tsources = torch.as_tensor(np.asarray(sources), device=device, dtype=dtype)
    tbase = torch.as_tensor(original, device=device, dtype=dtype)
    tspace = torch.as_tensor(space, device=device, dtype=dtype)
    times = torch.as_tensor(system.tau, device=device, dtype=dtype)
    scale = torch.as_tensor(m.utility_bundle(space[:, 0])[1], device=device, dtype=dtype)
    stream = torch.Generator().manual_seed(seed+config['finite']['time_stream_offset'])
    sequence = torch.randint(1, nt+1, (config['evaluation_steps'],), generator=stream).tolist()
    np.save(dest/'sampled_time_indices.npy', sequence)
    modes = config['finite']['representations']
    if seed % 2 == 0:
        modes = list(reversed(modes))
    rows = []; started = time.perf_counter(); dt = float(system.tau[1])
    initial_check = []
    for mode in modes:
        folder = dest/mode; folder.mkdir()
        torch.manual_seed(seed+config['finite']['correction_seed_offset'])
        torch.cuda.manual_seed_all(seed+config['finite']['correction_seed_offset'])
        net = EvaluationCorrection(m, **config['network'], mode=mode).cuda()
        initial_values = network_values(net, points).reshape(nt+1, n)+original
        initial_check.append(float(np.max(np.abs(initial_values-original))))
        optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate']['initial'], fused=True)
        history = []
        for step, k in enumerate(sequence, 1):
            optimizer.param_groups[0]['lr'] = learning_rate(config, config['warmup_steps']+step-1)
            optimizer.zero_grad(set_to_none=True)
            x = torch.cat([times[k].expand(n, 1), tspace], dim=1)
            xp = torch.cat([times[k-1].expand(n, 1), tspace], dim=1)
            v = tbase[k]+net(x)[:, 0]
            vp = tbase[k-1]+net(xp)[:, 0]
            h = SparseAction.apply(v[:, None], top[k-1], ttranspose[k-1])[:, 0]+tsources[k-1]
            residual = ((v-vp)/dt-h)/scale
            loss = torch.mean(residual**2)
            if not torch.isfinite(loss):
                raise FloatingPointError((scenario['name'], seed, mode, step))
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            if step in config['checkpoints']:
                checkpoint(folder/f'step_{step}.pt', net, seed=seed, step=step, mode=mode)
                values = original+network_values(net, points).reshape(nt+1, n)
                row, arrays = finite_audit(system, values, indices, uq)
                row = dict(scenario=scenario['name'], seed=seed, mesh=grid_config['name'], mode=mode, step=step, **row)
                arrays['candidate_values'] = values
                np.savez_compressed(folder/f'audit_{step}.npz', **arrays)
                rows.append(row); write_csv(dest/'summary.csv', rows)
                print('FINITE', json.dumps({k:row[k] for k in ['scenario','seed','mesh','mode','step','eval_rms_full','gain_vs_frozen_bp','directional_rms_core']}), flush=True)
            if step % 50 == 0:
                history.append(dict(step=step, time_index=k, sampled_rms=float(loss.detach().sqrt())))
        write_csv(folder/'training_history.csv', history)
    assert max(initial_check) == 0.
    write_json(dest/'completion.json', dict(status='complete', seconds=time.perf_counter()-started,
               initial_value_differences=initial_check, frozen_solve_residual=solve_residual))


def simulate_case(out, config, scenario, seed):
    source = out/'continuous'/scenario['name']/f'seed{seed}'
    dest = out/'simulation'/scenario['name']/f'seed{seed}'
    if (dest/'completion.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=False)
    m = load_net(source/'step_0.pt', config).market
    settings = config['continuous']; rows=[]; start=time.perf_counter()
    for nsteps in settings['mc_steps']:
        payoffs = {}
        for step in [0, *config['primary_comparison']]:
            pg = np.load(source/f'policy_{step}.npz')
            controller = ScalarFeedback([pg['tau'], pg['y'], pg['z']], pg['q'])
            result = simulate(m, controller, npaths=settings['mc_paths'], nsteps=nsteps,
                              fine_steps=settings['mc_fine_steps'], seed=settings['mc_seed'])
            np.savez_compressed(dest/f'paths_{step}_{nsteps}.npz', **result)
            payoffs[step] = result['payoff']
        for after, before in [(50, 0), (1000, 0), (1000, 50)]:
            ce = paired_ce(m, payoffs[after], payoffs[before])
            gain = -1e4*ce['relative_CE_shortfall_vs_reference']; se = 1e4*ce['paired_mc_se']
            adjusted_z = norm.ppf(1-.05/(2*len(config['seeds'])*len(config['scenarios'])))
            row = dict(scenario=scenario['name'], seed=seed, nsteps=nsteps, after=after, before=before,
                       gain_bp=gain, se_bp=se, low_bp=gain-1.96*se, high_bp=gain+1.96*se,
                       simultaneous_low_bp=gain-adjusted_z*se, simultaneous_high_bp=gain+adjusted_z*se)
            rows.append(row)
        write_csv(dest/'summary.csv', rows)
        print('SIMULATION', json.dumps(rows[-1]), flush=True)
    write_json(dest/'completion.json', dict(status='complete', seconds=time.perf_counter()-start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/npi_focused_strengthening_20260926')
    parser.add_argument('--phase', required=True, choices=['continuous', 'finite', 'simulation'])
    parser.add_argument('--seeds', type=int, nargs='+')
    args = parser.parse_args()
    out = ROOT/args.output; config = initialize(out)
    phase_files = [Path(__file__), *sorted((ROOT/'src/npi').glob('*.py')),
                   ROOT/'configs/focused_strengthening.json']
    manifest = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in phase_files}
    record = out/f'{args.phase}_source_sha256.json'
    if record.exists():
        if json.loads(record.read_text()) != manifest:
            raise ValueError('Phase sources changed; preserve an explicit amendment before rerunning')
    else:
        write_json(record, manifest)
        with zipfile.ZipFile(out/f'{args.phase}_sources.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in phase_files:
                z.write(p, p.relative_to(ROOT))
    torch.set_num_threads(2)
    seeds = config['seeds'] if args.seeds is None else args.seeds
    if not set(seeds) <= set(config['seeds']):
        raise ValueError('Only locked study seeds may be requested')
    if args.phase in ('continuous', 'simulation'):
        operation = continuous_case if args.phase == 'continuous' else simulate_case
        for scenario in config['scenarios']:
            for seed in seeds:
                operation(out, config, scenario, seed)
    else:
        original = json.loads((ROOT/config['model_source']).read_text())
        for grid in config['finite']['grids']:
            for scenario in config['scenarios']:
                if scenario['name'] not in grid['scenarios']:
                    continue
                m = Model.from_dict({**original['model'], **scenario['changes']})
                system = FiniteSystem.create(m, **{k:grid[k] for k in ['ny','nz','nt','na']})
                for seed in seeds:
                    finite_case(out, config, scenario, seed, grid, system)
    print('PHASE COMPLETE', args.phase, flush=True)


if __name__ == '__main__':
    main()
