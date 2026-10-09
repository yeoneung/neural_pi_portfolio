"""Prospective equal-wall-time comparison of simple NPI evaluation rules.

The directional rule is a diagnostic heuristic. Its acceptance is not an
application of a certified-error theorem. All methods and seeds are retained.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import random
import sys
import time
import zipfile

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_npi_iteration import write_csv,write_json
from npi.model import Model
from npi.neural import ValueNetwork,sample,derivatives,greedy,residual
from npi.inexact import PolicyBank,coefficients,disagreement_metrics


def train_one(m,settings,method,seed,out,budget,check_times):
    out.mkdir(exist_ok=False)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=ValueNetwork(m,settings['width'],settings['depth']).cuda()
    optimizer=torch.optim.Adam(net.parameters(),lr=settings['lr'],fused=True)
    stream=torch.Generator().manual_seed(seed+17000)
    val=sample(m,2048,torch.Generator().manual_seed(seed+29000),'cuda',torch.float32)
    scale=torch.exp((1-m.gamma1)*val[:,1])+m.mixture*torch.exp((1-m.gamma2)*val[:,1])
    # The clock includes initial policy copying and all subsequent diagnostics.
    torch.cuda.synchronize();start=time.perf_counter()
    bank=PolicyBank(net)
    previous=coefficients(net,val)
    old_q=bank.controls(val)
    start_eval=float(torch.mean((previous['hjb']/scale)**2).sqrt())
    rows=[];snapshots=[];inner=0;step=0;updates=0;forced=0;next_check=0
    while True:
        x=sample(m,1024,stream,'cuda',torch.float32)
        qfixed=None if method=='pinn' else bank.controls(x)
        # All methods use exactly the same batches for common update prefixes.
        for _ in range(50):
            fraction=min((time.perf_counter()-start)/budget,1.)
            lr=1e-5+.5*(.001-1e-5)*(1+np.cos(np.pi*fraction))
            optimizer.param_groups[0]['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            d=derivatives(net,x)
            q=greedy(m,d).detach() if method=='pinn' else qfixed
            ss=torch.exp((1-m.gamma1)*x[:,1])+m.mixture*torch.exp((1-m.gamma2)*x[:,1])
            raw=residual(m,d,q)
            loss=torch.mean((raw/ss)**2+torch.relu(-d['vy']/ss)**2+torch.relu((d['vyy']-d['vy'])/ss)**2)
            if not torch.isfinite(loss):raise FloatingPointError((method,seed,step))
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            step+=1;inner+=1
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        adaptive=method in ['directional','forcing']
        due=adaptive or step%200==0 or elapsed>=budget or (next_check<len(check_times) and elapsed>=check_times[next_check])
        metric={};accept=False;reason='continue'
        if due:
            current=coefficients(net,val)
            metric=disagreement_metrics(current,previous,old_q,scale)
            if method in ['pi200','damped']:
                accept=inner>=200 or elapsed>=budget
                reason='fixed interval' if accept else reason
            elif method=='pi1000':
                accept=inner>=1000 or elapsed>=budget
                reason='fixed interval' if accept else reason
            elif adaptive:
                passed=(metric['disagreement_ratio']<=.5 if method=='directional' else metric['eval_rms']<=.5*start_eval)
                accept=inner>=200 and (passed or inner>=1000)
                reason=('directional proxy' if method=='directional' else 'relative residual') if passed else 'inner cap'
                if accept and not passed:forced+=1
            if accept:
                bank.update(net,damping=.5 if method=='damped' else 1.)
                old_q=bank.controls(val);inner=0;updates+=1
                # Recompute the new evaluation residual (damped q may differ).
                adv=(current['q']-old_q)*(current['b']-.5*current['c']*(current['q']+old_q))
                start_eval=float(torch.mean(((current['hjb']-adv)/scale)**2).sqrt())
            previous=current
            rows.append(dict(step=step,elapsed_seconds=time.perf_counter()-start,inner_steps=inner,
                accepted=accept,reason=reason,updates=updates,forced_updates=forced,**metric))
        # Serialization and policy copying are charged to the same clock.
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        while next_check<len(check_times) and elapsed>=check_times[next_check]:
            nominal=check_times[next_check]
            if method=='pinn':bank.update(net)
            state=dict(model=m.to_dict(),settings=settings,method=method,seed=seed,
                state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                policy_bank=bank.state(),steps=step,elapsed_seconds=elapsed,nominal_seconds=nominal)
            torch.save(state,out/f'budget_{int(nominal)}.pt')
            snapshots.append(dict(nominal_seconds=nominal,elapsed_seconds=elapsed,steps=step))
            next_check+=1
        if elapsed>=budget:break
    torch.cuda.synchronize()
    meta=dict(method=method,seed=seed,steps=step,wall_seconds=time.perf_counter()-start,
              nominal_seconds=budget,accepted_updates=updates,forced_updates=forced,snapshots=snapshots,
              unfinished_inner_steps=inner,policy_bank_size=len(bank.members),
              scope='same nominal training wall budget; 50-step deadline polling; initialization, decisions and saves charged; final policy is deployed bank')
    write_csv(out/'history.csv',rows);write_json(out/'metadata.json',meta)
    return meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    original=json.loads((ROOT/'configs/confirmatory_study.json').read_text())
    plan=dict(seeds=[201,202,203],methods=['pinn','pi200','pi1000','damped','forcing','directional'],
        nominal_wall_seconds=30.,check_times=[10.,20.,30.],minimum_adaptive_inner=200,
        maximum_adaptive_inner=1000,adaptive_check_every=50,disagreement_multiplier=2.,
        disagreement_ratio_threshold=.5,forcing_eval_reduction=.5,damping=.5,
        mc_seed=9264001,mc_paths=32768,mc_steps=[480,960],mc_fine_steps=960,policy_grid=[41,81,81],
        observed_target_gain_bp=10.,model=original['model'],training=original['training'],
        status='prospective small comparison, new seeds; thresholds fixed without tuning on these seeds; directional proxy not certified',
        schedule='cosine by elapsed fraction of 30 seconds, same for all methods',
        timing='sequential GPU runs, no simultaneous audit jobs; 50-step polling; CPU serialization charged; post-training MC excluded for all methods',
        target_scope='first saved checkpoint with estimated CE gain >=10 bp over myopic; noisy post-hoc measurement, not online stopping')
    write_json(out/'protocol.json',plan)
    source=[f for sub in ['src','scripts'] for f in (ROOT/sub).rglob('*.py')]
    write_json(out/'source_sha256.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source})
    with zipfile.ZipFile(out/'sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in source:z.write(f,str(f.relative_to(ROOT)))
        z.write(out/'protocol.json','protocol.json')
    torch.set_num_threads(4);m=Model.from_dict(plan['model']);status=[]
    # Warm CUDA and the common derivative/optimizer kernels before all timing.
    warm=ValueNetwork(m,64,3).cuda();xx=torch.rand(16,3,device='cuda')
    for _ in range(3):
        dd=derivatives(warm,xx);rr=residual(m,dd,greedy(m,dd).detach());(rr*rr).mean().backward();warm.zero_grad(set_to_none=True)
    del warm;torch.cuda.synchronize()
    for i,seed in enumerate(plan['seeds']):
        methods=plan['methods'][i:]+plan['methods'][:i]
        if i%2:methods=list(reversed(methods))
        for method in methods:
            print('START',method,seed,flush=True)
            meta=train_one(m,plan['training'],method,seed,out/f'{method}_seed{seed}',30.,plan['check_times'])
            status.append(meta);write_json(out/'status.json',status)
            print('END',json.dumps(meta),flush=True)
    write_json(out/'completion.json',dict(status='complete',runs=len(status)))


if __name__=='__main__':main()
