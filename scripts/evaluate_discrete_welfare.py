"""Evaluate all completed neural policies independently on their reference MDP."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npi.reference import Reference
from npi.neural import ValueNetwork,predict
from npi.discrete_evaluation import evaluate_policy,certainty_equivalent


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);rows=[]
    for run in map(Path,args.runs):
        ref=Reference.load(run/'reference.npz');m=ref.model
        grid=np.meshgrid(ref.tau,ref.y,ref.z,indexing='ij')
        points=np.stack([g.ravel() for g in grid],axis=1)
        center=np.array([[m.horizon,0.,0.]])
        vstar=float(ref.interpolate(center)[0]);cestar=float(certainty_equivalent(m,vstar))
        for checkpoint in sorted(run.glob('*_seed*/checkpoint.pt')):
            data=torch.load(checkpoint,map_location='cpu',weights_only=True)
            meta=data['metadata'];setting=meta['settings']
            net=ValueNetwork(m,setting['width'],setting['depth'])
            net.load_state_dict(data['state_dict'])
            pred=predict(net,points)
            evaluated=evaluate_policy(ref,pred['q'].reshape(ref.value.shape),pred['v'].reshape(ref.value.shape))
            name=f"{meta['method']}_seed{meta['seed']}"
            evaluated.save(out/f'{name}.npz')
            j=float(evaluated.interpolate(center)[0]);ce=float(certainty_equivalent(m,j))
            row={'method':meta['method'],'seed':meta['seed'],'run':str(run),
                 'initial_state':{'tau':m.horizon,'wealth':1.0,'factor':0.0},
                 'discrete_optimal_utility':vstar,'discrete_policy_utility':j,
                 'discrete_optimal_CE':cestar,'discrete_policy_CE':ce,
                 'relative_discrete_CE_loss':1-ce/cestar,
                 **{key:value for key,value in evaluated.diagnostics.items()
                    if key not in ['history','boundary','policy_representation']},
                 'status':'pilot; finite stopped MDP only; no continuum or simulation welfare claim'}
            rows.append(row);print(json.dumps(row),flush=True)
            (out/'welfare_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')


if __name__=='__main__':main()
