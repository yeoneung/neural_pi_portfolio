from pathlib import Path
import argparse,hashlib,json,sys,zipfile
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.reference import Reference
from npi.neural import ValueNetwork,predict
from npi.transport_bounds import propagate

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sources={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
             for sub in ['src','scripts'] for f in (ROOT/sub).rglob('*.py')}
    (out/'source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    with zipfile.ZipFile(out/'run_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for file in sources:archive.write(ROOT/file,file)
    (out/'inputs.json').write_text(json.dumps({'reference':a.reference,'checkpoints':a.checkpoints,
        'input_sha256':{str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [a.reference,*a.checkpoints]}},indent=2),encoding='utf-8')
    ref=Reference.load(a.reference);m=ref.model;torch.set_num_threads(2)
    points=np.stack([g.ravel() for g in np.meshgrid(ref.tau,ref.y,ref.z,indexing='ij')],axis=1)
    rows=[]
    for path in map(Path,a.checkpoints):
        data=torch.load(path,map_location='cpu',weights_only=True);meta=data['metadata']
        net=ValueNetwork(m,meta['settings']['width'],meta['settings']['depth']);net.load_state_dict(data['state_dict'])
        pred=predict(net,points)
        result=propagate(ref,pred['q'].reshape(ref.value.shape),pred['v'].reshape(ref.value.shape))
        center=lambda arr:float(RegularGridInterpolator((ref.y,ref.z),arr)([[0,0]])[0])
        row={'name':path.parent.name,'global_value_bound':result['global_value_bound'],'global_policy_bound':result['global_policy_bound'],
             'initial_value_envelope':center(result['value_envelope']),
             'initial_policy_evaluation_envelope':center(result['policy_evaluation_envelope']),
             'initial_policy_loss_bound':center(result['policy_loss_bound']),
             'initial_deficit_upper_envelope':center(result['deficit_upper_envelope']),
             'initial_deficit_policy_loss_bound':center(result['deficit_policy_loss_bound']),
             'initial_actual_discrete_policy_loss':center(ref.value[-1]-result['policy_value']),
             'initial_actual_discrete_value_error':abs(center(ref.value[-1])-float(predict(net,np.array([[m.horizon,0,0]]))['v'][0])),
             'max_envelope_bellman_residual':max(h['envelope_residual'] for h in result['history']),
             'max_deficit_bellman_residual':max(h['deficit_envelope_residual'] for h in result['history']),
             'max_fixed_policy_linear_system_residual':max(h['fixed_policy_linear_system_residual'] for h in result['history']),
             'scope':'finite MDP; exact-solve theorem, floating point diagnostics; no continuum certificate'}
        for arr in ['value_envelope','policy_evaluation_envelope','policy_loss_bound']:
            if np.min(result[arr])< -1e-8:raise AssertionError('Envelope positivity failed')
        if np.max(ref.value[-1]-result['policy_value']-result['policy_loss_bound'])>1e-6:raise AssertionError('Observed discrete loss exceeds bound')
        if np.max(ref.value[-1]-result['policy_value']-result['deficit_policy_loss_bound'])>1e-6:raise AssertionError('Deficit bound failed')
        np.savez_compressed(out/(path.parent.name+'.npz'),y=ref.y,z=ref.z,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        (out/(path.parent.name+'_history.json')).write_text(json.dumps(result['history'],indent=2),encoding='utf-8')
        rows.append(row);(out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)

if __name__=='__main__':main()
