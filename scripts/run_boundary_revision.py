from pathlib import Path
import argparse,hashlib,json,sys,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.cheap_bounds import FiniteSystem
from npi.boundary_neural import train_boundary,grid_values

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False)
manifest=json.loads((ROOT/'configs/cost_boundary_revision.json').read_text())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
hashes={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
        for sub in ['src','scripts','configs'] for f in (ROOT/sub).rglob('*') if f.suffix in ['.py','.json']}
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
with zipfile.ZipFile(out/'run_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in hashes:z.write(ROOT/name,name)
config=manifest['boundary'];m=Model.from_dict(json.loads((ROOT/config['model_source']).read_text())['model'])
rows=[]
for grid in config['grids']:
    system=FiniteSystem.create(m,**{k:grid[k] for k in ['ny','nz','nt','na']})
    for seed in grid['seeds']:
        modes=config['representations'] if seed%2 else list(reversed(config['representations']))
        for mode in modes:
            name=f"{grid['name']}_{mode}_seed{seed}"
            print('START',name,flush=True)
            net,meta=train_boundary(system,mode,seed,config['training'],out/name)
            values=grid_values(net,system)
            np.savez_compressed(out/name/'values.npz',values=values,tau=system.tau,y=system.y,z=system.z,ids=system.ids)
            row={'name':name,'grid':grid['name'],'mode':mode,'seed':seed,'status':'success',
                 'steps':config['training']['steps'],'seconds':meta['training_seconds']}
            rows.append(row);(out/'status.json').write_text(json.dumps(rows,indent=2))
            print('END',json.dumps(row),flush=True)
