"""Prespecified paired study; training does not consult a numerical reference."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import traceback
import time
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.neural import train,predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--after-study',default=None)
    args=p.parse_args();manifest=json.loads(Path(args.manifest).read_text())
    if args.after_study:
        before=Path(args.after_study)
        previous=json.loads((before/'manifest.json').read_text())
        expected=sum(len(v['seeds'])*len(v['methods']) for v in previous['variants'])
        while True:
            try:ready=len(json.loads((before/'status.json').read_text()))>=expected
            except (FileNotFoundError,json.JSONDecodeError):ready=False
            if ready:break
            time.sleep(5)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sources={str(f.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
             for sub in ['src','scripts'] for f in (ROOT/sub).rglob('*.py')}
    (out/'source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    with zipfile.ZipFile(out/'run_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sources:z.write(ROOT/f,f)
        z.write(out/'manifest.json','manifest.json')
    rows=[]
    for variant in manifest['variants']:
        for seed in variant['seeds']:
            order=variant['methods'] if seed%2 else list(reversed(variant['methods']))
            for method in order:
                name=f"{variant['name']}_{method}_seed{seed}"
                m=Model.from_dict({**manifest['model'],**variant.get('model',{})})
                settings={**manifest['training'],**variant.get('training',{})}
                folder=out/name
                print(f'START {name}',flush=True)
                try:
                    net,meta=train(m,settings,method,seed,folder)
                    axes=[np.linspace(0,m.horizon,11),np.linspace(-.7,.7,31),np.linspace(-.3,.3,21)]
                    points=np.stack([g.ravel() for g in np.meshgrid(*axes,indexing='ij')],axis=1)
                    pred=predict(net,points)
                    np.savez_compressed(folder/'evaluation.npz',points=points,**pred)
                    row={'name':name,'variant':variant['name'],'method':method,'seed':seed,'status':'success',
                         'steps':meta['total_optimizer_steps'],'seconds':meta['wall_seconds_including_diagnostics']}
                except Exception as e:
                    folder.mkdir(exist_ok=True)
                    (folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
                    row={'name':name,'variant':variant['name'],'method':method,'seed':seed,'status':'failed','error':str(e)}
                rows.append(row)
                (out/'status.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
                print('END '+json.dumps(row),flush=True)


if __name__=='__main__':main()
