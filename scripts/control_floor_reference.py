from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.reference import solve
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
m=Model.from_dict(json.loads((ROOT/'configs/control_floor_study.json').read_text())['model'])
rows=[]
for n,nt,na in [(80,240,121),(120,360,161)]:
    ref=solve(m,ny=n+1,nz=3*n//4+1,nt=nt,na=na,generator_kind='bounded',tolerance=1e-11)
    ref.save(out/f'floor_{n}.npz')
    rows.append({k:v for k,v in ref.diagnostics.items() if k!='history'})
    (out/'audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows[-1]),flush=True)
