from pathlib import Path
import argparse,json,sys
import numpy as np
import torch
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.reference import Reference
from npi.neural import ValueNetwork,predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--checkpoints',nargs='+',required=True)
    args=p.parse_args();ref=Reference.load(args.reference);m=ref.model
    s=2.;L=min(-m.zmin,m.zmax);timechange=np.expm1(2*m.kappa*s)/(2*m.kappa)
    survival=max(0.,1-4*norm.sf(L/(m.sigma_z*np.sqrt(timechange))))
    gap=survival*(m.utility(m.ymin+m.rate*s)-m.utility(m.ymin))
    info={'survival_interval':s,'survival_probability_lower_bound':float(survival),
          'lower_boundary_value_jump_lower_bound':float(gap),
          'global_HJB_sup_lower_bound_for_continuous_hard_boundary_candidates':float(gap/m.horizon),
          'scope':'analytic conservative bound; not estimated from Monte Carlo',
          'source_of_obstruction':'stopped liquidation problem introduced in this revision; zero exposure and positive safe rate'}
    (ROOT/'notes/boundary_obstruction.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    torch.set_num_threads(2)
    ys=np.linspace(m.ymin,m.ymin+.7,200);points=np.column_stack([np.full(len(ys),m.horizon),ys,np.zeros(len(ys))])
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    ax.plot(np.exp(ys),ref.interpolate(points),label='Finite-grid reference',color='black')
    ax.plot(np.exp(ys),m.utility(ys),label='Immediate liquidation',ls=':')
    lower=m.utility(ys)+survival*(m.utility(ys+m.rate*s)-m.utility(ys))
    ax.plot(np.exp(ys[1:]),lower[1:],label='Zero-exposure bound (interior)',ls='--')
    ax.scatter([np.exp(ys[0])],[lower[0]],facecolors='none',edgecolors='tab:orange',zorder=5)
    for path in map(Path,args.checkpoints):
        data=torch.load(path,map_location='cpu',weights_only=True);meta=data['metadata']
        net=ValueNetwork(m,meta['settings']['width'],meta['settings']['depth']);net.load_state_dict(data['state_dict'])
        ax.plot(np.exp(ys),predict(net,points)['v'],label=meta['method'].upper()+f", seed {meta['seed']}")
    ax.set(xlabel='Wealth near the lower liquidation barrier',ylabel='Value / expected utility',title='Boundary trace obstruction with zero exposure admissible')
    ax.legend(fontsize=8)
    fig.savefig(ROOT/'reports/figures/boundary_obstruction.pdf');fig.savefig(ROOT/'reports/figures/boundary_obstruction.png',dpi=170)
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
