"""Check paired payoffs and original inputs before reporting the follow-up."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np
import torch
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.simulation import paired_ce
from followup_io import BOUNDARY_DEFAULT, DERIVATIVE_DEFAULT, project_path, verify_inputs


def read(path):return list(csv.DictReader(path.open(encoding='utf-8')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--boundary-results',type=project_path,default=BOUNDARY_DEFAULT)
    parser.add_argument('--derivative-results',type=project_path,default=DERIVATIVE_DEFAULT)
    parser.add_argument('--tables-dir',type=project_path,default=ROOT/'reports/tables')
    args=parser.parse_args()
    out=args.boundary_results
    cfg=json.loads((out/'protocol.json').read_text())
    completion=json.loads((out/'simulation_completion.json').read_text())
    assert completion['seeds']==cfg['seeds']
    m=Model.from_dict(json.loads((ROOT/cfg['model_source']).read_text())['model'])
    provenance={str(directory):verify_inputs(directory) for directory in [out,args.derivative_results]}
    (out/'source_validation.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    derivative_out=args.derivative_results
    derivative_rows=read(derivative_out/'summary.csv')
    assert len(derivative_rows)==252
    cached={}
    for row in derivative_rows:
        key=(row['seed'],row['region'])
        if key not in cached:cached[key]=dict(np.load(derivative_out/f'seed{key[0]}_{key[1]}.npz'))
        a=cached[key];mesh=row['mesh'];step=row['step'];delta={}
        for field in ['v','vy','vyy','vyz']:
            delta[field]=a[f'step{step}_{field}']-a[f'{mesh}_{field}']
            error=float(np.sqrt(np.mean((delta[field]/a['scale'])**2)))
            difference=float(np.sqrt(np.mean(((a[f'fine_{field}']-a[f'finest_{field}'])/a['scale'])**2)))
            assert abs(error-float(row[field+'_rms']))<1e-12
            assert abs(difference-float(row[field+'_mesh_difference']))<1e-12
        p=a[f'step{step}_q'];q=a['fixed_q']
        b=m.phi(a['points'][:,2])*delta['vy']+m.rho*m.sigma_z*delta['vyz'];c=delta['vy']-delta['vyy']
        direction=float(np.sqrt(np.mean(((p-q)*(b-.5*(p+q)*c)/a['scale'])**2)))
        assert abs(direction-float(row['directional_rms']))<1e-12
    rows=read(out/'simulation_summary.csv');assert len(rows)==72
    contrasts={'frozen_length_hard','frozen_length_split','frozen_representation',
               'iteration_representation','iteration_early_hard','iteration_early_split'}
    expected={(str(seed),str(nt),name) for seed in cfg['seeds'] for nt in cfg['mc_steps'] for name in contrasts}
    assert {(r['seed'],r['time_steps'],r['comparison']) for r in rows}==expected
    checked=0;crit=norm.ppf(1-.05/48)
    for row in rows:
        seed=int(row['seed']);nt=int(row['time_steps']);c=row['comparison']
        if c.startswith('frozen_length_'):
            mode=c.removeprefix('frozen_length_');a=('frozen',mode,1000);b=('frozen',mode,50)
        elif c.startswith('iteration_early_'):
            mode=c.removeprefix('iteration_early_');a=('iteration',mode,400);b=('iteration',mode,200)
        else:
            phase=c.removesuffix('_representation');step=1000 if phase=='frozen' else 2000
            a=(phase,'split',step);b=(phase,'hard',step)
        def payoff(k):return np.load(out/k[0]/f'seed{seed}'/k[1]/f'paths_{k[2]}_{nt}.npz')['payoff']
        ce=paired_ce(m,payoff(a),payoff(b));gain=-1e4*ce['relative_CE_shortfall_vs_reference'];se=1e4*ce['paired_mc_se']
        for key,val in dict(gain_bp=gain,se_bp=se,low_bp=gain-1.96*se,high_bp=gain+1.96*se,adjusted_low_bp=gain-crit*se,adjusted_high_bp=gain+crit*se).items():
            assert abs(float(row[key])-val)<1e-9,(seed,c,key)
        checked+=1
    history=[]
    for phase in ['frozen','iteration']:
        for seed in cfg['seeds']:
            init=[]
            for mode in cfg['modes']:
                d=out/phase/f'seed{seed}'/mode
                assert json.loads((d/'completion.json').read_text())['status']=='complete'
                checkpoint=torch.load(d/'step_0.pt',weights_only=True,map_location='cpu')
                assert checkpoint['model']==m.to_dict(),(phase,seed,mode,'model mismatch')
                init.append(checkpoint['state_dict'])
                history+=read(d/'history.csv')
                for path in d.glob('policy_*.npz'):
                    q=np.load(path)['q'];assert np.isfinite(q).all() and q.min()>=m.qmin-1e-6 and q.max()<=m.qmax+1e-6
            assert init[0].keys()==init[1].keys()
            assert all(torch.equal(init[0][k],init[1][k]) for k in init[0]),(phase,seed)
    assert len(history)==108
    summary={}
    for c in sorted({r['comparison'] for r in rows}):
        r=[x for x in rows if x['comparison']==c and x['time_steps']=='960']
        gains=[float(x['gain_bp']) for x in r]
        summary[c]=dict(median_bp=float(np.median(gains)),min_bp=min(gains),max_bp=max(gains),
            negative_points=sum(x<0 for x in gains),positive_points=sum(x>0 for x in gains),
            adjusted_negative=sum(float(x['adjusted_high_bp'])<0 for x in r),
            adjusted_positive=sum(float(x['adjusted_low_bp'])>0 for x in r))
        if c.startswith('iteration_early_'):
            del summary[c]['adjusted_negative'],summary[c]['adjusted_positive']
            summary[c]['interpretation']='Descriptive comparison, outside the primary multiplicity family.'
    for mode in cfg['modes']:
        ratios=[]
        for seed in cfg['seeds']:
            h={int(x['step']):x for x in history if x['phase']=='frozen' and x['mode']==mode and int(x['seed'])==seed}
            ratios.append(float(h[1000]['eval_rms_full'])/float(h[50]['eval_rms_full']))
        summary['frozen_length_'+mode]['median_residual_ratio']=float(np.median(ratios))
    by={(r['seed'],r['comparison'],r['time_steps']):float(r['gain_bp']) for r in rows}
    max_time_difference=max(abs(v-by[(s,c,'480')]) for (s,c,t),v in by.items() if t=='960')
    summary['checks']=dict(paired_contrasts_recomputed=checked,derivative_rows_recomputed=len(derivative_rows),initial_parameter_pairs_equal=12,
        history_rows=len(history),maximum_time_contrast_difference_bp=max_time_difference,
        nonpositive_curvature_samples=sum(int(r['curvature_nonpositive']) for r in history),
        scope='Conditional MC intervals; diagnostic comparison on previously inspected baseline seeds; no runtime claim.')
    (out/'verified_summary.json').write_text(json.dumps(summary,indent=2))
    names=[('frozen_length_hard','Full: 1000 versus 50 steps'),('frozen_length_split','Relaxed: 1000 versus 50 steps'),
           ('frozen_representation','Relaxed versus full: frozen'),('iteration_representation','Relaxed versus full: NPI')]
    tex=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{5pt}',
        r'\caption{Continuous-residual boundary comparisons, six paired seeds. Frozen comparisons use a fixed base plus trainable correction; repeated NPI uses the single-network trial functions for ten policy updates (2000 optimizer steps). CE differences use 960 simulation steps. Adjusted intervals cover the 24 primary seedwise contrasts.}',
        r'\label{tab:continuous-boundary}',r'\begin{tabular}{lrrr}',r'\toprule',
        r'Comparison & Median CE (bp) & Adjusted $<0$ & Adjusted $>0$ \\',r'\midrule']
    for key,label in names[2:]:
        r=summary[key];tex.append(f"{label} & {r['median_bp']:.2f} & {r['adjusted_negative']}/6 & {r['adjusted_positive']}/6 \\")
        tex[-1]+='\\'
    tex.append(r'\midrule')
    for key,label in [('iteration_early_hard',r'Full NPI: $200\to400$'),('iteration_early_split',r'Relaxed NPI: $200\to400$')]:
        r=summary[key]
        tex.append(f"{label} & {r['median_bp']:.2f} & "+r'\multicolumn{2}{c}{'+f"{r['negative_points']}/6 negative estimates"+r'} \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\footnotesize The last two rows are descriptive; their counts refer to point estimates, not adjusted intervals.',r'\end{table}']
    args.tables_dir.mkdir(parents=True,exist_ok=True)
    (args.tables_dir/'npi_continuous_boundary.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    tex=[r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Individual primary continuous-residual contrasts at 960 simulation steps. Brackets give Bonferroni-adjusted conditional 95\% Monte Carlo intervals across the 24 primary contrasts.}',
        r'\label{tab:continuous-boundary-pairs}',r'\begin{tabular}{rrr}',r'\toprule',r'Seed & CE difference (bp) & Adjusted interval (bp) \\',r'\midrule']
    for key,label in names:
        tex.append(r'\multicolumn{3}{l}{'+label+r'} \\')
        for row in rows:
            if row['comparison']==key and row['time_steps']=='960':
                tex.append(f"{row['seed']} & {float(row['gain_bp']):.2f} & [{float(row['adjusted_low_bp']):.2f}, {float(row['adjusted_high_bp']):.2f}] "+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (args.tables_dir/'npi_continuous_boundary_pairs.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
