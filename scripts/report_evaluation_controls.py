"""Validate saved control comparisons and generate compact reports tables."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from evaluation_controls import ROOT, OUT, BOUNDARY, initialize, contrast, decomposition, rms
from followup_io import verify_inputs


def read(p):
    return list(csv.DictReader(p.open(encoding='utf-8')))


def table(path, caption, label, columns, header, rows):
    text=[r'\begin{table}[htbp]',r'\centering\small\setlength{\tabcolsep}{5pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',
          r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']
    text += [' & '.join(map(str,row))+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    cfg,design,m=initialize();provenance=verify_inputs(OUT)
    decomposition(cfg,m)
    matched=read(OUT/'matched_comparisons.csv');damped=read(OUT/'damping_comparisons.csv')
    assert len(matched)==84 and len(damped)==18
    assert len({(r['seed'],r['time_steps'],r['comparison']) for r in matched})==84
    assert len({(r['seed'],r['time_steps'],r['comparison']) for r in damped})==18
    checked=0
    for r in matched+damped:
        seed=int(r['seed']);nt=int(r['time_steps']);name=r['comparison']
        if r in matched:
            dest=OUT/'cpu_single'/f'seed{seed}'
            def load(mode,step):
                if mode in ['cpu','gpu']:p=dest/f'paths_{mode}_{step}_{nt}.npz'
                else:p=BOUNDARY/'frozen'/f'seed{seed}'/mode/f'paths_{step}_{nt}.npz'
                return np.load(p)['payoff']
            if name.endswith('_length'):
                mode=name.removesuffix('_length');a,b=load(mode,1000),load(mode,50);family=24
            else:
                mode=name.removesuffix('_minus_cpu_long');a,b=load(mode,1000),load('cpu',1000);family=1
        else:
            dest=OUT/'damping'/f'seed{seed}'
            keys={'damped_minus_short':('damped','50'),'long_minus_short':('1000','50'),'damped_minus_long':('damped','1000')}
            k,j=keys[name];a=np.load(dest/f'paths_{k}_{nt}.npz')['payoff'];b=np.load(dest/f'paths_{j}_{nt}.npz')['payoff']
            family=3 if name=='damped_minus_short' else 1
        for key,value in contrast(m,a,b,family).items():assert abs(value-float(r[key]))<1e-10
        checked+=1
    summary={}
    for data,prefix in [(matched,'matched'),(damped,'damping')]:
        for name in sorted({r['comparison'] for r in data}):
            rows=[r for r in data if r['comparison']==name and r['time_steps']=='960']
            vals=[float(r['gain_bp']) for r in rows]
            summary[prefix+'_'+name]=dict(median_bp=float(np.median(vals)),min_bp=min(vals),max_bp=max(vals),
                negative_points=sum(v<0 for v in vals),positive_points=sum(v>0 for v in vals),
                adjusted_negative=sum(float(r['adjusted_high_bp'])<0 for r in rows),
                adjusted_positive=sum(float(r['adjusted_low_bp'])>0 for r in rows))
    h=[read(OUT/'cpu_single'/f'seed{s}'/'history.csv') for s in cfg['seeds']]
    for seed in cfg['seeds']:
        cpu=torch.load(OUT/'cpu_single'/f'seed{seed}'/'step_0.pt',map_location='cpu',weights_only=True)['state_dict']
        for mode in cfg['modes']:
            correction=torch.load(BOUNDARY/'frozen'/f'seed{seed}'/mode/'step_0.pt',map_location='cpu',weights_only=True)['state_dict']
            assert all(torch.equal(value,correction['base.'+key]) for key,value in cpu.items())
            assert not torch.count_nonzero(correction['correction.net.6.weight'])
            assert not torch.count_nonzero(correction['correction.net.6.bias'])
    ratios=[float(next(r for r in x if r['step']=='1000')['eval_rms_full'])/float(next(r for r in x if r['step']=='50')['eval_rms_full']) for x in h]
    summary['matched_cpu_length']['median_residual_ratio']=float(np.median(ratios))
    dd=read(OUT/'quotient_decomposition.csv'); assert len(dd)==18
    co=read(OUT/'damping_coefficients.csv'); assert len(co)==3
    # The stored network actions and multiplication by lambda use float32.
    assert all(abs(float(r['short_rms'])-float(r['matched_rms']))<1e-9 for r in co)
    pairs={ (r['seed'],r['comparison'],r['time_steps']):float(r['gain_bp']) for r in matched+damped}
    max_dt=max(abs(v-pairs[s,c,'480']) for (s,c,t),v in pairs.items() if t=='960')
    summary['checks']=dict(paired_contrasts_recomputed=checked,exact_identity_max_error=max(float(r['identity_max_error']) for r in dd),
        all_core_points_interior=all(r['points']==r['interior_points'] for r in dd),
        maximum_time_contrast_difference_bp=max_dt,coefficient_rms_matching=True,identical_initial_function_pairs=12,input_hashes=provenance)
    (OUT/'verified_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    tables=ROOT/'reports/tables'
    labels=[('gpu','Single network, GPU'),('cpu','Single network, CPU'),('hard','Full correction, CPU'),('split','Relaxed correction, CPU')]
    rows=[]
    for mode,label in labels:
        s=summary['matched_'+mode+'_length']
        rows.append([label,f"{s['median_bp']:.2f}",f"{s['negative_points']}/6",f"{s['adjusted_negative']}/6",f"{s['adjusted_positive']}/6"])
    table(tables/'npi_matched_controls.tex',
        r'Evaluation length under common path sampling, seeds 701--706. CE compares 1000 with 50 evaluation steps. GPU uses fused Adam; all CPU runs use unfused Adam. The three CPU designs share the initial function, frozen feedback, training draws, and learning-rate schedule. Adjusted signs use a family of 24 conditional Monte Carlo intervals; these are diagnostic comparisons on reused seeds.',
        'tab:matched-controls','lrrrr',r'Evaluator & Median CE (bp) & Points $<0$ & Adj. $<0$ & Adj. $>0$',rows)
    rows=[]
    for r in dd:
        if r['mesh']!='finest':continue
        rows.append([r['seed'],r['step']]+[f"{1e3*float(r[k]):.2f}" for k in ['a1_rms','a2_rms','a3_rms','greedy_error_rms']]+[f"{1e6*float(r['sum_cross_terms']):.2f}"])
    table(tables/'npi_quotient_controls.tex',
        r'Exact quotient decomposition on the 546 core points using the finest finite reference. The four RMS columns are multiplied by $10^3$; the cross-term sum $2\sum_{i<j}\langle a_i,a_j\rangle_\nu$ is multiplied by $10^6$. Every point has positive candidate and reference curvature and interior greedy actions. Cross terms are retained in the squared error.',
        'tab:quotient-controls','rrrrrrr',r'Seed & Step & $\|a_1\|_\nu$ & $\|a_2\|_\nu$ & $\|a_3\|_\nu$ & $\|\sum_i a_i\|_\nu$ & Cross terms',rows)
    rows=[]
    for c in co:
        s=c['seed'];r=next(r for r in damped if r['seed']==s and r['time_steps']=='960' and r['comparison']=='damped_minus_short')
        raw=next(r for r in damped if r['seed']==s and r['time_steps']=='960' and r['comparison']=='long_minus_short')
        rows.append([s,f"{float(c['lambda_value']):.3f}",f"{float(raw['gain_bp']):.2f}",f"{float(r['gain_bp']):.2f}",f"[{float(r['adjusted_low_bp']):.2f}, {float(r['adjusted_high_bp']):.2f}]"])
    table(tables/'npi_damping_controls.tex',
        r'Empirical matching of policy-update RMS on fixed core points. CE differences use 960 simulation steps and compare with $p_{50}$. Intervals adjust the three damped-versus-short comparisons. The rule uses no policy-performance estimates and supplies no certified improvement guarantee.',
        'tab:damping-controls','rrrrr',r'Seed & $\lambda$ & Long CE (bp) & Damped CE (bp) & Adjusted interval',rows)
    rows=[]
    for r in dd:
        if r['mesh']!='finest':continue
        rows.append([r['seed'],r['step']]+[f"{float(r[k]):.5f}" for k in ['ey_rms','eyy_rms','eyz_rms','d_rms','eta_scaled_rms']])
    table(tables/'npi_derivative_controls.tex',
        r'Initial, short, and long evaluations against the finest finite reference. Derivative errors and $\eta$ are divided by $S$ before taking the core RMS; $d=p-q$ is unscaled. The last two columns separate update magnitude from the directional derivative discrepancy.',
        'tab:derivative-controls','rrrrrrr',r'Seed & Step & $e_y/S$ & $e_{yy}/S$ & $e_{yz}/S$ & $d$ & $\eta/S$',rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
