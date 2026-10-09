"""Verify matched derivative arrays, report every seed, and plot paired changes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from matched_derivative_bridge import ROOT,DEFAULT,initialize,diagnostics,write_csv,write_json,reference,load_net,predict
from followup_io import project_path,sha256,verify_inputs


def read(p):return list(csv.DictReader(p.open(encoding='utf-8')))


def table(path,caption,label,columns,header,rows):
    lines=[r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',r'\caption{'+caption+'}',
           r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r' \\',r'\midrule']
    lines += [' & '.join(map(str,r))+r' \\' for r in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def family_sensitivity(out):
    sources=[ROOT/'results/npi_focused_strengthening_20260926/continuous_contrasts.csv',
             ROOT/'results/continuous_boundary_followup_20260929/simulation_summary.csv',
             ROOT/'results/npi_evaluation_controls_20260929/matched_comparisons.csv',
             ROOT/'results/npi_evaluation_controls_20260929/damping_comparisons.csv']
    families=[read(sources[0]),*[ [r for r in read(p) if r['time_steps']=='960' and r['primary']=='True'] for p in sources[1:]]]
    assert list(map(len,families))==[18,24,24,3]
    rows=[];counts={};crit=norm.ppf(1-.05/(2*69))
    for name,n,data in zip(['parameters','boundary','matched','damping'],[18,24,24,3],families):
        for r in data:
            key=r.get('comparison',r.get('scenario'));gain=float(r['gain_bp']);se=float(r['se_bp'])
            rows.append(dict(family=name,comparison=key,seed=r['seed'],family_size=n,gain_bp=gain,se_bp=se,
                             low_69=gain-crit*se,high_69=gain+crit*se))
        for key in sorted({r.get('comparison',r.get('scenario')) for r in data}):
            subset=[r for r in rows if r['family']==name and r['comparison']==key]
            counts[name+'_'+key]=dict(negative_69=int(sum(r['high_69']<0 for r in subset)),positive_69=int(sum(r['low_69']>0 for r in subset)))
    write_csv(out/'family_sensitivity.csv',rows)
    write_json(out/'family_sensitivity.json',dict(reported_slots=69,scope='The four specified families, counting repeated contrasts; excludes ancillary solver/timing comparisons. Retrospective sensitivity, not a new prespecified family.',
        input_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in sources},counts=counts))
    return counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=project_path,default=DEFAULT)
    p.add_argument('--report-dir',type=project_path,default=ROOT/'reports');args=p.parse_args()
    out=args.output;cfg,m=initialize(out);saved=read(out/'summary.csv');rows=[];references=[]
    assert len(saved)==96
    for seed in cfg['seeds']:
        dest=out/f'seed{seed}'
        for mesh in cfg['meshes']:
            marker=json.loads((dest/f'{mesh}_completion.json').read_text());assert marker['sha256']==sha256(dest/f'{mesh}_value.npz')
            references.append(marker)
        a=dict(np.load(dest/'diagnostic_arrays.npz'))
        for mesh in cfg['meshes']:
            for field,value in reference(dest/f'{mesh}_value.npz',m,a['points']).items():
                np.testing.assert_array_equal(value,a[mesh+'_'+field])
        for mode in cfg['modes']:
            for step in cfg['steps']:
                pred=predict(load_net(cfg,m,mode,seed,step),a['points'])
                for field in ['v','vy','vyy','vyz','q']:np.testing.assert_array_equal(pred[field],a[f'{mode}_{step}_'+field])
        for field in ['v','vy','vyy','vyz','q']:np.testing.assert_array_equal(a['single_0_'+field],a['full_0_'+field])
        current=diagnostics(a,cfg,m,seed)
        for r in current:
            original=next(x for x in saved if int(x['seed'])==seed and x['mesh']==r['mesh'] and x['mode']==r['mode'] and int(x['step'])==r['step'])
            for key,value in r.items():
                if isinstance(value,str):assert original[key]==value
                else:assert abs(float(original[key])-float(value))<1e-12,(seed,key)
        rows+=current
    index={(r['seed'],r['mesh'],r['mode'],r['step']):r for r in rows}
    changes=[]
    for seed in cfg['seeds']:
        for mesh in cfg['meshes']:
            for mode in cfg['modes']:
                short=index[seed,mesh,mode,50];long=index[seed,mesh,mode,1000]
                changes.append(dict(seed=seed,mesh=mesh,mode=mode,error_ratio=long['greedy_error_rms']/short['greedy_error_rms'],
                    cancellation_change=long['combined_to_component_ratio']-short['combined_to_component_ratio'],
                    error_change=long['greedy_error_rms']-short['greedy_error_rms'],cross_change=long['cross_sum']-short['cross_sum'],
                    component_mse_ratio=long['component_mse']/short['component_mse'],eta_ratio=long['eta_scaled_rms']/short['eta_scaled_rms']))
    write_csv(out/'paired_changes.csv',changes)
    summary={}
    for mesh in cfg['meshes']:
        summary[mesh]={}
        for mode in cfg['modes']:
            r=[x for x in changes if x['mesh']==mesh and x['mode']==mode]
            summary[mesh][mode]=dict(median_error_ratio=float(np.median([x['error_ratio'] for x in r])),
                error_increase=int(sum(x['error_ratio']>1 for x in r)),ratio_increase=int(sum(x['cancellation_change']>0 for x in r)),
                median_cancellation_change=float(np.median([x['cancellation_change'] for x in r])),
                median_component_mse_ratio=float(np.median([x['component_mse_ratio'] for x in r])),
                median_eta_ratio=float(np.median([x['eta_ratio'] for x in r])))
    summary['checks']=dict(rows_recomputed=len(rows),references=len(references),neural_checkpoints_recomputed=48,reference_derivative_extractions_recomputed=12,initial_functions_equal=6,
        common_interior_points=sorted({r['common_interior_points'] for r in rows}),
        identity_max_error=max(r['identity_error'] for r in rows),maximum_linear_residual=max(r['maximum_linear_residual'] for r in references),
        inputs=verify_inputs(out),same_error_direction_across_meshes=all((index[s,'fine',mode,1000]['greedy_error_rms']>index[s,'fine',mode,50]['greedy_error_rms'])==(index[s,'finest',mode,1000]['greedy_error_rms']>index[s,'finest',mode,50]['greedy_error_rms']) for s in cfg['seeds'] for mode in cfg['modes']))
    summary['family_sensitivity']=family_sensitivity(out)
    write_json(out/'verified_summary.json',summary)
    tables=args.report_dir/'tables';figures=args.report_dir/'figures';tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    contrasts=[]
    for seed in cfg['seeds']:
        rr={mode:next(r for r in changes if r['seed']==seed and r['mesh']=='finest' and r['mode']==mode) for mode in cfg['modes']}
        contrasts.append([seed,*[f"{rr[mode]['error_ratio']:.2f}" for mode in cfg['modes']],*[f"{rr[mode]['cancellation_change']:+.3f}" for mode in cfg['modes']]])
    table(tables/'npi_matched_derivative_pairs.tex',
        r'Matched derivative changes from 50 to 1000 evaluation steps on the finest reference. $Q=\|Gv_{1000}-p_\star\|_\nu/\|Gv_{50}-p_\star\|_\nu$. $K_m=\|\sum_i a_i\|_\nu/(\sum_i\|a_i\|_\nu^2)^{1/2}$ and $\Delta K=K_{1000}-K_{50}$. Each seed shares its reference and diagnostic points across the two CPU designs.',
        'tab:matched-derivative-pairs','rrrrr',r'Seed & $Q$, single & $Q$, full & $\Delta K$, single & $\Delta K$, full',contrasts)
    comp=[]
    for r in rows:
        if r['mesh']!='finest' or r['step'] not in [50,1000]:continue
        comp.append([r['seed'],r['mode'],r['step'],*[f"{1e3*r[k]:.2f}" for k in ['a1_rms','a2_rms','a3_rms','greedy_error_rms']],f"{1e6*r['cross_sum']:+.2f}"])
    table(tables/'npi_matched_derivative_components.tex',
        r'Individual derivative contributions for both CPU designs against the finest shared reference. RMS columns are multiplied by $10^3$; the sum of the three cross terms is multiplied by $10^6$. All components, including cancellation, enter the exact squared greedy error. The archive additionally retains steps 0 and 200 and both reference grids.',
        'tab:matched-derivative-components','rlrrrrrr',r'Seed & Design & Step & $\|a_1\|_\nu$ & $\|a_2\|_\nu$ & $\|a_3\|_\nu$ & Error & Cross sum',comp)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.2))
    colors=plt.get_cmap('tab10').colors
    for i,seed in enumerate(cfg['seeds']):
        for ax,field,base in zip(axes,['error_ratio','cancellation_change'],[1,0]):
            for mesh in cfg['meshes']:
                vals=[next(r[field] for r in changes if r['seed']==seed and r['mesh']==mesh and r['mode']==mode) for mode in cfg['modes']]
                offset=(i-2.5)*.018
                ax.plot(np.array([0,1])+offset,vals,color=colors[i],alpha=.8 if mesh=='finest' else .45,
                    marker='o' if mesh=='finest' else 's',markersize=4,markerfacecolor=colors[i] if mesh=='finest' else 'white',
                    linestyle='-' if mesh=='finest' else ':',linewidth=.8,label=str(seed) if ax is axes[0] and mesh=='finest' else None)
            ax.axhline(base,color='.7',linewidth=.5,zorder=0) if i==0 else None
    for ax in axes:ax.set_xticks([0,1],['Single network','Full correction']);ax.set_xlim(-.22,1.22);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Greedy error ratio: 1000 / 50')
    axes[1].set_ylabel(r'Change in $K$: 1000 minus 50')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=6,frameon=False,title='Shared frozen-policy seed',fontsize=8,title_fontsize=8)
    fig.tight_layout(rect=[0,.16,1,1]);fig.savefig(figures/'npi_matched_derivatives.pdf',bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
