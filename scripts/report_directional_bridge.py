"""Build reports attribution tables and a refinement figure from saved audits."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.simulation import paired_ce


def read(path):
    return [{k:(v if k in ['mesh','region'] else float(v)) for k,v in row.items()}
            for row in csv.DictReader(path.open(encoding='utf-8'))]


def main():
    folder=ROOT/'results/npi_directional_bridge_20260926';man=ROOT/'reports'
    meshes=['coarse','medium','fine','finest','time','action','alternate']
    results={};regions={};source_checks=0
    for mesh in meshes:
        run=folder/mesh
        assert json.loads((run/'completion.json').read_text())['status']=='complete'
        for name,digest in json.loads((run/'input_sha256.json').read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            source_checks+=1
        rows=read(run/'summary.csv');assert len(rows)==6
        results[mesh]={(int(r['seed']),int(r['inner_steps'])):r for r in rows}
        rr=read(run/'regions.csv');assert len(rr)==30
        regions[mesh]={(int(r['seed']),int(r['inner_steps']),r['region']):r for r in rr}
        for r in rows:
            assert abs(r['candidate_integral']-r['error_integral']-r['value_difference'])<1e-9
            assert max(r[k] for k in ['maximum_local_identity_error','maximum_propagated_identity_error',
                'maximum_linear_solve_residual','partition_identity_error'])<1e-8
            final=np.load(run/f"seed{int(r['seed'])}/final_{int(r['inner_steps'])}.npz")
            fixed=np.load(run/f"seed{int(r['seed'])}/fixed_value.npz")
            assert np.max(np.abs(final['value']-fixed['value'][-1]-final['candidate_integral']+final['error_integral']))<1e-8
    model=Model.from_dict(json.loads((ROOT/'configs/confirmatory_study.json').read_text())['model'])
    mc={}
    for seed in [101,102,103]:
        base=ROOT/f'results/fixed_policy_accuracy_20260926/seed{seed}'
        a=np.load(base/'paths_1000_960.npz')['payoff'];b=np.load(base/'paths_50_960.npz')['payoff']
        ce=paired_ce(model,a,b)
        mc[seed]={'gain':-1e4*ce['relative_CE_shortfall_vs_reference'],'low':-1e4*ce['paired_mc_95_high'],'high':-1e4*ce['paired_mc_95_low']}
    changes={}
    for mesh,rr in results.items():
        changes[mesh]={seed:1e4*((1+rr[seed,1000]['ce_gain_bp']/1e4)/(1+rr[seed,50]['ce_gain_bp']/1e4)-1) for seed in [101,102,103]}
    finest=results['finest']
    correction_dominates=all(results[mesh][s,1000]['error_integral']>results[mesh][s,1000]['candidate_integral']>0 for mesh in meshes if mesh!='alternate' for s in [101,102,103])
    decline=all(x<0 for ch in changes.values() for x in ch.values())
    assert correction_dominates and decline,'Revise the narrative if the sensitivity checks change its sign.'
    ratios=[regions['finest'][s,1000,'core']['directional_error_rms_scaled']/regions['finest'][s,50,'core']['directional_error_rms_scaled'] for s in [101,102,103]]
    shares=[100*finest[s,1000]['small_q_error_integral']/finest[s,1000]['error_integral'] for s in [101,102,103]]
    maximum_identity=max(r['maximum_propagated_identity_error'] for rr in results.values() for r in rr.values())
    # Main table reports utility components, not a linearized CE decomposition.
    lines=[r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Finite-model attribution at $(t,y,z)=(0,0,0)$ on the $161\times121$ spatial grid with 480 time intervals and 321 actions. The three utility columns are multiplied by $10^4$; CE changes are computed from the policy values. This is a deterministic discrete calculation.}',
        r'\label{tab:directional-attribution}',r'\begin{tabular}{rrrrrr}',r'\toprule',
        r'Seed & Inner steps & $C$ & $E$ & $C-E$ & CE vs frozen (bp) \\',r'\midrule']
    for s in [101,102,103]:
        for j in [50,1000]:
            r=finest[s,j]
            lines.append(f"{s} & {j} & {1e4*r['candidate_integral']:.3f} & {1e4*r['error_integral']:.3f} & {1e4*r['value_difference']:.3f} & {r['ce_gain_bp']:.3f}"+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (man/'tables/npi_directional_attribution.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Discrete CE change of the 1000-step proposal relative to the 50-step proposal (bp). Joint refinements change space, time, and actions. The last three rows change only the indicated part of the medium-grid calculation.}',
        r'\label{tab:attribution-sensitivity}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'Case & $(N_y,N_z)$ & $N_\tau$ & $N_a$ & 101 & 102 & 103 \\',r'\midrule']
    names={'coarse':'Coarse','medium':'Medium','fine':'Fine','finest':'Finest','time':'Time refinement','action':'Action refinement','alternate':'Alternative stencil'}
    for mesh in meshes:
        r=results[mesh][101,50]
        lines.append(f"{names[mesh]} & $({int(r['ny'])},{int(r['nz'])})$ & {int(r['nt'])} & {int(r['na'])} & "+' & '.join(f'{changes[mesh][s]:.3f}' for s in [101,102,103])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (man/'tables/npi_attribution_sensitivity.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axs=plt.subplots(1,3,figsize=(8.2,3.0),sharey=True,constrained_layout=True)
    for ax,s in zip(axs,[101,102,103]):
        ax.plot(range(4),[changes[mesh][s] for mesh in meshes[:4]],'o-',color='#245a81',label='Finite model')
        r=mc[s];ax.errorbar([4],[r['gain']],yerr=[[r['gain']-r['low']],[r['high']-r['gain']]],fmt='s',capsize=4,color='#c86432',label='Path simulation')
        ax.axhline(0,color='0.5',lw=.7);ax.set_title(f'Seed {s}')
        ax.set_xticks(range(5),['41','81','121','161','MC']);ax.set_xlabel('Wealth nodes / MC')
        ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('CE change (bp)');axs[-1].legend(fontsize=8,loc='lower right')
    fig.savefig(man/'figures/npi_directional_refinement.pdf');fig.savefig(folder/'directional_refinement.png',dpi=180);plt.close(fig)
    macros={'BridgeErrorRatioLow':f'{min(ratios):.1f}','BridgeErrorRatioHigh':f'{max(ratios):.1f}',
        'BridgeSmallShareLow':f'{min(shares):.1f}','BridgeSmallShareHigh':f'{max(shares):.1f}'}
    (man/'npi_bridge_macros.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key,value in macros.items())+'\n',encoding='utf-8')
    summary=dict(cases=sum(len(rr) for rr in results.values()),source_hash_checks=source_checks,
        every_bounded_stencil_long_proposal_positive_candidate_advantage_and_larger_error=correction_dominates,
        every_long_short_difference_negative=decline,core_error_growth_factors=ratios,
        signed_small_exposure_correction_percent=shares,maximum_propagated_identity_error=maximum_identity,
        ce_long_short_by_mesh=changes,monte_carlo_long_short=mc,
        continuum_certified=False,floating_point_verified=False)
    (folder/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
