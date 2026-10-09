"""Verify retained outcomes and build the focused robustness tables and figure."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from npi.model import Model
from npi.simulation import paired_ce

OUT = ROOT/'results/npi_focused_strengthening_20260926'
MAN = ROOT/'reports'
LABELS = {'baseline':'Baseline', 'short_horizon':r'$T=1.5$', 'larger_mixture':r'$a=0.6$'}


def read_csv(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def group_summary(rows, continuous=False):
    gains=np.array([r['gain_bp'] for r in rows]);ratios=np.array([r['residual_ratio'] for r in rows])
    out=dict(n=len(rows), median_gain_bp=float(np.median(gains)), min_gain_bp=float(gains.min()),
             max_gain_bp=float(gains.max()), median_residual_ratio=float(np.median(ratios)),
             residual_decrease=int(np.count_nonzero(ratios<1)), negative_gain=int(np.count_nonzero(gains<0)),
             decreased_residual_and_negative_gain=int(np.count_nonzero((ratios<1)&(gains<0))))
    if continuous:
        out.update(negative_pointwise_interval=int(sum(r['high_bp']<0 for r in rows)),
                   negative_simultaneous_interval=int(sum(r['simultaneous_high_bp']<0 for r in rows)),
                   max_time_step_difference_bp=max(abs(r['gain_bp']-r['gain_480_bp']) for r in rows))
    return out


def main():
    config=json.loads((OUT/'protocol.json').read_text())
    assert config==json.loads((ROOT/'configs/focused_strengthening.json').read_text())
    source_checks=0
    for phase in ['finite','simulation']:
        for path, expected in json.loads((OUT/f'{phase}_source_sha256.json').read_text()).items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
            source_checks+=1
    # The initial source archive preserves the runner used for continuous training.
    initial=json.loads((OUT/'input_sha256.json').read_text())
    with zipfile.ZipFile(OUT/'sources_before_run.zip') as z:
        for path, expected in initial.items():
            assert hashlib.sha256(z.read(path)).hexdigest()==expected,path
    original=json.loads((ROOT/config['model_source']).read_text())
    continuous=[]; finite=[]; max_mc_difference=0.; max_saved_identity=0.; max_runtime_identity=0.
    cases=0; initial_checks=[]
    for scenario in config['scenarios']:
        name=scenario['name'];m=Model.from_dict({**original['model'],**scenario['changes']})
        for seed in config['seeds']:
            source=OUT/'continuous'/name/f'seed{seed}'
            assert json.loads((source/'completion.json').read_text())['status']=='complete'
            history={int(r['step']):r for r in read_csv(source/'history.csv')}
            dest=OUT/'simulation'/name/f'seed{seed}'
            assert json.loads((dest/'completion.json').read_text())['status']=='complete'
            rows=read_csv(dest/'summary.csv');contrasts={}
            for nsteps in config['continuous']['mc_steps']:
                payoffs={step:np.load(dest/f'paths_{step}_{nsteps}.npz')['payoff'] for step in [0,50,1000]}
                for after,before in [(50,0),(1000,0),(1000,50)]:
                    ce=paired_ce(m,payoffs[after],payoffs[before]);gain=-1e4*ce['relative_CE_shortfall_vs_reference']
                    se=1e4*ce['paired_mc_se'];z=norm.ppf(1-.05/(2*18))
                    saved=next(r for r in rows if int(r['nsteps'])==nsteps and int(r['after'])==after and int(r['before'])==before)
                    expected=dict(gain_bp=gain,se_bp=se,low_bp=gain-1.96*se,high_bp=gain+1.96*se,
                                  simultaneous_low_bp=gain-z*se,simultaneous_high_bp=gain+z*se)
                    max_mc_difference=max(max_mc_difference,max(abs(float(saved[k])-v) for k,v in expected.items()))
                    if (after,before)==(1000,50):contrasts[nsteps]=expected
            row=dict(scenario=name,seed=seed,residual_50=float(history[50]['eval_rms_full']),
                     residual_1000=float(history[1000]['eval_rms_full']),
                     residual_ratio=float(history[1000]['eval_rms_full'])/float(history[50]['eval_rms_full']),
                     gain_480_bp=contrasts[480]['gain_bp'],**contrasts[960])
            continuous.append(row)
    for grid in config['finite']['grids']:
        for scenario in grid['scenarios']:
            for seed in config['seeds']:
                dest=OUT/'finite'/grid['name']/scenario/f'seed{seed}'
                status=json.loads((dest/'completion.json').read_text());assert status['status']=='complete'
                initial_checks.extend(status['initial_value_differences'])
                fixed=np.load(dest/'frozen.npz');rows=read_csv(dest/'summary.csv')
                yy,zz=np.meshgrid(fixed['y'],fixed['z'],indexing='ij')
                ids=fixed['ids'];center=np.flatnonzero((abs(yy.ravel()[ids])<1e-12)&(abs(zz.ravel()[ids])<1e-12)).item()
                for mode in config['finite']['representations']:
                    history={int(r['step']):r for r in rows if r['mode']==mode}
                    for step,r in history.items():
                        arrays=np.load(dest/mode/f'audit_{step}.npz')
                        identity=arrays['value']-fixed['value'][-1]-arrays['candidate_integral']+arrays['error_integral']
                        max_saved_identity=max(max_saved_identity,float(np.max(abs(identity))))
                        max_runtime_identity=max(max_runtime_identity,float(r['maximum_identity_error']))
                        np.testing.assert_allclose(arrays['value'][center],float(r['proposal_value']),atol=1e-12)
                        np.testing.assert_allclose(arrays['candidate_integral'][center],float(r['candidate_integral']),atol=1e-12)
                        np.testing.assert_allclose(arrays['error_integral'][center],float(r['error_integral']),atol=1e-12)
                        assert float(r['minimum_candidate_advantage'])>-1e-8
                        cases+=1
                    a,b=history[50],history[1000]
                    row=dict(mesh=grid['name'],scenario=scenario,seed=seed,mode=mode,
                        gain_bp=1e4*(float(b['proposal_ce'])/float(a['proposal_ce'])-1),
                        residual_ratio=float(b['eval_rms_full'])/float(a['eval_rms_full']),
                        directional_core_ratio=float(b['directional_rms_core'])/max(float(a['directional_rms_core']),1e-30),
                        candidate_change=float(b['candidate_integral'])-float(a['candidate_integral']),
                        error_change=float(b['error_integral'])-float(a['error_integral']),
                        value_change=float(b['proposal_value'])-float(a['proposal_value']),
                        gain_50_vs_frozen_bp=float(a['gain_vs_frozen_bp']),gain_1000_vs_frozen_bp=float(b['gain_vs_frozen_bp']))
                    assert abs(row['value_change']-row['candidate_change']+row['error_change'])<1e-8
                    finite.append(row)
    assert len(continuous)==18 and len(finite)==48 and cases==144
    assert max(initial_checks)==0 and max_mc_difference<1e-10 and max_runtime_identity<1e-8
    write_csv(OUT/'continuous_contrasts.csv',continuous);write_csv(OUT/'finite_contrasts.csv',finite)
    cs={s['name']:group_summary([r for r in continuous if r['scenario']==s['name']],True) for s in config['scenarios']}
    fs={}
    for grid in config['finite']['grids']:
        for scenario in grid['scenarios']:
            for mode in config['finite']['representations']:
                key=f"{grid['name']}/{scenario}/{mode}"
                fs[key]=group_summary([r for r in finite if r['mesh']==grid['name'] and r['scenario']==scenario and r['mode']==mode])
    summary=dict(continuous_cases=18,independent_seed_blocks=6,finite_training_runs=48,finite_audited_checkpoints=cases,
        continuous=cs,finite=fs,maximum_mc_recompute_difference_bp=max_mc_difference,
        maximum_saved_identity_error=max_saved_identity,maximum_all_layer_identity_error=max_runtime_identity,
        maximum_initial_function_difference=max(initial_checks),source_hash_checks=source_checks,
        original_archived_source_files=len(initial),continuum_certified=False,
        interpretation='Prespecified contrasts, all cases retained; finite representation intervention is not a continuous causal certificate')
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

    lines=[r'\begin{table}[htbp]',r'\centering\small',
           r'\caption{New-seed frozen-policy checks. $R$ is the full-domain evaluation RMS at 1000 steps divided by that at 50 steps. CE differences compare the long and short proposals at 960 simulation steps. The last two columns count negative pointwise and Bonferroni-adjusted intervals across the 18 contrasts. Each row uses the same six seed blocks.}',
           r'\label{tab:focused-continuous}',r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Setting & Median $R$ & Median CE (bp) & Pointwise $<0$ & Adjusted $<0$ \\',r'\midrule']
    for scenario,s in cs.items():
        lines.append(f"{LABELS[scenario]} & {s['median_residual_ratio']:.3f} & {s['median_gain_bp']:.3f} & {s['negative_pointwise_interval']}/6 & {s['negative_simultaneous_interval']}/6 \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (MAN/'tables/npi_focused_continuous.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{table}[htbp]',r'\centering\small',
           r'\caption{Matched finite evaluation with the lower-face constraint retained (hard) or released (split). All rows use six seeds and compare steps 1000 and 50. $R$ is the full-grid residual ratio; negative changes are deterministic finite-model CE differences, not continuous confidence statements.}',
           r'\label{tab:focused-representation}',r'\begin{tabular}{lllrrr}',r'\toprule',
           r'Grid & Setting & Correction & Median $R$ & Median CE (bp) & Negative \\',r'\midrule']
    for key,s in fs.items():
        grid,scenario,mode=key.split('/')
        lines.append(f"{grid.capitalize()} & {LABELS[scenario]} & {mode.capitalize()} & {s['median_residual_ratio']:.3f} & {s['median_gain_bp']:.3f} & {s['negative_gain']}/6 \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (MAN/'tables/npi_focused_representation.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines=[r'\begin{table}[htbp]',r'\centering\small',
           r'\caption{All paired representation contrasts. Entries compare step 1000 with step 50. $R$ is the full-grid evaluation-residual ratio. Each hard/split pair starts from the same value and frozen policy and uses the same sampled time indices.}',
           r'\label{tab:focused-all-pairs}',r'\begin{tabular}{llrrrrr}',r'\toprule',
           r'Grid & Setting & Seed & $R_{\rm hard}$ & $R_{\rm split}$ & CE hard (bp) & CE split (bp) \\',r'\midrule']
    for grid in config['finite']['grids']:
        for scenario in grid['scenarios']:
            for seed in config['seeds']:
                pair={r['mode']:r for r in finite if r['mesh']==grid['name'] and r['scenario']==scenario and r['seed']==seed}
                a,b=pair['hard'],pair['split']
                lines.append(f"{grid['name'].capitalize()} & {LABELS[scenario]} & {seed} & {a['residual_ratio']:.3f} & {b['residual_ratio']:.3f} & {a['gain_bp']:.3f} & {b['gain_bp']:.3f} \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (MAN/'tables/npi_focused_all_pairs.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(6.6,2.65),sharey=True)
    for ax,scenario in zip(axes,config['scenarios']):
        rows=[r for r in continuous if r['scenario']==scenario['name']]
        x=np.arange(6);y=np.array([r['gain_bp'] for r in rows]);err=np.array([1.96*r['se_bp'] for r in rows])
        ax.errorbar(x,y,yerr=err,fmt='o',color='#245b80',capsize=3,ms=4)
        ax.axhline(0,color='0.35',lw=.8,ls='--');ax.set_xticks(x,config['seeds'],rotation=45)
        ax.set_title(LABELS[scenario['name']]);ax.set_xlabel('Training seed');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('CE change (bp)')
    fig.tight_layout();fig.savefig(MAN/'figures/npi_focused_robustness.pdf');fig.savefig(OUT/'robustness.png',dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
