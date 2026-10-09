"""Generate NPI reports tables/figures directly from completed experiments."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from npi.model import Model
from npi.discrete_evaluation import certainty_equivalent
from npi.simulation import paired_ce

FIXED=ROOT/'results/fixed_policy_accuracy_20260926'
STUDY=ROOT/'results/npi_walltime_20260926'
AUDIT=ROOT/'results/npi_walltime_audit_20260926'
EARLY=ROOT/'results/npi_iteration_diagnostic_20260926/independent_early_update_check'
MAN=ROOT/'reports'
NAMES={'pinn':'Direct PINN','pi200':'NPI-200','pi1000':'NPI-1000','damped':'Half-damped','forcing':'Residual forcing','directional':'Directional test'}


def csv_rows(path):return list(csv.DictReader(path.open(encoding='utf-8')))


def recorded_input(name):
    """Resolve historical Windows manifests after relocating the research files."""
    normalized=str(name).replace('\\','/')
    for marker in ['/results/', '/scripts/', '/src/', '/configs/']:
        if marker in normalized and ':' in normalized:
            return ROOT / (marker.lstrip('/') + normalized.split(marker, 1)[1])
    path=Path(normalized)
    return path if path.is_absolute() else ROOT/path


def table(name,caption,columns,head,rows):
    text='\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\n'
    text+='\\begin{tabular}{'+columns+'}\n\\toprule\n'+' & '.join(head)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (MAN/'tables'/name).write_text(text,encoding='utf-8')


def interval(x):return rf"${x['mean']:.3f}\ [{x['low']:.3f}, {x['high']:.3f}]$"


def main():
    for folder in [FIXED,STUDY,AUDIT,EARLY]:
        assert json.loads((folder/'completion.json').read_text())['status']=='complete'
    for folder,filename,relative in [(FIXED,'input_sha256.json',False),(STUDY,'source_sha256.json',True),(AUDIT,'input_sha256.json',False),(EARLY,'input_sha256.json',False)]:
        hashes=json.loads((folder/filename).read_text())
        bad=[f for f,s in hashes.items() if hashlib.sha256(recorded_input(f).read_bytes()).hexdigest()!=s]
        assert not bad,(str(folder),bad)
    cfg=json.loads((STUDY/'protocol.json').read_text());m=Model.from_dict(cfg['model'])
    stats=json.loads((STUDY/'status.json').read_text());performance=csv_rows(AUDIT/'policy_performance.csv')
    assert len(stats)==18 and len(performance)==72
    fixed=csv_rows(FIXED/'policy_changes.csv');hist=csv_rows(FIXED/'evaluation_history.csv')
    early=csv_rows(EARLY/'paired_changes.csv')
    er=[[r['seed'],f"{float(r['ce_gain_bp']):.3f}",f"[{float(r['ci_low_bp']):.3f}, {float(r['ci_high_bp']):.3f}]"] for r in early if r['time_steps']=='960']
    table('npi_early_update.tex','First-to-second NPI policy update: fresh 65536 paths, 960 time steps. Intervals are paired conditional 95\\% Monte Carlo intervals.','rrl',['Seed','CE change (bp)','95\\% interval (bp)'],er)
    shutil.copy2(EARLY/'early_update_check.pdf',MAN/'figures/npi_early_update.pdf')
    fr=[];longshort=[]
    for seed in [101,102,103]:
        for steps in [50,200,1000]:
            r=next(r for r in fixed if int(r['seed'])==seed and int(r['inner_steps'])==steps and r['time_steps']=='960')
            h=next(r for r in hist if int(r['seed'])==seed and int(r['inner_steps'])==steps and r['region']=='full')
            fr.append([seed,steps,f"{float(h['eval_rms']):.4f}",f"{float(r['gain_bp']):.3f}",f"[{float(r['low_bp']):.3f}, {float(r['high_bp']):.3f}]"])
        a=np.load(FIXED/f'seed{seed}/paths_1000_960.npz')['payoff'];b=np.load(FIXED/f'seed{seed}/paths_50_960.npz')['payoff']
        ce=paired_ce(m,a,b);longshort.append(dict(seed=seed,gain_bp=-1e4*ce['relative_CE_shortfall_vs_reference'],low=-1e4*ce['paired_mc_95_high'],high=-1e4*ce['paired_mc_95_low']))
        r50=next(float(r['eval_rms']) for r in hist if int(r['seed'])==seed and int(r['inner_steps'])==50 and r['region']=='full')
        r1000=next(float(r['eval_rms']) for r in hist if int(r['seed'])==seed and int(r['inner_steps'])==1000 and r['region']=='full')
        assert r1000<r50
    table('npi_fixed_evaluation.tex','Strictly frozen-policy intervention. Evaluation residuals use independent full-domain points. CE changes compare the proposed greedy policy with the frozen policy on 32768 common paths at 960 time steps.','rrrll',['Seed','Inner steps','Eval. RMS','CE change (bp)','95\\% MC interval'],fr)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for i,seed in enumerate([101,102,103]):
        color=f'C{i}'
        for region,style in [('full','-'),('core','--')]:
            rr=[r for r in hist if int(r['seed'])==seed and r['region']==region]
            axs[0].plot([int(r['inner_steps']) for r in rr],[float(r['eval_rms']) for r in rr],style,color=color,label=f'{seed}, {region}')
        rr=[r for r in fixed if int(r['seed'])==seed and r['time_steps']=='960']
        y=np.array([float(r['gain_bp']) for r in rr]);lo=np.array([float(r['low_bp']) for r in rr]);hi=np.array([float(r['high_bp']) for r in rr])
        axs[1].errorbar(np.array([0,1,2])+(i-1)*.09,y,yerr=np.vstack([y-lo,hi-y]),fmt='o-',capsize=3,color=color,label=f'Seed {seed}')
    axs[0].set(xlabel='Inner optimizer updates',ylabel='Scaled evaluation residual RMS');axs[0].legend(fontsize=8,ncol=2)
    axs[1].axhline(0,color='gray',lw=.8);axs[1].set(xticks=[0,1,2],xticklabels=['50','200','1000'],xlabel='Inner optimizer updates',ylabel='CE change vs frozen policy (bp)');axs[1].legend(fontsize=9)
    fig.savefig(MAN/'figures/npi_fixed_evaluation.pdf');fig.savefig(FIXED/'evaluation_length.png',dpi=180);plt.close(fig)

    pay={};bases={s:np.load(AUDIT/f'myopic_{s}.npz')['payoff'] for s in [480,960]}
    max_recompute=0.
    for r in performance:
        method=r['method'];seed=int(r['seed']);budget=int(float(r['nominal_seconds']));steps=int(r['time_steps'])
        arr=np.load(AUDIT/f'{method}_seed{seed}_budget_{budget}_paths_{steps}.npz')['payoff']
        assert len(arr)==cfg['mc_paths'] and np.isfinite(arr).all()
        ce=paired_ce(m,arr,bases[steps]);max_recompute=max(max_recompute,abs(float(r['gain_bp'])+1e4*ce['relative_CE_shortfall_vs_reference']))
        pay[method,seed,budget,steps]=arr
    assert max_recompute<1e-9
    # Resample training seeds and common path groups jointly, never pretend
    # that policies sharing Brownian paths provide independent MC samples.
    rng=np.random.default_rng(9264004);boot=2000;groups=128
    si=rng.integers(0,3,(boot,3));gi=rng.integers(0,groups,(boot,groups))
    def joint(a,b):
        aa=a.reshape(3,groups,-1).mean(axis=2);bb=b.reshape(3,groups,-1).mean(axis=2)
        ja=aa[si[:,:,None],gi[:,None,:]].mean(axis=2);jb=bb[si[:,:,None],gi[:,None,:]].mean(axis=2)
        samples=(1e4*(certainty_equivalent(m,ja)/certainty_equivalent(m,jb)-1)).mean(axis=1)
        point=float(np.mean(1e4*(certainty_equivalent(m,a.mean(axis=1))/certainty_equivalent(m,b.mean(axis=1))-1)))
        low,high=np.quantile(samples,[.025,.975]);return dict(mean=point,low=float(low),high=float(high))
    aggregate=[];tr=[];wr=[]
    for method in cfg['methods']:
        values=np.stack([pay[method,s,30,960] for s in cfg['seeds']])
        base=np.stack([bases[960]]*3);fixedvals=np.stack([pay['pi200',s,30,960] for s in cfg['seeds']])
        pinnvals=np.stack([pay['pinn',s,30,960] for s in cfg['seeds']])
        gain=joint(values,base);vsfixed=joint(values,fixedvals);vspinn=joint(values,pinnvals)
        successes=[]
        for seed in cfg['seeds']:
            rr=sorted([r for r in performance if r['method']==method and int(r['seed'])==seed and r['time_steps']=='480'],key=lambda r:float(r['nominal_seconds']))
            passed=next((float(r['elapsed_seconds']) for r in rr if float(r['gain_bp'])>=10),None)
            if passed is not None:successes.append(passed)
        sr=[r for r in stats if r['method']==method]
        item=dict(method=method,gain_vs_myopic=gain,gain_vs_pi200=vsfixed,gain_vs_pinn=vspinn,
            mean_steps=float(np.mean([r['steps'] for r in sr])),mean_seconds=float(np.mean([r['wall_seconds'] for r in sr])),
            mean_refreshes=float(np.mean([r['accepted_updates'] for r in sr])),cap_updates=sum(r['forced_updates'] for r in sr),
            total_updates=sum(r['accepted_updates'] for r in sr),target_reached=len(successes),target_median_seconds=float(np.median(successes)) if successes else None)
        aggregate.append(item)
        target_time=f"{item['target_median_seconds']:.2f}" if successes else '--'
        tr.append([NAMES[method],f"{item['mean_steps']:.0f}",f"{item['mean_seconds']:.3f}",f"{item['mean_refreshes']:.1f}",f"{item['cap_updates']}/{item['total_updates']}" if item['total_updates'] else '--',target_time])
        wr.append([NAMES[method],interval(gain),interval(vsfixed),f"{len(successes)}/3"])
    table('npi_walltime.tex','Equal nominal 30-second training budget, three new seeds. Counts and seconds are seed means; cap updates are pooled counts. Target time is the median observed checkpoint time exceeding the 10-bp point-estimate target. Actual training time includes update decisions and saving.','lrrrrr',['Method','Steps','Seconds','Refreshes','Cap/total','Target (s)'],tr)
    table('npi_budget_comparison.tex','Endpoint performance at 960 simulation steps: mean CE gains and joint seed/path-group bootstrap 95\\% intervals (bp). Target counts use the first stored 480-step point estimate exceeding 10 bp over myopic; they are not certified stopping decisions.','llll',['Method','Gain vs myopic','Gain vs NPI-200','Target'],wr)
    fig,ax=plt.subplots(figsize=(7.5,4),constrained_layout=True)
    for method in cfg['methods']:
        vals=[np.mean([float(r['gain_bp']) for r in performance if r['method']==method and float(r['nominal_seconds'])==b and r['time_steps']=='480']) for b in [10,20,30]]
        ax.plot([10,20,30],vals,'o-',label=NAMES[method])
    ax.axhline(10,color='gray',ls='--',lw=.8);ax.set(xlabel='Nominal training wall time (s)',ylabel='Mean CE gain over myopic (bp)');ax.legend(fontsize=9,ncol=2)
    fig.savefig(MAN/'figures/npi_walltime.pdf');fig.savefig(AUDIT/'walltime_comparison.png',dpi=180);plt.close(fig)
    direction=next(r for r in aggregate if r['method']=='directional');comparison=direction['gain_vs_pi200']
    forcing=next(r for r in aggregate if r['method']=='forcing')
    verdict=('The directional test has a positive joint interval against NPI-200 in this small comparison.' if comparison['low']>0 else
             'The directional test is worse than NPI-200 under this protocol.' if comparison['high']<0 else
             'The directional test does not establish an endpoint advantage over NPI-200 under this protocol.')
    text=(f"The estimated endpoint CE difference of the directional rule from NPI-200 is {comparison['mean']:.3f} bp, with a joint seed-and-path bootstrap interval $[{comparison['low']:.3f},{comparison['high']:.3f}]$ bp. {verdict} "
          f"It makes {direction['total_updates']} policy refreshes across the three runs, of which {direction['cap_updates']} are forced by the inner cap. "
          f"Residual forcing reaches its cap in {forcing['cap_updates']} of {forcing['total_updates']} refreshes. Both adaptive sequences therefore include updates that do not satisfy their respective acceptance tests.\n")
    (MAN/'npi_results_interpretation.tex').write_text(text,encoding='utf-8')
    (MAN/'npi_macros.tex').write_text('% Generated by report_npi_refocus.py\n'+f'\\newcommand{{\\NpiDirectionGain}}{{{comparison["mean"]:.3f}}}\n',encoding='utf-8')
    summary=dict(status='complete',new_fixed_evaluations=3,new_walltime_runs=18,policy_bank_audits=len(performance),
        fixed_long_vs_short=longshort,walltime=aggregate,
        actual_wall_time_range=[min(r['wall_seconds'] for r in stats),max(r['wall_seconds'] for r in stats)],
        max_policy_interpolation_rms=max(float(r['interpolation_rms_core']) for r in performance),
        verification=dict(source_and_checkpoint_hashes_match=True,raw_payoff_recomputed_max_abs_bp=max_recompute),
        directional_verdict=verdict)
    (ROOT/'results/npi_refocus_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
