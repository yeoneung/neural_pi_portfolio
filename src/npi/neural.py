"""Shared PINN and PINN-PI implementation with matched training machinery."""
import copy
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .model import Model


class ValueNetwork(nn.Module):
    """Hard terminal AND lateral liquidation data; no CRRA curvature prior.

    v = g + tau * spatial_bubble * g_y * network.
    Scaling by the given terminal derivative improves numerical conditioning;
    no homogeneity or fixed relative risk aversion is imposed on v.
    """
    def __init__(self, model, width=64, depth=3):
        super().__init__()
        self.market=model
        layers=[]
        for j in range(depth):
            layers.extend([nn.Linear(3 if j==0 else width,width),nn.Tanh()])
        layers.append(nn.Linear(width,1))
        self.net=nn.Sequential(*layers)
        for module in self.net:
            if isinstance(module,nn.Linear):
                nn.init.xavier_normal_(module.weight);nn.init.zeros_(module.bias)
        nn.init.zeros_(self.net[-1].weight)

    def forward(self,x):
        m=self.market
        tau,y,z=x[:,0:1],x[:,1:2],x[:,2:3]
        yn=2*(y-m.ymin)/(m.ymax-m.ymin)-1
        zn=2*(z-m.zmin)/(m.zmax-m.zmin)-1
        inp=torch.cat([2*tau/m.horizon-1,yn,zn],dim=1)
        a=torch.exp((1-m.gamma1)*y);b=m.mixture*torch.exp((1-m.gamma2)*y)
        g=a/(1-m.gamma1)+b/(1-m.gamma2)
        bubble=(1-yn**2)*(1-zn**2)
        return g+tau*bubble*(a+b)*self.net(inp)


def derivatives(network,x):
    x=x.detach().clone().requires_grad_(True)
    v=network(x)
    dv=torch.autograd.grad(v.sum(),x,create_graph=True)[0]
    dy=torch.autograd.grad(dv[:,1].sum(),x,create_graph=True)[0]
    dz=torch.autograd.grad(dv[:,2].sum(),x,create_graph=True)[0]
    return {'x':x,'v':v[:,0],'vt':dv[:,0],'vy':dv[:,1],'vz':dv[:,2],
            'vyy':dy[:,1],'vyz':dy[:,2],'vzz':dz[:,2]}


def greedy(model,d):
    linear=model.phi(d['x'][:,2])*d['vy']+model.rho*model.sigma_z*d['vyz']
    quad=d['vyy']-d['vy']
    safe=torch.where(quad < -1e-12,quad,-torch.ones_like(quad))
    interior=torch.clamp(-linear/safe,model.qmin,model.qmax)
    lo=linear*model.qmin+0.5*quad*model.qmin**2
    hi=linear*model.qmax+0.5*quad*model.qmax**2
    boundary=torch.where(hi>lo,torch.full_like(hi,model.qmax),torch.full_like(lo,model.qmin))
    return torch.where(quad < -1e-12,interior,boundary)


def residual(model,d,q):
    z=d['x'][:,2]
    return (-d['vt']+(model.rate+q*model.phi(z)-0.5*q*q)*d['vy']
            +model.kappa*(model.zbar-z)*d['vz']+0.5*q*q*d['vyy']
            +model.rho*q*model.sigma_z*d['vyz']+0.5*model.sigma_z**2*d['vzz'])


def sample(model,n,generator,device,dtype):
    u=torch.rand((n,3),generator=generator,dtype=dtype)
    lower=torch.tensor([0,model.ymin,model.zmin],dtype=dtype)
    span=torch.tensor([model.horizon,model.ymax-model.ymin,model.zmax-model.zmin],dtype=dtype)
    return (lower+u*span).to(device)


def predict(network,points,chunk=2048):
    device=next(network.parameters()).device;dtype=next(network.parameters()).dtype
    fields={k:[] for k in ['v','q','vy','vyy','vyz','residual','curvature']}
    network.eval()
    for first in range(0,len(points),chunk):
        x=torch.as_tensor(points[first:first+chunk],device=device,dtype=dtype)
        d=derivatives(network,x);q=greedy(network.market,d)
        data={**d,'q':q,'residual':residual(network.market,d,q),'curvature':d['vy']-d['vyy']}
        for key in fields:fields[key].append(data[key].detach().cpu().numpy())
    return {k:np.concatenate(v) for k,v in fields.items()}


def train(model,settings,method,seed,out_dir):
    if method not in ['pinn','pi']:
        raise ValueError(method)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(settings.get('cpu_threads',4)))
    device=settings.get('device','cuda' if torch.cuda.is_available() else 'cpu')
    dtype=getattr(torch,settings.get('dtype','float32'))
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    network=ValueNetwork(model,settings['width'],settings['depth']).to(device=device,dtype=dtype)
    optimizer=torch.optim.Adam(network.parameters(),lr=settings['lr'],
                               fused=bool(settings.get('fused_adam',False)) if str(device).startswith('cuda') else False)
    planned_steps=settings['steps_per_outer']*settings['outer_iterations']
    scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=planned_steps,
               eta_min=settings.get('min_lr',settings['lr']/100))
               if settings.get('lr_schedule','constant')=='cosine' else None)
    # Independent streams guarantee identical collocation draws across methods.
    stream=torch.Generator().manual_seed(seed+17000)
    valstream=torch.Generator().manual_seed(seed+29000)
    validation=sample(model,settings['validation_points'],valstream,device,dtype)
    steps_per_outer=settings['steps_per_outer'];nouter=settings['outer_iterations']
    refresh=settings['resample_every'];rows=[];total_steps=0;policy_points=0
    initial_state={k:v.detach().cpu().clone() for k,v in network.state_dict().items()}
    start=time.perf_counter()
    for outer in range(nouter):
        frozen=copy.deepcopy(network).eval() if method=='pi' else None
        if frozen is not None:
            for p in frozen.parameters():p.requires_grad_(False)
        for inner in range(steps_per_outer):
            if inner % refresh==0:
                x=sample(model,settings['batch_size'],stream,device,dtype)
                if frozen is not None:
                    qfixed=greedy(model,derivatives(frozen,x)).detach()
                    policy_points+=len(x)
            optimizer.zero_grad(set_to_none=True)
            d=derivatives(network,x)
            # Danskin's envelope derivative: q* is held fixed in backprop.
            # This is the exact gradient of the maximized residual a.e.
            q=qfixed if method=='pi' else greedy(model,d).detach()
            raw=residual(model,d,q)
            y=x[:,1]
            scale=torch.exp((1-model.gamma1)*y)+model.mixture*torch.exp((1-model.gamma2)*y)
            pde=torch.mean((raw/scale)**2)
            shape=torch.mean(torch.relu(-d['vy']/scale)**2+torch.relu((d['vyy']-d['vy'])/scale)**2)
            loss=pde+settings.get('shape_weight',1.0)*shape
            if not torch.isfinite(loss):raise FloatingPointError(f'{method} nonfinite loss at {total_steps}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),settings.get('gradient_clip',1.0))
            optimizer.step();total_steps+=1
            if scheduler is not None:scheduler.step()
            if total_steps in settings.get('save_steps',[]):
                torch.save({'state_dict':network.state_dict(),'model':model.to_dict(),
                            'settings':settings,'method':method,'seed':seed,'step':total_steps},
                           out_dir/f'step_{total_steps}.pt')
        vd=derivatives(network,validation)
        vq=greedy(model,vd)
        vr=residual(model,vd,vq)
        vscale=torch.exp((1-model.gamma1)*validation[:,1])+model.mixture*torch.exp((1-model.gamma2)*validation[:,1])
        if frozen is not None:
            oldq=greedy(model,derivatives(frozen,validation)).detach()
            evalr=residual(model,vd,oldq)
            gap=vr-evalr
        else:
            evalr=vr;gap=torch.zeros_like(vr)
        row={'outer':outer+1,'steps':total_steps,'train_loss':float(loss.detach()),
             'validation_hjb_rms_scaled':float(torch.mean((vr/vscale)**2).sqrt().detach()),
             'validation_hjb_max_raw':float(vr.abs().max().detach()),
             'validation_eval_max_raw':float(evalr.abs().max().detach()),
             'validation_update_gap_max_raw':float(gap.max().detach()),
             'validation_eval_rms_scaled':float(torch.mean((evalr/vscale)**2).sqrt().detach()),
             'validation_update_gap_rms_scaled':float(torch.mean((gap/vscale)**2).sqrt().detach()),
             'learning_rate':optimizer.param_groups[0]['lr'],
             'curvature_violation_fraction':float(((vd['vy']-vd['vyy'])<=0).float().mean().detach()),
             'elapsed_seconds':time.perf_counter()-start}
        rows.append(row)
        print(f"{method} seed={seed} outer={outer+1}/{nouter} scaled HJB={row['validation_hjb_rms_scaled']:.3e}",flush=True)
        with (out_dir/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    wall=time.perf_counter()-start
    meta={'method':method,'seed':seed,'model':model.to_dict(),'settings':settings,
          'total_optimizer_steps':total_steps,'training_residual_points':total_steps*settings['batch_size'],
          'extra_frozen_policy_points':policy_points,'wall_seconds_including_diagnostics':wall,
          'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if str(device).startswith('cuda') else None,
          'checkpoint_selection':'final iterate for BOTH methods','lr_schedule':settings.get('lr_schedule','constant'),
          'curvature_identity_penalty':0,'status':settings.get('status','pilot, not publication evidence')}
    torch.save({'state_dict':network.state_dict(),'initial_state_dict':initial_state,'metadata':meta},out_dir/'checkpoint.pt')
    (out_dir/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return network,meta
