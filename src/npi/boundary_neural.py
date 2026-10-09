"""Separate interior traces from assigned stopped-boundary payoffs.

Both representations train against the SAME finite Bellman operator, payoff,
action set and domain. No numerical reference values enter training.
"""
import copy,csv,json,time
from pathlib import Path
import numpy as np
import torch
from .neural import ValueNetwork


class BoundaryValueNetwork(ValueNetwork):
    def __init__(self,model,width=64,depth=3,boundary_mode='hard'):
        super().__init__(model,width,depth)
        if boundary_mode not in ['hard','split']:raise ValueError(boundary_mode)
        self.boundary_mode=boundary_mode

    def interior(self,x):
        m=self.market;tau,y,z=x[:,0:1],x[:,1:2],x[:,2:3]
        yn=2*(y-m.ymin)/(m.ymax-m.ymin)-1;zn=2*(z-m.zmin)/(m.zmax-m.zmin)-1
        inp=torch.cat([2*tau/m.horizon-1,yn,zn],dim=1)
        a=torch.exp((1-m.gamma1)*y);b=m.mixture*torch.exp((1-m.gamma2)*y)
        g=a/(1-m.gamma1)+b/(1-m.gamma2)
        bubble=((1-yn**2) if self.boundary_mode=='hard' else (1-yn))*(1-zn**2)
        return g+tau*bubble*(a+b)*self.net(inp)

    def forward(self,x):
        # The interior trace may differ from an assigned boundary-state value.
        m=self.market;y=x[:,1:2];z=x[:,2:3]
        g=torch.exp((1-m.gamma1)*y)/(1-m.gamma1)+m.mixture*torch.exp((1-m.gamma2)*y)/(1-m.gamma2)
        boundary=(y<=m.ymin)|(y>=m.ymax)|(z<=m.zmin)|(z>=m.zmax)
        return torch.where(boundary,g,self.interior(x))


class SparseAction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,operator,transpose):
        ctx.transpose=transpose
        return torch.sparse.mm(operator,x)

    @staticmethod
    def backward(ctx,grad):
        return torch.sparse.mm(ctx.transpose,grad),None,None


def torch_sparse(matrix,device,dtype):
    coo=matrix.tocoo()
    return torch.sparse_coo_tensor(torch.as_tensor(np.stack([coo.row,coo.col]),device=device),
                                  torch.as_tensor(coo.data,device=device,dtype=dtype),coo.shape).coalesce()


def train_boundary(system,mode,seed,settings,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    device=settings.get('device','cuda');dtype=torch.float32
    net=BoundaryValueNetwork(system.model,settings['width'],settings['depth'],mode).to(device)
    initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    optimizer=torch.optim.Adam(net.parameters(),lr=settings['lr'],fused=device=='cuda')
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,settings['steps'],eta_min=settings.get('min_lr',1e-5))
    operator=torch_sparse(system.ay,device,dtype);transpose=torch_sparse(system.ay.T,device,dtype)
    spatial=np.stack(np.meshgrid(system.y,system.z,indexing='ij'),axis=-1).reshape(-1,2)[system.ids]
    space=torch.as_tensor(spatial,device=device,dtype=dtype);n=len(space);na=len(system.actions)
    source=torch.as_tensor(system.sources,device=device,dtype=dtype)
    scale=torch.as_tensor(system.model.utility_bundle(spatial[:,0])[1],device=device,dtype=dtype)
    dt=float(system.tau[1]);times=torch.as_tensor(system.tau,device=device,dtype=dtype)
    generator=torch.Generator().manual_seed(seed+45000)
    sequence=torch.randint(1,len(times),(settings['steps'],),generator=generator).tolist()
    terminal=torch.as_tensor(system.terminal.ravel()[system.ids],device=device,dtype=dtype)
    history=[];started=time.perf_counter()
    for step,k in enumerate(sequence,start=1):
        optimizer.zero_grad(set_to_none=True)
        x=torch.cat([times[k].expand(n,1),space],dim=1)
        previous=torch.cat([times[k-1].expand(n,1),space],dim=1)
        v=net(x)[:,0];vp=terminal if k==1 else net(previous)[:,0]
        h=SparseAction.apply(v[:,None],operator,transpose).reshape(na,n)+source
        r=(v-vp)/dt-h.max(dim=0).values
        loss=torch.mean((r/scale)**2)
        if not torch.isfinite(loss):raise FloatingPointError(step)
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        optimizer.step();scheduler.step()
        if step%settings.get('log_every',200)==0 or step==1:
            row={'step':step,'sampled_time_index':k,'scaled_rms':float(loss.detach().sqrt()),'seconds':time.perf_counter()-started}
            history.append(row);print(mode,seed,json.dumps(row),flush=True)
        if step in settings.get('save_steps',[]):
            torch.save({'state_dict':net.state_dict(),'model':system.model.to_dict(),'mode':mode,'settings':settings,'seed':seed,'step':step},out/f'step_{step}.pt')
    if device=='cuda':torch.cuda.synchronize()
    meta={'model':system.model.to_dict(),'mode':mode,'method':'monotone_residual_network','seed':seed,'settings':settings,
          'grid':{'ny':len(system.y),'nz':len(system.z),'nt':len(system.tau)-1,'na':na},
          'training_seconds':time.perf_counter()-started,'checkpoint_selection':'final iterate',
          'boundary_payoff':'same stopped liquidation rule in both modes','reference_values_used_in_training':False}
    torch.save({'state_dict':net.state_dict(),'initial_state_dict':initial,'metadata':meta},out/'checkpoint.pt')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    return net,meta


def grid_values(net,system):
    spatial=np.stack(np.meshgrid(system.y,system.z,indexing='ij'),axis=-1).reshape(-1,2)[system.ids]
    device=next(net.parameters()).device;dtype=next(net.parameters()).dtype;result=[]
    with torch.no_grad():
        for tau in system.tau:
            x=torch.as_tensor(np.column_stack([np.full(len(spatial),tau),spatial]),device=device,dtype=dtype)
            result.append(net(x)[:,0].cpu().numpy().astype(float))
    values=np.asarray(result);values[0]=system.terminal.ravel()[system.ids]
    return values
