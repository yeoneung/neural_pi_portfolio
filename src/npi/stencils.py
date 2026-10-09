"""Positive generators with uniformly shrinking physical diffusion stencils.

bounded: rational mixed stencil with a cap; minimal nonnegative diagonal
perturbation is allowed and recorded. semi_lagrangian: bilinear interpolation
of symmetric covariance-factor jumps, with length parameter sqrt(max(h)).
Both approximate drift by the same positive upwind generator. Boundary jumps
are clamped explicitly; interior consistency is distinct from boundary error.
"""
import numpy as np
from scipy import sparse


def build(model,y,z,actions,kind='bounded'):
    ny,nz=len(y),len(z);hy=y[1]-y[0];hz=z[1]-z[0];h=max(hy,hz)
    ii,jj=np.meshgrid(np.arange(1,ny-1),np.arange(1,nz-1),indexing='ij')
    ii,jj=ii.ravel(),jj.ravel();n=len(ii);ids=ii*nz+jj
    cap=max(1,int(np.ceil(1/np.sqrt(h))))
    matrices=[];max_added=0.;max_radius=0.;max_offset=1
    for q in actions:
        rowlist=[];collist=[];vallist=[];diagonal=np.zeros(n)
        def add_target(ti,tj,rate):
            rate=np.broadcast_to(np.maximum(rate,0),(n,))
            if not np.any(rate):return
            rowlist.append(np.arange(n));collist.append(np.clip(ti,0,ny-1)*nz+np.clip(tj,0,nz-1));vallist.append(rate)
            diagonal[:]-=rate
        def add(di,dj,rate):add_target(ii+di,jj+dj,rate)
        if kind=='bounded':
            cross=model.rho*q*model.sigma_z
            rem_y=q*q;rem_z=model.sigma_z**2
            if abs(cross)>1e-15:
                candidates=[]
                for iy in range(1,cap+1):
                    for iz in range(1,cap+1):
                        slope=iy*hy/(iz*hz)
                        ry=q*q-abs(cross)*slope;rz=model.sigma_z**2-abs(cross)/slope
                        extra=max(-ry,0)+max(-rz,0)
                        candidates.append((round(extra,14),max(iy,iz),iy+iz,iy,iz,ry,rz,extra))
                _,_,_,iy,iz,rem_y,rem_z,extra=min(candidates)
                max_added=max(max_added,extra);max_radius=max(max_radius,np.hypot(iy*hy,iz*hz))
                max_offset=max(max_offset,iy,iz)
                sign=1 if cross>0 else -1;rate=abs(cross)/(2*iy*iz*hy*hz)
                add(iy,sign*iz,rate);add(-iy,-sign*iz,rate)
            add(1,0,max(rem_y,0)/(2*hy*hy));add(-1,0,max(rem_y,0)/(2*hy*hy))
            add(0,1,max(rem_z,0)/(2*hz*hz));add(0,-1,max(rem_z,0)/(2*hz*hz))
        elif kind=='semi_lagrangian':
            delta=np.sqrt(h)
            for by,bz in [(q,model.rho*model.sigma_z),(0.,np.sqrt(1-model.rho**2)*model.sigma_z)]:
                max_radius=max(max_radius,delta*np.hypot(by,bz)+np.hypot(hy,hz))
                max_offset=max(max_offset,int(np.ceil(abs(delta*by)/hy)),int(np.ceil(abs(delta*bz)/hz)))
                for sign in [-1,1]:
                    fy=np.clip(ii+sign*delta*by/hy,0,ny-1);fz=np.clip(jj+sign*delta*bz/hz,0,nz-1)
                    iy=np.minimum(np.floor(fy).astype(int),ny-2);iz=np.minimum(np.floor(fz).astype(int),nz-2)
                    wy=fy-iy;wz=fz-iz
                    for dy,dz,w in [(0,0,(1-wy)*(1-wz)),(1,0,wy*(1-wz)),(0,1,(1-wy)*wz),(1,1,wy*wz)]:
                        add_target(iy+dy,iz+dz,w/(2*delta**2))
        else:raise ValueError(kind)
        by=model.rate+q*model.phi(z[jj])-.5*q*q;bz=model.kappa*(model.zbar-z[jj])
        add(1,0,np.maximum(by,0)/hy);add(-1,0,np.maximum(-by,0)/hy)
        add(0,1,np.maximum(bz,0)/hz);add(0,-1,np.maximum(-bz,0)/hz)
        rowlist.append(np.arange(n));collist.append(ids);vallist.append(diagonal)
        matrix=sparse.csr_matrix((np.concatenate(vallist),(np.concatenate(rowlist),np.concatenate(collist))),shape=(n,ny*nz))
        matrix.eliminate_zeros();matrices.append(matrix)
    return sparse.vstack(matrices,format='csr'),ids,max_offset,{
        'generator_kind':kind,'direction_cap':cap,'max_added_diagonal_variance_sum':max_added if kind=='bounded' else None,
        'max_diffusion_jump_radius_bound':float(max_radius),'spatial_step_max':float(h),
        'uniform_interior_consistency_design':True,'boundary_convergence_certified':False}
