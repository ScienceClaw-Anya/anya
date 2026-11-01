"""Six-state Epileptor / zero-delay Heun kernel.

Equations and noise/integration conventions adapted from TVB library 2.10.0,
models/epileptor.py, noise.py and integrators.py. Copyright (c) 2012-2025,
Baycrest Centre for Geriatric Care and others. GPL-3.0-or-later; see LICENSE.
Changes: sparse compiled loop, explicit paired seeds, whole-parcel removal,
and a stream of one-time-unit averages. Not a clinical intervention model.
"""
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix


def normalized_weights(raw):
    w=np.array(raw,dtype=float,copy=True)
    if w.ndim!=2 or w.shape[0]!=w.shape[1] or not np.isfinite(w).all() or (w<0).any():
        raise ValueError('Expected square finite nonnegative connectivity')
    np.fill_diagonal(w,0)
    scale=float(w.max())
    return w/scale if scale else w, scale


def resect(w, action):
    """W[target,source]; remove incident edges without renormalization."""
    out=np.array(w,copy=True)
    action=np.asarray(action,dtype=int)
    if len(np.unique(action))!=len(action) or (action<0).any() or (action>=len(w)).any():
        raise ValueError('Invalid resection indices')
    out[action,:]=0;out[:,action]=0
    return out


@njit(cache=True)
def coupling(y, indptr, indices, data, factor):
    c=np.zeros((2,y.shape[1]))
    for i in range(y.shape[1]):
        for e in range(indptr[i],indptr[i+1]):
            j=indices[e]
            c[0,i]+=factor*data[e]*(y[0,j]-y[0,i])
            c[1,i]+=factor*data[e]*(y[3,j]-y[3,i])
    return c


@njit(cache=True)
def rhs(y,c,p):
    out=np.empty_like(y)
    for i in range(y.shape[1]):
        x,y1,z,x2,y2,g=y[:,i]
        x0,I1,I2,slope,r,Ks,Kf,Kvf=p[:,i]
        f=-x*x+3*x if x<0 else slope-x2+0.6*(z-4)**2
        out[0,i]=y1-z+I1+Kvf*c[0,i]+f*x
        out[1,i]=1-5*x*x-y1
        h=4*(x-x0)-(0.1*z**7 if z<0 else 0)
        out[2,i]=r*(h-z+Ks*c[0,i])
        out[3,i]=-y2+x2-x2**3+I2+2*g-0.3*(z-3.5)+Kf*c[1,i]
        out[4,i]=(-y2+(6*(x2+0.25) if x2>=-0.25 else 0))/10
        out[5,i]=-0.01*(g-0.1*x)
    return out


@njit(cache=True)
def _run(p,indptr,indices,data,factor,initial,nsig,dt,steps,stride,seed,removed,ntau):
    np.random.seed(seed)
    n=p.shape[1];y=initial.copy()
    eta=np.random.normal(0.,1.,(6,n))
    E=np.exp(-dt/ntau) if ntau>0 else 0.
    out=np.zeros((steps//stride,6,n));acc=np.zeros((6,n))
    for step in range(steps):
        random=np.random.normal(0.,1.,(6,n))
        if ntau>0:
            eta=E*eta+np.sqrt(1-E*E)*random
            inc=dt/np.sqrt(ntau)*eta
        else:
            inc=np.sqrt(dt)*random
        for v in range(6):
            inc[v,:]*=np.sqrt(2*nsig[v])
        c=coupling(y,indptr,indices,data,factor)
        d1=rhs(y,c,p)
        pred=y+dt*d1+inc
        # TVB's Heun holds long-range coupling fixed within the step.
        y=y+dt*0.5*(d1+rhs(pred,c,p))+inc
        for j in removed:
            y[:,j]=initial[:,j]
        acc+=y
        if (step+1)%stride==0:
            out[step//stride]=acc/stride
            acc[:]=0
    for j in removed:
        out[:,:,j]=0
    return out


def simulate(w,p,initial,nsig,factor=-0.01,dt=0.05,horizon=8000,period=1.,seed=101,action=(),ntau=1.):
    if not np.isclose(horizon/dt,round(horizon/dt)) or not np.isclose(period/dt,round(period/dt)):
        raise ValueError('Horizon/period must be integer multiples of dt')
    wr=resect(w,action);csr=csr_matrix(wr)
    initial=np.broadcast_to(np.asarray(initial).reshape(6,-1),(6,len(w))).copy()
    if p.shape!=(8,len(w)) or np.asarray(nsig).shape!=(6,):
        raise ValueError('Parameter or noise shape mismatch')
    result=_run(np.ascontiguousarray(p),csr.indptr,csr.indices,csr.data,factor,initial,
                np.asarray(nsig),dt,int(round(horizon/dt)),int(round(period/dt)),seed,np.asarray(action,dtype=np.int64),ntau)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite trajectory')
    return result
