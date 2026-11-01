import dataclasses
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from intervention.model import coupling,normalized_weights,resect,rhs,simulate
from intervention.metrics import burden,reduction
from intervention.data import Observation


def fixture():
    w=np.array([[0.,0.3,0.],[0.1,0.,0.8],[0.2,0.,0.]])
    p=np.tile(np.array([-2.5,3.1,0.45,0.2,0.00015,0.001,0.0001,0.001])[:,None],(1,3))
    initial=np.array([-1.46242601,-9.69344913,2.99029597,-1.11181819,0.,-0.438727802])
    return w,p,initial


def test_directed_orientation_and_normalization():
    w,p,initial=fixture();a,scale=normalized_weights(w)
    r=resect(a,[1]);assert np.array_equal(r[[0,2]][:,[0,2]],a[[0,2]][:,[0,2]])
    assert not r[1].any() and not r[:,1].any()
    y=np.tile(initial[:,None],(1,3));y[0]=[1,2,4];csr=csr_matrix(w)
    c=coupling(y,csr.indptr,csr.indices,csr.data,0.5)
    assert c[0,0]==pytest.approx(0.5*0.3*(2-1))
    assert c[0,1]==pytest.approx(0.5*(0.1*(1-2)+0.8*(4-2)))


def test_null_seed_pair_and_resection():
    w,p,i=fixture();kwargs=dict(horizon=20,seed=3)
    nsig=np.array([.001,.001,0,.0002,.0002,0.])
    a=simulate(w,p,i,nsig,**kwargs);b=simulate(w,p,i,nsig,action=(),**kwargs)
    assert np.array_equal(a,b)
    assert not np.array_equal(a,simulate(w,p,i,nsig,horizon=20,seed=4))
    wr=resect(w,[1]);a=simulate(wr,p,i,nsig,**kwargs)
    b=simulate(w,p,i,nsig,action=(1,),**kwargs)
    np.testing.assert_allclose(a[:,:,[0,2]],b[:,:,[0,2]],atol=0,rtol=0)
    assert not b[:,:,1].any()


def test_permutation_and_parameter_isolation():
    w,p,i=fixture();perm=[2,0,1]
    a=simulate(w,p,i,np.zeros(6),horizon=30)
    b=simulate(w[np.ix_(perm,perm)],p[:,perm],i,np.zeros(6),horizon=30)
    np.testing.assert_allclose(a[:,:,perm],b,atol=1e-12)
    assert not {'p','x0','EZ','PZ','initial','nsig'} & {x.name for x in dataclasses.fields(Observation)}
    with pytest.raises(ValueError):resect(w,[1,1])
    with pytest.raises(ValueError):reduction(1,0)


def test_original_denominator_and_removed_tissue():
    s=np.zeros((1500,6,3));s[:,0,:]=-0.5
    assert burden(s,transient=1000)==1500
    assert burden(s,action=(1,),transient=1000)==1000
    assert reduction(1000,1500)==pytest.approx(1/3)


def test_rhs_and_trajectory_against_tvb():
    from tvb.simulator.models.epileptor import Epileptor
    from tvb.simulator.integrators import HeunStochastic
    from tvb.simulator.noise import Additive
    w,p,i=fixture();n=len(w)
    m=Epileptor(**{k:p[j] for j,k in enumerate(['x0','Iext','Iext2','slope','r','Ks','Kf','Kvf'])});m.configure()
    y=np.tile(i[:,None],(1,n));c=np.ones((2,n))*.25
    np.testing.assert_allclose(rhs(y,c,p),m.dfun(y[...,None],c[...,None])[:,:,0],atol=1e-12)
    nsig=np.array([.001,.001,0,.0002,.0002,0.]);dt=.05
    noise=Additive(nsig=nsig[:,None,None],ntau=1,random_stream=np.random.RandomState(7))
    noise.configure_coloured(dt,(6,n,1))
    integ=HeunStochastic(dt=dt,noise=noise);integ.configure()
    integ.bounded_state_variable_indices=np.array([],dtype=int)
    integ.clamped_state_variable_indices=np.array([],dtype=int)
    y=y[...,None];samples=[];avg=np.zeros_like(y)
    for step in range(200):
        # Independent dense coupling calculation, then TVB integration.
        c=np.stack([-.01*(w@y[k,:,0]-w.sum(axis=1)*y[k,:,0]) for k in [0,3]])[...,None]
        y=integ.scheme(y,m.dfun,c,0.,0.);avg+=y
        if (step+1)%20==0:samples.append(avg[:,:,0]/20);avg=np.zeros_like(y)
    actual=simulate(w,p,i,nsig,seed=7,horizon=10)
    np.testing.assert_allclose(actual,np.array(samples),atol=2e-12,rtol=2e-12)
