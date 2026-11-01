"""Simple independent-source covariance inversion for a development baseline."""
import numpy as np
from scipy.optimize import nnls


def source_variances(covariance,gain,ridge=1e-3):
    g=np.asarray(gain,dtype=float);c=np.asarray(covariance,dtype=float)
    if c.shape!=(len(g),len(g)) or ridge<=0: raise ValueError('Invalid covariance or ridge')
    # Vectorize the full matrix: off-diagonal terms correctly appear twice in
    # the Frobenius objective. Normalize columns before a fixed ridge penalty.
    design=np.einsum('ci,di->cdi',g,g).reshape(len(g)**2,g.shape[1])
    norms=np.linalg.norm(design,axis=0)
    if (norms==0).any(): raise ValueError('Unobserved gain column')
    a=design/norms
    solution,_=nnls(np.vstack([a,np.sqrt(ridge)*np.eye(g.shape[1])]),
                    np.concatenate([c.ravel(),np.zeros(g.shape[1])]),maxiter=20*g.shape[1])
    q=solution/norms
    reconstructed=(g*q)@g.T
    return q,float(np.linalg.norm(reconstructed-c)/np.linalg.norm(c))


def covariance(signal):
    x=np.asarray(signal)-np.asarray(signal).mean(axis=0)
    return x.T@x/(len(x)-1)


def feature_error(predicted,observed):
    pc=covariance(predicted);oc=covariance(observed)
    ps=np.sqrt(np.diag(pc));os=np.sqrt(np.diag(oc))
    return {'covariance_relative_error':float(np.linalg.norm(pc-oc)/np.linalg.norm(oc)),
            'sensor_std_relative_error':float(np.linalg.norm(ps-os)/np.linalg.norm(os))}
