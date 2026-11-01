"""Observation-only linear forecasting baseline, not a seizure/resection model.

Fit y[t+1]-mean = a*(y[t]-mean) + b*M*(y[t]-mean), where
M = G*(W-diag(row_sum(W)))*G_ridge_inverse. The source-to-sensor
operator is the released anatomical gain. No source trajectory or generating
parameter is read. Anatomy is static; ridge and dynamics use training only.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class LinearForecast:
    transition: np.ndarray
    mean: np.ndarray
    a: float
    b: float
    spectral_radius: float


def sensor_network_operator(weights,gain,ridge):
    if ridge<=0: raise ValueError('Positive inverse regularization required')
    w=np.asarray(weights,dtype=float)
    g=np.asarray(gain,dtype=float)
    if w.shape!=(g.shape[1],g.shape[1]): raise ValueError('Atlas shape mismatch')
    # Scale Laplacian by maximum node strength, retaining topology. This is a
    # linear baseline convention, not the Epileptor coupling parameter.
    scale=max(float(w.sum(axis=1).max()),1e-30)
    lap=(w-np.diag(w.sum(axis=1)))/scale
    u,s,vt=np.linalg.svd(g,full_matrices=False)
    lam=ridge*s[0]**2
    inverse=(vt.T*(s/(s*s+lam)))@u.T
    return g@lap@inverse


def fit_linear(training,operator):
    y=np.asarray(training,dtype=float)
    if len(y)<3 or not np.isfinite(y).all(): raise ValueError('Invalid training data')
    mean=y.mean(axis=0)
    x=y[:-1]-mean; target=y[1:]-mean
    coupled=x@operator.T
    design=np.column_stack([x.ravel(),coupled.ravel()])
    # Bound persistence and coupling coefficient. Stability is separately
    # inspected; do not silently clip a failed forecast after evaluation.
    opt=lsq_linear(design,target.ravel(),bounds=([0.,0.],[1.,1.]),tol=1e-10)
    a,b=map(float,opt.x)
    transition=a*np.eye(y.shape[1])+b*operator
    radius=float(np.max(abs(np.linalg.eigvals(transition))))
    return LinearForecast(transition,mean,a,b,radius)


def predict(model,origins,horizon):
    if horizon<1: raise ValueError('Positive forecast horizon required')
    return (np.asarray(origins)-model.mean)@np.linalg.matrix_power(model.transition,horizon).T+model.mean


def forecast_errors(model,block,horizon):
    if len(block)<=horizon: raise ValueError('Forecast block too short')
    truth=block[horizon:]
    prediction=predict(model,block[:-horizon],horizon)
    return {'mse':float(np.mean((prediction-truth)**2)),
            'persistence_mse':float(np.mean((block[:-horizon]-truth)**2)),
            'training_mean_mse':float(np.mean((model.mean-truth)**2))}
