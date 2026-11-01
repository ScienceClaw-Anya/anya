"""State-paired losses and fixed context/candidate interactions."""
from itertools import combinations
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from experiments.spes_agent_v14.tools import FEATURES,VIEWS

ALPHA=100.
POLICIES=['pair_experts','pair_monolithic','point_monolithic']


def expand(z,columns):
    z=np.asarray(z,dtype=float); columns=list(columns)
    if z.ndim!=2 or z.shape[1]!=44 or not np.isfinite(z).all(): raise ValueError('Finite44-feature inputs required')
    context=[j for j in columns if j<10]; candidate=[j for j in columns if j>=10]
    products=(z[:,context,None]*z[:,None,candidate]).reshape(len(z),-1)
    return np.column_stack([z[:,columns],products])


def pair_data(x,gain,groups):
    x=np.asarray(x,dtype=float); gain=np.asarray(gain,dtype=float)
    if x.ndim!=2 or gain.shape!=(len(x),) or len(groups)!=len(x) or not np.isfinite(x).all() or not np.isfinite(gain).all(): raise ValueError('Invalid paired training input')
    buckets={}
    for i,key in enumerate(groups): buckets.setdefault(tuple(key),[]).append(i)
    features=[]; targets=[]; weights=[]
    for indices in buckets.values():
        if len(indices)!=6: raise ValueError('Exactly six frozen candidates per patient/state required')
        for i,j in combinations(indices,2):
            features.append(x[i]-x[j]); targets.append(gain[i]-gain[j]); weights.append(6/15)
    return np.vstack(features),np.array(targets),np.array(weights)


def fit_models(x,gain,groups):
    scaler=StandardScaler().fit(x); z=scaler.transform(x); models={}
    for name,columns in VIEWS.items():
        features=expand(z,columns); px,py,pw=pair_data(features,gain,groups)
        models[name]=Ridge(alpha=ALPHA,fit_intercept=False).fit(px,py,sample_weight=pw)
    models['point']=Ridge(alpha=ALPHA,fit_intercept=True).fit(expand(z,VIEWS['monolithic']),gain)
    return {'scaler':scaler,'models':models,'views':VIEWS,'alpha':ALPHA}


def predict(artifact,x):
    z=artifact['scaler'].transform(x); values={name:artifact['models'][name].predict(expand(z,columns)) for name,columns in VIEWS.items()}
    values['pair_monolithic']=values['monolithic']
    values['point_monolithic']=artifact['models']['point'].predict(expand(z,VIEWS['monolithic']))
    values['pair_experts']=np.mean([rankdata(values[name])/(len(x)+1) for name in ['geometry','neural','waveform']],axis=0)
    if any(not np.isfinite(v).all() for v in values.values()): raise ValueError('Nonfinite ranking utility')
    return values


def choose(ids,values):
    if len(ids)!=len(values) or not len(ids) or not np.isfinite(values).all(): raise ValueError('Invalid candidate ranking')
    return min(zip(ids,values),key=lambda row:(-row[1],row[0]))[0]
