"""Complementary observed-only feature views and fixed value learners."""
import numpy as np
from scipy.special import expit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiments.spes_agent_v8.features import feature_vector, FEATURES as BASE
from experiments.spes_agent_v10.features import contact_features, FEATURES as CONTACT
from experiments.spes_agent_v13.tools import location_scale

FEATURES = ['relative_'+s if s.startswith('score_') or 'weighted_score' in s or 'uncertainty' in s else s for s in BASE]+['gain_weighted_'+s for s in CONTACT[2:]]
VIEWS = {'geometry':[0,1,2,10,11,12,13,16],
         'neural':[0,1,2,3,4,5,6,10,11,12,13,14,15,16,18],
         'waveform':[0,1,2,7,8,9,10,11,12,13,16,17,19]+list(range(20,44)),
         'monolithic':list(range(44))}
POLICIES = ['expert_consensus','monolithic','ridge']


def table(distance, observed, scores, selected, remaining):
    """Distance is verified known geometry; observed contains selected pairs only."""
    z,center,scale = location_scale(scores)
    relative = expit((z-center)/scale)
    morphology = contact_features(observed[...,-512:],observed[...,0],scores)[:,2:]
    kernel = np.where(distance>0,np.exp(-distance/40),0.)
    current = kernel[:,selected].max(axis=1)
    rows=[]
    for q in remaining:
        if q in selected: raise ValueError('Candidate is already observed')
        base = feature_vector(distance,observed[...,-512:],observed[...,0]>1e-12,relative,selected,q)
        weights = .01+np.maximum(kernel[:,q]-current,0.)
        rows.append(np.r_[base,np.average(morphology,axis=0,weights=weights)])
    result=np.vstack(rows)
    if result.shape!=(len(remaining),44) or not np.isfinite(result).all(): raise ValueError('Invalid query-value features')
    return result


def fit_models(x,y,patient,outer):
    unique=np.unique(patient)
    weights=np.array([len(y)/(len(unique)*np.sum(patient==s)) for s in patient])
    models={}
    for i,(name,columns) in enumerate(VIEWS.items()):
        m=GradientBoostingRegressor(n_estimators=150,max_depth=2,min_samples_leaf=30,learning_rate=.03,
                                   loss='huber',alpha=.9,random_state=14000+outer*10+i)
        models[name]=m.fit(x[:,columns],y,sample_weight=weights)
    ridge=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=100.))])
    models['ridge']=ridge.fit(x,y,scale__sample_weight=weights,model__sample_weight=weights)
    return models


def predict(models,x):
    out={name:models[name].predict(x[:,columns]) for name,columns in VIEWS.items()}
    out['ridge']=models['ridge'].predict(x)
    expert=np.vstack([out[k] for k in ['geometry','neural','waveform']])
    out['expert_mean']=expert.mean(axis=0)
    out['expert_disagreement']=expert.std(axis=0)
    out['expert_consensus']=expert.mean(axis=0)-.5*expert.std(axis=0)
    return out


def choose(ids,values):
    if len(ids)!=len(values) or not len(ids) or not np.isfinite(values).all(): raise ValueError('Invalid candidate utilities')
    return min(zip(ids,values),key=lambda row:(-row[1],row[0]))[0]
