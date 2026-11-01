"""Unlabeled candidate features. Future waveforms/labels are not arguments."""
import numpy as np
from scipy.stats import rankdata

FEATURES=['query_fraction','log_contacts','observed_contact_fraction','score_mean','score_std','score_entropy','score_90_10_gap',
 'observed_log_rms_mean','observed_log_rms_std','observed_trial_rms_CV','candidate_available_fraction',
 'candidate_log_distance_mean','candidate_log_distance_std','coverage_gain','score_focus_gain','uncertainty_focus_gain',
 'maximum_profile_similarity','interpolated_observed_log_rms','candidate_weighted_score','candidate_weighted_observed_log_rms']

def distance_catalog(raw_distance):
    """Raw distance/availability only; no signal amplitude is inspected."""
    raw_distance=np.asarray(raw_distance,dtype=float);valid=np.abs(raw_distance)>1e-12
    distance=np.divide(raw_distance.sum(axis=1),valid.sum(axis=1),out=np.zeros(raw_distance.shape[::2]),where=valid.sum(axis=1)>0)
    return distance,[int(i) for i in np.flatnonzero((distance>0).any(axis=0))]

def feature_vector(distance,observed_signal,observed_valid,scores,selected,candidate):
    distance=np.asarray(distance,dtype=float);signal=np.asarray(observed_signal,dtype=float);valid=np.asarray(observed_valid,dtype=bool);scores=np.asarray(scores,dtype=float)
    selected=list(map(int,selected));candidate=int(candidate);n=len(scores);k=len(selected)
    if distance.ndim!=2 or scores.ndim!=1 or distance.shape[0]!=n or signal.ndim!=4 or signal.shape[:3]!=valid.shape or signal.shape[0]!=n or signal.shape[2]!=k:raise ValueError('Observed-only feature shapes differ')
    if not 1<=k<=7 or len(set(selected))!=k or any(i<0 or i>=distance.shape[1] for i in selected) or candidate in selected or not 0<=candidate<distance.shape[1]:raise ValueError('Invalid current/candidate query')
    if not np.isfinite(distance).all() or not np.isfinite(signal).all() or not np.isfinite(scores).all() or (distance<0).any() or ((scores<0)|(scores>1)).any():raise ValueError('Invalid unlabeled inputs')
    kernel=np.where(distance>0,np.exp(-distance/40.),0.);q=kernel[:,candidate]
    if q.sum()<=0:raise ValueError('Unavailable candidate')
    current=kernel[:,selected].max(axis=1);gain=np.maximum(q-current,0.);rank=rankdata(scores)/(n+1)
    uncertainty=1-2*np.abs(scores-.5);clipped=np.clip(scores,1e-6,1-1e-6)
    rms=np.sqrt(np.square(signal).mean(axis=-1));count=valid.sum(axis=1)
    pair_mean=np.divide((rms*valid).sum(axis=1),count,out=np.zeros((n,k)),where=count>0)
    pair_second=np.divide((rms*rms*valid).sum(axis=1),count,out=np.zeros((n,k)),where=count>0)
    pair_sd=np.sqrt(np.maximum(pair_second-pair_mean*pair_mean,0.));cv=pair_sd/(pair_mean+1e-8)
    observed_values=np.log1p(rms[valid]);observed_count=(count>0).sum(axis=1)
    per_contact=np.divide((np.log1p(pair_mean)*(count>0)).sum(axis=1),observed_count,out=np.zeros(n),where=observed_count>0)
    query_count=valid.sum(axis=(0,1));query_mean=np.divide((rms*valid).sum(axis=(0,1)),query_count,out=np.zeros(k),where=query_count>0)
    profiles=kernel[:,selected];denom=np.linalg.norm(profiles,axis=0)*np.linalg.norm(q)
    similarity=np.divide((profiles*q[:,None]).sum(axis=0),denom,out=np.zeros(k),where=denom>0)
    weight=np.exp(5*(similarity-similarity.max()));weight/=weight.sum()
    positive_distance=distance[:,candidate][distance[:,candidate]>0]
    def average(x,w):return float(np.average(x,weights=w))
    values=[k/8,np.log1p(n),float((observed_count>0).mean()),scores.mean(),scores.std(),
      -(clipped*np.log(clipped)+(1-clipped)*np.log(1-clipped)).mean(),np.quantile(scores,.9)-np.quantile(scores,.1),
      observed_values.mean() if len(observed_values) else 0.,observed_values.std() if len(observed_values) else 0.,
      np.clip(cv[count>0],0,10).mean() if (count>0).any() else 0.,float((distance[:,candidate]>0).mean()),
      np.log1p(positive_distance.mean()),np.log1p(positive_distance.std()),gain.mean(),average(gain,.25+rank),average(gain,.05+uncertainty),
      similarity.max(),float(weight@np.log1p(query_mean)),average(scores,q),average(per_contact,q)]
    result=np.array(values,dtype=float)
    if result.shape!=(len(FEATURES),) or not np.isfinite(result).all():raise ValueError('Nonfinite candidate features')
    return result

def from_revealed(raw,selected,scores,candidate):
    """The only full-array read before feature extraction is known geometry.

    Signal data passed to feature_vector contain selected columns only. The
    caller must compute scores from the same observed columns, never full data.
    """
    distance,_=distance_catalog(raw[...,0])
    return feature_vector(distance,raw[:,:,selected,-512:].copy(),np.abs(raw[:,:,selected,0])>1e-12,scores,selected,candidate)
