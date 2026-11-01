"""Contact morphology features; no labels, future traces or clinical time labels."""
import numpy as np
from scipy.stats import rankdata

FEATURES=['hat_score','hat_rank','query_fraction','observed_trial_fraction','log_distance_mean','log_distance_min',
 'log_rms_mean','log_rms_sd','log_rms_max','log_mean_signed_response_rms','mean_absolute_trial_coherence',
 'relative_difference_energy']+[f'time_index_energy_fraction_{i}' for i in range(8)]+[f'fft_bin_energy_fraction_{i}' for i in range(6)]


def contact_features(observed_signal,observed_distance,scores):
    signal=np.asarray(observed_signal,dtype=float);distance=np.asarray(observed_distance,dtype=float);scores=np.asarray(scores,dtype=float)
    if signal.ndim!=4 or signal.shape[-1]!=512 or distance.shape!=signal.shape[:3] or scores.shape!=(len(signal),) or not np.isfinite(signal).all() or not np.isfinite(distance).all() or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():raise ValueError('Invalid observed-only input')
    valid=distance>1e-12;signal=signal*valid[...,None];counts=valid.sum(axis=(1,2));safe=np.maximum(counts,1);n=len(scores)
    rms=np.sqrt(np.square(signal).mean(axis=-1));lr=np.log1p(rms)
    mean=(lr*valid).sum(axis=(1,2))/safe;second=(lr*lr*valid).sum(axis=(1,2))/safe
    energy=np.square(signal).sum(axis=(1,2))/safe[:,None];total=energy.sum(axis=1)
    mean_trace=signal.sum(axis=(1,2))/safe[:,None]
    coherence=np.divide(np.abs(mean_trace).mean(axis=1),(np.abs(signal).sum(axis=(1,2,3))/(safe*512))+1e-8)
    diff=np.square(np.diff(signal,axis=-1)).sum(axis=(1,2,3))/safe
    minimum=np.min(np.where(valid,distance,np.inf),axis=(1,2));minimum[~np.isfinite(minimum)]=0
    values=[scores,rankdata(scores)/(n+1),np.full(n,signal.shape[2]/8),counts/(signal.shape[1]*signal.shape[2]),
      np.log1p((distance*valid).sum(axis=(1,2))/safe),np.log1p(minimum),mean,np.sqrt(np.maximum(second-mean*mean,0)),lr.max(axis=(1,2)),
      np.log1p(np.sqrt(np.square(mean_trace).mean(axis=1))),coherence,np.log1p(diff/(total+1e-8))]
    values += [energy[:,i*64:(i+1)*64].sum(axis=1)/(total+1e-8) for i in range(8)]
    spectrum=np.abs(np.fft.rfft(signal,axis=-1))**2;power=spectrum.sum(axis=(1,2))/safe[:,None];denominator=power[:,1:].sum(axis=1)
    bounds=[1,8,16,32,64,128,257]
    values += [power[:,bounds[i]:bounds[i+1]].sum(axis=1)/(denominator+1e-8) for i in range(6)]
    result=np.column_stack(values)
    if result.shape!=(n,len(FEATURES)) or not np.isfinite(result).all():raise ValueError('Nonfinite contact features')
    return result


def from_revealed(raw,selected,scores):
    if len(selected)!=len(set(selected)) or not 1<=len(selected)<=8:raise ValueError('Invalid observed query list')
    return contact_features(raw[:,:,selected,-512:].copy(),raw[:,:,selected,0].copy(),scores)


def expected_rank_gain(posterior,before,after):
    """Ratio-of-expected pair counts, not E[AUROC] or a calibrated clinical risk.

    Independent label marginals are a computational surrogate. No true labels
    enter this function. A common denominator preserves within-state ordering.
    """
    q=np.asarray(posterior,dtype=float);before=np.asarray(before);after=np.asarray(after)
    if q.ndim!=1 or q.shape!=before.shape or q.shape!=after.shape or not np.isfinite(q).all() or not np.isfinite(before).all() or not np.isfinite(after).all() or ((q<0)|(q>1)).any():raise ValueError('Invalid posterior/rank inputs')
    weight=q[:,None]*(1-q[None,:]);np.fill_diagonal(weight,0);denominator=weight.sum()
    def concordance(scores):
        delta=scores[:,None]-scores[None,:]
        return ((delta>0)+.5*(delta==0))*weight
    return float((concordance(after).sum()-concordance(before).sum())/denominator) if denominator>1e-12 else 0.
