"""Two response hypotheses constructed without candidate waveforms or labels."""
import numpy as np
from scipy.stats import rankdata
from experiments.spes_agent_v8.features import FEATURES as BASE_FEATURES
from experiments.spes_agent_v8.numeric import feature_matrix

METHODS=['profile_average','distance_corrected']
SUMMARY=['future_score_mean','future_score_sd','entropy_change','mean_absolute_score_change','mean_score_change',
 'mean_absolute_rank_change','pairwise_rank_flip_fraction','high_rank_mean_score_change','low_rank_mean_score_change',
 'rank_change_sd','future_log_rms_mean','future_log_rms_sd']
FEATURES=BASE_FEATURES+[m+'_'+s for m in METHODS for s in SUMMARY]+['hypothesis_score_disagreement','hypothesis_rank_disagreement']


def impute(distance_trials,observed_signal,selected,candidate,method):
    """Return an explicitly hypothetical next waveform, from observed traces.

    distance_trials contains geometry/availability only; observed_signal contains
    only selected columns. Trial indices across sites are aligned computationally,
    not assumed to be simultaneous physiological observations.
    """
    d=np.asarray(distance_trials,dtype=float);signal=np.asarray(observed_signal,dtype=float)
    selected=list(map(int,selected));candidate=int(candidate)
    if method not in METHODS or candidate in selected or len(set(selected))!=len(selected):raise ValueError('Invalid hypothesis/query')
    if d.ndim!=3 or signal.shape!=d[:,:,selected].shape+(512,) or not np.isfinite(d).all() or not np.isfinite(signal).all():raise ValueError('Observed-only input shapes/values differ')
    valid=d>1e-12;counts=valid.sum(axis=1)
    dist=np.divide(d.sum(axis=1),counts,out=np.zeros(counts.shape,dtype=float),where=counts>0)
    kernel=np.where(dist>0,np.exp(-dist/40),0);profiles=kernel[:,selected];target=kernel[:,candidate]
    denom=np.linalg.norm(profiles,axis=0)*np.linalg.norm(target)
    similarity=np.divide((profiles*target[:,None]).sum(axis=0),denom,out=np.zeros(len(selected)),where=denom>0)
    weight=np.exp(20*(similarity-similarity.max()))[None,None,:]*valid[:,:,selected]
    if method=='distance_corrected':
        radial=np.exp(np.clip((d[:,:,selected]-d[:,:,candidate,None])/40,-np.log(4),np.log(4)))
        signal=signal*radial[...,None]
        weight=weight*np.exp(-np.abs(d[:,:,selected]-d[:,:,candidate,None])/40)
    weight=np.divide(weight,weight.sum(axis=2,keepdims=True),out=np.zeros_like(weight),where=weight.sum(axis=2,keepdims=True)>0)
    result=(signal*weight[...,None]).sum(axis=2)
    result[~valid[:,:,candidate]]=0
    return np.asarray(result,dtype=np.float32)


def hypothetical_input(raw,selected,candidate,method):
    # The only full-array read is known geometry. Candidate signal is never read.
    observed=np.array(raw[:,:,selected,:],copy=True)
    predicted=impute(raw[...,0],observed[...,-512:],selected,candidate,method)
    added=np.zeros(observed.shape[:2]+(1,observed.shape[3]),dtype=raw.dtype)
    added[:,:,0,0]=raw[:,:,candidate,0];added[:,:,0,-512:]=predicted
    return np.concatenate([observed,added],axis=2),predicted


def summarize(before,after,waveform,valid):
    before=np.asarray(before,dtype=float);after=np.asarray(after,dtype=float);n=len(before)
    r0=rankdata(before)/(n+1);r1=rankdata(after)/(n+1);change=after-before
    ix=np.triu_indices(n,1);a=before[:,None]-before[None,:];b=after[:,None]-after[None,:]
    def entropy(s):
        s=np.clip(s,1e-6,1-1e-6);return float(-(s*np.log(s)+(1-s)*np.log(1-s)).mean())
    rms=np.sqrt(np.square(np.asarray(waveform,dtype=float)).mean(axis=-1));value=np.log1p(rms[valid])
    return [after.mean(),after.std(),entropy(after)-entropy(before),np.abs(change).mean(),change.mean(),
      np.abs(r1-r0).mean(),float(((a[ix]*b[ix])<0).mean()),float(change[r0>=.75].mean()) if (r0>=.75).any() else 0.,
      float(change[r0<=.25].mean()) if (r0<=.25).any() else 0.,float((r1-r0).std()),value.mean() if len(value) else 0.,value.std() if len(value) else 0.]


def normalise_hypothesis(array,stats):
    # The upstream HAT explicitly crops to distance + last 512 samples. Normalize
    # those exact two segments; the unused middle 128 values never enter HAT.
    x=np.concatenate([array[...,:1],array[...,-512:]],axis=-1)
    for segment,stat in [(slice(0,1),stats[0]),(slice(1,None),stats[2])]:
        view=x[...,segment];mask=np.abs(view)>1e-12;view[mask]=(view[mask]-stat['mean'])/stat['std']
    return x


def score_hypotheses(model,arrays,stats):
    import torch
    x=np.concatenate([normalise_hypothesis(a,stats) for a in arrays]);result=[]
    with torch.no_grad():
        for begin in range(0,len(x),32):result.append(torch.sigmoid(model(torch.from_numpy(x[begin:begin+32]).cuda())[:,0]).cpu().numpy())
    value=np.concatenate(result)
    if not np.isfinite(value).all():raise ValueError('Nonfinite hypothetical score')
    return np.split(value,np.cumsum([len(a) for a in arrays])[:-1])


def make_features(raw,selected,before,candidates,model,stats):
    base=feature_matrix(raw,selected,before,candidates);rows=[];details=[];cache={}
    # Batch four candidates / eight hypotheses, bounding host memory. This does
    # not acquire additional real measurements. All queries retain the same set.
    for begin in range(0,len(candidates),4):
        group=candidates[begin:begin+4];arrays=[];waves=[]
        for candidate in group:
            for method in METHODS:
                a,w=hypothetical_input(raw,selected,candidate,method);arrays.append(a);waves.append(w)
        values=score_hypotheses(model,arrays,stats)
        for j,(value,wave) in enumerate(zip(values,waves)):cache[group[j//2],METHODS[j%2]]=(value,wave)
    for i,candidate in enumerate(candidates):
        scores={};parts=[]
        for method in METHODS:
            after,waveform=cache[candidate,method]
            scores[method]=after;parts+=summarize(before,after,waveform,np.abs(raw[:,:,candidate,0])>1e-12)
        a,b=[scores[m] for m in METHODS]
        row=np.concatenate([base[i],np.array(parts),[np.abs(a-b).mean(),np.abs(rankdata(a)-rankdata(b)).mean()/(len(a)+1)]])
        if row.shape!=(len(FEATURES),) or not np.isfinite(row).all():raise ValueError('Nonfinite hypothetical features')
        rows.append(row);details.append(scores)
    return np.vstack(rows),details
