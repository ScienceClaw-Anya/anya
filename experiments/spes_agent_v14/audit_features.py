"""Independent cached reconstruction; audit only, never the frozen controller."""
import numpy as np
from scipy.special import expit,logit
from scipy.stats import rankdata
from experiments.spes_agent_v10.features import contact_features


def audit_table(distance,observed,scores,selected,remaining):
    z=logit(np.clip(np.asarray(scores,dtype=float),1e-12,1-1e-12)); median=np.median(z)
    relative=expit((z-median)/max(float(np.median(np.abs(z-median))),1e-6))
    signal=np.asarray(observed[...,-512:],dtype=float); valid=observed[...,0]>1e-12; n=len(scores); k=len(selected)
    morphology=contact_features(observed[...,-512:],observed[...,0],scores)[:,2:]
    kernel=np.where(distance>0,np.exp(-distance/40),0.); profiles=kernel[:,selected]; current=profiles.max(axis=1)
    rank=rankdata(relative)/(n+1); uncertainty=1-2*np.abs(relative-.5); clipped=np.clip(relative,1e-6,1-1e-6)
    rms=np.sqrt(np.square(signal).mean(axis=-1)); count=valid.sum(axis=1)
    pair_mean=np.divide((rms*valid).sum(axis=1),count,out=np.zeros((n,k)),where=count>0)
    pair_second=np.divide((rms*rms*valid).sum(axis=1),count,out=np.zeros((n,k)),where=count>0)
    pair_sd=np.sqrt(np.maximum(pair_second-pair_mean*pair_mean,0.)); cv=pair_sd/(pair_mean+1e-8)
    values=np.log1p(rms[valid]); observed_count=(count>0).sum(axis=1)
    per_contact=np.divide((np.log1p(pair_mean)*(count>0)).sum(axis=1),observed_count,out=np.zeros(n),where=observed_count>0)
    query_count=valid.sum(axis=(0,1)); query_mean=np.divide((rms*valid).sum(axis=(0,1)),query_count,out=np.zeros(k),where=query_count>0)
    context=[k/8,np.log1p(n),float((observed_count>0).mean()),relative.mean(),relative.std(),
             -(clipped*np.log(clipped)+(1-clipped)*np.log(1-clipped)).mean(),np.quantile(relative,.9)-np.quantile(relative,.1),
             values.mean() if len(values) else 0.,values.std() if len(values) else 0.,np.clip(cv[count>0],0,10).mean() if (count>0).any() else 0.]
    rows=[]
    for candidate in remaining:
        if candidate in selected: raise ValueError('Audit candidate is already observed')
        q=kernel[:,candidate]; gain=np.maximum(q-current,0.); positive=distance[:,candidate][distance[:,candidate]>0]
        denom=np.linalg.norm(profiles,axis=0)*np.linalg.norm(q)
        similarity=np.divide((profiles*q[:,None]).sum(axis=0),denom,out=np.zeros(k),where=denom>0)
        weight=np.exp(5*(similarity-similarity.max())); weight/=weight.sum()
        candidate_features=[float((distance[:,candidate]>0).mean()),np.log1p(positive.mean()),np.log1p(positive.std()),
                            gain.mean(),float(np.average(gain,weights=.25+rank)),float(np.average(gain,weights=.05+uncertainty)),
                            similarity.max(),float(weight@np.log1p(query_mean)),float(np.average(relative,weights=q)),float(np.average(per_contact,weights=q))]
        rows.append(np.r_[context,candidate_features,np.average(morphology,axis=0,weights=.01+gain)])
    return np.vstack(rows)
