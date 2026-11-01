"""Finite timing fit; all new scientific computation runs on GCP."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from intervention.data import load_recording
from intervention.temporal import causal_variance, training_scale, envelope, phase_search, fit_ar, ar_forecast, trend_forecast

ROOT = Path(__file__).resolve().parents[1]
assert str(ROOT) == '/home/devstar3022/anya-intervention-research'
OUT = ROOT/'outputs/development-v3'
OLD = ROOT/'outputs/development-v2/epileptor-fit'
OUT.mkdir(exist_ok=False)
start = time.time()
manifest = json.loads((ROOT/'manifests/development-v3.json').read_text())
(OUT/'budget.json').write_text(json.dumps(dict(start_epoch=start,max_wall_seconds=3600,maximum_new_simulator_calls=6))+'\n')
accessed = set()
def audit(event,args):
    if event == 'open' and isinstance(args[0],(str,bytes)):
        path = str(args[0])
        if '/parameters/' in path or 'source_timeseries' in path:
            raise PermissionError('Released evaluator assets forbidden')
        if '/data/vec/' in path: accessed.add(path)
sys.addaudithook(audit)
o = load_recording(ROOT/'data/vec',manifest['subject'])
y = o.sensors/1e6
window = manifest['feature']['window_samples']; stride = manifest['feature']['stride_samples']
times = np.arange(window-1,len(y),stride)
train_boundary = int(.6*len(y)); final_boundary = int(.8*len(y))
i = int(np.sum(times < train_boundary)); j = int(np.sum(times < final_boundary))
variance = causal_variance(y,window)
scale = training_scale(variance,times[:i])
observed = np.log1p(variance[times]/scale)
training = observed[:i]; prefix = observed[:j]
val_mask = times[i:j] >= train_boundary+window-1
test_mask = times[j:] >= final_boundary+window-1
threshold = float(np.mean(np.quantile(training.mean(axis=1),[.2,.8])))
offsets = np.arange(0,3301,25)
np.savez(OUT/'features.npz',observed=observed,times=times,scale=scale,i=i,j=j,
         val_mask=val_mask,test_mask=test_mask,threshold=threshold,offsets=offsets)

def loss(pred):
    return float(np.mean((pred[val_mask]-observed[i:j][val_mask])**2))

def fixed(name,block,n):
    if name == 'persistence': return np.repeat(block[-1:],n,axis=0)
    if name == 'prefix_mean': return np.repeat(block.mean(axis=0,keepdims=True),n,axis=0)
    if name == 'zero_envelope': return np.zeros((n,block.shape[1]))
    raise ValueError(name)

baselines = []
for name in ['persistence','prefix_mean','zero_envelope']:
    val_pred = fixed(name,training,j-i); prediction = fixed(name,prefix,len(times)-j)
    baselines.append(dict(name=name,validation_mse=loss(val_pred)))
    np.savez(OUT/f'baseline-{name}.npz',validation=val_pred,prediction=prediction)
trend_candidates = []
for damping in [.8,.95,1.]:
    pred = trend_forecast(training,damping,j-i)
    trend_candidates.append(dict(damping=damping,validation_mse=loss(pred)))
trend_choice = min(trend_candidates,key=lambda x:x['validation_mse'])
baselines.append(dict(name='damped_trend',**trend_choice,candidates=trend_candidates))
np.savez(OUT/'baseline-damped_trend.npz',
         validation=trend_forecast(training,trend_choice['damping'],j-i),
         prediction=trend_forecast(prefix,trend_choice['damping'],len(times)-j))
ar_candidates = []
for lag in [1,5,10]:
    for ridge in [1e-6,.001,.1]:
        coef,mean,radius = fit_ar(training,lag,ridge)
        stable = radius <= 1.000001
        ar_candidates.append(dict(lag=lag,ridge=ridge,spectral_radius=radius,
                                  stable=stable,validation_mse=loss(ar_forecast(training,coef,mean,j-i)) if stable else None))
stable_candidates = [r for r in ar_candidates if r['stable']]
if stable_candidates:
    ar_choice = min(stable_candidates,key=lambda r:r['validation_mse'])
    val_coef,val_mean,_ = fit_ar(training,ar_choice['lag'],ar_choice['ridge'])
    coef,mean,radius = fit_ar(prefix,ar_choice['lag'],ar_choice['ridge'])
    fallback = radius > 1.000001
    prediction = fixed('persistence',prefix,len(times)-j) if fallback else ar_forecast(prefix,coef,mean,len(times)-j)
    baselines.append(dict(name='autoregression',**ar_choice,candidates=ar_candidates,
                          refit_spectral_radius=radius,refit_unstable_persistence_fallback=fallback))
    np.savez(OUT/'baseline-autoregression.npz',prediction=prediction,
             validation=ar_forecast(training,val_coef,val_mean,j-i),coefficients=coef,mean=mean)
baseline_choice = min(baselines,key=lambda r:r['validation_mse'])['name']

networks = []; hashes = {}
for network in manifest['mechanistic_fit']['networks']:
    candidates = []
    for candidate in range(4):
        assert time.time()-start < 3500
        path = OLD/f'{network}-candidate{candidate}.npz'
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = np.load(path)
        sim = envelope(artifact['sensor'],scale,window)
        offset,profile,diagnostic = phase_search(sim,training,times[:i],offsets)
        val_pred = sim[times[i:j]+offset]
        candidates.append(dict(candidate=candidate,training_phase=diagnostic,validation_mse=loss(val_pred)))
        np.savez(OUT/f'{network}-candidate{candidate}-profile.npz',offsets=offsets,training_losses=profile)
    choice = min(candidates,key=lambda r:r['validation_mse'])
    artifact = np.load(OLD/f"{network}-candidate{choice['candidate']}.npz")
    sim = envelope(artifact['sensor'],scale,window)
    offset,profile,diagnostic = phase_search(sim,prefix,times[:j],offsets)
    row = dict(network=network,candidate=choice['candidate'],validation_mse=choice['validation_mse'],
               training_phase=choice['training_phase'],prefix_phase=diagnostic,candidates=candidates)
    networks.append(row)
    np.savez(OUT/f'{network}-reference.npz',prediction=sim[times[j:]+offset],
             prefix_prediction=sim[times[:j]+offset],offsets=offsets,prefix_losses=profile,
             validation=sim[times[i:j]+choice['training_phase']['offset']])
    print(json.dumps(row),flush=True)

selection = dict(phase=manifest['phase'],subject=manifest['subject'],raw_samples=len(y),channels=y.shape[1],
                 train_features=i,prefix_features=j,validation_scored_features=int(val_mask.sum()),
                 final_scored_features=int(test_mask.sum()),proxy_threshold=threshold,
                 baselines=baselines,selected_baseline=baseline_choice,networks=networks,
                 input_paths=sorted(accessed),forbidden_asset_guard=True,
                 reused_artifact_sha256=hashes,source_commit=(ROOT/'outputs/source-commit.txt').read_text().strip(),
                 manifest_sha256=hashlib.sha256((ROOT/'manifests/development-v3.json').read_bytes()).hexdigest(),
                 runtime_seconds=time.time()-start,new_simulator_calls=0)
(OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print(json.dumps(dict(selected_baseline=baseline_choice,runtime_seconds=time.time()-start)),flush=True)
