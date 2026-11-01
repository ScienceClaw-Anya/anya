"""Independent direct-window replay; deliberately does not import temporal.py."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from intervention.data import load_recording

ROOT = Path(__file__).resolve().parents[1]
assert str(ROOT) == '/home/devstar3022/anya-intervention-research'
OUT = ROOT/'outputs/development-v3'; OLD = ROOT/'outputs/development-v2/epileptor-fit'
assert not (OUT/'replay.json').exists()
start = time.time()
selection = json.loads((OUT/'selection.json').read_text())
evaluation = json.loads((OUT/'evaluation.json').read_text())
f = np.load(OUT/'features.npz'); times = f['times']; i=int(f['i']); j=int(f['j'])
offsets=f['offsets']; mask=f['test_mask']; val_mask=f['val_mask']
o=load_recording(ROOT/'data/vec','sub-010'); raw=o.sensors/1e6
def windows(signal,indices):
    return np.stack([np.var(signal[t-49:t+1],axis=0) for t in indices])
variance=windows(raw,times)
scale=variance[:i].mean(axis=0)
scale=np.maximum(scale,.01*np.median(scale[scale>0]))
observed=np.log1p(variance/scale)
np.testing.assert_allclose(observed,f['observed'],atol=1e-9,rtol=1e-8)
np.testing.assert_allclose(scale,f['scale'],atol=1e-10,rtol=1e-8)
threshold=float(np.mean(np.quantile(observed[:i].mean(axis=1),[.2,.8])))
assert abs(threshold-float(f['threshold'])) < 1e-9
profiles=0; fresh_profiles=0
for row in selection['networks']:
    name=row['network']; validation_losses=[]
    for choice in row['candidates']:
        k=choice['candidate']; a=np.load(OLD/f'{name}-candidate{k}.npz')
        # Independent direct windows for all required offsets/times.
        sim=np.full_like(a['sensor'],np.nan)
        sim[49:]=np.log1p(windows(a['sensor'],np.arange(49,len(sim)))/scale)
        losses=np.array([np.square(sim[times[:i]+d]-observed[:i]).mean() for d in offsets])
        saved=np.load(OUT/f'{name}-candidate{k}-profile.npz')
        np.testing.assert_allclose(losses,saved['training_losses'],rtol=1e-8,atol=1e-9)
        offset=int(offsets[np.argmin(losses)]); assert offset==choice['training_phase']['offset']
        val=float(np.square(sim[times[i:j][val_mask]+offset]-observed[i:j][val_mask]).mean())
        assert np.isclose(val,choice['validation_mse'],rtol=1e-8,atol=1e-9)
        validation_losses.append(val); profiles+=1
        if k==row['candidate']:
            prefix_losses=np.array([np.square(sim[times[:j]+d]-observed[:j]).mean() for d in offsets])
            prefix_offset=int(offsets[np.argmin(prefix_losses)])
            assert prefix_offset==row['prefix_phase']['offset']
            saved=np.load(OUT/f'{name}-reference.npz')
            np.testing.assert_allclose(prefix_losses,saved['prefix_losses'],rtol=1e-8,atol=1e-9)
            np.testing.assert_allclose(sim[times[j:]+prefix_offset],saved['prediction'],rtol=1e-8,atol=1e-9)
    assert int(np.argmin(validation_losses))==row['candidate']
    for seed in [701,702]:
        a=np.load(OUT/f'{name}-seed{seed}.npz'); sim=np.full_like(a['sensor'],np.nan)
        sim[49:]=np.log1p(windows(a['sensor'],np.arange(49,len(sim)))/scale)
        losses=np.array([np.square(sim[times[:j]+d]-observed[:j]).mean() for d in offsets])
        offset=int(offsets[np.argmin(losses)])
        saved=next(x for x in evaluation['fresh_phase_profiles'] if x['network']==name and x['seed']==seed)
        assert offset==saved['phase']['offset']
        np.testing.assert_allclose(losses,a['prefix_losses'],rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(sim[times[j:]+offset],a['prediction'],rtol=1e-8,atol=1e-9)
        fresh_profiles+=1

for name in ['persistence','prefix_mean','zero_envelope']:
    saved=np.load(OUT/f'baseline-{name}.npz')
    for key,block,n in [('validation',observed[:i],j-i),('prediction',observed[:j],len(times)-j)]:
        last=block[-1] if name=='persistence' else block.mean(axis=0) if name=='prefix_mean' else np.zeros(observed.shape[1])
        np.testing.assert_allclose(saved[key],np.broadcast_to(last,(n,len(last))),rtol=1e-8,atol=1e-9)
trend=next(x for x in selection['baselines'] if x['name']=='damped_trend')
for key,block,n in [('validation',observed[:i],j-i),('prediction',observed[:j],len(times)-j)]:
    slope=np.polyfit(np.arange(10),block[-10:],1)[0]
    distance=np.array([sum(trend['damping']**k for k in range(1,h+1)) for h in range(1,n+1)])
    expected=np.maximum(0,block[-1]+distance[:,None]*slope)
    np.testing.assert_allclose(expected,np.load(OUT/'baseline-damped_trend.npz')[key],rtol=1e-8,atol=1e-9)
ar=next((x for x in selection['baselines'] if x['name']=='autoregression'),None)
if ar:
    a=np.load(OUT/'baseline-autoregression.npz')
    if ar['refit_unstable_persistence_fallback']:
        expected=np.broadcast_to(observed[j-1],a['prediction'].shape)
    else:
        history=list(observed[:j]); coefficient=a['coefficients']; mean=a['mean']
        for h in range(len(times)-j):
            value=mean.copy()
            for lag,c in enumerate(coefficient): value+=c*(history[-lag-1]-mean)
            history.append(np.maximum(0,value))
        expected=np.array(history[j:])
    np.testing.assert_allclose(expected,a['prediction'],rtol=1e-8,atol=1e-9)
for row in selection['baselines']:
    pred=np.load(OUT/f"baseline-{row['name']}.npz")['validation']
    val=np.square(pred[val_mask]-observed[i:j][val_mask]).mean()
    assert np.isclose(val,row['validation_mse'],rtol=1e-8,atol=1e-9)
assert min(selection['baselines'],key=lambda r:r['validation_mse'])['name']==selection['selected_baseline']

truth=observed[j:][mask]; replay_rows=[]
for row in evaluation['rows']:
    name=row['method']; kind=row['kind']
    if kind=='baseline': pred=np.load(OUT/f'baseline-{name}.npz')['prediction'][mask]
    elif kind=='reference_seed401': pred=np.load(OUT/f'{name}-reference.npz')['prediction'][mask]
    elif kind=='fresh_seed_mean_prediction':
        pred=(np.load(OUT/f'{name}-seed701.npz')['prediction'][mask]+np.load(OUT/f'{name}-seed702.npz')['prediction'][mask])/2
    else: pred=np.load(OUT/f'{name}-seed{kind[-3:]}.npz')['prediction'][mask]
    mse=float(np.square(pred-truth).sum()/truth.size)
    mae=float(np.abs(pred-truth).sum()/truth.size)
    mismatch=float(np.mean((pred.mean(axis=1)>threshold)!=(truth.mean(axis=1)>threshold)))
    assert np.isclose(mse,row['errors']['mse'],rtol=1e-8,atol=1e-9)
    assert np.isclose(mae,row['errors']['mae'],rtol=1e-8,atol=1e-9)
    assert mismatch==row['errors']['high_activity_proxy_mismatch']
    replay_rows.append(dict(method=name,kind=kind,mse=mse))
for name,digest in evaluation['artifact_sha256'].items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest
for path,digest in selection['reused_artifact_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
assert not any('/parameters/' in p or 'source_timeseries' in p for p in selection['input_paths'])
summary=dict(passed=True,independent_direct_window_features=True,candidate_profiles_replayed=profiles,
             fresh_seed_profiles_replayed=fresh_profiles,final_metric_rows_replayed=len(replay_rows),
             baseline_predictions_checked=True,artifact_hashes_verified=True,
             rows=replay_rows,runtime_seconds=time.time()-start)
(OUT/'replay.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'}))
