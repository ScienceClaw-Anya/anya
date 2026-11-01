"""Fit sub-010 observations with explicit priors; reject evaluator asset access."""
import hashlib
import itertools
import json
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from intervention.data import load_recording
from intervention.model import simulate,normalized_weights
from intervention.covariance_fit import covariance,source_variances,feature_error

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2/epileptor-fit'
OUT.mkdir(exist_ok=False)
manifest=json.loads((ROOT/'manifests/epileptor-fit-v2.json').read_text())
budget=json.loads((OUT.parent/'budget.json').read_text())
accessed=set()
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0])
        if '/parameters/' in path or 'source_timeseries' in path:
            raise PermissionError('Evaluator assets forbidden in observation-only fit')
        if '/data/vec/' in path: accessed.add(path)
sys.addaudithook(audit)
start=time.monotonic();calls=0
o=load_recording(ROOT/'data/vec',manifest['subject'])
y=o.sensors/1e6 # Verified synthetic exporter V-to-microV scale, not biological calibration.
training=y[:int(.6*len(y))]
q,inverse_error=source_variances(covariance(training),o.gain)
np.save(OUT/'inferred-source-variances.npy',q)
with zipfile.ZipFile(ROOT/'data/vec/VirtualEpilepticCohort/derivatives/tvb/sub-003/struct/sub-003_connectome.zip') as z:
    labels=tuple(l.split()[0] for l in z.read('centres.txt').decode().splitlines());assert labels==o.labels
    donor,_=normalized_weights(np.loadtxt(z.open('weights.txt')))
networks={'personal':o.weights,'donor':donor,'zero':np.zeros_like(o.weights)}
priors=manifest['fixed_priors']
initial=np.array([-1.46242601,-9.69344913,2.99029597,-1.11181819,0.,-0.438727802])
grid=list(itertools.product(manifest['global_candidates']['r'],manifest['global_candidates']['slope']))
x0=np.asarray(manifest['x0_grid'])
def parameters(x,r,slope):
    n=len(x)
    return np.stack([x,np.full(n,priors['Iext']),np.full(n,priors['Iext2']),np.full(n,slope),
                     np.full(n,r),np.full(n,priors['Ks']),np.full(n,priors['Kf']),np.full(n,priors['Kvf'])])
def run(w,p,seed):
    global calls
    assert time.time()-budget['start_epoch']<budget['max_wall_seconds']-100,'Shared wall cap'
    calls+=1
    assert calls<=14 # 2 libraries + 12 candidate trajectories; evaluator gets 6 more.
    return simulate(w,p,initial,np.array(priors['nsig']),factor=priors['coupling_factor'],
                    dt=priors['dt'],horizon=priors['horizon'],seed=seed,ntau=priors['ntau'])
library_params=np.concatenate([parameters(x0,r,s) for r,s in grid],axis=1)
library=[]
for seed in manifest['library_seeds']:
    states=run(np.zeros((library_params.shape[1],library_params.shape[1])),library_params,seed)
    library.append((states[1000:,3]-states[1000:,0]).var(axis=0,ddof=1))
variances=np.mean(library,axis=0).reshape(len(grid),len(x0))
np.savez(OUT/'library.npz',variances=variances,x0=x0)
rows=[];selected=[]
for name,w in networks.items():
    choices=[]
    for k,(r,slope) in enumerate(grid):
        assignments=np.argmin(abs(q[:,None]-variances[k][None,:]),axis=1)
        p=parameters(x0[assignments],r,slope)
        states=run(w,p,manifest['fit_seed'])
        sensor=(states[1000:,3]-states[1000:,0])@o.gain.T
        errors=feature_error(sensor,training)
        path=OUT/f'{name}-candidate{k}.npz'
        np.savez(path,parameters=p,sensor=sensor,weights=w)
        row=dict(network=name,candidate=k,r=r,slope=slope,training_errors=errors,
                 q_above_library_range=int((q>variances[k].max()).sum()),
                 x0_range=[float(p[0].min()),float(p[0].max())])
        rows.append(row);choices.append((errors['covariance_relative_error'],k))
        print(json.dumps(row),flush=True)
    loss,k=min(choices)
    selected.append({'network':name,'candidate':k,'training_loss':loss})
    (OUT/f'{name}-selection.json').write_text(json.dumps(selected[-1],indent=2)+'\n')
summary=dict(phase=manifest['phase'],subject=o.subject,rows=rows,selections=selected,
             independent_source_covariance_relative_error=inverse_error,
             input_paths=sorted(accessed),forbidden_input_guard_enabled=True,
             simulator_calls=calls,runtime_seconds=time.monotonic()-start,
             manifest_sha256=hashlib.sha256((ROOT/'manifests/epileptor-fit-v2.json').read_bytes()).hexdigest())
(OUT/'fit-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
