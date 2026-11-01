"""GCP development forecast using only recordings and anatomy."""
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from intervention.data import load_recording
from intervention.model import normalized_weights
from intervention.forecast import sensor_network_operator,fit_linear,forecast_errors

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2/forecast'
OUT.mkdir(exist_ok=False)
budget=json.loads((OUT.parent/'budget.json').read_text())
assert time.time()-budget['start_epoch']<3400
start=time.monotonic();reports=[]
manifest=json.loads((ROOT/'manifests/forecast-v2.json').read_text())
for subject in manifest['subjects']:
    o=load_recording(ROOT/'data/vec',subject)
    other='sub-010' if subject=='sub-003' else 'sub-003'
    with zipfile.ZipFile(ROOT/f'data/vec/VirtualEpilepticCohort/derivatives/tvb/{other}/struct/{other}_connectome.zip') as z:
        labels=tuple(l.split()[0] for l in z.read('centres.txt').decode().splitlines())
        assert labels==o.labels
        donor,_=normalized_weights(np.loadtxt(z.open('weights.txt')))
    perm=np.random.default_rng(3001).permutation(len(o.labels))
    networks={'personal':o.weights,'donor':donor,'shuffled':o.weights[np.ix_(perm,perm)],'zero':np.zeros_like(o.weights)}
    y=o.sensors
    i,j=int(len(y)*.6),int(len(y)*.8)
    # Global numerical scale from training only. No per-channel fit using test data.
    scale=float(np.sqrt(np.mean((y[:i]-y[:i].mean(axis=0))**2)))
    y=y/scale
    training=y[:i];validation=y[i+50:j];test=y[j+50:]
    for name,w in networks.items():
        candidates=[]
        for ridge in [1e-6,1e-4,1e-2,1.]:
            op=sensor_network_operator(w,o.gain,ridge)
            model=fit_linear(training,op)
            val=forecast_errors(model,validation,10)['mse']
            candidates.append((val,ridge,model))
        val,ridge,model=min(candidates,key=lambda r:r[0])
        # Save the selected model before test errors are read/computed.
        namebase=f'{subject}-{name}'
        np.savez(OUT/(namebase+'.npz'),transition=model.transition,mean=model.mean,scale=scale)
        (OUT/(namebase+'-selection.json')).write_text(json.dumps({'ridge':ridge,'a':model.a,'b':model.b,'validation_mse':val})+'\n')
        report=dict(subject=subject,network=name,train_samples=len(training),validation_samples=len(validation),test_samples=len(test),
                    selected_ridge=ridge,a=model.a,b=model.b,spectral_radius=model.spectral_radius,
                    validation_candidates=[{'ridge':r,'mse':v} for v,r,m in candidates],
                    test_errors={str(h):forecast_errors(model,test,h) for h in manifest['evaluation_horizons_samples']})
        reports.append(report);print(json.dumps(report),flush=True)
summary={'phase':manifest['phase'],'results':reports,'runtime_seconds':time.monotonic()-start,
         'manifest_sha256':hashlib.sha256((ROOT/'manifests/forecast-v2.json').read_bytes()).hexdigest(),
         'limitation':'Linear forecasting of the released sensors; unresolved source-file discrepancies prevent physical source-validation claims. No fitted seizure parameters or intervention choices.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
