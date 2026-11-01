"""Separate evaluator opens generator only after all observation fits are saved."""
import json
import time
from pathlib import Path
import numpy as np
from intervention.data import load_recording,load_generator
from intervention.model import simulate
from intervention.covariance_fit import feature_error,covariance

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2/epileptor-fit'
f=json.loads((ROOT/'manifests/epileptor-fit-v2.json').read_text())
fit=json.loads((OUT/'fit-summary.json').read_text())
budget=json.loads((OUT.parent/'budget.json').read_text())
start=time.monotonic();rows=[]
o=load_recording(ROOT/'data/vec',f['subject']);y=o.sensors/1e6
training=y[:int(.6*len(y))];test=y[int(.8*len(y))+50:]
truth=load_generator(ROOT/'data/vec',f['subject'])['p'][0]
priors=f['fixed_priors']
initial=np.array([-1.46242601,-9.69344913,2.99029597,-1.11181819,0.,-0.438727802])
for choice in fit['selections']:
    artifact=np.load(OUT/f"{choice['network']}-candidate{choice['candidate']}.npz")
    for seed in f['evaluation_seeds']:
        assert time.time()-budget['start_epoch']<budget['max_wall_seconds']-100
        s=simulate(artifact['weights'],artifact['parameters'],initial,np.array(priors['nsig']),
                   factor=priors['coupling_factor'],dt=priors['dt'],horizon=priors['horizon'],seed=seed,ntau=priors['ntau'])
        sensor=(s[1000:,3]-s[1000:,0])@o.gain.T
        np.savez(OUT/f"{choice['network']}-evaluation-seed{seed}.npz",sensor=sensor)
        row=dict(network=choice['network'],seed=seed,errors_against_development_test_block=feature_error(sensor,test),
                 x0_mae=float(np.mean(abs(artifact['parameters'][0]-truth))),
                 constant_minus2p7_x0_mae=float(np.mean(abs(-2.7-truth))))
        rows.append(row);print(json.dumps(row),flush=True)
summary=dict(phase='development_feature_prediction_not_intervention',rows=rows,
             training_feature_baseline=feature_error(training,test),
             training_covariance_norm=float(np.linalg.norm(covariance(training))),
             test_covariance_norm=float(np.linalg.norm(covariance(test))),
             simulator_calls=len(rows),runtime_seconds=time.monotonic()-start,
             limitation='Stationary simulated feature distributions versus a later temporal block, not time-aligned forecasts or an independently recorded seizure. The final block was already used by the linear development experiment.')
(OUT/'evaluation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
