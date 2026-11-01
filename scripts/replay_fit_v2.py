"""Independent feature and parameter-error replay, confined to the GCP evaluator."""
import json
from pathlib import Path
import numpy as np
from intervention.data import load_recording,load_generator

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2/epileptor-fit'
fit=json.loads((OUT/'fit-summary.json').read_text())
evaluation=json.loads((OUT/'evaluation-summary.json').read_text())
assert fit['forbidden_input_guard_enabled']
assert all('/parameters/' not in p and 'source_timeseries' not in p for p in fit['input_paths'])
y=load_recording(ROOT/'data/vec','sub-010').sensors/1e6
training=y[:int(.6*len(y))];test=y[int(.8*len(y))+50:]
truth=load_generator(ROOT/'data/vec','sub-010')['p'][0]
def independent_errors(a,b):
    ca=np.cov(a,rowvar=False,ddof=1);cb=np.cov(b,rowvar=False,ddof=1)
    va=np.std(a,axis=0,ddof=1);vb=np.std(b,axis=0,ddof=1)
    return {'covariance_relative_error':float(np.sqrt(np.sum((ca-cb)**2)/np.sum(cb**2))),
            'sensor_std_relative_error':float(np.sqrt(np.sum((va-vb)**2)/np.sum(vb**2)))}
checks=0
for row in fit['rows']:
    z=np.load(OUT/f"{row['network']}-candidate{row['candidate']}.npz")
    errors=independent_errors(z['sensor'],training)
    for k,v in errors.items(): assert np.isclose(v,row['training_errors'][k],rtol=1e-10,atol=1e-12)
    checks+=1
for selection in fit['selections']:
    rows=[r for r in fit['rows'] if r['network']==selection['network']]
    best=min(rows,key=lambda r:(r['training_errors']['covariance_relative_error'],r['candidate']))
    assert best['candidate']==selection['candidate']
for row in evaluation['rows']:
    selected=next(s for s in fit['selections'] if s['network']==row['network'])
    params=np.load(OUT/f"{row['network']}-candidate{selected['candidate']}.npz")['parameters']
    sensor=np.load(OUT/f"{row['network']}-evaluation-seed{row['seed']}.npz")['sensor']
    errors=independent_errors(sensor,test)
    for k,v in errors.items(): assert np.isclose(v,row['errors_against_development_test_block'][k],rtol=1e-10,atol=1e-12)
    assert np.isclose(np.abs(params[0]-truth).sum()/len(truth),row['x0_mae'],rtol=1e-12)
    checks+=1
for k,v in independent_errors(training,test).items():
    assert np.isclose(v,evaluation['training_feature_baseline'][k],rtol=1e-10,atol=1e-12)
result=dict(feature_comparisons_replayed=checks,choices_match_training_scores=True,feature_errors_agree=True,
            parameter_errors_agree=True,fit_input_log_excludes_sources_and_generators=True)
(OUT/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
