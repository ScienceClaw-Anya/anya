"""Independent GCP checks of saved masks and forecast errors; no metric imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from intervention.data import load_recording

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2'
rows=json.loads((OUT/'network/summary.json').read_text())['rows']
masks={};comparisons=[]
for row in rows:
    stem=f"{row['subject']}-{row['network']}-m{row['coupling_multiplier']:g}-seed{row['seed']}"
    path=OUT/'network'/f'{stem}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['output_sha256']
    x=np.load(path)['x1']
    padded=np.pad(x,((50,49),(0,0)),mode='edge')
    total=np.concatenate([np.zeros((1,x.shape[1])),np.cumsum(padded,axis=0)])
    mask=((total[100:]-total[:-100])/100>-1)[1000:]
    assert int(mask.sum())==row['burden']
    masks[(row['subject'],row['network'],row['coupling_multiplier'],row['seed'])]=mask
for subject in ['sub-003','sub-010']:
    for seed in [301,302]:
        a=masks[(subject,'personal',1.,seed)];b=masks[(subject,'zero',1.,seed)]
        comparisons.append({'subject':subject,'seed':seed,'paired_zero_mask_disagreement':float(np.mean(a!=b)),
                            'burden_difference_personal_minus_zero':int(a.sum())-int(b.sum())})
    a=masks[(subject,'personal',1.,301)];b=masks[(subject,'personal',1.,302)]
    comparisons.append({'subject':subject,'personal_seed_change_mask_disagreement':float(np.mean(a!=b))})

forecasts=json.loads((OUT/'forecast/summary.json').read_text())['results']
count=0
for row in forecasts:
    o=load_recording(ROOT/'data/vec',row['subject'])
    saved=np.load(OUT/'forecast'/f"{row['subject']}-{row['network']}.npz")
    y=o.sensors/float(saved['scale']);j=int(len(y)*.8);test=y[j+50:]
    A=saved['transition'];mean=saved['mean']
    power=np.eye(len(mean))
    for horizon in range(1,51):
        power=power@A
        if horizon not in [1,10,50]: continue
        expected=row['test_errors'][str(horizon)]
        prediction=(power@(test[:-horizon]-mean).T).T+mean
        mse=float(np.square(prediction-test[horizon:]).sum()/prediction.size)
        assert np.isclose(mse,expected['mse'],rtol=1e-10,atol=1e-12)
        assert np.isclose(np.mean((test[:-horizon]-test[horizon:])**2),expected['persistence_mse'],rtol=1e-12)
        count+=1
result=dict(network_trajectories_replayed=len(rows),network_burdens_exact=True,
            paired_network_comparisons=comparisons,forecast_error_checks=count,forecast_errors_agree=True,
            note='Numerical replay only; development patients and temporal forecast blocks do not establish intervention or clinical benefit.')
(OUT/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
