"""Post-pilot exploratory controls on GCP, within the same wall-time cap."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from intervention.data import load_observation,load_generator
from intervention.model import simulate
from intervention.metrics import burden,ictal_mask

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
os.chdir(ROOT);out=ROOT/'outputs/pilot-v1'
budget=json.loads((ROOT/'outputs/pilot-budget.json').read_text())
assert time.time()-budget['start_epoch']<3400,'Not enough remaining pilot time'
start=time.monotonic();o=load_observation(ROOT/'data/vec');g=load_generator(ROOT/'data/vec')
base=json.loads((ROOT/'outputs/baseline-summary.json').read_text())
repeated=simulate(o.weights,**g,seed=101,horizon=8000)
digest=hashlib.sha256(repeated.tobytes()).hexdigest()
assert digest==base['states_sha256'],'Untreated rerun did not reproduce'
del repeated
zero=simulate(np.zeros_like(o.weights),**g,seed=201,horizon=8000)
zero_mask=ictal_mask(zero);np.save(out/'zero-coupling-seed201-x1.npy',zero[:,0,:])
summary=json.loads((out/'summary.json').read_text())
rows=[json.loads(x) for x in (out/'rows.jsonl').read_text().splitlines()]
base_eval=next(x for x in rows if x['stage']=='evaluation' and x['seed']==201 and not x['action'])
coupled=np.load(ROOT/base_eval['x1_path']);cmask=uniform_filter1d(coupled,100,axis=0)>-1
comparison=[]
for action in sorted(set(tuple(a) for a in summary['selections'].values())):
    # Exact zero-coupling intervention result follows from independent nodes;
    # label as analytic removal, not a separately simulated trajectory.
    zm=zero_mask.copy();zm[:,list(action)]=False
    treated=next(x for x in rows if x['stage']=='evaluation' and x['seed']==201 and x['action']==list(action))
    direct=int(cmask[1000:,list(action)].sum())
    total=base_eval['burden']-treated['burden']
    comparison.append({'action':list(action),'coupled_total_reduction':total,'direct_removed_baseline_burden':direct,
                       'survivor_burden_change':total-direct,'zero_coupling_analytic_R':1-float(zm[1000:].sum())/float(zero_mask[1000:].sum())})
projected=o.source[100:100+len(o.sensors)]@o.gain.T
per_channel=[]
for i in range(len(o.gain)):
    pred=projected[:,i];obs=o.sensors[:,i]
    per_channel.append(float(np.corrcoef(pred,obs)[0,1]))
report={'phase':'exploratory_development_controls','reason':'Initial development runs showed no detected non-susceptible recruitment; examine direct-removal contribution without changing pilot-v1',
        'untreated_repeated_hash_matches':True,'zero_coupling_untreated_burden':burden(zero),'coupled_untreated_burden':base_eval['burden'],
        'coupled_zero_ictal_mask_disagreement_fraction':float(np.mean(cmask[1000:]!=zero_mask[1000:])),
        'selected_action_decomposition':comparison,'sensor_projection_channel_correlation_quantiles':np.quantile(per_channel,[0,.05,.25,.5,.75,.95,1]).tolist(),
        'sensor_projection_channels_below_099':int((np.asarray(per_channel)<.99).sum()),
        'runtime_seconds':time.monotonic()-start,'elapsed_since_first_baseline_seconds':time.time()-budget['start_epoch']}
(out/'controls.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
