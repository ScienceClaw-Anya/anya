"""Run only on authorized GCP host. No interventions; development diagnostics."""
import hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from intervention.data import load_observation,load_generator
from intervention.model import simulate
from intervention.policies import sensor_onset_scores,ranking
from intervention.metrics import burden

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research','GCP execution only'
os.chdir(ROOT);out=ROOT/'outputs';out.mkdir(exist_ok=True)
budget=out/'pilot-budget.json'
if not budget.exists():budget.write_text(json.dumps({'start_epoch':time.time(),'max_wall_seconds':3600}))
limit=json.loads(budget.read_text());assert time.time()-limit['start_epoch']<3600
o=load_observation(ROOT/'data/vec');g=load_generator(ROOT/'data/vec')
score=sensor_onset_scores(o)
audit={'subject':o.subject,'weights_shape':list(o.weights.shape),'gain_shape':list(o.gain.shape),'sensor_shape':list(o.sensors.shape),'source_shape':list(o.source.shape),'volume_units':'unverified','volume_positive':bool((o.volumes_unverified>0).all()),'volume_range':[float(o.volumes_unverified.min()),float(o.volumes_unverified.max())],'sensor_onset_positive_parcels':int((score>0).sum()),'onset_top_indices':ranking(score,o.labels)[:12]}
projected=o.source@o.gain.T
# Search only offsets allowed by source vs sensor length; inspect one energetic channel.
channel=int(np.argmax(o.sensors.std(axis=0)));length=len(o.sensors)
scores=[np.corrcoef(projected[k:k+length,channel],o.sensors[:,channel])[0,1] for k in range(len(projected)-length+1)]
lag=int(np.argmax(scores));prediction=projected[lag:lag+length]
scale=float(np.sum(prediction*o.sensors)/np.sum(prediction**2))
audit.update(source_sensor_offset=lag,source_sensor_correlation=float(max(scores)),projection_scale=scale,projection_relative_rmse=float(np.linalg.norm(scale*prediction-o.sensors)/np.linalg.norm(o.sensors)))
t=time.monotonic();states=simulate(o.weights,**g,seed=101,horizon=8000)
np.save(out/'untreated-101.npy',states)
summary={'runtime_seconds':time.monotonic()-t,'host':platform.node(),'phase':'known_parameter_development','horizon_model_units':8000,'dt':g['dt'],'seed':101,'burden_node_model_time':burden(states),'x1_mean_quantiles':np.quantile(states[1000:,0,:].mean(axis=0),[0,.25,.5,.75,1]).tolist(),'x1_smoothed_max_quantiles':np.quantile(uniform_filter1d(states[:,0,:],100,axis=0)[1000:].max(axis=0),[0,.25,.5,.75,1]).tolist(),'source_std_correlation':float(np.corrcoef((states[:,3,:]-states[:,0,:]).std(axis=0),o.source.std(axis=0))[0,1]),'states_sha256':hashlib.sha256(states.tobytes()).hexdigest(),'audit':audit}
(out/'baseline-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
