"""Compare release source representations and sensor projections on GCP."""
import csv
import json
import re
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from intervention.data import load_recording

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2'
start=time.monotonic()
results=[]
for subject, run in [('sub-003','01'),('sub-010','01'),('sub-010','02'),('sub-010','03')]:
    o=load_recording(ROOT/'data/vec',subject,run)
    p=ROOT/f'data/vec/VirtualEpilepticCohort/derivatives/tvb/{subject}/ses-01/clinicalhypothesis'
    prefix=p/f'{subject}_simulated_source_timeseries_run-{run}'
    z=np.load(prefix.with_suffix('.npz'))
    source=z['source_signal']; times=z['time_steps']
    row=next(csv.DictReader(prefix.with_suffix('.tsv').open(),delimiter='\t'))
    def numbers(s): return np.array([float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?',s)])
    tt=numbers(row['time_steps'])
    excerpt=np.array([numbers(line) for line in row['source_signal'].splitlines() if len(numbers(line))])
    indices=[int(np.argmin(abs(times-t))) for t in tt]
    actual=source[np.ix_(indices,[0,1,2,len(o.labels)-3,len(o.labels)-2,len(o.labels)-1])]
    assert excerpt.shape==actual.shape
    errors=abs(actual-excerpt)
    # Expected start time comes from TSV metadata; scan crop only if it fails length.
    lag=indices[0]
    assert lag+len(o.sensors)<=len(source)
    prediction=source[lag:lag+len(o.sensors)]@o.gain.T
    pc=prediction-prediction.mean(axis=0)
    yc=o.sensors-o.sensors.mean(axis=0)
    scale=float(np.sum(prediction*o.sensors)/np.sum(prediction**2))
    centered_scale=float(np.sum(pc*yc)/np.sum(pc**2))
    corrs=np.sum(pc*yc,axis=0)/(np.linalg.norm(pc,axis=0)*np.linalg.norm(yc,axis=0))
    smooth=[]
    for window in [1,5,25,100]:
        a=uniform_filter1d(pc,window,axis=0,mode='nearest')
        b=uniform_filter1d(yc,window,axis=0,mode='nearest')
        ss=float(np.sum(a*b)/np.sum(a*a))
        smooth.append({'window_samples':window,'centered_relative_rmse':float(np.linalg.norm(a*ss-b)/np.linalg.norm(b))})
    singular=np.linalg.svd(o.gain,compute_uv=False)
    results.append(dict(subject=subject,run=run,source_samples=len(source),sensor_samples=len(o.sensors),
                        crop=lag,npz_tsv_excerpt_max_abs_error=float(errors.max()),
                        npz_tsv_excerpt_fraction_beyond_5e_minus9=float((errors>5e-9).mean()),
                        same_excerpt_within_rounding=bool((errors<=5e-9).all()),
                        projection_scale=scale,raw_projection_relative_rmse=float(np.linalg.norm(prediction*scale-o.sensors)/np.linalg.norm(o.sensors)),
                        centered_projection_relative_rmse=float(np.linalg.norm(pc*centered_scale-yc)/np.linalg.norm(yc)),
                        projection_corr_quantiles=np.quantile(corrs,[0,.05,.5,.95,1]).tolist(),
                        gain_shape=list(o.gain.shape),gain_numerical_rank=int((singular>singular[0]*1e-10).sum()),
                        gain_condition_number=float(singular[0]/singular[-1]),smoothing_diagnostics=smooth))
    print(json.dumps(results[-1]),flush=True)
summary=dict(phase='development_release_consistency_audit',results=results,runtime_seconds=time.monotonic()-start,
             note='Source snippets are evaluator diagnostics; no source-derived observation matrix is used by a planner.')
(OUT/'source-audit.json').write_text(json.dumps(summary,indent=2)+'\n')
