"""GCP-only exploratory observation audit; source arrays never enter a planner."""
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from intervention.data import load_observation

ROOT = Path(__file__).resolve().parents[1]
assert str(ROOT) == '/home/devstar3022/anya-intervention-research', 'GCP only'
OUT = ROOT / 'outputs/development-v2'
OUT.mkdir(exist_ok=True)
budget = OUT / 'budget.json'
if not budget.exists():
    budget.write_text(json.dumps({'start_epoch': time.time(), 'max_wall_seconds': 3600}))
assert time.time() - json.loads(budget.read_text())['start_epoch'] < 3500
start = time.monotonic()
o = load_observation(ROOT / 'data/vec')
root = ROOT / 'data/vec/VirtualEpilepticCohort'
mono = np.loadtxt(root / 'derivatives/tvb/sub-003/struct/sub-003_gain.txt')
names = [r['name'] for r in csv.DictReader((root / 'sub-003/sub-003_electrodes.tsv').open(), delimiter='\t')]
header = (root / 'sub-003/ses-01/ieeg/sub-003_ses-01_task-simulatedseizure_acq-clinicalhypothesis_run-01_ieeg.vhdr').read_text()
channels = [line.split('=', 1)[1].split(',')[0] for line in header.splitlines() if re.match(r'^Ch\d+=', line)]
mapping = {}
for i in range(len(names) - 1):
    a, b = [re.fullmatch(r"([A-Za-z']+)(\d+)", n) for n in names[i:i+2]]
    if a and b and a[1] == b[1]:
        mapping[f'{a[1]}{a[2]}-{b[2]}'] = (i, i+1)
indices = np.array([mapping[c] for c in channels])
lo, hi = mono[indices[:, 0]], mono[indices[:, 1]]
gains = {'absolute_difference': np.abs(hi-lo), 'signed_difference': hi-lo,
         'mean': (hi+lo)/2, 'first_contact': lo, 'second_contact': hi}
observed = o.sensors
yc = observed - observed.mean(axis=0)
yn = np.linalg.norm(yc, axis=0)
length = len(observed)
channel = int(np.argmax(observed.std(axis=0)))
results = []
for name, gain in gains.items():
    full = o.source @ gain.T
    # Candidate crop chosen on one energetic channel; all-channel checks follow.
    corrs = []
    for lag in range(len(full)-length+1):
        x = full[lag:lag+length, channel]
        corrs.append(float(np.corrcoef(x, observed[:, channel])[0, 1]))
    lag = int(np.argmax(np.abs(corrs)))
    prediction = full[lag:lag+length]
    pc = prediction-prediction.mean(axis=0)
    pn = np.linalg.norm(pc, axis=0)
    channel_corr = np.sum(pc*yc, axis=0)/(pn*yn)
    scale = float(np.sum(prediction*observed)/np.sum(prediction**2))
    centered_scale = float(np.sum(pc*yc)/np.sum(pc**2))
    channel_scale = np.sum(pc*yc, axis=0)/np.sum(pc**2, axis=0)
    residual = yc-pc*channel_scale
    results.append(dict(gain=name, lag=lag, global_scale=scale,
                        raw_relative_rmse=float(np.linalg.norm(prediction*scale-observed)/np.linalg.norm(observed)),
                        centered_global_relative_rmse=float(np.linalg.norm(pc*centered_scale-yc)/np.linalg.norm(yc)),
                        channel_affine_relative_rmse=float(np.linalg.norm(residual)/np.linalg.norm(yc)),
                        corr_quantiles=np.quantile(channel_corr, [0, .05, .5, .95, 1]).tolist(),
                        scale_quantiles=np.quantile(channel_scale, [0, .05, .5, .95, 1]).tolist()))
    if name == 'absolute_difference':
        # Is the residual just a constant/gain, or another dynamic mixture?
        np.savez(OUT/'projection-diagnostics.npz', prediction=prediction, residual=residual,
                 channel_correlations=channel_corr, channel_scales=channel_scale)

# A fitted unconstrained linear observation map is a diagnostic, not a valid
# replacement anatomy. Fit first half, predict second half to distinguish
# inconsistent saved realizations from a stable alternative projection.
source = o.source[100:100+length]
cut = length//2
X = np.column_stack([source, np.ones(length)])
coef, _, rank, singular = np.linalg.lstsq(X[:cut], observed[:cut], rcond=1e-8)
prediction = X[cut:] @ coef
unconstrained = dict(fit_samples=cut, validation_samples=length-cut, rank=int(rank),
                     singular_ratio=float(singular[-1]/singular[0]),
                     validation_relative_rmse=float(np.linalg.norm(prediction-observed[cut:])/np.linalg.norm(observed[cut:])),
                     caveat='Temporal blocks in one release; diagnostic mapping learned from evaluator source; never planner input')
result = dict(phase='exploratory_observation_audit', subject=o.subject, variants=results,
              unconstrained_source_projection=unconstrained, runtime_seconds=time.monotonic()-start,
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'projection-summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
