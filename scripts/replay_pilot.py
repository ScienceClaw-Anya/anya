"""Independent GCP replay using cumulative sums, without importing pilot metrics."""
import hashlib,json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
os.chdir(ROOT)
out=ROOT/'outputs/pilot-v1'
rows=[json.loads(s) for s in (out/'rows.jsonl').read_text().splitlines()]
checks=[]
for row in rows:
    file=ROOT/row['x1_path']
    assert hashlib.sha256(file.read_bytes()).hexdigest()==row['x1_sha256']
    x=np.load(file);padded=np.pad(x,((50,49),(0,0)),mode='edge')
    summed=np.concatenate([np.zeros((1,x.shape[1])),np.cumsum(padded,axis=0)],axis=0)
    avg=(summed[100:]-summed[:-100])/100
    active=avg>-1.;active[:,row['action']]=False
    counted=int(np.count_nonzero(active[1000:]))
    assert counted==row['burden'],(file,counted,row['burden'])
    # Pilot sensitivity calls used SciPy's default reflect boundary, whereas
    # the primary detector explicitly uses nearest. Replay the saved definition,
    # preserving this discrepancy instead of silently changing original results.
    symmetric=np.pad(x,((50,49),(0,0)),mode='symmetric')
    sc=np.concatenate([np.zeros((1,x.shape[1])),np.cumsum(symmetric,axis=0)],axis=0)
    alternate_average=(sc[100:]-sc[:-100])/100
    for th,expected in row['detector_sensitivity'].items():
        alternate=alternate_average>float(th);alternate[:,row['action']]=False
        assert int(np.count_nonzero(alternate[1000:]))==expected,(file,th)
    assert len(row['action'])==row['cost_parcels']
    checks.append(dict(stage=row['stage'],seed=row['seed'],action=row['action'],burden=counted))
summary=json.loads((out/'summary.json').read_text())
for result in summary['results']:
    seed=result['scenario']['seed'];action=summary['selections'][result['policy']]
    untreated=next(c['burden'] for c in checks if c['stage']=='evaluation' and c['seed']==seed and not c['action'])
    treated=next(c['burden'] for c in checks if c['stage']=='evaluation' and c['seed']==seed and c['action']==action)
    assert abs((untreated-treated)/untreated-result['R'])<1e-12
report=dict(replayed_trajectories=len(checks),thresholds=[-1.2,-1.,-.8],hashes_verified=True,burdens_exact=True,reductions_agree=True,
            initial_failure='Sensitivity boundary handling: primary nearest, sensitivity default reflect. Original output preserved; replay corrected to match actual definitions.',
            limitation='Numerical replay is not biological validation; one development patient provides no population confidence interval.')
(out/'replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
