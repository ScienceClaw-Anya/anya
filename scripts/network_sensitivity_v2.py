"""Known-parameter development controls, separate from observation-only fitting."""
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from intervention.data import load_observation, load_generator
from intervention.model import simulate, normalized_weights
from intervention.metrics import ictal_mask

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research', 'GCP only'
OUT=ROOT/'outputs/development-v2/network'
OUT.mkdir(exist_ok=False)
budget=json.loads((OUT.parent/'budget.json').read_text())
start=time.monotonic()
rows=[]
for subject in ['sub-003', 'sub-010']:
    o=load_observation(ROOT/'data/vec',subject)
    g=load_generator(ROOT/'data/vec',subject)
    other='sub-010' if subject=='sub-003' else 'sub-003'
    with zipfile.ZipFile(ROOT/f'data/vec/VirtualEpilepticCohort/derivatives/tvb/{other}/struct/{other}_connectome.zip') as z:
        labels=tuple(l.split()[0] for l in z.read('centres.txt').decode().splitlines())
        assert labels==o.labels
        donor,_=normalized_weights(np.loadtxt(z.open('weights.txt')))
    perm=np.random.default_rng(3001).permutation(len(o.labels))
    shuffled=o.weights[np.ix_(perm,perm)]
    configurations=[('personal',o.weights,1.,301),('zero',np.zeros_like(o.weights),1.,301),
                    ('donor',donor,1.,301),('shuffled',shuffled,1.,301),
                    ('personal',o.weights,1.,302),('zero',np.zeros_like(o.weights),1.,302)]
    if subject=='sub-003':
        configurations += [('stress',o.weights,m,301) for m in [10.,100.,1000.]]
    references={}
    for name,w,mult,seed in configurations:
        assert time.time()-budget['start_epoch']<budget['max_wall_seconds']-100, 'Wall cap'
        t=time.monotonic()
        args={**g,'factor':g['factor']*mult}
        s=simulate(w,**args,seed=seed,horizon=8000)
        mask=ictal_mask(s)[1000:]
        source=s[1000:,3]-s[1000:,0]
        sensor=source@o.gain.T
        feature=sensor.std(axis=0)
        # Coupling RHS relative to intrinsic slow RHS, evaluated on saved averages.
        y=s[1000::10]
        c=args['factor']*(y[:,0]@w.T-y[:,0]*w.sum(axis=1))
        intrinsic=4*(y[:,0]-g['p'][0])-np.where(y[:,2]<0,.1*y[:,2]**7,0)-y[:,2]
        ratio=np.linalg.norm(g['p'][5]*c)/max(np.linalg.norm(intrinsic),1e-30)
        if name=='personal' and seed==301: references['base']=(mask,feature)
        base,bf=references['base']
        row=dict(subject=subject,network=name,coupling_multiplier=mult,seed=seed,
                 burden=int(mask.sum()),active_regions=int(mask.any(axis=0).sum()),
                 mask_disagreement_vs_personal301=float(np.mean(mask!=base)),
                 sensor_std_relative_change_vs_personal301=float(np.linalg.norm(feature-bf)/np.linalg.norm(bf)),
                 slow_coupling_to_intrinsic_rhs_norm_ratio=float(ratio),
                 runtime_seconds=time.monotonic()-t)
        path=OUT/f'{subject}-{name}-m{mult:g}-seed{seed}.npz'
        np.savez_compressed(path,x1=s[:,0],sensor_std=feature,source_std=source.std(axis=0))
        row['output_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
        with (OUT/'rows.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
summary=dict(phase='known_parameter_development_sensitivity',rows=rows,
             runtime_seconds=time.monotonic()-start,
             caveats=['Stress multipliers are synthetic sensitivity cases, not release corrections or fitted biological values.',
                      'Comparisons use a development detector and model time; neither clinical SOZ nor clinical seizure freedom.',
                      'Donor and shuffled networks are controls, not trained population models.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
