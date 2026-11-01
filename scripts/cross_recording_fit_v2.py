"""Exploratory other-recording check after fit choices, using common channels."""
import json
import re
from pathlib import Path
import numpy as np
from intervention.data import load_recording,load_generator

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2/epileptor-fit'
fit=json.loads((OUT/'fit-summary.json').read_text())
evaluation=json.loads((OUT/'evaluation-summary.json').read_text())
def channels(run):
    header=(ROOT/f'data/vec/VirtualEpilepticCohort/sub-010/ses-01/ieeg/sub-010_ses-01_task-simulatedseizure_acq-clinicalhypothesis_run-{run}_ieeg.vhdr').read_text()
    return [line.split('=',1)[1].split(',')[0] for line in header.splitlines() if re.match(r'^Ch\d+=',line)]
def error(a,b):
    ca=np.cov(a,rowvar=False);cb=np.cov(b,rowvar=False)
    return float(np.linalg.norm(ca-cb)/np.linalg.norm(cb))
reference=load_recording(ROOT/'data/vec','sub-010','01')
g1=load_generator(ROOT/'data/vec','sub-010','01')
names1=channels('01');rows=[]
for run in ['02','03']:
    o=load_recording(ROOT/'data/vec','sub-010',run)
    names=channels(run);shared=sorted(set(names1)&set(names))
    a=[names1.index(c) for c in shared];b=[names.index(c) for c in shared]
    training=reference.sensors[:int(.6*len(reference.sensors)),a]/1e6
    observed=o.sensors[:,b]/1e6
    truth=load_generator(ROOT/'data/vec','sub-010',run)
    mismatch=dict(x0_regions_different=int(np.count_nonzero(g1['p'][0]!=truth['p'][0])),
                  other_parameter_entries_different=int(np.count_nonzero(g1['p'][1:]!=truth['p'][1:])),
                  coupling_factor_different=bool(g1['factor']!=truth['factor']),
                  noise_different=bool(not np.array_equal(g1['nsig'],truth['nsig'])))
    for row in evaluation['rows']:
        sensor=np.load(OUT/f"{row['network']}-evaluation-seed{row['seed']}.npz")['sensor'][:,a]
        rows.append(dict(recording=run,network=row['network'],seed=row['seed'],common_channels=len(shared),
                         covariance_relative_error=error(sensor,observed),
                         training_feature_baseline_error=error(training,observed),
                         evaluator_generator_comparison=mismatch))
result=dict(phase='exploratory_other_recording_validation',subject='sub-010',rows=rows,
            selection_changes=0,additional_simulations=0,
            limitation='Other recordings were audited before this check but never entered fit selection. Same development patient, potentially different generator conditions; not independent patients or confirmatory evidence.')
(OUT/'cross-recording.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
