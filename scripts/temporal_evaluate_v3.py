"""Fresh-seed timing fits then final-block evaluation, without hidden assets."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from intervention.model import simulate
from intervention.data import load_recording
from intervention.temporal import envelope,phase_search,errors

ROOT = Path(__file__).resolve().parents[1]
assert str(ROOT) == '/home/devstar3022/anya-intervention-research'
OUT = ROOT/'outputs/development-v3'
OLD = ROOT/'outputs/development-v2/epileptor-fit'
assert not (OUT/'evaluation.json').exists()
manifest = json.loads((ROOT/'manifests/development-v3.json').read_text())
selection = json.loads((OUT/'selection.json').read_text())
budget = json.loads((OUT/'budget.json').read_text())
old_manifest = json.loads((ROOT/'manifests/epileptor-fit-v2.json').read_text())
start = time.time(); calls = 0
def audit(event,args):
    if event == 'open' and isinstance(args[0],(str,bytes)):
        if '/parameters/' in str(args[0]) or 'source_timeseries' in str(args[0]):
            raise PermissionError('Released evaluator assets forbidden in v3')
sys.addaudithook(audit)
f = np.load(OUT/'features.npz'); observed = f['observed']; times = f['times']; j = int(f['j'])
mask = f['test_mask']; threshold = float(f['threshold'])
o = load_recording(ROOT/'data/vec',manifest['subject'])
priors = old_manifest['fixed_priors']
initial = np.array([-1.46242601,-9.69344913,2.99029597,-1.11181819,0.,-.438727802])
fresh_profiles = []
# Every fresh prediction is saved before any final-block errors are calculated.
for row in selection['networks']:
    name = row['network']
    artifact = np.load(OLD/f"{name}-candidate{row['candidate']}.npz")
    for seed in manifest['mechanistic_fit']['fresh_seeds']:
        assert calls < budget['maximum_new_simulator_calls']
        assert time.time()-budget['start_epoch'] < budget['max_wall_seconds']-100
        path = OUT/f'{name}-seed{seed}.npz'; assert not path.exists()
        calls += 1
        states = simulate(artifact['weights'],artifact['parameters'],initial,np.array(priors['nsig']),
                          factor=priors['coupling_factor'],dt=priors['dt'],horizon=priors['horizon'],
                          seed=seed,ntau=priors['ntau'])
        sensor = (states[1000:,3]-states[1000:,0])@o.gain.T
        sim = envelope(sensor,f['scale'],manifest['feature']['window_samples'])
        offset,profile,diagnostic = phase_search(sim,observed[:j],times[:j],f['offsets'])
        np.savez(path,sensor=sensor,prediction=sim[times[j:]+offset],
                 prefix_prediction=sim[times[:j]+offset],prefix_losses=profile)
        record = dict(network=name,seed=seed,phase=diagnostic)
        fresh_profiles.append(record)
        print(json.dumps(record),flush=True)
(OUT/'fresh-selections.json').write_text(json.dumps(fresh_profiles,indent=2)+'\n')
truth = observed[j:][mask]; target_times = times[j:][mask]
rows = []
for baseline in selection['baselines']:
    name = baseline['name']; pred = np.load(OUT/f'baseline-{name}.npz')['prediction'][mask]
    rows.append(dict(method=name,kind='baseline',errors=errors(pred,truth,target_times,threshold)))
baseline_mse = next(r['errors']['mse'] for r in rows if r['method']==selection['selected_baseline'])
for network in selection['networks']:
    name = network['network']
    pred = np.load(OUT/f'{name}-reference.npz')['prediction'][mask]
    rows.append(dict(method=name,kind='reference_seed401',errors=errors(pred,truth,target_times,threshold)))
    predictions = []
    for seed in manifest['mechanistic_fit']['fresh_seeds']:
        pred = np.load(OUT/f'{name}-seed{seed}.npz')['prediction'][mask]; predictions.append(pred)
        rows.append(dict(method=name,kind=f'fresh_seed{seed}',errors=errors(pred,truth,target_times,threshold)))
    rows.append(dict(method=name,kind='fresh_seed_mean_prediction',
                     errors=errors(np.mean(predictions,axis=0),truth,target_times,threshold)))
for row in rows: row['mse_ratio_to_selected_baseline'] = row['errors']['mse']/baseline_mse
personal = [r for r in rows if r['method']=='personal']
prediction_gate = (next(r['errors']['mse'] for r in personal if r['kind']=='reference_seed401') <= .9*baseline_mse
                   and all(r['errors']['mse'] < baseline_mse for r in personal if r['kind'] in ['fresh_seed701','fresh_seed702']))
personal_mean = next(r['errors']['mse'] for r in personal if r['kind']=='fresh_seed_mean_prediction')
zero_mean = next(r['errors']['mse'] for r in rows if r['method']=='zero' and r['kind']=='fresh_seed_mean_prediction')
hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.npz'))}
summary = dict(phase=manifest['phase'],selected_baseline=selection['selected_baseline'],rows=rows,
               fresh_phase_profiles=fresh_profiles,provisional_personal_prediction_gate=prediction_gate,
               personal_to_zero_mean_prediction_mse_ratio=personal_mean/zero_mean,
               provisional_network_screen=bool(prediction_gate and personal_mean <= .9*zero_mean),
               new_simulator_calls=calls,evaluation_runtime_seconds=time.time()-start,
               wall_seconds_to_evaluation=time.time()-budget['start_epoch'],
               artifact_sha256=hashes,selection_sha256=hashlib.sha256((OUT/'selection.json').read_bytes()).hexdigest(),
               held_out_patients_inspected=0)
(OUT/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','artifact_sha256','fresh_phase_profiles']}),flush=True)
