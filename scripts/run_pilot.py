"""Bounded GCP-only known-parameter DEVELOPMENT pilot; no clinical claims."""
import argparse,hashlib,json,os,platform,time,zipfile
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from intervention.data import load_observation,load_generator
from intervention.model import simulate,normalized_weights
from intervention.policies import sensor_onset_scores,network_scores,ranking,choose_ranked
from intervention.metrics import burden,ictal_mask,reduction

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research','GCP execution only'
os.chdir(ROOT)


def prepare():
    o=load_observation(ROOT/'data/vec');score=sensor_onset_scores(o)
    net=network_scores(o,score)
    # Common candidate set uses only pretreatment sensor proxy and anatomy.
    ids=list(dict.fromkeys(ranking(score,o.labels)[:4]+ranking(net,o.labels)[:2]+ranking(o.weights.sum(axis=0),o.labels)[:2]))
    candidates=[(i,) for i in ids]
    f=dict(version='pilot-v1',phase='known_parameter_development',subject='sub-003',
           reference_subject='sub-010',cost='one whole parcel; equal count proxy, not tissue volume',
           horizon=8000,transient=1000,detector='100-sample mean x1 > -1.0',
           planning_seed=101,evaluation_scenarios=[{'seed':201,'coupling_multiplier':1.,'x0_shift':0.},{'seed':202,'coupling_multiplier':1.2,'x0_shift':0.05}],
           candidates=[list(a) for a in candidates],candidate_generation='top 4 sensor proxy union top 2 localization-weighted strength union top 2 outgoing strength; alphabetical ties',
           localization='exploratory simulated-SEEG variance-onset proxy; not independently annotated clinical SOZ',
           policies={'SOZ_proxy':list(choose_ranked(score,o.labels,candidates)),'network_heuristic':list(choose_ranked(net,o.labels,candidates))},
           information_boundary='SOZ and heuristic use only Observation; personal and donor planners explicitly access evaluator parameters as ORACLE controls; not an equal-information primary comparison',
           mapping_gate='atlas order exact; sensor projection approximate (18.6% residual); physical time and volume unverified',
           intervention='zero incoming/outgoing edges; same intact max-weight normalization; silent removed parcel; fixed original denominator',
           workers=1,threads=1,timeout_seconds=3600,host=platform.node(),
           tests='5 scientific-invariant tests passed on GCP before baseline',
           limits='no held-out patients, no fitted realistic planner, no established-method replication, no clinical or counterfactual validation',
           code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))})
    path=ROOT/'manifests/pilot-v1.json'
    if path.exists():raise RuntimeError('Do not overwrite a prepared run')
    path.write_text(json.dumps(f,indent=2)+'\n');print(json.dumps(f,indent=2))


def run():
    f=json.loads((ROOT/'manifests/pilot-v1.json').read_text())
    for name,digest in f['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Source changed since prepare'
    o=load_observation(ROOT/'data/vec');g=load_generator(ROOT/'data/vec')
    with zipfile.ZipFile(ROOT/'data/vec/VirtualEpilepticCohort/derivatives/tvb/sub-010/struct/sub-010_connectome.zip') as z:
        labels=tuple(s.split()[0] for s in z.read('centres.txt').decode().splitlines())
        assert labels==o.labels
        donor,_=normalized_weights(np.loadtxt(z.open('weights.txt')))
    candidates=[tuple(a) for a in f['candidates']]
    out=ROOT/'outputs/pilot-v1';out.mkdir(exist_ok=True)
    start=time.monotonic();budget=json.loads((ROOT/'outputs/pilot-budget.json').read_text())
    deadline=budget['start_epoch']+budget['max_wall_seconds'];rows=[]
    def evaluate(stage,w,action,seed,multiplier=1.,shift=0.):
        if deadline-time.time()<90:raise TimeoutError('Pilot wall-time cap: insufficient reserve for next run')
        t=time.monotonic();params={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in g.items()}
        params['p'][0]+=shift;params['factor']*=multiplier
        states=simulate(w,**params,seed=seed,action=action,horizon=f['horizon'])
        filename=f'{stage}-seed{seed}-action'+('_'.join(map(str,action)) if action else 'none')+'.npy'
        np.save(out/filename,states[:,0,:])
        mask=ictal_mask(states,action)
        # Onset/propagation split is evaluator-only generator x0 > -2.1, explicitly oracle.
        ez=g['p'][0]>-2.1
        row=dict(stage=stage,action=list(action),seed=seed,cost_parcels=len(action),
                 burden=burden(states,action),onset_generator_burden=int(mask[1000:,ez].sum()),
                 nononset_generator_burden=int(mask[1000:,~ez].sum()),
                 detector_sensitivity={str(th):int(((uniform_filter1d(states[:,0,:],100,axis=0)>th) & ~np.isin(np.arange(len(w)),action))[1000:].sum()) for th in [-1.2,-.8]},
                 runtime_seconds=time.monotonic()-t,x1_path='outputs/pilot-v1/'+filename,
                 x1_sha256=hashlib.sha256((out/filename).read_bytes()).hexdigest())
        rows.append(row)
        with (out/'rows.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        return row['burden']
    failure=None
    try:
        # Same number of candidate simulations and same parameters/seeds across model planners.
        selections=dict(f['policies'])
        for name,w in [('personal_oracle',o.weights),('donor_oracle',donor)]:
            scores=[evaluate(name,w,a,101) for a in candidates]
            selections[name]=list(min(zip(scores,candidates),key=lambda z:(z[0],tuple(o.labels[i] for i in z[1])))[1])
        # Selections are saved before evaluator outcomes from fresh scenarios are opened.
        (out/'selections.json').write_text(json.dumps(selections,indent=2)+'\n')
        for scenario in f['evaluation_scenarios']:
            evaluate('evaluation',o.weights,(),scenario['seed'],scenario['coupling_multiplier'],scenario['x0_shift'])
            for a in candidates:evaluate('evaluation',o.weights,a,scenario['seed'],scenario['coupling_multiplier'],scenario['x0_shift'])
        summary={'phase':f['phase'],'selections':selections,'results':[]}
        for scenario in f['evaluation_scenarios']:
            s=[r for r in rows if r['stage']=='evaluation' and r['seed']==scenario['seed']]
            untreated=next(r['burden'] for r in s if not r['action'])
            best=min(r['burden'] for r in s if r['action'])
            for policy,a in selections.items():
                value=next(r['burden'] for r in s if r['action']==a)
                summary['results'].append(dict(policy=policy,scenario=scenario,R=reduction(value,untreated),regret_fraction=(value-best)/untreated,cost_parcels=1))
    except Exception as e:
        failure=repr(e);summary={'phase':f['phase'],'failure':failure}
        raise
    finally:
        summary.update(completed_calls=len(rows),runtime_seconds=time.monotonic()-start,
                       elapsed_since_first_baseline_seconds=time.time()-budget['start_epoch'],failure=failure)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);a=p.parse_args()
    prepare() if a.mode=='prepare' else run()
