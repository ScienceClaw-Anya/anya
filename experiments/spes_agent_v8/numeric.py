"""Frozen same-measurement-budget evaluation of learned query-value policies."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v4.acquisition import Acquisition,numeric_choice
from experiments.spes_agent_v4.screen import bootstrap_delta
from .crossfit import write_json
from .features import distance_catalog,feature_vector
from .learner import load_model,predict

BASELINES=['random_0','random_1','coverage','score_focus','uncertainty']
LEARNED=['phase0_bag_mean','phase1_bag_mean','phase1_relative_mean','phase1_ridge','phase1_robust']

def feature_matrix(raw,selected,scores,candidates):
    """Cache observed RMS once; frozen features use only RMS of each trace.

    HAT scoring still receives full revealed waveforms. This equivalent feature
    calculation avoids re-squaring 512 samples for every candidate. No future
    waveform enters the cache. Equivalence is regression-tested.
    """
    distance,_=distance_catalog(raw[...,0]);signal=np.asarray(raw[:,:,selected,-512:],dtype=float)
    rms=np.sqrt(np.square(signal).mean(axis=-1))[...,None];valid=np.abs(raw[:,:,selected,0])>1e-12
    return np.vstack([feature_vector(distance,rms,valid,scores,selected,q) for q in candidates])

def freeze(project,output):
    project,output=Path(project),Path(output);original=read_json(project/'outputs/spes_agent_v4/active/active_protocol.json')
    model_hashes={}
    for phase in [0,1]:
        for outer in range(5):
            folder=output/f'learners/phase{phase}/outer{outer}';r=read_json(folder/'receipt.json')
            if sha(folder/'models.joblib')!=r['models_sha256'] or r['query_protocol_sha256']!=sha(output/'query_protocol.json'):raise ValueError('Learner provenance differs')
            model_hashes[f'{phase}/{outer}']=r['models_sha256']
    sources=['experiments/spes_agent_v8/'+f for f in ['numeric.py','features.py','learner.py','targets.py','crossfit.py']]
    sources+=['experiments/spes_agent_v4/acquisition.py','experiments/spes_agent_v4/neural.py']
    value={'study':'v8 learned numeric acquisition policies','scope':'Exploratory 35-patient evaluation after both declared development phases; no LLM or multi-agent claim',
      'cases':original['cases'],'policies':BASELINES+LEARNED,'primary':'phase1_bag_mean minus random_0, mean patient AUROC',
      'secondary':'Report every baseline and all five fixed learned policies; no post-hoc primary replacement.',
      'measurement_budget':8,'initial_common_measurements':4,'adaptive_queries':4,'final_scorer':'Unchanged original v4 outer HAT',
      'sampling':'Common old random_0 first-four prefix, then four adaptive choices; exact eight measurements, no early stopping.',
      'LLM_budget':0,'source_sha256':{s:sha(project/s) for s in sources},'models_sha256':model_hashes,
      'neural_model_sha256':original['model_sha256'],'query_protocol_sha256':sha(output/'query_protocol.json'),
      'future_LLM_gate':'Primary phase1_bag_mean exceeds the strongest fixed numerical baseline by >0.005 and its one-sided 95% patient-bootstrap lower bound versus random_0 is positive. At least ten patients have different final query sets from random_0. This gates further spending, not clinical validity.',
      'prefix_sha256':{c['case_id']:sha(project/'outputs/spes_agent_v4/active/trajectories'/c['case_id']/'random_0/metrics.json') for c in original['cases']}}
    path=output/'numeric_protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen numeric evaluation changed')
    write_json(path,value);return value

def run_case(project,output,case,policy):
    import torch
    from experiments.spes_agent_v4.neural import load_array
    project,output=Path(project),Path(output);protocol=read_json(output/'numeric_protocol.json');torch.set_num_threads(2)
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Frozen source differs')
    folder=output/'numeric'/case['case_id']/policy;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return read_json(folder/'metrics.json')
    training=project/'outputs/spes_agent_v4/multitrial';outer=case['outer']
    for name in ['baseline','mte','hat']:
        if sha(training/f'neural/outer{outer}/{name}/checkpoint.pth')!=protocol['neural_model_sha256'][f'{outer}/{name}']:raise ValueError('Scorer changed')
    env=Acquisition(project,training,outer,case['subject'])
    prefix_path=project/'outputs/spes_agent_v4/active/trajectories'/case['case_id']/'random_0/metrics.json'
    if sha(prefix_path)!=protocol['prefix_sha256'][case['case_id']]:raise ValueError('Common prefix changed')
    original=read_json(prefix_path);prefix=original['queries'][:4]
    for q in prefix:env.observe(int(q[1:]))
    learner=raw=None;phase=None
    if policy in LEARNED:
        phase=0 if policy.startswith('phase0') else 1
        if sha(output/f'learners/phase{phase}/outer{outer}/models.joblib')!=protocol['models_sha256'][f'{phase}/{outer}']:raise ValueError('Learner changed')
        learner=load_model(output,phase,outer)
        if int(case['subject'][-2:]) in learner['fit_patients']:raise ValueError('Test patient entered value regressor')
        raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20]
    seed=6190+outer*100+int(case['subject'][-2:])+(10000 if policy=='random_1' else 0)
    for step in range(4):
        public=env.public()
        if learner is None:
            chosen=numeric_choice(public,policy,np.random.default_rng(seed+(4+step)*100003))
            trace={'step':step+1,'choice':chosen,'rule':policy}
        else:
            candidates=[int(r['id'][1:]) for r in public['evidence']['geometry']['candidates']]
            x=feature_matrix(raw,env.selected,env.predictions['hat'],candidates);values=predict(learner,x)
            key=policy.removeprefix(f'phase{phase}_');chosen=f's{candidates[int(np.argmax(values[key]))]:03d}'
            trace={'step':step+1,'choice':chosen,'rule':policy,'feature_rows':x.tolist(),
               'candidates':candidates,'predicted_values':{n:v.tolist() for n,v in values.items()},
               'limitations':['Relative scores are centered within sampled training states.','Bag spread is not a calibrated confidence interval.']}
        write_json(folder/f'{step:02d}_choice.json',trace);observation=env.observe(int(chosen[1:]));write_json(folder/f'{step:02d}_observation.json',observation)
    result={**env.final_judge(),'case_id':case['case_id'],'policy':policy,'measurement_budget':8,'actual_measurements':8,'initial_common_queries':prefix,'new_LLM_calls':0}
    if policy=='random_0' and (abs(result['AUROC']-original['AUROC'])>1e-12 or not np.allclose(result['prediction'],original['prediction'],rtol=0,atol=1e-7)):raise ValueError('Exact baseline reproduction failed')
    write_json(folder/'metrics.json',result);print('VALUE_POLICY_DONE',case['case_id'],policy,round(result['AUROC'],6),flush=True);return result

def summarize(output):
    output=Path(output);protocol=read_json(output/'numeric_protocol.json');lookup={}
    for c in protocol['cases']:
        for p in protocol['policies']:lookup[c['case_id'],p]=read_json(output/'numeric'/c['case_id']/p/'metrics.json')
    ids=[c['case_id'] for c in protocol['cases']];summary={};base=np.array([lookup[c,'random_0']['AUROC'] for c in ids])
    for p in protocol['policies']:
        values=np.array([lookup[c,p]['AUROC'] for c in ids]);delta=values-base
        summary[p]={'mean_patient_AUROC':float(values.mean()),'gain_over_random_0':float(delta.mean()),'paired_95CI':bootstrap_delta(delta),
            'different_query_sets':sum(set(lookup[c,p]['queries'])!=set(lookup[c,'random_0']['queries']) for c in ids)}
    primary=np.array([lookup[c,'phase1_bag_mean']['AUROC'] for c in ids]);delta=primary-base;rng=np.random.default_rng(49103)
    lower=float(np.quantile(delta[rng.integers(0,len(ids),(20000,len(ids)))].mean(axis=1),.05))
    eligible=primary.mean()>max(summary[p]['mean_patient_AUROC'] for p in BASELINES)+.005 and lower>0 and summary['phase1_bag_mean']['different_query_sets']>=10
    result={'patients':35,'policies':summary,'primary':protocol['primary'],'primary_one_sided_95_lower_vs_random0':lower,
      'further_LLM_eligible':bool(eligible),'no_API_calls':True,'scope':'Exploratory numeric evidence; not multi-agent or clinical benefit',
      'rows':[{k:v for k,v in lookup[c,p].items() if k not in ['prediction','SOZ']} for c in ids for p in protocol['policies']]}
    write_json(output/'numeric_results.json',result);print(summary);print('Further LLM eligible:',eligible)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.freeze_only:freeze(a.project,a.output);print('Numeric evaluation frozen before new outer policy outcomes');return
    protocol=read_json(a.output/'numeric_protocol.json')
    if not 1<=a.workers<=4:raise ValueError('One to four numeric workers')
    with exclusive(a.output/'numeric.lock'):
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run_case,a.project,a.output,c,p) for c in protocol['cases'] for p in protocol['policies']]
            for f in as_completed(futures):f.result()
        summarize(a.output)

if __name__=='__main__':main()
