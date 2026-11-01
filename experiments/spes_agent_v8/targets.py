"""Learnable query-value targets from inner-held-out development patients.

Labels and added candidate responses are used ONLY after candidate features
and the rollout choice have been computed. No outer test patient is processed.
Phase 0 uses random prefixes; phase 1 uses the fixed phase-0 learned policy.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,exclusive,check_groups
from experiments.spes_agent_v2.verify import auc
from experiments.spes_agent_v4.neural import load_array,upstream_models
from experiments.spes_agent_v4.acquisition import gated_input
from experiments.spes_agent_v5.neural import fixed_queries
from .crossfit import write_json
from .features import FEATURES,distance_catalog,from_revealed
from .learner import SETTINGS,load_model,predict

def freeze(project,output):
    project,output=Path(project),Path(output)
    sources=['experiments/spes_agent_v8/'+f for f in ['targets.py','features.py','learner.py','crossfit.py']]
    sources+=['experiments/spes_agent_v4/neural.py','experiments/spes_agent_v4/acquisition.py','experiments/spes_agent_v5/neural.py']
    value={'study':'v8 supervised value of a diagnostic SPES query','scope':'Own SPES adaptation inspired by learning acquisition value, not a reproduction of LAL or EDDI',
      'features':FEATURES,'prefix_sizes':[4,5,6,7],'candidates_per_state':6,'phases':2,
      'phase0':'Fixed random_0 patient prefix; include its next query plus up to five seeded alternatives.',
      'phase1':'Same initial four queries; roll out the phase-0 bag-mean utility policy. Include chosen query plus up to five seeded alternatives.',
      'label':'After-query minus current within-patient SOZ AUROC; future response used for supervised target only.',
      'feature_gate':'Only known geometry/availability, observed post-stimulus waveforms and scores recomputed from the same observed columns. No labels, full-response predictions or future signal summaries.',
      'inner_source':'20 isolated HAT models; target patient was absent from its source fitting AND early stopping. Outer test patients never enter banks or regressors.',
      'learner_settings':SETTINGS,'policy_candidates':['phase0_bag_mean','phase1_bag_mean','phase1_relative_mean','phase1_ridge','phase1_robust'],
      'evaluation_plan':'Common four-measurement prefixes, four adaptive queries, unchanged outer v4 HAT. Complete both phases before opening outer utility-policy results; compare all fixed numerical controls. No new LLM calls until numeric value learning is evaluated.',
      'new_API_calls':0,'sources':{p:sha(project/p) for p in sources},'crossfit_protocol_sha256':sha(output/'crossfit_protocol.json')}
    path=output/'query_protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen query-value protocol changed')
    write_json(path,value);return value

def score_subset(model,raw,stats,selected):
    import torch
    x=gated_input(raw,selected)
    for segment,stat in zip([slice(0,1),slice(1,129),slice(129,None)],stats):
        view=x[...,segment];mask=np.abs(view)>1e-12;view[mask]=(view[mask]-stat['mean'])/stat['std']
    model.eval()
    with torch.no_grad():
        values=[torch.sigmoid(model(torch.from_numpy(x[b:b+8]).cuda())[:,0]).cpu().numpy() for b in range(0,len(x),8)]
    prediction=np.concatenate(values)
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite observed-only scores')
    return prediction

def rows_for_patient(raw,y,model,stats,outer,patient,phase,learner=None):
    _,available=distance_catalog(raw[...,0]);random_sequence=fixed_queries(available,outer,patient,0);selected=random_sequence[:4]
    rows=[];scores=score_subset(model,raw,stats,selected)
    for k in [4,5,6,7]:
        if len(selected)!=k:raise ValueError('Rollout length differs')
        remaining=[q for q in available if q not in selected]
        # Choice and all features are fixed before inspecting candidate responses
        # or computing this state's supervised loss target.
        table=np.vstack([from_revealed(raw,selected,scores,q) for q in remaining])
        chosen=random_sequence[k] if phase==0 else remaining[int(np.argmax(predict(learner,table)['bag_mean']))]
        rng=np.random.default_rng(92300+outer*10000+patient*100+k+phase*1000000)
        others=[q for q in remaining if q!=chosen];candidates=[chosen]+[int(q) for q in rng.choice(others,min(5,len(others)),replace=False)]
        current_auc=auc(y,scores);chosen_prediction=None
        for candidate in candidates:
            feature=table[remaining.index(candidate)].copy()
            after=score_subset(model,raw,stats,selected+[candidate]);after_auc=auc(y,after)
            rows.append({'features':feature,'gain':after_auc-current_auc,'before_auc':current_auc,'after_auc':after_auc,
              'prefix':selected+[-1]*(7-k),'candidate':candidate,'before_scores':scores.copy(),'after_scores':after.copy()})
            if candidate==chosen:chosen_prediction=after.copy()
        selected=selected+[chosen];scores=chosen_prediction
    return {'features':np.vstack([r['features'] for r in rows]),'gain':np.array([r['gain'] for r in rows]),
        'before_auc':np.array([r['before_auc'] for r in rows]),'after_auc':np.array([r['after_auc'] for r in rows]),
        'prefix':np.array([r['prefix'] for r in rows]),'candidate':np.array([r['candidate'] for r in rows]),
        'before_scores':np.vstack([r['before_scores'] for r in rows]),'after_scores':np.vstack([r['after_scores'] for r in rows]),'y':y}

def generate_job(project,output,outer,inner,phase):
    import torch
    project,output=Path(project),Path(output);protocol=read_json(output/'query_protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['sources'].items()):raise ValueError('Frozen query sources changed')
    source=read_json(output/'crossfit_protocol.json');job=next(j for j in source['jobs'] if j['outer']==outer and j['inner']==inner)
    check_groups(job['train'],job['early_stop'],job['predict'],job['outer_test'])
    target=output/f'targets/phase{phase}/outer{outer}_inner{inner}';target.mkdir(parents=True,exist_ok=True)
    folder=output/f'crossfit/outer{outer}_inner{inner}';receipt=read_json(folder/'receipt.json')
    if (target/'receipt.json').exists():
        previous=read_json(target/'receipt.json')
        if previous['query_protocol_sha256']!=sha(output/'query_protocol.json') or previous['source_receipt_sha256']!=sha(folder/'receipt.json') or previous['source_checkpoint_sha256']!=sha(folder/'checkpoint.pth'):
            raise ValueError('Completed target provenance changed')
        if any(sha(target/f'{s}.npz')!=h for s,h in previous['patient_file_sha256'].items()):raise ValueError('Completed target bank changed')
        return
    for key in ['train','early_stop','predict','outer_test']:
        if receipt[key]!=job[key]:raise ValueError('Source provenance differs')
    if sha(folder/'checkpoint.pth')!=receipt['checkpoint_sha256']:raise ValueError('Source checkpoint changed')
    torch.set_num_threads(2);mods=upstream_models(project);model,_,_=mods.create_model('hat')
    saved=torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=True);model.load_state_dict(saved['model_state_dict']);model=model.cuda().eval()
    learner=load_model(output,0,outer) if phase==1 else None
    hashes={};inputs={};total=0
    for subject in job['predict']:
        if subject in job['train']+job['early_stop']+job['outer_test']:raise ValueError('Target patient was not fully source-held-out')
        raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz';raw=load_array(raw_path)[:,:20]
        y=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
        bank=rows_for_patient(raw,y,model,receipt['normalisation'],outer,int(subject[-2:]),phase,learner)
        np.savez_compressed(target/f'{subject}.npz',**bank);hashes[subject]=sha(target/f'{subject}.npz');inputs[subject]=sha(raw_path);total+=len(bank['gain'])
        print('TARGET_PATIENT',phase,outer,inner,subject,len(bank['gain']),flush=True)
    learner_sha=sha(output/f'learners/phase0/outer{outer}/models.joblib') if phase==1 else None
    write_json(target/'receipt.json',{'outer':outer,'inner':inner,'phase':phase,'patients':job['predict'],'outer_test':job['outer_test'],
      'rows':total,'feature_names':FEATURES,'patient_file_sha256':hashes,'raw_input_sha256':inputs,
      'source_receipt_sha256':sha(folder/'receipt.json'),'source_checkpoint_sha256':receipt['checkpoint_sha256'],
      'phase0_learner_sha256':learner_sha,'query_protocol_sha256':sha(output/'query_protocol.json')})
    print('TARGET_JOB_DONE',phase,outer,inner,total,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',type=int,choices=[0,1],default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--freeze-only',action='store_true');p.add_argument('--ready-only',action='store_true');a=p.parse_args()
    if a.freeze_only:freeze(a.project,a.output);print('Query-value protocol frozen before targets');return
    if not 1<=a.workers<=2:raise ValueError('One or two inference workers')
    protocol=read_json(a.output/'crossfit_protocol.json');jobs=protocol['jobs']
    if a.ready_only:jobs=[j for j in jobs if (a.output/f'crossfit/outer{j["outer"]}_inner{j["inner"]}/receipt.json').exists()]
    elif not all((a.output/f'crossfit/outer{j["outer"]}_inner{j["inner"]}/receipt.json').exists() for j in jobs):raise ValueError('All source fits must complete, or explicitly request ready-only generation')
    with exclusive(a.output/f'targets_phase{a.phase}.lock'):
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(generate_job,a.project,a.output,j['outer'],j['inner'],a.phase) for j in jobs]
            for f in as_completed(futures):f.result()

if __name__=='__main__':main()
