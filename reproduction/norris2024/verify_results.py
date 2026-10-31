"""Independently replay saved predictions, split membership, budgets and source hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

from train_reproduction import MODELS, score, paper_threshold


def check_rows(saved, actual):
    if len(saved)!=len(actual):raise ValueError('Per-patient row count mismatch')
    for left,right in zip(saved,actual):
        for key,value in right.items():
            if key not in left or not np.isclose(left[key],value,rtol=0,atol=1e-12,equal_nan=False):
                raise ValueError(f'Recomputed {key} differs')


def verify(project, allow_incomplete=False):
    output=project/'outputs/reproduction/norris2024'
    track=project/'reproduction/norris2024'
    upstream=project/'vendor/norris2024'
    lock=json.loads((track/'source-lock.json').read_text())
    for name,expected in lock['reconstruction_scripts_sha256'].items():
        if hashlib.sha256((track/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Reconstruction source hash changed: '+name)
    protocol=json.loads((output/'training/protocol.json').read_text())
    for name,expected in protocol['source_sha256'].items():
        if hashlib.sha256((upstream/'src'/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Original numerical source changed: '+name)
    seen={name:set() for name in MODELS}
    rows=[]
    checkpoints={}
    for file in sorted((output/'training').glob('*/metrics.json')):
        saved=json.loads(file.read_text())
        name,seed,fold=saved['model'],saved['seed'],saved['fold']
        key=(seed,fold)
        if name not in seen or key in seen[name]:raise ValueError('Duplicate/unknown model run')
        seen[name].add(key)
        split=json.loads((output/'training'/f'split_seed{seed}_fold{fold}.json').read_text())
        groups=[set(split[k]) for k in ['train','val','test']]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Patient leakage between splits')
        data=np.load(file.parent/'predictions.npz')
        expected={int(subject[-2:]) for subject in split['test']}
        if set(data['patient'])!=expected:raise ValueError('Held-out prediction IDs differ from split')
        y,prediction,patient=data['y'],data['prediction'],data['patient']
        if not np.isfinite(prediction).all() or (prediction<0).any() or (prediction>1).any():
            raise ValueError('Invalid numerical probabilities')
        for subject in np.unique(patient):
            original=np.load(upstream/'data/main'/f'y_ccepAgeUMCU{int(subject):02d}.npy')
            if not np.array_equal(original,y[patient==subject]):raise ValueError('Contact label order differs')
        fpr,tpr,thresholds=roc_curve(data['val_y'],data['val_prediction'])
        check_rows(saved['released_code_threshold_per_patient'],score(y,prediction,patient,thresholds[int(np.argmax(tpr-fpr))]))
        if saved['paper_mad_n'] is not None:
            n=paper_threshold(data['val_y'],data['val_prediction'],data['val_patient'])
            if not np.isclose(n,saved['paper_mad_n'],rtol=0,atol=1e-12):raise ValueError('Validation threshold mismatch')
            check_rows(saved['paper_threshold_per_patient'],score(y,prediction,patient,lambda v:np.median(v)+n*np.median(np.abs(v-np.median(v)))))
        else:
            check_rows(saved['paper_threshold_per_patient'],score(y,prediction,patient))
        checkpoints[file.parent.name]=hashlib.sha256((file.parent/'checkpoint.pth').read_bytes()).hexdigest()
        rows.append(saved)
    expected={(seed,fold) for seed in range(1,6) for fold in range(5)}
    if not allow_incomplete and any(values!=expected for values in seen.values()):
        raise ValueError('All 75 primary runs are required')
    agent_rows=[]
    agent_keys=set()
    for file in sorted((output/'agents').glob('*/judge_metrics.json')):
        saved=json.loads(file.read_text())
        case=file.parent
        seed,fold,patient=saved['seed'],saved['fold'],saved['patient']
        if (seed,fold,patient) in agent_keys:raise ValueError('Duplicate held-out agent case')
        agent_keys.add((seed,fold,patient))
        slug='Transformer_base'
        data=np.load(output/'training'/f'{slug}_seed{seed}_fold{fold}'/'predictions.npz')
        truth=data['y'][data['patient']==patient]
        for policy in ['single','multi']:
            result=saved['strategies'][policy]
            folder=case/policy
            if result['calls']!=8 or len(list(folder.glob('request_*.json')))!=8 or len(list(folder.glob('response_*.json')))!=8:
                raise ValueError('Agent request/response budget mismatch')
            if len(list(folder.glob('api_usage_*.json')))!=8:raise ValueError('Provider usage records missing')
            for usage_file in folder.glob('api_usage_*.json'):
                usage=json.loads(usage_file.read_text())
                if usage['max_tokens']!=8192 or usage['reasoning_effort']!='low' or usage['request_model']!='gemini-3.6-flash':
                    raise ValueError('Formal API model/token/reasoning settings differ')
            for step in range(8):
                request=json.loads((folder/f'request_{step:02d}.json').read_text())
                expected_role='single' if policy=='single' else ['signal','network','uncertainty','critic'][step%4]
                if request['role']!=expected_role:raise ValueError('Role schedule mismatch')
            proposals=[json.loads((folder/f'response_{step:02d}.json').read_text())['scores'] for step in range(8)]
            states=[json.loads((folder/f'response_{step:02d}.json').read_text()) for step in range(8)]
            if sum(row['accepted_agent_output'] for row in states)!=result['accepted_agent_outputs'] or sum(not row['accepted_agent_output'] for row in states)!=result['schema_failures']:
                raise ValueError('Accepted/invalid output accounting mismatch')
            prediction=np.mean(proposals[-4:],axis=0)
            if not np.allclose(prediction,result['prediction'],rtol=0,atol=1e-12):raise ValueError('Agent score aggregation mismatch')
            check_rows([result['metrics']],score(truth,prediction,np.zeros(len(truth),dtype=int)))
        payload=json.loads((case/'payload.json').read_text())
        if set(payload)!={'task','feature_notes','models','contacts','required_json_shape','constraints'}:
            raise ValueError('Unapproved agent payload fields')
        for contact in payload['contacts']:
            if set(contact)!={'id','model_scores','incoming_RMS_ranks','outgoing_RMS_ranks','neighbors'}:
                raise ValueError('Unapproved contact fields')
        import sys
        sys.path.insert(0,str(project))
        from neurotarget.scienceclaw_runtime import upstream_root
        upstream_root(project)
        from artifacts.artifact import Artifact
        artifact_root=case/'scienceclaw/artifacts'
        seen_artifacts=set()
        stores={}
        for line in (artifact_root/'global_index.jsonl').read_text().splitlines():
            entry=json.loads(line)
            producer=entry['producer_agent']
            if producer not in {'recordings','paper-single','paper-signal','paper-network','paper-uncertainty','paper-critic'}:
                raise ValueError('Unknown artifact producer')
            if producer not in stores:
                stores[producer]={item['artifact_id']:item for item in [json.loads(text) for text in (artifact_root/producer/'store.jsonl').read_text().splitlines()]}
            item=stores[producer][entry['artifact_id']]
            if item['content_hash']!=Artifact._hash_payload(item['payload']):raise ValueError('Native artifact content hash mismatch')
            if not set(item['parent_artifact_ids']).issubset(seen_artifacts):raise ValueError('Native artifact lineage mismatch')
            seen_artifacts.add(item['artifact_id'])
        if len(seen_artifacts)!=saved['native_artifacts'] or len(seen_artifacts)!=17:
            raise ValueError('Native artifact count mismatch')
        agent_rows.append(saved)
    if not allow_incomplete and len(agent_rows)!=175:raise ValueError('All 175 held-out agent cases are required')
    result={'verified_primary_runs':len(rows),'verified_agent_cases':len(agent_rows),
            'source_hashes_verified':True,'patient_splits_verified':True,'scores_recomputed':True,
            'checkpoint_sha256':checkpoints,'exact_paper_cohort':protocol['cohort_counts_match'],
            'status':'partial verification' if allow_incomplete else 'complete public-code and agent verification'}
    (output/'verification.json').write_text(json.dumps(result,indent=2))
    return {key:value for key,value in result.items() if key!='checkpoint_sha256'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.project.resolve(),args.allow_incomplete),indent=2))
