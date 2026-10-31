"""Continue a halted episode within its original eight-attempt budget; never replay a request."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from agent_comparison import public_case, bounded_board, call_or_carry, validate, ROLES, EXPERTISE
from train_reproduction import MODELS, score


def identify(output,case_id):
    for seed in range(1,6):
        for fold in range(5):
            data=np.load(output/'training'/f'Transformer_base_seed{seed}_fold{fold}'/'predictions.npz')
            for patient in np.unique(data['patient']):
                if hashlib.sha256(f'{seed}:{fold}:{int(patient)}'.encode()).hexdigest()[:16]==case_id:
                    return seed,fold,int(patient)
    raise ValueError('Unknown case ID')


def recover(project,case_id,dry_run=False):
    sys.path.insert(0,str(project))
    from neurotarget.scienceclaw_runtime import Framework,Gemini
    output=project/'outputs/reproduction/norris2024';upstream=project/'vendor/norris2024'
    seed,fold,patient=identify(output,case_id)
    case=output/'agents'/case_id
    if (case/'judge_metrics.json').exists():return {'already_complete':True}
    payload,judge=public_case(project,output,upstream,seed,fold,patient)
    if json.loads((case/'payload.json').read_text())!=payload:raise ValueError('Observation changed; cannot continue')
    plan={}
    for policy in ['single','multi']:
        folder=case/policy
        requests=sorted(folder.glob('request_*.json'))
        count=len(requests)
        if count>8:raise ValueError('Original attempt budget was exceeded')
        if [f.stem for f in requests]!=[f'request_{i:02d}' for i in range(count)]:raise ValueError('Noncontiguous request history')
        failure=folder/'failure.json'
        if count<8 and count and (not failure.exists() or json.loads(failure.read_text())['calls']!=count):
            raise ValueError('No counter evidence for already-spent attempts')
        plan[policy]={'already_attempted':count,'new_requests_remaining':8-count}
    if dry_run:return {'case':case_id,'budget_plan':plan,'API_calls':0}
    framework=Framework(project,case/'scienceclaw')
    from artifacts.artifact import Artifact
    root=case/'scienceclaw/artifacts'
    entries=[json.loads(line) for line in (root/'global_index.jsonl').read_text().splitlines()]
    artifacts={}
    for entry in entries:
        producer=entry['producer_agent']
        if producer not in framework.stores:framework.stores[producer]=framework.store_class(producer)
        item=framework.stores[producer].get(entry['artifact_id'])
        if item is None or item.content_hash!=Artifact._hash_payload(item.payload):raise ValueError('Existing artifact is invalid')
        artifacts[item.artifact_id]=item
    observation=artifacts[entries[0]['artifact_id']]
    if observation.producer_agent!='recordings' or observation.payload!=payload:raise ValueError('Observation identity mismatch')
    existing={'single':[],'multi':[]}
    for entry in entries[1:]:
        existing['single' if entry['producer_agent']=='paper-single' else 'multi'].append(artifacts[entry['artifact_id']])
    results={}
    for policy in ['single','multi']:
        folder=case/policy;folder.mkdir(exist_ok=True)
        attempted=plan[policy]['already_attempted']
        failure_path=folder/'failure.json'
        evidence=json.loads(failure_path.read_text()) if failure_path.exists() else None
        gemini=Gemini(project,max_calls=8,max_tokens=2800);gemini.calls=attempted
        board=[];parents=[observation.artifact_id];started=time.perf_counter()
        try:
            for step in range(8):
                role='single' if policy=='single' else ROLES[step%4]
                prompt={**payload,'role':role,'expertise':EXPERTISE[role],'round':step//4+1,
                        'shared_board':bounded_board(board),
                        'instruction':'Review the same observations and earlier proposals, revise scores only with an evidence-based explanation, and return only the required JSON.'}
                request=folder/f'request_{step:02d}.json';response=folder/f'response_{step:02d}.json';usage=folder/f'api_usage_{step:02d}.json'
                previous=board[-1]['scores'] if board else [c['model_scores'][2] for c in payload['contacts']]
                if step<attempted:
                    if json.loads(request.read_text())!=prompt:raise ValueError('Original request differs from reconstructed context')
                    if response.exists():state=json.loads(response.read_text())
                    elif step<len(existing[policy]):state=existing[policy][step].payload
                    elif usage.exists():
                        recorded=json.loads(usage.read_text());text=recorded.get('response_content') or ''
                        try:
                            state={**validate(json.loads(gemini.llm_class._strip_wrapping_code_fence(text)),len(previous)),
                                   'accepted_agent_output':True,'state_origin':'agent'}
                        except ValueError:
                            state={'scores':list(previous),'reason':'A captured response failed validation; previous scores are retained.',
                                   'reflection':'Recovered policy state; no extra request was made.','accepted_agent_output':False,
                                   'state_origin':'carried_previous_scores','failure_type':'ValueError'}
                    else:
                        if evidence is None or evidence['calls']<=step:raise ValueError('Unproven failed attempt')
                        state={'scores':list(previous),'reason':'This earlier request failed without a captured output. Its attempt remains counted and previous scores are retained.',
                               'reflection':'Recovery record for an already-spent attempt; no generated agent explanation is available.',
                               'accepted_agent_output':False,'state_origin':'carried_after_failed_attempt','failure_kind':'transport',
                               'failure_type':evidence['error_type']}
                        descriptor={'captured_response':False,'attempt_counted':True,'usage':None,'response_content':None,
                                    'response_model':None,'request_model':None,'max_tokens':None,'reasoning_effort':None,'seconds':None,
                                    'finish_reason':None,'configured_model':'gemini-3.6-flash','configured_max_tokens':8192,
                                    'configured_reasoning_effort':'low','original_step':step,'failure_evidence':evidence}
                        usage.write_text(json.dumps(descriptor,indent=2))
                else:
                    request.write_text(json.dumps(prompt,indent=2))
                    state=call_or_carry(gemini,'paper-'+role,prompt,folder,step,previous)
                if step<len(existing[policy]):
                    artifact=existing[policy][step]
                    if artifact.content_hash!=Artifact._hash_payload(state):raise ValueError('State differs from original artifact')
                else:
                    artifact=framework.artifact('paper-'+role,'spes-soz-classification',state,'soz_policy_state',parents)
                parents=[observation.artifact_id,artifact.artifact_id]
                board.append({'step':step+1,'role':role,**state})
                if not response.exists():response.write_text(json.dumps(state,indent=2))
            final=np.mean([r['scores'] for r in board[-4:]],axis=0)
            records=[json.loads(f.read_text()) for f in sorted(folder.glob('api_usage_*.json'))]
            tokens={};captured={}
            for key in ['prompt_tokens','completion_tokens','total_tokens']:
                values=[r['usage'].get(key) if r.get('usage') else None for r in records]
                tokens[key]=sum(values) if all(v is not None for v in values) else None
                captured[key]=sum(v for v in values if v is not None)
            first_request=(folder/'request_00.json').stat().st_mtime
            results[policy]={'calls':gemini.calls,'seconds':time.time()-first_request,'seconds_include_recovery_delay':bool(attempted),
                             'accepted_agent_outputs':sum(r['accepted_agent_output'] for r in board),
                             'schema_failures':sum(not r['accepted_agent_output'] and r.get('failure_kind')!='transport' for r in board),
                             'transport_failures':sum(r.get('failure_kind')=='transport' for r in board),
                             'api_tokens':tokens,'captured_tokens_lower_bound':captured,
                             'unknown_usage_attempts':sum(not r.get('captured_response',True) for r in records),
                             'preexisting_attempts':attempted,'new_requests_after_resume':gemini.calls-attempted,
                             'response_models':sorted({r['response_model'] for r in records if isinstance(r.get('response_model'),str)}),
                             'prediction':final.tolist(),'metrics':score(judge['y'],final,np.zeros(len(final),dtype=int))[0]}
            if failure_path.exists():
                archive=folder/f'failure_recovered_at_attempt_{evidence["calls"]:02d}.json'
                if archive.exists():raise ValueError('Recovery archive already exists')
                failure_path.rename(archive)
        except Exception as error:
            failure_path.write_text(json.dumps({'calls':gemini.calls,'error_type':type(error).__name__,
                                                'message':'Continuation stopped; already-spent attempts will never be replayed.'}))
            raise
    base=judge['model_scores']
    baselines={name:score(judge['y'],base[:,i],np.zeros(len(base),dtype=int))[0] for i,name in enumerate(MODELS)}
    baselines['fixed_mean_ensemble']=score(judge['y'],base.mean(axis=1),np.zeros(len(base),dtype=int))[0]
    result={'seed':seed,'fold':fold,'patient':patient,'strategies':results,'numerical_baselines':baselines,
            'native_artifacts':framework.verify_artifacts(),'continuation':{'original_budget_preserved':True,'plan':plan}}
    temporary=case/'judge_metrics.recovered.tmp';temporary.write_text(json.dumps(result,indent=2));temporary.replace(case/'judge_metrics.json')
    return {'case':case_id,'completed':True,'requests_after_resume':sum(r['new_requests_after_resume'] for r in results.values()),
            'all_policy_attempts':sum(r['calls'] for r in results.values()),'native_artifacts':result['native_artifacts']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--case-id',required=True);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();print(json.dumps(recover(args.project.resolve(),args.case_id,args.dry_run),indent=2))
