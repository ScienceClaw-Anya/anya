"""Physical re-inference, independent budget/merger checks and native lineage."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.environment import PhysicalAcquisition
from .experiment import verified,summarize
from .logic import prompts,baseline_ranking
from .provider import response_format,validate,parse_json
from .public import recording_catalog,public_state,check_public


def close_tree(expected,actual,path='public'):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(expected)!=set(actual): raise ValueError('Public keys differ: '+path)
        for key in expected: close_tree(expected[key],actual[key],path+'.'+key)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual): raise ValueError('Public length differs: '+path)
        for i,(a,b) in enumerate(zip(expected,actual)): close_tree(a,b,path+f'[{i}]')
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not isinstance(actual,(float,int)) or abs(expected-actual)>2e-6: raise ValueError('Public number differs: '+path)
    elif expected!=actual: raise ValueError('Public value differs: '+path)


def native_artifacts(project,folder,ids):
    from neurotarget.scienceclaw_runtime import upstream_root
    upstream_root(project)
    from artifacts.artifact import Artifact
    state=folder/'scienceclaw'; entries=[json.loads(line) for line in (state/'artifacts/global_index.jsonl').read_text(encoding='utf8').splitlines()]
    stores={}; artifacts={}; seen=set()
    for entry in entries:
        agent=entry['producer_agent']
        if agent not in stores:
            path=state/'artifacts'/agent/'store.jsonl'; stores[agent]={a['artifact_id']:a for a in [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]}
        artifact=stores[agent][entry['artifact_id']]
        if artifact['content_hash']!=Artifact._hash_payload(artifact['payload']) or not set(artifact['parent_artifact_ids'])<=seen: raise ValueError('Native hash/lineage differs')
        if artifact['artifact_id'] in seen: raise ValueError('Duplicate native artifact')
        artifacts[artifact['artifact_id']]=artifact; seen.add(artifact['artifact_id'])
    if seen!=set(ids.values()) or len(seen)!=28: raise ValueError('Native artifact accounting differs')
    for r in range(4):
        for slot in [0,1]:
            item=artifacts[ids[f'proposal_{r}_{slot}']]
            if item['parent_artifact_ids']!=[ids[f'tool_{r}_{slot}']]: raise ValueError('Current peer proposal leaked through native parentage')
        item=artifacts[ids[f'merge_{r}']]
        if item['parent_artifact_ids']!=[ids[f'proposal_{r}_0'],ids[f'proposal_{r}_1']]: raise ValueError('Joint merger parents differ')
    return artifacts


def audit(project,output,full=False):
    import torch
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2)
    approval=read_json(output/('expansion_transmission_approval.json' if full else 'transmission_approval.json'))
    if approval.get('status')!='explicit_user_approval_received' or approval.get('destination_host')!='generativelanguage.googleapis.com' or approval.get('maximum_paid_attempts',0)<(840 if full else 120): raise ValueError('Specific transmission/paid-attempt authorization scope missing')
    cases=[c for c in p['cases'] if full or c['pilot']]; known=[]; attempts=valid=artifacts_count=measurements=0; unknown=0; prompt_tokens=completion_tokens=0
    for case in cases:
        map_path=project/'outputs/spes_physical/maps'/case['subject']/'receipt.json'
        if sha(map_path)!=p['mapping_receipt_sha256'][case['subject']]: raise ValueError('Map changed')
        mapping=read_json(map_path); catalog_path=output/'catalog'/f'{case["case_id"]}.json'; catalog=recording_catalog(project,case['subject'],mapping)
        if sha(catalog_path)!=p['catalog_sha256'][case['case_id']] or read_json(catalog_path)!=catalog: raise ValueError('Anatomy catalog differs')
        neural=read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json')['jobs'][case['outer']]
        if case['subject'] not in neural['predict'] or case['subject'] in neural['train']+neural['early_stop']: raise ValueError('Outer patient isolation differs')
        for family in ['baseline','mte','hat']:
            if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{case["outer"]}/{family}/checkpoint.pth')!=p['model_sha256'][f'{case["outer"]}/{family}']: raise ValueError('Neural weights differ')
        raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20]; labels=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz')
        for policy in p['policies']:
            folder=output/'trajectories'/case['case_id']/policy; metric=read_json(folder/'metrics.json'); ids=read_json(folder/'ids.json'); native=native_artifacts(project,folder,ids); artifacts_count+=len(native)
            if metric['protocol_sha256']!=sha(output/'protocol.json') or metric['LLM_attempts']!=8 or metric['physical_response_groups']!=8 or metric['native_public_queries']!=8: raise ValueError('Frozen protocol/budgets differ')
            env=PhysicalAcquisition(project,case['outer'],case['subject']); prefix=fixed_queries(env.available,case['outer'],int(case['subject'][-2:]),0)[:4]
            for i,q in enumerate(prefix): close_tree(env.observe(q),read_json(folder/f'warmup_{i}.json'),'warmup')
            history=[]; previous=None; accepted=0
            for r in range(4):
                public=check_public(read_json(folder/f'{r:02d}_public.json')); close_tree(public_state(env,catalog,raw,previous),public)
                batch=prompts(public,policy,r,history); proposals=[]
                for slot,prompt in enumerate(batch):
                    if read_json(folder/f'{r:02d}_{slot}_request.json')!=prompt or read_json(folder/f'{r:02d}_{slot}_tool.json')!=public: raise ValueError('Independent history/common evidence differs')
                    usage_path=folder/f'{r:02d}_{slot}_usage.json'; known.append(usage_path.resolve()); saved=read_json(usage_path); proposal=read_json(folder/f'{r:02d}_{slot}_decision.json')
                    if saved['prompt_sha256']!=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest() or saved['response_format']!=response_format(public): raise ValueError('Transport request schema/hash differs')
                    if urlparse(saved['endpoint']).hostname!='generativelanguage.googleapis.com' or saved['model']!='gemini-4-argon' or saved['retries']!=0 or saved['max_tokens']!=32768 or saved['reasoning']!='medium': raise ValueError('Provider/destination/budget differs')
                    accepted_response=False
                    try:
                        if not saved.get('captured_response') or saved.get('response_model')!='gemini-4-argon' or saved.get('finish_reason')!='stop': raise ValueError('Not accepted')
                        parsed=validate(parse_json(saved.get('response_content')),public); accepted_response=True
                    except (ValueError,TypeError,KeyError): parsed=None
                    if bool(proposal['accepted_output'])!=accepted_response: raise ValueError('Accepted decision accounting differs')
                    if parsed is not None:
                        for key in parsed:
                            if proposal[key]!=parsed[key]: raise ValueError('Saved model proposal differs from captured response')
                    elif proposal['ranked_queries']!=baseline_ranking(public) or proposal['origin']!='spent_default': raise ValueError('Unaccepted attempt did not use declared fallback')
                    if proposal['round']!=r+1 or proposal['slot']!=slot or proposal['role']!=prompt['role']: raise ValueError('Role cadence differs')
                    if native[ids[f'proposal_{r}_{slot}']]['payload']!=proposal or native[ids[f'tool_{r}_{slot}']]['payload']!=public: raise ValueError('Native evidence/decision differs')
                    accepted+=int(accepted_response); attempts+=1
                    usage=saved.get('usage')
                    if not isinstance(usage,dict): unknown+=1
                    else: prompt_tokens+=int(usage.get('prompt_tokens') or 0); completion_tokens+=int(usage.get('completion_tokens') or 0)
                    proposals.append(proposal)
                # Independently recompute rank votes rather than call the production merger.
                scores={}; geometry={row['id']:row for row in public['evidence']['geometry']['candidates']}
                for proposal in proposals:
                    for q,weight in zip(proposal['ranked_queries'],[3,2,1]): scores[q]=scores.get(q,0)+weight
                ranking=sorted(scores,key=lambda q:(-scores[q],-geometry[q]['score_focus_gain'],q)); joint=read_json(folder/f'{r:02d}_merge.json')
                if joint['choice']!=ranking[0] or joint['ranking']!=ranking or joint['votes']!=scores or native[ids[f'merge_{r}']]['payload']!=joint: raise ValueError('Joint decision differs')
                previous=env.predictions['hat'].copy(); obs=env.observe(int(joint['choice'][1:])); feedback=read_json(folder/f'{r:02d}_feedback.json'); close_tree(obs,feedback,'feedback')
                if native[ids[f'feedback_{r}']]['payload']!=feedback: raise ValueError('Native feedback differs')
                history.append({'round':r+1,'proposals':proposals,'executed':joint['choice'],'feedback':feedback})
            final=env.final_judge()
            if metric['queries']!=final['queries'] or len(set(metric['queries']))!=8 or metric['query_contacts']!=final['query_contacts'] or metric['accepted_model_outputs']!=accepted: raise ValueError('Physical budget/identity/validity differs')
            np.testing.assert_allclose(final['prediction'],metric['prediction'],rtol=0,atol=1e-6)
            np.testing.assert_allclose(roc_auc_score(labels,metric['prediction']),metric['AUROC'],rtol=0,atol=1e-12)
            valid+=accepted; measurements+=8; print('V16_AUDIT',case['case_id'],policy,flush=True)
    all_actual={path.resolve() for path in (output/'trajectories').glob('*/*/*_usage.json')}
    allowed={output.resolve()/'trajectories'/c['case_id']/name/f'{r:02d}_{slot}_usage.json' for c in p['cases'] for name in p['policies'] for r in range(4) for slot in [0,1]}
    if not all_actual<=allowed or len(all_actual)>(840 if full else 120) or set(known)!=all_actual: raise ValueError('Unexpected/out-of-budget API-attempt record')
    old=read_json(output/('full_results.json' if full else 'pilot_results.json')); summarize(project,output,pilot=not full)
    if old!=read_json(output/('full_results.json' if full else 'pilot_results.json')): raise ValueError('Result summary/gate replay differs')
    result={'status':'verified','scope':'full' if full else 'pilot','patients':len(cases),'physical_trajectories':len(cases)*3,
            'specific_transmission_approval_verified':True,
            'actual_API_attempts':attempts,'accepted_model_outputs':valid,'physical_response_groups':measurements,'native_artifacts_verified':artifacts_count,
            'reported_prompt_tokens':prompt_tokens,'reported_completion_tokens':completion_tokens,'attempts_without_token_usage':unknown,
            'checks':['Every actual physical trajectory re-inferred with all three original own neural scorers; all public anatomy, observed-only quality and feedback reconstructed.',
                      'All source/model/catalog/raw hashes and patient isolation; zero test labels or unobserved summaries in requests.',
                      'Both independent requests, delayed histories and reason-hidden ablation replayed; all API schemas/identities/reservations/budgets checked.',
                      'All native artifact hashes, parent graphs, independent merger votes, physical identities and AUROC verified.'],
            'limits':['Re-inference permits small floating point differences; source identities remain the same repeatedly inspected cohort.',
                      'Equal call/token caps do not imply identical actual token consumption. Pilot is not a confirmatory clinical or collaboration trial.']}
    write_json(output/('audit_full.json' if full else 'audit_pilot.json'),result); print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--full',action='store_true')
    a=parser.parse_args(); audit(a.project,a.output,a.full)
