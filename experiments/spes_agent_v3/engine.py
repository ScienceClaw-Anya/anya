"""Common information, exact numerical execution and matched decision/query budgets."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from .common import exclusive, read_json, sha, source_hashes, write_json
from experiments.spes_agent_v2.providers import Provider, ProviderUnavailable, parse_json

POLICIES=['numeric_transformer','numeric_selector','single','multi_serial','multi_independent','multi_no_exchange']
ROLES=['ranking','validation','distribution','auditor']
TOOLS=['baseline','selector','validation','distribution']
EXPERTISE={'single':'Assess all numerical, validation and distribution evidence and review earlier decisions.',
 'ranking':'Compare executable rankings. Do not invent probabilities or physiological interpretations.',
 'validation':'Inspect patient-isolated OOF evidence, eligibility and the limits of fitted-selector generalization.',
 'distribution':'Inspect measured predictor disagreement and distribution shift; do not treat it as a waveform-quality biomarker.',
 'auditor':'Check evidence and unsupported changes to the Transformer default. In an independent first round, audit data sufficiency rather than unseen proposals.'}

def visible_indices(policy,step):
    if policy=='multi_no_exchange' or (policy=='multi_independent' and step<4):return []
    return list(range(step))

def prompt_for(public,policy,step,states,tool):
    role='single' if policy=='single' else ROLES[step%4]
    return {**public,'role':role,'expertise':EXPERTISE[role],'round':step//4+1,
        'shared_board':[{key:states[i][key] for key in ['step','role','action','evidence_ids','reason','counterargument','accepted_output']}
                        for i in visible_indices(policy,step)],
        'executed_tool':tool,
        'instruction':'Choose transformer or conservative_selector. Transformer is the exact default. Scores are executed by code. Use only supplied evidence. No label/accuracy feedback occurs between rounds. Write English; each text field must be complete and at most 900 characters. Do not output probabilities or weights.',
        'required_json':{'action':'transformer | conservative_selector','evidence_ids':['one or more exact keys from evidence'],
            'reason':'Evidence-grounded justification','counterargument':'A limitation or concrete reason to retain the default'}}

def validate_answer(answer,public):
    if not isinstance(answer,dict) or set(answer)!={'action','evidence_ids','reason','counterargument'}:
        raise ValueError('Only executable actions and structured evidence are permitted')
    if answer['action'] not in public['allowed_actions']:raise ValueError('Unknown scorer')
    ids=answer['evidence_ids']
    if not isinstance(ids,list) or not ids or not all(isinstance(v,str) for v in ids) or not set(ids)<=set(public['evidence']):raise ValueError('Unknown evidence citation')
    for key in ['reason','counterargument']:
        if not isinstance(answer[key],str) or not 0<len(answer[key].strip())<=900:raise ValueError('Missing or oversized structured explanation')
    return {**answer,'accepted_output':True,'origin':'model_selected_executable'}

def execute_tool(framework,case,operation):
    name='neurotarget-spes-v3';meta=framework.registry.get_skill(name)
    if not meta:raise ValueError('Native SPES v3 skill missing')
    response=framework.executor.execute_skill(name,meta,{'project':str(framework.project),'public_case':str(case/'tools.json'),
        'operation':operation,'format':'json'},timeout=90)
    if response.get('status')!='success' or not isinstance(response.get('result'),dict):raise RuntimeError('Native evidence tool failed')
    return response['result']

def run_policy(project,output,case_id,policy):
    from neurotarget.scienceclaw_runtime import Framework
    from experiments.spes_agent_v2.verify import read_native
    project,output=Path(project),Path(output);case=output/'cases'/case_id;folder=output/'policies'/case_id/policy
    if (folder/'metrics.json').exists():return read_json(folder/'metrics.json')
    folder.mkdir(parents=True,exist_ok=True);public=read_json(case/'public.json');scores=np.load(case/'scores.npz')['scores']
    framework=Framework(project,folder/'scienceclaw');ids_path=folder/'native_ids.json'
    ids=read_json(ids_path) if ids_path.exists() else {'observation':None,'states':[],'tools':[]}
    if ids['observation'] is None:
        ids['observation']=framework.artifact('observations','spes-v3-public',public,'spes_observation').artifact_id;write_json(ids_path,ids)
    states=[];client=None;numeric=policy.startswith('numeric_')
    for step in range(8):
        state_path=folder/f'state_{step:02d}.json';usage_path=folder/f'usage_{step:02d}.json'
        if state_path.exists():
            if len(ids['states'])<=step:raise ValueError('Missing native decision provenance')
            states.append(read_json(state_path));continue
        if len(ids['states'])>step:
            state=read_native(folder/'scienceclaw')[ids['states'][step]]['payload'];write_json(state_path,state);states.append(state);continue
        tool_path=folder/f'tool_{step:02d}.json'
        if tool_path.exists():tool=read_json(tool_path)
        else:tool=execute_tool(framework,case,TOOLS[step%4]);write_json(tool_path,tool)
        if len(ids['tools'])<=step:
            artifact=framework.artifact('scientific-tools','neurotarget-spes-v3',tool,'scientific_tool_result',[ids['observation']])
            ids['tools'].append(artifact.artifact_id);write_json(ids_path,ids)
        if numeric:
            state={'action':'transformer' if policy=='numeric_transformer' else 'conservative_selector',
                'evidence_ids':['baseline','candidate_evidence','selector_decision'],
                'reason':'Fixed numerical policy; no language model is called.','counterargument':'Generalization is evaluated only by the hidden final judge.',
                'accepted_output':True,'origin':'numeric_rule'}
        else:
            prompt=prompt_for(public,policy,step,states,tool);request_path=folder/f'request_{step:02d}.json'
            if request_path.exists() and read_json(request_path)!=prompt:raise ValueError('Prompt changed on resume')
            if not request_path.exists():write_json(request_path,prompt)
            if usage_path.exists():usage=read_json(usage_path);text=usage.get('response_content') or ''
            else:
                if client is None:client=Provider(project,'argon')
                # Reserved failures stay spent. The coordinator halts; resume only executes later steps.
                text=client.ask(prompt,usage_path);usage=read_json(usage_path)
            try:
                if not usage.get('captured_response'):raise ValueError('Uncaptured transport response')
                if usage.get('response_model')!='gemini-4-argon' or usage.get('finish_reason')!='stop':raise ValueError('Unexpected or incomplete model response')
                state=validate_answer(parse_json(text),public)
            except (ValueError,TypeError,KeyError):
                state={'action':'transformer','evidence_ids':['baseline'],'reason':'The spent attempt yielded no accepted decision; use the explicit Transformer default.',
                    'counterargument':'No replacement request was issued.','accepted_output':False,'origin':'policy_default',
                    'failure_kind':'schema' if usage.get('captured_response') else 'transport'}
        state.update(step=step+1,role='numeric' if numeric else ('single' if policy=='single' else ROLES[step%4]))
        parents=[ids['observation'],ids['tools'][step]]+[ids['states'][i] for i in visible_indices(policy,step)]
        artifact=framework.artifact(state['role'],'spes-v3-action',state,'executable_scorer_decision',parents)
        ids['states'].append(artifact.artifact_id);write_json(ids_path,ids);write_json(state_path,state);states.append(state)
    prediction=np.mean([scores[:,0 if s['action']=='transformer' else 1] for s in states[-4:]],axis=0)
    # Only the completed decision trajectory can reach the hidden labels.
    judge=read_json(case/'judge.json')
    result={'case_id':case_id,'policy':policy,'patient':judge['patient'],'outer':judge['outer'],
        'AUROC':float(roc_auc_score(judge['y'],prediction)),'prediction':prediction.tolist(),'decision_slots':8,'tool_queries':8,
        'model_attempts':0 if numeric else 8,'accepted_model_outputs':0 if numeric else sum(s['accepted_output'] for s in states),
        'distinct_actions':public['evidence']['limitations']['score_vectors_differ'],'native_artifacts':17}
    write_json(folder/'metrics.json',result);print('COMPLETE',case_id,policy,flush=True);return result

def freeze_agents(project,output):
    critical=['experiments/spes_agent_v3/'+n for n in ['common.py','crossfit.py','selector.py','prepare.py','engine.py']]
    critical+=['experiments/spes_agent_v2/providers.py','experiments/spes_agent_v2/verify.py','neurotarget/scienceclaw_runtime.py',
        'skills/neurotarget-spes-v3/SKILL.md','skills/neurotarget-spes-v3/scripts/evidence.py','config/upstream-lock.json']
    protocol={'version':3,'patients':35,'outer_repeat':1,'policies':POLICIES,'model':'gemini-4-argon','reasoning':'medium',
        'model_cap_per_patient_policy':8,'decision_slots':8,'native_tool_queries':8,'actual_numeric_model_calls':0,
        'budget_definition':'Same eight decision/query slots and an upper limit of eight model calls. Numerical baselines use zero model calls; monetary cost is not equalized by padding.',
        'aggregation':'Programmatic mean of the final four executable scorer vectors, in full precision',
        'primary_metric':'Mean patient AUROC; patient-bootstrap paired intervals conditional on recorded models and generations',
        'secondary_metrics':['Average precision','Fraction of patients worse than Transformer','Effective distinct-action opportunities','Actual model attempts and captured token usage'],
        'primary':'multi_independent minus single; secondary contrasts versus numeric_selector and numeric_transformer',
        'visibility':'Independent first four boards are empty; no-exchange always empty. All other observations and evidence are identical.',
        'default':'Transformer for every policy and every invalid or uncaptured attempt; no handwritten probabilities or weights',
        'selector_protocol_sha256':sha(output/'selector_protocol.json'),'prepared_sha256':sha(output/'prepared.json'),
        'source_sha256_LF':source_hashes(project,critical),
        'input_sha256':{p.relative_to(output).as_posix():sha(p) for p in sorted((output/'cases').glob('*/*')) if p.is_file()}}
    target=output/'agent_protocol.json'
    if target.exists():
        if read_json(target)!=protocol:raise ValueError('Frozen agent protocol changed')
    else:write_json(target,protocol)

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args();project,output=a.project.resolve(),a.output.resolve()
    if not 1<=a.workers<=24:raise ValueError('Invalid worker count')
    Provider(project,'argon')
    with exclusive(output/'agent.lock'):
        freeze_agents(project,output)
        tasks=iter((r['case_id'],policy) for r in sorted(read_json(output/'prepared.json')['cases'],key=lambda r:(r['outer'],r['patient'])) for policy in POLICIES)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            active={}
            def submit():
                item=next(tasks,None)
                if item is not None:active[pool.submit(run_policy,project,output,*item)]=item
            for _ in range(a.workers):submit()
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:future.result()
                for future in done:del active[future]
                for _ in done:submit()
    print('ALL SIX POLICIES COMPLETE',flush=True)

if __name__=='__main__':main()
