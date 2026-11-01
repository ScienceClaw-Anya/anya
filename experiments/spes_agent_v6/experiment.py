"""Four shared warm-up measurements, then four propose/review/observe cycles.

Single and multi agents have identical evidence, eight attempts, four adaptive
decisions and the same physiological feedback cadence. The previous studies
remain frozen. This experiment retains the original v4 HAT scorer throughout.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
from pathlib import Path
import time
import numpy as np
from experiments.spes_agent_v3.common import read_json,write_json as _atomic_write,sha,exclusive
from experiments.spes_agent_v4.acquisition import Acquisition,numeric_choice
from experiments.spes_agent_v4.active import SPECIALTY

NUMERIC=['random_0','random_1','coverage','score_focus','uncertainty']
LLM=['single','multi','multi_reason_hidden']

def write_json(path,value):
    # Retry only a local atomic-write transaction, never a model request.
    for attempt in range(10):
        try:return _atomic_write(path,value)
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05)

def freeze(project,output):
    project,output=Path(project),Path(output);previous=project/'outputs/spes_agent_v4/active/active_protocol.json';v4=read_json(previous)
    cases=[]
    for outer in range(5):
        candidates=[c for c in v4['cases'] if c['outer']==outer and not c['pilot']]
        for i,c in enumerate(candidates):cases.append({**c,'pilot':i==0})
    sources=['experiments/spes_agent_v6/experiment.py','experiments/spes_agent_v4/acquisition.py','experiments/spes_agent_v4/neural.py',
       'experiments/spes_agent_v4/active.py','experiments/spes_agent_v2/providers.py','neurotarget/scienceclaw_runtime.py',
       'skills/neurotarget-spes-v6/SKILL.md','skills/neurotarget-spes-v6/scripts/observe.py','config/upstream-lock.json']
    value={'study':'v6 feedback-matched proposal/review collaboration','scope':'Adaptive exploratory follow-up. New fixed pilot excludes the five v4 LLM pilot patients; all cohort identities previously inspected.',
      'cases':cases,'scorer':'Original v4 full-response-trained HAT, unchanged across arms; no adoption of v5 sparse fits in this study',
      'prefix':'First four random_0 measurements from v4, identical within patient across every arm',
      'budget':{'total_measurements':8,'shared_warmup':4,'adaptive_measurements':4,'LLM_attempts_per_LLM_arm':8,'native_public_queries':8,'numerical_LLM_attempts':0},
      'model':'gemini-4-argon','reasoning':'medium','retries':0,
      'cadence':'For each adaptive measurement: proposal call, review call, execute reviewer choice, receive real waveform feedback. Identical for single and multi.',
      'roles':{'single':'One generalist performs both proposal and self-review with all four specialties supplied.',
        'multi':'Hypothesis/uncertainty proposers alternate; audit/coverage reviewers alternate. Full history and measured feedback shared.',
        'multi_reason_hidden':'Same role schedule, query choices and measured feedback visible, but other roles textual rationales hidden.'},
      'schema':'Ranked available queries, exact evidence IDs, complete English reason/counterargument up to 6000 chars. Never probabilities. Invalid or uncaptured requests stay spent and use coverage.',
      'primary':'Mean patient AUROC at eight measurements: multi minus single; all same-patient numerical controls also reported',
      'pilot_rule':'Five patients, one per outer fold. Expand only if multi exceeds single by >0.005, is not >0.02 below the best fixed numerical pilot policy, and >=90% of model decisions are accepted. No significance claim from five patients.',
      'numerical_gate':'Before pilot calls, fixed numerical policies must yield distinct final rankings and mean per-patient AUROC range >=0.01 on these pilot cases.',
      'source_sha256':{s:sha(project/s) for s in sources},'v4_protocol_sha256':sha(previous),
      'model_sha256':v4['model_sha256'],
      'prefix_metric_sha256':{c['case_id']:sha(project/'outputs/spes_agent_v4/active/trajectories'/c['case_id']/'random_0/metrics.json') for c in cases}}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen v6 protocol changed')
    write_json(path,value);return value

def role_for(policy,step):
    if policy=='single':return 'single'
    return (['hypothesis','auditor','uncertainty','coverage']*2)[step]

def visible_states(states,policy,role):
    if policy!='multi_reason_hidden':return states
    return [{k:v for k,v in s.items() if s['role']==role or k not in ['reason','counterargument']} for s in states]

def prompt_for(public,policy,step,states):
    role=role_for(policy,step);stage='proposal' if step%2==0 else 'review'
    specialty=' '.join(SPECIALTY[k] for k in ['hypothesis','uncertainty','auditor','coverage']) if role=='single' else SPECIALTY[role]
    return {**public,'role':role,'expertise':specialty,'stage':stage,'adaptive_measurement':step//2+1,
       'history':visible_states(states,policy,role),'instruction':
       ('Propose the next informative measurement. The following reviewer may challenge it.' if stage=='proposal' else
        'Review the immediately preceding proposal against actual evidence. Keep or replace its first query; code executes your reviewed choice next.')+
        ' Every query choice must be currently available. Do not write contact probabilities. Use only supplied evidence. Return JSON only and write concise complete English.',
       'required_json':{'ranked_queries':['one to three ordered available stimulus IDs'],'evidence_ids':['exact evidence keys'],
           'reason':'A concise complete rationale, up to 6000 characters','counterargument':'A limitation and what new response would change your view, up to 6000 characters'}}

def validate(answer,public):
    if not isinstance(answer,dict) or set(answer)!={'ranked_queries','evidence_ids','reason','counterargument'}:raise ValueError('Invalid schema')
    q=answer['ranked_queries'];allowed={r['id'] for r in public['evidence']['geometry']['candidates']}
    if not isinstance(q,list) or not 1<=len(q)<=3 or not all(isinstance(x,str) for x in q) or len(q)!=len(set(q)) or not set(q)<=allowed:raise ValueError('Invalid queries')
    e=answer['evidence_ids']
    if not isinstance(e,list) or not e or not all(isinstance(i,str) for i in e) or not set(e)<=set(public['evidence']):raise ValueError('Invalid citations')
    for k in ['reason','counterargument']:
        if not isinstance(answer[k],str) or not 0<len(answer[k].strip())<=6000:raise ValueError('Invalid rationale')
    return {**answer,'accepted_output':True,'origin':'model'}

def run_case(project,output,case,policy):
    import torch
    from neurotarget.scienceclaw_runtime import Framework
    from experiments.spes_agent_v2.providers import Provider,ProviderUnavailable,parse_json
    project,output=Path(project),Path(output);protocol=read_json(output/'protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Frozen source differs')
    folder=output/'trajectories'/case['case_id']/policy;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return read_json(folder/'metrics.json')
    training=project/'outputs/spes_agent_v4/multitrial';torch.set_num_threads(2)
    for family in ['baseline','mte','hat']:
        if sha(training/f'neural/outer{case["outer"]}/{family}/checkpoint.pth')!=protocol['model_sha256'][f'{case["outer"]}/{family}']:raise ValueError('Frozen model differs')
    env=Acquisition(project,training,case['outer'],case['subject']);framework=Framework(project,folder/'scienceclaw')
    prefix_path=project/'outputs/spes_agent_v4/active/trajectories'/case['case_id']/'random_0/metrics.json'
    if sha(prefix_path)!=protocol['prefix_metric_sha256'][case['case_id']]:raise ValueError('Prefix input changed')
    prefix=read_json(prefix_path)['queries'][:4];ids=read_json(folder/'ids.json') if (folder/'ids.json').exists() else {}
    def artifact(key,agent,payload,kind,parents):
        if key not in ids:
            ids[key]=framework.artifact(agent,'neurotarget-spes-v6',payload,kind,[ids[k] for k in parents]).artifact_id;write_json(folder/'ids.json',ids)
    for i,q in enumerate(prefix):
        obs=env.observe(int(q[1:]));write_json(folder/f'warmup_{i}.json',obs)
        artifact(f'warmup_{i}','warmup',obs,'shared_real_measurement',[f'warmup_{j}' for j in range(i)])
    states=[];client=None;numeric=policy in NUMERIC
    seed=6190+case['outer']*100+int(case['subject'][-2:])+(10000 if policy=='random_1' else 0)
    for step in range(8):
        public_path=folder/f'{step:02d}_public.json'
        if public_path.exists():public=read_json(public_path)
        else:public=env.public();write_json(public_path,public)
        tool_path=folder/f'{step:02d}_tool.json'
        if tool_path.exists():tool=read_json(tool_path)
        else:
            metadata=framework.registry.get_skill('neurotarget-spes-v6')
            if not metadata:raise ValueError('Native v6 observation skill missing')
            result=framework.executor.execute_skill('neurotarget-spes-v6',metadata,{'project':str(project),'state_path':str(public_path),'format':'json'},timeout=90)
            if result.get('status')!='success' or not isinstance(result.get('result'),dict):raise RuntimeError('Observation tool failed')
            tool=result['result'];write_json(tool_path,tool)
        artifact(f'tool_{step}','tools',tool,'public_observation_tool',[f'warmup_{i}' for i in range(4)]+[f'observation_{i}' for i in range(step//2)])
        decision_path=folder/f'{step:02d}_decision.json'
        if decision_path.exists():state=read_json(decision_path)
        elif numeric:
            query=numeric_choice(public,policy,np.random.default_rng(seed+(4+step//2)*100003))
            state={'ranked_queries':[query],'evidence_ids':['geometry','model_uncertainty'],'reason':'Fixed numerical rule; no LLM call.',
               'counterargument':'Geometric gains and disagreement are uncalibrated heuristics.','accepted_output':True,'origin':'numeric'}
        else:
            prompt=prompt_for(tool,policy,step,states);request=folder/f'{step:02d}_request.json';usage=folder/f'{step:02d}_usage.json'
            if request.exists():prompt=read_json(request)
            else:write_json(request,prompt)
            if not usage.exists():
                if client is None:client=Provider(project,'argon')
                try:client.ask(prompt,usage)
                except ProviderUnavailable:pass
            saved=read_json(usage)
            try:
                if not saved.get('captured_response') or saved.get('response_model')!='gemini-4-argon' or saved.get('finish_reason')!='stop':raise ValueError('Spent incomplete request')
                state=validate(parse_json(saved.get('response_content')),public)
            except (ValueError,TypeError,KeyError):
                state={'ranked_queries':[numeric_choice(public,'coverage')],'evidence_ids':['geometry'],
                    'reason':'Spent request yielded no accepted executable decision; apply the declared coverage default.',
                    'counterargument':'No replacement request.','accepted_output':False,'origin':'spent_default'}
        state={**state,'step':step+1,'role':'numeric' if numeric else role_for(policy,step),'stage':'proposal' if step%2==0 else 'review'}
        write_json(decision_path,state);artifact(f'decision_{step}',state['role'],state,'proposal_or_review',[f'tool_{step}']+[f'decision_{i}' for i in range(step)])
        states.append(state)
        if step%2==1:
            obs=env.observe(int(state['ranked_queries'][0][1:]));write_json(folder/f'adaptive_{step//2}.json',obs)
            artifact(f'observation_{step//2}','observations',obs,'real_response_feedback',[f'decision_{step}'])
    metric={**env.final_judge(),'case_id':case['case_id'],'policy':policy,'pilot':case['pilot'],'prefix':prefix,
       'LLM_attempts':0 if numeric else 8,'accepted_model_outputs':0 if numeric else sum(s['accepted_output'] for s in states),
       'native_public_queries':8,'measurements':8,'adaptive_measurements':4,'native_artifacts':len(ids)}
    write_json(folder/'metrics.json',metric);print('V6_DONE',case['case_id'],policy,round(metric['AUROC'],6),flush=True);return metric

def numerical_gate(output,protocol):
    ranges=[];distinct=[]
    for c in protocol['cases']:
        if not c['pilot']:continue
        rows=[read_json(Path(output)/'trajectories'/c['case_id']/p/'metrics.json') for p in NUMERIC]
        ranges.append(max(r['AUROC'] for r in rows)-min(r['AUROC'] for r in rows))
        distinct.append(len({tuple(np.argsort(r['prediction'])) for r in rows}))
    result={'launch':bool(np.mean(ranges)>=.01 and all(n>1 for n in distinct)),
       'mean_pilot_AUROC_range':float(np.mean(ranges)),'distinct_rankings_per_case':distinct,'caveat':'Exploratory feasibility only, not proof of learnable benefit.'}
    write_json(Path(output)/'numerical_gate.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['freeze','numeric-pilot','agent-pilot','numeric-all','agent-remaining'],required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.stage=='freeze':freeze(a.project,a.output);print('V6 protocol frozen');return
    protocol=read_json(a.output/'protocol.json');agent=a.stage.startswith('agent')
    if agent and not numerical_gate(a.output,protocol)['launch']:print('Numerical gate failed; no LLM requests');return
    if a.stage=='agent-remaining':raise ValueError('Expansion requires a completed pilot audit and explicit recorded pilot-rule decision first')
    cases=[c for c in protocol['cases'] if a.stage=='numeric-all' or c['pilot']];policies=LLM if agent else NUMERIC
    if not 1<=a.workers<=8:raise ValueError('One to eight workers')
    with exclusive(a.output/f'{a.stage}.lock'):
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run_case,a.project,a.output,c,policy) for c in cases for policy in policies]
            for f in as_completed(futures):f.result()
    if not agent:print(numerical_gate(a.output,protocol))

if __name__=='__main__':main()
