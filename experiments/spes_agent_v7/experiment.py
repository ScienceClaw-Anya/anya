"""Common-prefix collaboration with API-enforced executable JSON output."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v4.acquisition import Acquisition,numeric_choice
from experiments.spes_agent_v4.active import SPECIALTY
from experiments.spes_agent_v6.experiment import write_json,role_for,visible_states
from .provider import StructuredProvider,ProviderUnavailable,parse_json,validate,EVIDENCE,response_format

NUMERIC=['random_0','random_1','coverage','score_focus','uncertainty']
LLM=['single','multi','multi_reason_hidden']

def freeze(project,output):
    project,output=Path(project),Path(output);v6=read_json(project/'outputs/spes_agent_v6/protocol.json')
    interface=read_json(output/'interface_preflight/summary.json')
    if not interface['passed']:raise ValueError('Structured interface checks must pass first')
    cases=[]
    for outer in range(5):
        remaining=[c for c in v6['cases'] if c['outer']==outer and not c['pilot']]
        cases.extend([{**c,'pilot':i==0} for i,c in enumerate(remaining)])
    sources=['experiments/spes_agent_v7/experiment.py','experiments/spes_agent_v7/provider.py',
      'experiments/spes_agent_v6/experiment.py','experiments/spes_agent_v4/acquisition.py','experiments/spes_agent_v4/neural.py',
      'experiments/spes_agent_v4/active.py','experiments/spes_agent_v2/providers.py','neurotarget/scienceclaw_runtime.py',
      'skills/neurotarget-spes-v7/SKILL.md','skills/neurotarget-spes-v7/scripts/observe.py','config/upstream-lock.json']
    value={'study':'v7 schema-constrained, feedback-matched proposal/review',
      'scope':'Exploratory new pilot excludes both earlier five-patient LLM pilots. All underlying patient identities were inspected in prior numerical work.',
      'cases':cases,'model':'gemini-4-argon','reasoning':'medium','retries':0,'total_measurements':8,'common_prefix_measurements':4,
      'adaptive_measurements':4,'LLM_attempts_per_LLM_arm':8,'native_public_queries':8,'numerical_LLM_attempts':0,
      'cadence':'Proposal, review, execute reviewer query, real feedback; identical in single/multi/hidden arms.',
      'scorer':'Unchanged original v4 HAT; sparse v5 scorers not adopted',
      'roles':'Same v6 specialty schedule. Single integrates all specialties. Hidden condition removes other roles rationale text but preserves choices and physiological feedback.',
      'interface':'Actual API response_format JSON Schema: next_query enum from current catalog, evidence_ids enum of four real roots, no probabilities or extra fields. Complete reasons kept, maximum 6000 characters.',
      'interface_smoke':'Two synthetic schema checks, then one full single-agent pilot trajectory must accept all eight attempts before the rest of the pilot launches. The full trajectory counts within the 120-attempt pilot, not extra.',
      'pilot_expansion_rule':'Five fixed patients; multi-minus-single >0.005, no more than 0.02 below best declared numeric policy, >=95% accepted model decisions. Pilot is not confirmatory.',
      'primary':'Mean SOZ AUROC at eight measurements, multi versus single, plus every same-patient numerical control',
      'numeric_gate':'Mean per-patient AUROC range across fixed numeric pilot policies >=0.01; all pilot cases have multiple distinct rankings.',
      'source_sha256':{s:sha(project/s) for s in sources},'model_sha256':v6['model_sha256'],
      'interface_preflight_sha256':sha(output/'interface_preflight/summary.json'),
      'prefix_metric_sha256':{c['case_id']:sha(project/'outputs/spes_agent_v4/active/trajectories'/c['case_id']/'random_0/metrics.json') for c in cases}}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen v7 protocol changed')
    write_json(path,value);return value

def prompt_for(public,policy,step,states):
    role=role_for(policy,step);stage='proposal' if step%2==0 else 'review'
    specialty=' '.join(SPECIALTY[k] for k in ['hypothesis','uncertainty','auditor','coverage']) if role=='single' else SPECIALTY[role]
    return {**public,'role':role,'expertise':specialty,'stage':stage,'adaptive_measurement':step//2+1,
      'history':visible_states(states,policy,role),'instruction':
      ('Propose the next informative measurement.' if stage=='proposal' else 'Review the latest proposal; keep or replace its next_query. Your choice is executed before new feedback.')+
      ' Choose one currently available ID. Do not generate contact probabilities. Cite only these exact evidence IDs: geometry, observed, model_uncertainty, limitations. A descriptive subfield may be discussed in the reason but is not an evidence ID. Write concise complete English.',
      'required_json':{'next_query':'one available ID','evidence_ids':EVIDENCE,'reason':'Evidence-grounded rationale in a few sentences','counterargument':'A limitation and what new response would change the plan'}}

def run_case(project,output,case,policy):
    import torch
    from neurotarget.scienceclaw_runtime import Framework
    project,output=Path(project),Path(output);protocol=read_json(output/'protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Frozen source changed')
    folder=output/'trajectories'/case['case_id']/policy;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return read_json(folder/'metrics.json')
    training=project/'outputs/spes_agent_v4/multitrial';torch.set_num_threads(2)
    for f in ['baseline','mte','hat']:
        if sha(training/f'neural/outer{case["outer"]}/{f}/checkpoint.pth')!=protocol['model_sha256'][f'{case["outer"]}/{f}']:raise ValueError('Scorer changed')
    env=Acquisition(project,training,case['outer'],case['subject']);framework=Framework(project,folder/'scienceclaw')
    prefix_path=project/'outputs/spes_agent_v4/active/trajectories'/case['case_id']/'random_0/metrics.json'
    if sha(prefix_path)!=protocol['prefix_metric_sha256'][case['case_id']]:raise ValueError('Prefix changed')
    prefix=read_json(prefix_path)['queries'][:4];ids=read_json(folder/'ids.json') if (folder/'ids.json').exists() else {}
    def artifact(key,agent,payload,kind,parents):
        if key not in ids:
            ids[key]=framework.artifact(agent,'neurotarget-spes-v7',payload,kind,[ids[p] for p in parents]).artifact_id;write_json(folder/'ids.json',ids)
    for i,q in enumerate(prefix):
        obs=env.observe(int(q[1:]));write_json(folder/f'warmup_{i}.json',obs)
        artifact(f'warmup_{i}','warmup',obs,'shared_real_measurement',[f'warmup_{j}' for j in range(i)])
    numeric=policy in NUMERIC;states=[];client=None;seed=6190+case['outer']*100+int(case['subject'][-2:])+(10000 if policy=='random_1' else 0)
    for step in range(8):
        public_path=folder/f'{step:02d}_public.json'
        if public_path.exists():public=read_json(public_path)
        else:public=env.public();write_json(public_path,public)
        tool_path=folder/f'{step:02d}_tool.json'
        if tool_path.exists():tool=read_json(tool_path)
        else:
            metadata=framework.registry.get_skill('neurotarget-spes-v7')
            if not metadata:raise ValueError('Native v7 skill missing')
            result=framework.executor.execute_skill('neurotarget-spes-v7',metadata,{'project':str(project),'state_path':str(public_path),'format':'json'},timeout=90)
            if result.get('status')!='success' or not isinstance(result.get('result'),dict):raise RuntimeError('Public observation tool failed')
            tool=result['result'];write_json(tool_path,tool)
        artifact(f'tool_{step}','tools',tool,'gated_public_observation',[f'warmup_{i}' for i in range(4)]+[f'observation_{i}' for i in range(step//2)])
        decision_path=folder/f'{step:02d}_decision.json'
        if decision_path.exists():state=read_json(decision_path)
        elif numeric:
            q=numeric_choice(public,policy,np.random.default_rng(seed+(4+step//2)*100003))
            state={'next_query':q,'evidence_ids':['geometry','model_uncertainty'],'reason':'Fixed numerical rule.',
              'counterargument':'Heuristic objective, no validated expected AUROC gain.','accepted_output':True,'origin':'numeric'}
        else:
            request=folder/f'{step:02d}_request.json';usage=folder/f'{step:02d}_usage.json'
            if request.exists():prompt=read_json(request)
            else:prompt=prompt_for(tool,policy,step,states);write_json(request,prompt)
            if not usage.exists():
                if client is None:client=StructuredProvider(project)
                try:client.ask(prompt,usage)
                except ProviderUnavailable:pass
            saved=read_json(usage)
            try:
                if not saved.get('captured_response') or saved.get('response_model')!='gemini-4-argon' or saved.get('finish_reason')!='stop' or saved.get('response_format')!=response_format(public):raise ValueError('Incomplete or unconstrained response')
                state=validate(parse_json(saved.get('response_content')),public)
            except (ValueError,TypeError,KeyError):
                state={'next_query':numeric_choice(public,'coverage'),'evidence_ids':['geometry'],'reason':'Spent attempt produced no accepted choice; execute declared default.',
                   'counterargument':'No replacement model request.','accepted_output':False,'origin':'spent_default'}
        state={**state,'step':step+1,'role':'numeric' if numeric else role_for(policy,step),'stage':'proposal' if step%2==0 else 'review'}
        write_json(decision_path,state);artifact(f'decision_{step}',state['role'],state,'executable_proposal_or_review',[f'tool_{step}']+[f'decision_{i}' for i in range(step)])
        states.append(state)
        if step%2:
            obs=env.observe(int(state['next_query'][1:]));write_json(folder/f'adaptive_{step//2}.json',obs)
            artifact(f'observation_{step//2}','observations',obs,'real_response_feedback',[f'decision_{step}'])
    metric={**env.final_judge(),'case_id':case['case_id'],'policy':policy,'pilot':case['pilot'],'prefix':prefix,
       'LLM_attempts':0 if numeric else 8,'accepted_model_outputs':0 if numeric else sum(s['accepted_output'] for s in states),
       'native_public_queries':8,'measurements':8,'adaptive_measurements':4,'native_artifacts':len(ids)}
    write_json(folder/'metrics.json',metric);print('V7_DONE',case['case_id'],policy,round(metric['AUROC'],6),flush=True);return metric

def gate(output,protocol):
    values=[];distinct=[]
    for c in protocol['cases']:
        if not c['pilot']:continue
        rows=[read_json(Path(output)/'trajectories'/c['case_id']/p/'metrics.json') for p in NUMERIC]
        values.append(max(r['AUROC'] for r in rows)-min(r['AUROC'] for r in rows));distinct.append(len({tuple(np.argsort(r['prediction'])) for r in rows}))
    result={'launch':bool(np.mean(values)>=.01 and all(v>1 for v in distinct)),'mean_AUROC_range':float(np.mean(values)),'distinct_rankings':distinct}
    write_json(Path(output)/'numeric_gate.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['freeze','numeric-pilot','smoke','agent-pilot','numeric-all'],required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.stage=='freeze':freeze(a.project,a.output);print('V7 protocol frozen');return
    protocol=read_json(a.output/'protocol.json');pilot=[c for c in protocol['cases'] if c['pilot']]
    if a.stage in ['smoke','agent-pilot'] and not gate(a.output,protocol)['launch']:raise ValueError('Numeric feasibility gate failed')
    if a.stage=='smoke':
        result=run_case(a.project,a.output,pilot[0],'single')
        if result['accepted_model_outputs']!=8:raise ValueError('Interface smoke failed; do not expand')
        print('Full smoke trajectory accepted 8/8 calls; included in pilot budget.');return
    if a.stage=='agent-pilot':
        smoke=a.output/'trajectories'/pilot[0]['case_id']/'single/metrics.json'
        if not smoke.exists() or read_json(smoke)['accepted_model_outputs']!=8:raise ValueError('Complete and inspect the eight-call smoke before launching remaining pilot')
    if not 1<=a.workers<=8:raise ValueError('One to eight workers')
    cases=protocol['cases'] if a.stage=='numeric-all' else pilot;policies=LLM if a.stage=='agent-pilot' else NUMERIC
    with exclusive(a.output/f'{a.stage}.lock'):
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run_case,a.project,a.output,c,policy) for c in cases for policy in policies]
            for f in as_completed(futures):f.result()
    if a.stage!='agent-pilot':print(gate(a.output,protocol))

if __name__=='__main__':main()
