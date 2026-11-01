"""Matched-budget ScienceClaw/Gemini acquisition experiment with real feedback."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,write_json,sha,exclusive
from .acquisition import Acquisition,case_id,numeric_choice
from .screen import bootstrap_delta

NUMERIC=['random_0','random_1','coverage','score_focus','uncertainty']
AGENTS=['single','multi','multi_no_exchange']
ROLES=['coverage','hypothesis','uncertainty','auditor']
SPECIALTY={
 'single':'Integrate coverage, diagnostic hypothesis, uncertainty and evidence quality. Reconsider your plan after each real observation.',
 'coverage':'Seek complementary geometric coverage and avoid redundant measurements. Explain what new recording contacts the query could inform.',
 'hypothesis':'Use currently observed HAT rankings to choose informative probes of competing high-ranking contact hypotheses. Do not assume high scores are true SOZ.',
 'uncertainty':'Consider disagreement among the three trained models and which measurement may inform uncertain contacts. Disagreement is not a calibrated error probability.',
 'auditor':'Look for weakly covered or poorly supported diagnostic claims, challenge premature focus, and choose a measurement that could falsify the current plan.'}

def freeze(project,output,training):
    project,output,training=Path(project),Path(output),Path(training);n=read_json(training/'neural_protocol.json')
    sources=['experiments/spes_agent_v4/'+f for f in ['active.py','acquisition.py','neural.py']]
    sources+=['skills/neurotarget-spes-v4/SKILL.md','skills/neurotarget-spes-v4/scripts/observe.py',
       'experiments/spes_agent_v2/providers.py','neurotarget/scienceclaw_runtime.py','config/upstream-lock.json']
    cases=[{'case_id':case_id(s,r['outer']),'subject':s,'outer':r['outer'],'pilot':i==0}
        for r in n['jobs'] for i,s in enumerate(r['predict'])]
    value={'experiment':'v4 active diagnostic SPES replay','scope':'New exploratory acquisition task on an inspected patient cohort, not the full-response paper task',
      'cases':cases,'policies':NUMERIC+AGENTS,'model':'gemini-4-argon','reasoning':'medium','max_measurements':8,'max_LLM_attempts':8,
      'native_public_queries':8,'numeric_actual_LLM_calls':0,'final_scorer':'Fixed HAT using observed columns only; identical across policies',
      'feedback':'Real released waveforms revealed after each selected stimulus column. Recompute all three models from observed columns only.',
      'known_geometry':'Distances and released preprocessing availability map assumed known in advance. This assumption can overstate real prospective knowledge.',
      'visibility':{'single':'Eight sequential calls with all earlier measurements and own decisions',
        'multi':'Four independent roles propose one query each, execute four deduplicated queries, then four roles see all real feedback and the first-round rationales before proposing four more',
        'multi_no_exchange':'Same two batches and shared real feedback; other roles rationales never visible'},
      'execution':'One selected query per model attempt. Up to three ordered unique alternatives allowed for deterministic deduplication. Invalid/spent failures use geometric coverage, with no replacement LLM request.',
      'primary':'Mean patient AUROC at eight measurements: multi versus single; also compare every same-budget numeric control',
      'stage_gate':'Before any LLM, numerical pilot trajectories must show mean per-patient AUROC range at least 0.01 across fixed policies, with at least three distinct query sequences. This is exploratory feasibility, not evidence of agent benefit.',
      'pilot':'Five fixed patients (first subject in each outer test fold), 120 maximum LLM attempts; kept separate in later reports',
      'retry':'No API retries, no replacing malformed or uncaptured attempts; all attempts recorded before transmission',
      'source_sha256':{s:sha(project/s) for s in sources},'neural_protocol_sha256':sha(training/'neural_protocol.json'),
      'model_sha256':{f'{r["outer"]}/{f}':sha(training/f'neural/outer{r["outer"]}/{f}/checkpoint.pth') for r in n['jobs'] for f in ['baseline','mte','hat']}}
    path=output/'active_protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen active protocol changed')
    write_json(path,value);return value

def check_sources(project,protocol):
    if any(sha(Path(project)/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Active source changed after freeze')

def validate_answer(answer,public):
    if not isinstance(answer,dict) or set(answer)!={'ranked_queries','evidence_ids','reason','counterargument'}:raise ValueError('Invalid decision schema')
    allowed={r['id'] for r in public['evidence']['geometry']['candidates']};ranked=answer['ranked_queries']
    if not isinstance(ranked,list) or not 1<=len(ranked)<=3 or len(set(ranked))!=len(ranked) or not set(ranked)<=allowed:raise ValueError('Invalid query choices')
    ids=answer['evidence_ids']
    if not isinstance(ids,list) or not ids or not all(isinstance(i,str) for i in ids) or not set(ids)<=set(public['evidence']):raise ValueError('Invalid evidence citations')
    for key in ['reason','counterargument']:
        if not isinstance(answer[key],str) or not 0<len(answer[key].strip())<=900:raise ValueError('Incomplete rationale')
    return {**answer,'accepted_output':True,'origin':'model'}

def make_prompt(public,policy,step,states):
    role='single' if policy=='single' else ROLES[step%4]
    board=[] if policy=='multi_no_exchange' or step<4 and policy!='single' else states
    return {**public,'role':role,'expertise':SPECIALTY[role],'decision':step+1,'shared_decisions':board,
       'instruction':'Choose actual next measurements, never generate contact probabilities. All evidence is supplied equally to every strategy. Write complete English rationales. Cite exact evidence keys. Return only the required JSON.',
       'required_json':{'ranked_queries':['one to three ordered available stimulus IDs; code takes the first not already queried'],
            'evidence_ids':['geometry | observed | model_uncertainty | limitations'],'reason':'At most 900 characters',
            'counterargument':'At most 900 characters; state what feedback could change your view'}}

def run_case(project,output,training,case,policy):
    import torch
    from neurotarget.scienceclaw_runtime import Framework
    from experiments.spes_agent_v2.providers import Provider,ProviderUnavailable,parse_json
    project,output,training=Path(project),Path(output),Path(training);protocol=read_json(output/'active_protocol.json');check_sources(project,protocol)
    folder=output/'trajectories'/case['case_id']/policy;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return read_json(folder/'metrics.json')
    if sha(training/'neural_protocol.json')!=protocol['neural_protocol_sha256']:raise ValueError('Neural protocol changed')
    for family in ['baseline','mte','hat']:
        if sha(training/f'neural/outer{case["outer"]}/{family}/checkpoint.pth')!=protocol['model_sha256'][f'{case["outer"]}/{family}']:
            raise ValueError('Frozen acquisition model changed')
    torch.set_num_threads(2);env=Acquisition(project,training,case['outer'],case['subject'])
    framework=Framework(project,folder/'scienceclaw');numeric=policy in NUMERIC;states=[];client=None
    random_seed=6190+case['outer']*100+int(case['subject'][-2:])+ (0 if policy=='random_0' else 10000)
    id_path=folder/'artifact_ids.json';ids=read_json(id_path) if id_path.exists() else {}
    def artifact(key,agent,skill,payload,kind,parents):
        if key not in ids:
            ids[key]=framework.artifact(agent,skill,payload,kind,[ids[k] for k in parents]).artifact_id;write_json(id_path,ids)
        return ids[key]
    batch_size=1 if numeric or policy=='single' else 4
    for start in range(0,8,batch_size):
        batch=[];common=env.public()
        for step in range(start,start+batch_size):
            path=folder/f'{step:02d}_public.json'
            if path.exists():public=read_json(path)
            else:public=common;write_json(path,public)
            # This native skill reads a public snapshot only. It has no access to
            # the environment object, model weights, raw data, or hidden judge.
            meta=framework.registry.get_skill('neurotarget-spes-v4')
            if not meta:raise ValueError('Native acquisition skill missing')
            tool_path=folder/f'{step:02d}_tool.json'
            if tool_path.exists():tool=read_json(tool_path)
            else:
                result=framework.executor.execute_skill('neurotarget-spes-v4',meta,{'project':str(project),'state_path':str(path),'format':'json'},timeout=90)
                if result.get('status')!='success' or not isinstance(result.get('result'),dict):raise RuntimeError('Native observation skill failed')
                tool=result['result'];write_json(tool_path,tool)
            prior=[f'observation_{i}' for i in range(start)]
            artifact(f'tool_{step}','tools','neurotarget-spes-v4',tool,'observation_gated_tool_result',prior)
            decision_path=folder/f'{step:02d}_decision.json'
            if decision_path.exists():state=read_json(decision_path)
            elif numeric:
                state={'ranked_queries':[numeric_choice(public,policy,np.random.default_rng(random_seed+step*100003))],'evidence_ids':['geometry','model_uncertainty'],
                    'reason':'Fixed predeclared numerical acquisition rule.','counterargument':'Coverage and score disagreement are heuristic acquisition objectives.',
                    'accepted_output':True,'origin':'numeric'}
            else:
                prompt=make_prompt(tool,policy,step,states);request=folder/f'{step:02d}_request.json';usage=folder/f'{step:02d}_usage.json'
                if request.exists():prompt=read_json(request)
                else:write_json(request,prompt)
                if not usage.exists():
                    if client is None:client=Provider(project,'argon')
                    try:client.ask(prompt,usage)
                    except ProviderUnavailable:pass
                record=read_json(usage)
                try:
                    if not record.get('captured_response') or record.get('response_model')!='gemini-4-argon' or record.get('finish_reason')!='stop':raise ValueError('Spent incomplete request')
                    state=validate_answer(parse_json(record.get('response_content')),public)
                except (ValueError,TypeError,KeyError):
                    state={'ranked_queries':[numeric_choice(public,'coverage')],'evidence_ids':['geometry'],
                        'reason':'The spent attempt produced no accepted choice; execute the predeclared coverage fallback.',
                        'counterargument':'No replacement request is issued.','accepted_output':False,'origin':'spent_default'}
            state={**state,'step':step+1,'role':'numeric' if numeric else ('single' if policy=='single' else ROLES[step%4])}
            write_json(decision_path,state);batch.append(state)
            board_parents=[] if policy=='multi_no_exchange' or (policy!='single' and start==0) else [f'decision_{i}' for i in range(start)]
            artifact(f'decision_{step}',state['role'],'query-proposal',state,'diagnostic_query_proposal',[f'tool_{step}']+board_parents)
        # Proposals in a four-role batch all precede feedback. The coordinator
        # only removes duplicate queries; it neither scores hidden labels nor
        # requests another model answer.
        for offset,state in enumerate(batch):
            step=start+offset;available={r['id'] for r in env.candidate_table()}
            choice=next((q for q in state['ranked_queries'] if q in available),None)
            repaired=choice is None
            if repaired:choice=numeric_choice(env.public(),'coverage')
            observation=env.observe(int(choice[1:]));observation['duplicate_repair']=repaired
            write_json(folder/f'{step:02d}_observation.json',observation)
            artifact(f'observation_{step}','observations','reveal-real-SPES',observation,'real_recorded_response_feedback',[f'decision_{step}'])
        states+=batch
    metrics={**env.final_judge(),'case_id':case['case_id'],'policy':policy,'pilot':case['pilot'],'measurements':8,
        'native_public_queries':8,'LLM_attempts':0 if numeric else 8,'accepted_model_outputs':0 if numeric else sum(s['accepted_output'] for s in states),
        'actual_native_artifacts':len(ids),'trace_sha256':{p.name:sha(p) for p in sorted(folder.glob('*_decision.json'))}}
    write_json(folder/'metrics.json',metrics);print('ACTIVE_DONE',case['case_id'],policy,round(metrics['AUROC'],5),flush=True)
    return metrics

def feasibility(output,protocol):
    rows=[]
    for case in protocol['cases']:
        if not case['pilot']:continue
        metrics=[read_json(Path(output)/'trajectories'/case['case_id']/p/'metrics.json') for p in NUMERIC]
        rows.append({'case_id':case['case_id'],'range':max(m['AUROC'] for m in metrics)-min(m['AUROC'] for m in metrics),
           'distinct_queries':len({tuple(m['queries']) for m in metrics})})
    launch=np.mean([r['range'] for r in rows])>=.01 and all(r['distinct_queries']>=3 for r in rows)
    value={'launch_agent_pilot':bool(launch),'mean_AUROC_range':float(np.mean([r['range'] for r in rows])),'rows':rows,
        'caveat':'Exploratory feasibility using fixed pilot outcomes. Not evidence of learnable headroom, LLM benefit or clinical efficacy.'}
    write_json(Path(output)/'active_feasibility.json',value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--stage',choices=['freeze','numeric-pilot','numeric-all','agent-pilot','agent-remaining'],required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.stage=='freeze':freeze(a.project,a.output,a.training);print('Active acquisition protocol frozen');return
    protocol=read_json(a.output/'active_protocol.json');check_sources(a.project,protocol)
    cases=[c for c in protocol['cases'] if a.stage.endswith('all') or (c['pilot'] if a.stage.endswith('pilot') else not c['pilot'])]
    policies=AGENTS if a.stage.startswith('agent') else NUMERIC
    if a.stage.startswith('agent') and not feasibility(a.output,protocol)['launch_agent_pilot']:
        print('STOP: numerical feasibility gate failed; zero new LLM calls.');return
    if not 1<=a.workers<=8:raise ValueError('One to eight bounded local workers')
    with exclusive(a.output/f'{a.stage}.lock'):
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run_case,a.project,a.output,a.training,c,policy) for c in cases for policy in policies]
            for future in as_completed(futures):future.result()
    if not a.stage.startswith('agent'):print(feasibility(a.output,protocol))

if __name__=='__main__':main()
