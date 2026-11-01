"""Audit actual schemas, complete decisions, feedback and matched scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v2.verify import read_native,auc
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v4.screen import bootstrap_delta
from .experiment import write_json,prompt_for,NUMERIC,LLM
from .provider import response_format,validate,parse_json

def audit(project):
    project=Path(project);output=project/'outputs/spes_agent_v7';protocol=read_json(output/'protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Frozen source changed')
    cases={c['case_id']:c for c in protocol['cases']};metrics=[];attempts=accepted=artifacts=0
    for path in sorted(output.glob('trajectories/*/*/metrics.json')):
        m=read_json(path);folder=path.parent;case=cases[m['case_id']];native=read_native(folder/'scienceclaw');ids=read_json(folder/'ids.json')
        if len(native)!=24 or len(ids)!=24:raise ValueError('Native artifact budget')
        old=read_json(project/'outputs/spes_agent_v4/active/trajectories'/case['case_id']/'random_0/metrics.json')
        if m['prefix']!=old['queries'][:4] or m['queries'][:4]!=m['prefix'] or len(set(m['queries']))!=8:raise ValueError('Shared prefix or measurement budget')
        numeric=m['policy'] in NUMERIC;usages=list(folder.glob('*_usage.json'))
        if len(usages)!=(0 if numeric else 8):raise ValueError('Actual API budget differs')
        states=[]
        for step in range(8):
            public=read_json(folder/f'{step:02d}_public.json');tool=read_json(folder/f'{step:02d}_tool.json');state=read_json(folder/f'{step:02d}_decision.json')
            if public!=tool or native[ids[f'tool_{step}']]['payload']!=tool or native[ids[f'decision_{step}']]['payload']!=state:raise ValueError('Native payload mismatch')
            if public['budget']['used']!=4+step//2 or len(public['evidence']['observed']['queries'])!=4+step//2:raise ValueError('Feedback schedule differs')
            if step%2 and public!=read_json(folder/f'{step-1:02d}_public.json'):raise ValueError('New feedback between proposal/review')
            if not numeric:
                request=read_json(folder/f'{step:02d}_request.json');usage=read_json(folder/f'{step:02d}_usage.json')
                if request!=prompt_for(public,m['policy'],step,states):raise ValueError('History/rationale leak')
                digest=hashlib.sha256(json.dumps(request,separators=(',',':')).encode()).hexdigest()
                if usage['prompt_sha256']!=digest or usage['response_format']!=response_format(public) or not usage['structured_output']:raise ValueError('Actual captured schema differs')
                if usage['model']!='gemini-4-argon' or usage['reasoning']!='medium' or usage['retries']!=0:raise ValueError('Model or retry policy')
                if state['accepted_output']:
                    if usage.get('response_model')!='gemini-4-argon' or usage.get('finish_reason')!='stop':raise ValueError('Incomplete accepted output')
                    decoded=validate(parse_json(usage['response_content']),public)
                    if any(state[k]!=v for k,v in decoded.items()):raise ValueError('Model decision or rationale changed during execution')
                    accepted+=1
            if step%2:
                obs=read_json(folder/f'adaptive_{step//2}.json')
                if native[ids[f'observation_{step//2}']]['payload']!=obs or obs['stimulus']!=state['next_query'] or obs['stimulus']!=m['queries'][4+step//2]:raise ValueError('Review not executed')
            states.append(state)
        y=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz')
        if abs(auc(y,m['prediction'])-m['AUROC'])>1e-12:raise ValueError('Independent AUROC')
        if m['policy']=='random_0' and (abs(m['AUROC']-old['AUROC'])>1e-12 or not np.allclose(m['prediction'],old['prediction'],atol=1e-7,rtol=0)):raise ValueError('Old random reference differs')
        metrics.append(m);attempts+=len(usages);artifacts+=len(native)
    pilot_ids=[c['case_id'] for c in protocol['cases'] if c['pilot']];lookup={(m['case_id'],m['policy']):m for m in metrics if m['pilot']}
    complete=all((c,p) in lookup for c in pilot_ids for p in NUMERIC+LLM)
    if complete:
        means={p:float(np.mean([lookup[c,p]['AUROC'] for c in pilot_ids])) for p in NUMERIC+LLM};comparisons={}
        for p in ['single','multi_reason_hidden']+NUMERIC:
            delta=[lookup[c,'multi']['AUROC']-lookup[c,p]['AUROC'] for c in pilot_ids];comparisons[p]={'mean_gain':float(np.mean(delta)),'paired_95CI':bootstrap_delta(delta)}
        rate=sum(lookup[c,p]['accepted_model_outputs'] for c in pilot_ids for p in LLM)/120
        decision={'pilot_complete':True,'expand':bool(means['multi']-means['single']>.005 and means['multi']>=max(means[p] for p in NUMERIC)-.02 and rate>=.95),
          'mean_AUROC':means,'accepted_fraction':rate,'comparisons_multi_minus':comparisons,
          'rule':protocol['pilot_expansion_rule'],'caveat':'Exploratory five-patient screen; no superiority or clinical claim.'}
        write_json(output/'pilot_decision.json',decision)
    result={'verified':True,'completed_trajectories':len(metrics),'native_artifacts':artifacts,'attempts_in_completed':attempts,
      'formal_reserved_attempts':len(list(output.glob('trajectories/*/*/*_usage.json'))),'separate_synthetic_preflight_calls':2,
      'accepted_in_completed':accepted,'pilot_complete':complete,
      'checks':['Same four real prefix measurements','Same proposal/review feedback schedule','Actual API JSON Schema captures',
                'Full raw model decisions/rationales equal executed states','No peer rationale leak','Independent AUROC','Exact old random_0 scores','Native hashes and backward lineage','No request replay']}
    write_json(output/'audit.json',result);print(result)
    if complete:print(decision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();audit(a.project)
