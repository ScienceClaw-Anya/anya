"""Replay scoring, shared prefixes, feedback schedules and actual API budgets."""
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

def audit(project):
    project=Path(project);output=project/'outputs/spes_agent_v6';protocol=read_json(output/'protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Frozen source differs')
    cases={c['case_id']:c for c in protocol['cases']};all_metrics=[];artifacts=attempts=accepted=0
    for path in sorted(output.glob('trajectories/*/*/metrics.json')):
        metric=read_json(path);folder=path.parent;case=cases[metric['case_id']];native=read_native(folder/'scienceclaw');ids=read_json(folder/'ids.json')
        if len(native)!=24 or len(ids)!=24:raise ValueError('Native budget differs')
        old=read_json(project/'outputs/spes_agent_v4/active/trajectories'/case['case_id']/'random_0/metrics.json')
        if metric['queries'][:4]!=old['queries'][:4] or metric['prefix']!=old['queries'][:4]:raise ValueError('Common prefix differs')
        if len(set(metric['queries']))!=8 or metric['measurements']!=8 or metric['adaptive_measurements']!=4:raise ValueError('Measurement budget')
        for i in range(4):
            if native[ids[f'warmup_{i}']]['payload']!=read_json(folder/f'warmup_{i}.json'):raise ValueError('Warm-up artifact mismatch')
        numeric=metric['policy'] in NUMERIC;usages=list(folder.glob('*_usage.json'))
        if len(usages)!=(0 if numeric else 8):raise ValueError('Actual request budget')
        states=[]
        for step in range(8):
            public=read_json(folder/f'{step:02d}_public.json');tool=read_json(folder/f'{step:02d}_tool.json');decision=read_json(folder/f'{step:02d}_decision.json')
            if public!=tool or native[ids[f'tool_{step}']]['payload']!=tool or native[ids[f'decision_{step}']]['payload']!=decision:raise ValueError('Native payload differs')
            if public['budget']['used']!=4+step//2 or len(public['evidence']['observed']['queries'])!=4+step//2:raise ValueError('Unequal feedback cadence or future observation leak')
            if step%2 and public!=read_json(folder/f'{step-1:02d}_public.json'):raise ValueError('Observation changed between proposal/review')
            if not numeric:
                request=read_json(folder/f'{step:02d}_request.json');usage=read_json(folder/f'{step:02d}_usage.json')
                if request!=prompt_for(public,metric['policy'],step,states):raise ValueError('History/rationale visibility differs')
                digest=hashlib.sha256(json.dumps(request,separators=(',',':')).encode()).hexdigest()
                if usage['prompt_sha256']!=digest or usage['model']!='gemini-4-argon' or usage['reasoning']!='medium' or usage['retries']!=0:raise ValueError('Provider differs')
                if decision['accepted_output']:
                    if usage.get('response_model')!='gemini-4-argon' or usage.get('finish_reason')!='stop':raise ValueError('Accepted incomplete response')
                    accepted+=1
            if step%2:
                obs=read_json(folder/f'adaptive_{step//2}.json')
                if native[ids[f'observation_{step//2}']]['payload']!=obs or obs['stimulus']!=decision['ranked_queries'][0] or obs['stimulus']!=metric['queries'][4+step//2]:raise ValueError('Reviewer choice not executed')
            states.append(decision)
        y=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz')
        if abs(auc(y,metric['prediction'])-metric['AUROC'])>1e-12:raise ValueError('Independent AUROC replay')
        if metric['policy']=='random_0' and (abs(metric['AUROC']-old['AUROC'])>1e-12 or not np.allclose(metric['prediction'],old['prediction'],atol=1e-7,rtol=0)):raise ValueError('Frozen random control changed')
        all_metrics.append(metric);artifacts+=len(native);attempts+=len(usages)
    comparisons={};pilot=[m for m in all_metrics if m['pilot']];lookup={(m['case_id'],m['policy']):m for m in pilot}
    pilot_ids=[c['case_id'] for c in protocol['cases'] if c['pilot']]
    complete=all((c,p) in lookup for c in pilot_ids for p in NUMERIC+LLM)
    if complete:
        means={p:float(np.mean([lookup[c,p]['AUROC'] for c in pilot_ids])) for p in NUMERIC+LLM}
        for p in ['single','multi_reason_hidden']+NUMERIC:
            delta=[lookup[c,'multi']['AUROC']-lookup[c,p]['AUROC'] for c in pilot_ids]
            comparisons[p]={'mean_gain':float(np.mean(delta)),'paired_95CI':bootstrap_delta(delta)}
        rates={p:sum(lookup[c,p]['accepted_model_outputs'] for c in pilot_ids)/40 for p in LLM}
        rate=sum(lookup[c,p]['accepted_model_outputs'] for c in pilot_ids for p in LLM)/120
        expand=means['multi']-means['single']>.005 and means['multi']>=max(means[p] for p in NUMERIC)-.02 and rate>=.9
        decision={'pilot_complete':True,'expand':bool(expand),'mean_AUROC':means,'accepted_fraction':rate,'accepted_fraction_by_policy':rates,
            'comparisons_multi_minus':comparisons,'rule':'Frozen >0.005 versus single; no more than 0.02 below best numerical pilot policy; >=90% accepted decisions',
            'caveat':'Five-patient exploratory pilot. No multi-agent superiority or clinical efficacy established.'}
        write_json(output/'pilot_decision.json',decision)
    result={'verified':True,'completed_trajectories':len(all_metrics),'native_artifacts':artifacts,'attempts_in_completed':attempts,
      'all_reserved_attempts':len(list(output.glob('trajectories/*/*/*_usage.json'))),'accepted_in_completed':accepted,'pilot_complete':complete,
      'checks':['Same four real initial measurements','Same 4,4,5,5,6,6,7,7 feedback schedule','Reviewers executed after proposals',
                'Independent AUROC','Random control reproduces old exact scores','Eight attempts per completed LLM arm, no retries',
                'Native hashes/backward lineage','No peer rationale leak in hidden condition']}
    write_json(output/'audit.json',result);print(result)
    if complete:print(decision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();audit(a.project)
