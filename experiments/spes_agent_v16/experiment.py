"""Native ScienceClaw pilot: independent proposals, delayed exchange, real pairs."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.environment import PhysicalAcquisition
from .public import recording_catalog,public_state,check_public
from .logic import POLICIES,prompts,merge,baseline_ranking
from .provider import RankedProvider,ProviderUnavailable,parse_json,response_format,validate


def freeze(project,output):
    project,output=Path(project),Path(output); parent=read_json(project/'outputs/spes_agent_v11/numeric_protocol.json'); numeric=read_json(project/'outputs/spes_agent_v11/numeric_results.json')
    if not numeric['numerical_query_sensitivity'] or read_json(project/'outputs/spes_agent_v11/audit.json')['status']!='verified': raise ValueError('Audited physical query sensitivity required')
    cases=[{**c,'pilot':c['subject'] in parent['future_agent_pilot']} for c in parent['cases']]
    if sum(c['pilot'] for c in cases)!=5: raise ValueError('Five previously fixed pilot patients required')
    ranges=[]; distinct=[]; catalogs={}; ambiguous=0
    for c in cases:
        path=project/'outputs/spes_physical/maps'/c['subject']/'receipt.json'
        if sha(path)!=parent['mapping_receipt_sha256'][c['subject']]: raise ValueError('Physical mapping changed')
        catalog=recording_catalog(project,c['subject'],read_json(path)); dest=output/'catalog'/f'{c["case_id"]}.json'
        if dest.exists() and read_json(dest)!=catalog: raise ValueError('Frozen recording catalog changed')
        write_json(dest,catalog); catalogs[c['case_id']]=sha(dest)
        ambiguous+=sum(row.get('electrode_identity_status')=='coordinate_ambiguous' for row in catalog['contacts'])
        if c['pilot']:
            rr=[read_json(project/'outputs/spes_agent_v11/numeric'/c['case_id']/name/'metrics.json') for name in parent['policies']]
            ranges.append(max(r['AUROC'] for r in rr)-min(r['AUROC'] for r in rr)); distinct.append(len({tuple(np.argsort(r['prediction'])) for r in rr}))
    numeric_gate={'mean_pilot_AUROC_range':float(np.mean(ranges)),'distinct_pilot_rankings':distinct,'passed':float(np.mean(ranges))>.03 and all(v>1 for v in distinct)}
    files=['experiments/spes_agent_v16/'+f for f in ['experiment.py','public.py','logic.py','provider.py']]
    files+=['skills/neurotarget-spes-v16/SKILL.md','skills/neurotarget-spes-v16/scripts/observe.py','neurotarget/scienceclaw_runtime.py',
            'experiments/spes_agent_v2/providers.py','experiments/spes_physical/environment.py','experiments/spes_physical/mapping.py',
            'experiments/spes_agent_v10/features.py','experiments/spes_agent_v4/neural.py','experiments/spes_agent_v5/neural.py','config/upstream-lock.json']
    units=project/'data/ds004080/events.json'; declared=read_json(units)
    expected={'electrical_stimulation_current':'electrical stimulation current in Ampere','electrical_stimulation_frequency':'electrical stimulation frequency in Hertz','electrical_stimulation_pulsewidth':'electrical stimulation pulse width in s'}
    if any(declared[k]['description']!=v for k,v in expected.items()): raise ValueError('Recorded units not verified')
    protocol={'study':'v16 independent physical proposals with cross-round rationale exchange','scope':'First physically valid LLM pilot; repeatedly inspected cohort, not independent confirmation.',
       'cases':cases,'policies':POLICIES,'numeric_gate':numeric_gate,'model':'gemini-4-argon','reasoning':'medium','max_completion_tokens':32768,'retries':0,
       'budgets':{'physical_response_groups':8,'common_initial_groups':4,'adaptive_rounds':4,'LLM_attempts_per_trajectory':8,'native_public_queries':8,'pilot_trajectories':15,'pilot_attempt_cap':120,'full_cohort_attempt_cap':840},
       'fairness':'Every LLM arm receives the same tools/evidence, two independent current-round submissions, four actual feedback updates and same request/token caps. Numeric controls use same eight groups with zero model API calls; cost is reported separately.',
       'roles':'Single: two passes by one integrating generalist. Multi: persistent anatomy/network and response/error roles. Both current prompts saved before either API call; no current-peer answer. Past rounds exchanged after both submit.',
       'communication_ablation':'multi_reason_hidden removes peer reason/counterargument only; own text, peer rankings, executed choices and measured feedback remain.',
       'merger':'3/2/1 rank votes; ties by original score-focus geometric gain then physicalID. Same fixed rule allarms. No reviewer override.',
       'primary':'Mean patient SOZ AUROC at eight physical groups: multi minus matched single, plus multi minus frozen score_focus; report all controls and rationale ablation.',
       'scorer':'Unchanged original own outer HAT; failed v12-v15 trained predictors are not adopted.',
       'training':'No new model fine-tuning. Communication uses actual observed feedback only, never accuracy/SOZ feedback on test patients.',
       'smoke':'First pilot single eight-call trajectory is inside120cap. Any unaccepted smoke response stops further smoke calls; no replay/replacement. Full pilot requires8/8accepted smoke.',
       'pilot_expansion_rule':'Multi-single mean>0.01, multi-score_focus mean>=0, >=95%valid decisions acrosspilot, and multi/hidden different finalrankings in>=2/5patients. Pilot not confirmatory. Expand only after audit, unchanged protocol, to remaining30.',
       'new_neural_fits':0,'new_small_fits':0,'source_sha256':{f:sha(project/f) for f in files},'catalog_sha256':catalogs,
       'mapping_receipt_sha256':parent['mapping_receipt_sha256'],'model_sha256':parent['model_sha256'],'event_units_sha256':sha(units),'recording_name_ambiguities':ambiguous,
       'v11_results_sha256':sha(project/'outputs/spes_agent_v11/numeric_results.json'),
       'limits':['Response groups are not pulses or equal charge/current/time.','Released availability is retrospective; no safety, treatment or prospective simulation claim.',
                 'Nominal bootstrap intervals do not adjust for the full adaptive research search.','Anatomy/morphology are common tools; no disease-domain tuning is asserted.']}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=protocol: raise ValueError('Frozen independent-proposal protocol differs')
    write_json(path,protocol); print('V16_FROZEN',sha(path),numeric_gate,flush=True)


def verified(project,output):
    project,output=Path(project),Path(output); p=read_json(output/'protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()) or sha(project/'data/ds004080/events.json')!=p['event_units_sha256']: raise ValueError('Frozen source/units changed')
    return p


def execute_trajectory(project,output,case,policy,env,framework,prefix,catalog,raw,strict_smoke=False,client_factory=RankedProvider,state_builder=public_state):
    project,output=Path(project),Path(output); folder=output/'trajectories'/case['case_id']/policy; folder.mkdir(parents=True,exist_ok=True)
    ids=read_json(folder/'ids.json') if (folder/'ids.json').exists() else {}
    def artifact(key,agent,payload,kind,parents):
        if key not in ids:
            ids[key]=framework.artifact(agent,'neurotarget-spes-v16',payload,kind,[ids[q] for q in parents]).artifact_id; write_json(folder/'ids.json',ids)
    for i,q in enumerate(prefix):
        obs=env.observe(q); write_json(folder/f'warmup_{i}.json',obs); artifact(f'warmup_{i}','warmup',obs,'physical_response_group',[] if i==0 else [f'warmup_{i-1}'])
    history=[]; client=None; previous=None; accepted=0
    for r in range(4):
        public_path=folder/f'{r:02d}_public.json'
        if public_path.exists(): public=check_public(read_json(public_path))
        else: public=state_builder(env,catalog,raw,previous); write_json(public_path,public)
        expected_selected=[env.pairs[q]['id'] for q in env.selected]
        if public['evidence']['observed']['selected']!=expected_selected: raise ValueError('Public state replay history differs')
        batch=prompts(public,policy,r,history)
        # Persist BOTH independent prompts before any response becomes available.
        for slot,prompt in enumerate(batch):
            path=folder/f'{r:02d}_{slot}_request.json'
            if path.exists() and read_json(path)!=prompt: raise ValueError('Saved independent request differs')
            if not path.exists(): write_json(path,prompt)
        proposals=[]
        for slot,prompt in enumerate(batch):
            tool_path=folder/f'{r:02d}_{slot}_tool.json'
            if tool_path.exists(): tool=read_json(tool_path)
            else:
                metadata=framework.registry.get_skill('neurotarget-spes-v16')
                if not metadata: raise ValueError('Native physical observation skill missing')
                result=framework.executor.execute_skill('neurotarget-spes-v16',metadata,{'project':str(project),'state_path':str(public_path),'format':'json'},timeout=90)
                if result.get('status')!='success' or result.get('result')!=public: raise RuntimeError('Native common public tool failed')
                tool=result['result']; write_json(tool_path,tool)
            artifact(f'tool_{r}_{slot}','tools',tool,'common_observed_evidence',['warmup_3']+([f'feedback_{r-1}'] if r else []))
            usage=folder/f'{r:02d}_{slot}_usage.json'; decision=folder/f'{r:02d}_{slot}_decision.json'
            if not usage.exists():
                if client is None: client=client_factory(project)
                try: client.ask(prompt,usage)
                except ProviderUnavailable: pass
            saved=read_json(usage)
            try:
                encoded=json.dumps(prompt,separators=(',',':')).encode()
                if saved.get('prompt_sha256')!=hashlib.sha256(encoded).hexdigest() or not saved.get('captured_response') or saved.get('response_model')!='gemini-4-argon' or saved.get('finish_reason')!='stop' or saved.get('response_format')!=response_format(public): raise ValueError('Uncaptured/incomplete/unconstrained response')
                proposal=validate(parse_json(saved.get('response_content')),public)
            except (ValueError,TypeError,KeyError):
                proposal={'ranked_queries':baseline_ranking(public),'evidence_ids':['geometry'],'reason':'Spent attempt had no accepted proposal; use declared score-focus ranking.',
                          'counterargument':'No replacement API request.','accepted_output':False,'origin':'spent_default'}
            proposal={**proposal,'round':r+1,'slot':slot,'role':prompt['role']}; write_json(decision,proposal)
            artifact(f'proposal_{r}_{slot}',prompt['role'],proposal,'independent_ranking',[f'tool_{r}_{slot}']); proposals.append(proposal); accepted+=int(proposal['accepted_output'])
            if strict_smoke and not proposal['accepted_output']:
                write_json(output/'smoke_interruption.json',{'case_id':case['case_id'],'policy':policy,'round':r+1,'slot':slot,'reason':'Spent smoke response not accepted; no further smoke or formal calls permitted under this interface.','attempts_so_far':r*2+slot+1})
                raise RuntimeError('Smoke interface failed; preserve spent attempt and stop')
        joint=merge(proposals,public); write_json(folder/f'{r:02d}_merge.json',joint)
        artifact(f'merge_{r}','merger',joint,'deterministic_joint_choice',[f'proposal_{r}_0',f'proposal_{r}_1'])
        previous=env.predictions['hat'].copy(); obs=env.observe(int(joint['choice'][1:])); write_json(folder/f'{r:02d}_feedback.json',obs)
        artifact(f'feedback_{r}','observations',obs,'actual_physical_feedback',[f'merge_{r}'])
        history.append({'round':r+1,'proposals':proposals,'executed':joint['choice'],'feedback':obs})
    metric={**env.final_judge(),'case_id':case['case_id'],'policy':policy,'pilot':case['pilot'],'LLM_attempts':8,'accepted_model_outputs':accepted,
            'native_public_queries':8,'physical_response_groups':8,'adaptive_groups':4,'native_artifacts':len(ids)}
    write_json(folder/'metrics.json',metric); print('V16_PHYSICAL_AGENT',case['case_id'],policy,round(metric['AUROC'],6),accepted,flush=True); return metric


def run_case(project,output,case,policy,strict_smoke=False):
    import torch
    from neurotarget.scienceclaw_runtime import Framework
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2)
    if policy not in p['policies'] or case not in p['cases']: raise ValueError('Case/arm outside frozen attempt catalog')
    if not case['pilot']:
        if not read_json(output/'pilot_results.json')['pilot_expansion_eligible'] or read_json(output/'audit_pilot.json')['status']!='verified': raise ValueError('Non-pilot attempts require audited expansion gate')
    folder=output/'trajectories'/case['case_id']/policy
    if (folder/'metrics.json').exists(): return read_json(folder/'metrics.json')
    if sha(project/'outputs/spes_physical/maps'/case['subject']/'receipt.json')!=p['mapping_receipt_sha256'][case['subject']]: raise ValueError('Physical map changed')
    for f in ['baseline','mte','hat']:
        if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{case["outer"]}/{f}/checkpoint.pth')!=p['model_sha256'][f'{case["outer"]}/{f}']: raise ValueError('Scorer changed')
    catalog_path=output/'catalog'/f'{case["case_id"]}.json'
    if sha(catalog_path)!=p['catalog_sha256'][case['case_id']]: raise ValueError('Frozen recording catalog changed')
    catalog=read_json(catalog_path)
    if any(sha(project/f)!=h for f,h in catalog['source_sha256'].items()): raise ValueError('Anatomical source changed')
    env=PhysicalAcquisition(project,case['outer'],case['subject']); raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20]
    framework=Framework(project,folder/'scienceclaw'); prefix=fixed_queries(env.available,case['outer'],int(case['subject'][-2:]),0)[:4]
    metric=execute_trajectory(project,output,case,policy,env,framework,prefix,catalog,raw,strict_smoke=strict_smoke)
    metric['protocol_sha256']=sha(output/'protocol.json'); write_json(folder/'metrics.json',metric); return metric


def summarize(project,output,pilot=True):
    project,output=Path(project),Path(output); p=verified(project,output); cases=[c for c in p['cases'] if not pilot or c['pilot']]; ids=[c['case_id'] for c in cases]
    rows=[read_json(output/'trajectories'/c/name/'metrics.json') for c in ids for name in p['policies']]
    values={name:np.array([next(r['AUROC'] for r in rows if r['policy']==name and r['case_id']==c) for c in ids]) for name in p['policies']}
    parent=read_json(project/'outputs/spes_agent_v11/numeric_results.json')
    if sha(project/'outputs/spes_agent_v11/numeric_results.json')!=p['v11_results_sha256']: raise ValueError('Numeric controls changed')
    controls={name:np.array([next(r['AUROC'] for r in parent['rows'] if r['policy']==name and r['case_id']==c) for c in ids]) for name in parent['policies']}
    summary={name:{'mean_patient_AUROC':float(v.mean())} for name,v in {**controls,**values}.items()}
    comparisons={}
    for name in ['single','multi_reason_hidden','random_0','random_1','coverage','score_focus','uncertainty']:
        delta=values['multi']-(values[name] if name in values else controls[name]); comparisons[name]={'multi_gain':float(delta.mean()),'paired_95CI':bootstrap_delta(delta)}
    accepted=sum(r['accepted_model_outputs'] for r in rows); diverse=0
    for c in ids:
        a=next(r['prediction'] for r in rows if r['case_id']==c and r['policy']=='multi'); b=next(r['prediction'] for r in rows if r['case_id']==c and r['policy']=='multi_reason_hidden')
        diverse+=not np.array_equal(np.argsort(a),np.argsort(b))
    expand=comparisons['single']['multi_gain']>.01 and comparisons['score_focus']['multi_gain']>=0 and accepted/(len(rows)*8)>=.95 and diverse>=2
    result={'scope':'five-patient exploratory pilot' if pilot else 'full previously inspected cohort','summary':summary,'comparisons':comparisons,
            'valid_model_decisions':accepted,'actual_attempts':len(rows)*8,'physical_trajectories':len(rows),'multi_hidden_different_rankings':diverse,
            'pilot_expansion_eligible':bool(expand) if pilot else None,'multi_agent_superiority_established':False,
            'rows':[{k:v for k,v in r.items() if k!='prediction'} for r in rows]}
    write_json(output/('pilot_results.json' if pilot else 'full_results.json'),result); print('V16_RESULT',{k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['freeze','smoke','pilot','expand'],required=True); parser.add_argument('--workers',type=int,default=6); a=parser.parse_args()
    if a.stage=='freeze': freeze(a.project,a.output)
    else:
        p=verified(a.project,a.output); cases=[c for c in p['cases'] if c['pilot']]
        if not p['numeric_gate']['passed']: raise ValueError('Physical query-sensitivity gate failed')
        if (a.output/'smoke_interruption.json').exists(): raise ValueError('Smoke is interrupted; never silently retry')
        with exclusive(a.output/f'{a.stage}.lock'):
            if a.stage=='smoke': run_case(a.project,a.output,cases[0],'single',strict_smoke=True)
            else:
                smoke=a.output/'trajectories'/cases[0]['case_id']/'single/metrics.json'
                if not smoke.exists() or read_json(smoke)['accepted_model_outputs']!=8: raise ValueError('Eight accepted smoke calls required')
                if a.stage=='expand':
                    if not read_json(a.output/'pilot_results.json')['pilot_expansion_eligible'] or read_json(a.output/'audit_pilot.json')['status']!='verified': raise ValueError('Audited pilot gate required')
                    cases=[c for c in p['cases'] if not c['pilot']]
                if not 1<=a.workers<=8: raise ValueError('One to eight local request workers')
                with ProcessPoolExecutor(max_workers=a.workers) as pool:
                    futures=[pool.submit(run_case,a.project,a.output,c,name) for c in cases for name in POLICIES]
                    for f in as_completed(futures): f.result()
                summarize(a.project,a.output,pilot=a.stage=='pilot')
