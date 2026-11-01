"""Frozen posthoc diagnostics, not new executable agent policies."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v4.neural import load_array,upstream_models
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.mapping import physical_input

NUMERIC=['random_0','random_1','coverage','score_focus','uncertainty']
AGENTS=['single','multi','multi_reason_hidden']


def paired_summary(delta):
    delta=np.asarray(delta,dtype=float)
    if delta.ndim!=1 or not len(delta) or not np.isfinite(delta).all(): raise ValueError('Finite patient-level differences required')
    rng=np.random.default_rng(17671); draws=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
    return {'mean':float(delta.mean()),'paired_95CI':np.quantile(draws,[.025,.975]).tolist(),'patients':len(delta)}


def slate(selected,remaining,proposals,actual,outer,patient,policy_index):
    if len(selected)!=7 or len(set(selected))!=7 or actual in selected or actual not in remaining: raise ValueError('Seven actual prior groups required')
    pool=list(dict.fromkeys(q for proposal in proposals for q in proposal))
    if not pool or actual not in pool or not set(pool)<=set(remaining): raise ValueError('Executed choice must be inside the proposed eligible pool')
    random=[]
    for rep in range(5):
        rng=np.random.default_rng(173000+outer*10000+patient*100+policy_index*10+rep)
        random.append(list(map(int,rng.choice(remaining,len(pool),replace=False))))
    return {'actual_prior_seven':list(selected),'proposal_pool':pool,'actual_final_query':int(actual),'matched_random_pools':random}


def pool_metrics(auc_by_candidate,entry,score_focus):
    actual=auc_by_candidate[entry['actual_final_query']]
    oracle=max(auc_by_candidate[q] for q in entry['proposal_pool'])
    random=float(np.mean([max(auc_by_candidate[q] for q in pool) for pool in entry['matched_random_pools']]))
    return {'actual_AUROC8':float(actual),'proposal_oracle_AUROC8':float(oracle),'matched_random_oracle_mean_AUROC8':random,
            'score_focus_same_prefix_AUROC8':float(auc_by_candidate[score_focus]),'proposal_pool_size':len(entry['proposal_pool']),
            'oracle_minus_actual':float(oracle-actual),'oracle_minus_matched_random_oracle':float(oracle-random),
            'score_focus_minus_actual':float(auc_by_candidate[score_focus]-actual)}


def freeze(project,output):
    project,output=Path(project),Path(output); v11=read_json(project/'outputs/spes_agent_v11/numeric_protocol.json'); v16=read_json(project/'outputs/spes_agent_v16/protocol.json')
    if read_json(project/'outputs/spes_agent_v16/audit_pilot.json')['status']!='verified': raise ValueError('Completed audited physical agent pilot required')
    sources=['experiments/spes_agent_v16_supplement/analysis.py','experiments/spes_physical/mapping.py','experiments/spes_agent_v4/neural.py','vendor/norris2026_hat/src/models.py']
    inputs={}
    for c in v11['cases']:
        path=f'outputs/spes_agent_v12/evaluation/{c["case_id"]}/predictions.npz'; inputs[path]=sha(project/path)
        for name in NUMERIC:
            path=f'outputs/spes_agent_v11/numeric/{c["case_id"]}/{name}/metrics.json'; inputs[path]=sha(project/path)
    pilot=[c for c in v16['cases'] if c['pilot']]
    for c in pilot:
        for name in AGENTS:
            for file in ['metrics.json','03_public.json','03_0_decision.json','03_1_decision.json','03_merge.json']:
                path=f'outputs/spes_agent_v16/trajectories/{c["case_id"]}/{name}/{file}'; inputs[path]=sha(project/path)
    p={'study':'Posthoc physical budget and final-step candidate-pool diagnostics','scope':'Presentation and bottleneck analysis after all v16 outcomes; not prespecified agent improvement or new policy evidence.',
       'cases':v11['cases'],'pilot_cases':pilot,'numeric':NUMERIC,'agents':AGENTS,'source_sha256':{f:sha(project/f) for f in sources},'input_sha256':inputs,
       'mapping_receipt_sha256':v16['mapping_receipt_sha256'],'HAT_sha256':{str(i):v16['model_sha256'][f'{i}/hat'] for i in range(5)},
       'HAT_receipt_sha256':{str(i):sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{i}/hat/receipt.json') for i in range(5)},
       'budget_analysis':'Common four random0 physical groups versus actual eight; full35 numerical and five-patient pilot separately. Patient-paired bootstrap10000/seed17671.',
       'candidate_analysis':'At each actual seven-group final state only, evaluate the two proposals union, actual executed query, score-focus leader, and five seeded uniformly sampled pools of the same size.',
       'random_slates':'Seed173000+outer*10000+patient*100+policy_index*10+rep; all slates fixed before new candidate inference or label evaluation.',
       'oracle_definition':'Highest true-label AUROC8 within each pool conditional on unchanged actual prior7 groups. Random reference averages five within-pool maxima, not a max over all five pools.',
       'limits':['True test labels are used only by this posthoc judge, never sent to or returned as feedback to an agent.',
                 'Oracle uses unavailable labels; it is not executable agent accuracy or a full multi-step attainable bound.',
                 'Multiple random pools and policy arms are not independent new patients.','All original failed results and budgets remain unchanged.'],
       'new_API_calls':0,'new_training_fits':0,'parent_v16_protocol_sha256':sha(project/'outputs/spes_agent_v16/protocol.json'),
       'parent_v16_audit_sha256':sha(project/'outputs/spes_agent_v16/audit_pilot.json')}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=p: raise ValueError('Frozen diagnostic protocol differs')
    write_json(path,p); print('DIAGNOSTIC_PROTOCOL_FROZEN',sha(path),flush=True)


def verified(project,output):
    project,output=Path(project),Path(output); p=read_json(output/'protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()): raise ValueError('Frozen diagnostic source differs')
    return p


def budget(project,output):
    project,output=Path(project),Path(output); p=verified(project,output); rows=[]; pilot_ids={c['case_id'] for c in p['pilot_cases']}
    for c in p['cases']:
        reference=f'outputs/spes_agent_v12/evaluation/{c["case_id"]}/predictions.npz'
        if sha(project/reference)!=p['input_sha256'][reference]: raise ValueError('Four-group reference changed')
        with np.load(project/reference) as d:
            labels=d['label']; before=float(roc_auc_score(labels,d['4/hat']))
            np.testing.assert_array_equal(labels,load_array(project/f'vendor/norris2026_hat/data/y_{c["subject"]}.npz'))
        common=read_json(project/'outputs/spes_agent_v11/numeric'/c['case_id']/'random_0/metrics.json')['queries'][:4]
        for name in NUMERIC+(AGENTS if c['case_id'] in pilot_ids else []):
            path=(f'outputs/spes_agent_v11/numeric/{c["case_id"]}/{name}/metrics.json' if name in NUMERIC else f'outputs/spes_agent_v16/trajectories/{c["case_id"]}/{name}/metrics.json')
            if sha(project/path)!=p['input_sha256'][path]: raise ValueError('Completed trajectory changed')
            result=read_json(project/path)
            if len(result['queries'])!=8 or len(set(result['queries']))!=8 or result['queries'][:4]!=common: raise ValueError('Physical budget/common prefix changed')
            after=float(roc_auc_score(labels,result['prediction']))
            np.testing.assert_allclose(after,result['AUROC'],rtol=0,atol=1e-12)
            rows.append({'case_id':c['case_id'],'pilot':c['case_id'] in pilot_ids,'policy':name,'AUROC4':before,'AUROC8':after,'gain':after-before})
    summary={}
    for population,names in [('full35',NUMERIC),('pilot5',NUMERIC+AGENTS)]:
        summary[population]={}
        for name in names:
            rr=[r for r in rows if r['policy']==name and (population=='full35' or r['pilot'])]
            summary[population][name]={**paired_summary([r['gain'] for r in rr]),'mean_AUROC4':float(np.mean([r['AUROC4'] for r in rr])),
                'mean_AUROC8':float(np.mean([r['AUROC8'] for r in rr])),'patients_improved':sum(r['gain']>0 for r in rr),'patients_worse':sum(r['gain']<0 for r in rr)}
    write_json(output/'budget_results.json',{'scope':'Posthoc paired four-to-eight-group gain; cohort summaries remain separate.','summary':summary,'rows':rows,'new_API_calls':0})
    print('BUDGET_DIAGNOSTIC',summary,flush=True)


def infer(model,raw,index,selected,stats):
    import torch
    x=physical_input(raw,index,selected)
    for segment,stat in zip([slice(0,1),slice(1,129),slice(129,None)],stats):
        view=x[...,segment]; mask=np.abs(view)>1e-12; view[mask]=(view[mask]-stat['mean'])/stat['std']
    with torch.no_grad():
        score=np.concatenate([torch.sigmoid(model(torch.from_numpy(x[b:b+8]).cuda())[:,0]).cpu().numpy() for b in range(0,len(x),8)])
    if not np.isfinite(score).all(): raise ValueError('Nonfinite counterfactual score')
    return score


def oracle_case(project,output,case):
    import torch
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2); folder=output/'oracle'/case['case_id']; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists(): return
    mapping_path=project/'outputs/spes_physical/maps'/case['subject']/'receipt.json'; mapping=read_json(mapping_path)
    if sha(mapping_path)!=p['mapping_receipt_sha256'][case['subject']] or sha(mapping_path.with_name('index.npz'))!=mapping['index_sha256']: raise ValueError('Physical mapping changed')
    index=load_array(mapping_path.with_name('index.npz')); available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
    entries={}
    for pi,name in enumerate(AGENTS):
        origin=project/'outputs/spes_agent_v16/trajectories'/case['case_id']/name
        for file in ['metrics.json','03_public.json','03_0_decision.json','03_1_decision.json','03_merge.json']:
            rel=(origin/file).relative_to(project).as_posix()
            if sha(origin/file)!=p['input_sha256'][rel]: raise ValueError('Frozen final-state evidence changed')
        public=read_json(origin/'03_public.json'); metric=read_json(origin/'metrics.json'); selected=[int(q[1:]) for q in public['evidence']['observed']['selected']]
        remaining=[q for q in available if q not in selected]
        if remaining!=[int(row['id'][1:]) for row in public['evidence']['geometry']['candidates']]: raise ValueError('Eligible final candidates differ')
        proposals=[[int(q[1:]) for q in read_json(origin/f'03_{slot}_decision.json')['ranked_queries']] for slot in [0,1]]
        actual=int(read_json(origin/'03_merge.json')['choice'][1:]); entry=slate(selected,remaining,proposals,actual,case['outer'],int(case['subject'][-2:]),pi)
        leader=min(public['evidence']['geometry']['candidates'],key=lambda row:(-row['score_focus_gain'],row['id']))['id']
        entry['score_focus_query']=int(leader[1:]); entry['recorded_metric_AUROC8']=metric['AUROC']; entries[name]=entry
    slates_path=folder/'slates.json'; write_json(slates_path,{'protocol_sha256':sha(output/'protocol.json'),'entries':entries})
    # Slates are already fixed. No labels are needed to score any counterfactual.
    source=project/f'outputs/spes_agent_v4/multitrial/neural/outer{case["outer"]}/hat'; receipt=read_json(source/'receipt.json')
    if sha(source/'checkpoint.pth')!=p['HAT_sha256'][str(case['outer'])] or sha(source/'receipt.json')!=p['HAT_receipt_sha256'][str(case['outer'])] or case['subject'] not in receipt['predict'] or case['subject'] in receipt['train']+receipt['early_stop']: raise ValueError('HAT lineage/patient exclusion differs')
    raw_path=project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz'
    if sha(raw_path)!=mapping['raw_sha256']: raise ValueError('Raw source changed')
    raw=load_array(raw_path)[:,:20]; model,_,_=upstream_models(project).create_model('hat')
    model.load_state_dict(torch.load(source/'checkpoint.pth',map_location='cpu',weights_only=True)['model_state_dict']); model=model.cuda().eval(); cache={}
    for entry in entries.values():
        selected=entry['actual_prior_seven']; candidates=list(dict.fromkeys(entry['proposal_pool']+[q for pool in entry['matched_random_pools'] for q in pool]+[entry['score_focus_query']]))
        for q in candidates:
            key=tuple(selected+[q])
            if key not in cache: cache[key]=infer(model,raw,index,list(key),receipt['normalisation'])
    sequences=list(cache); predictions=np.vstack([cache[key] for key in sequences]); np.savez_compressed(folder/'predictions.npz',selected=np.array(sequences),prediction=predictions)
    # Only the posthoc evaluator now reads clinical labels; nothing is returned to agents.
    labels=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz'); rows=[]
    for name,entry in entries.items():
        selected=entry['actual_prior_seven']; values={key[-1]:float(roc_auc_score(labels,score)) for key,score in cache.items() if list(key[:7])==selected}
        actual_score=cache[tuple(selected+[entry['actual_final_query']])]; old=read_json(project/'outputs/spes_agent_v16/trajectories'/case['case_id']/name/'metrics.json')
        np.testing.assert_allclose(actual_score,old['prediction'],rtol=0,atol=1e-6)
        metrics=pool_metrics(values,entry,entry['score_focus_query']); np.testing.assert_allclose(metrics['actual_AUROC8'],old['AUROC'],rtol=0,atol=1e-12)
        rows.append({'case_id':case['case_id'],'policy':name,**metrics})
    write_json(folder/'metrics.json',{'rows':rows,'counterfactual_scorer_calls':len(cache),'slates_sha256':sha(slates_path),'predictions_sha256':sha(folder/'predictions.npz'),
                                    'protocol_sha256':sha(output/'protocol.json'),'labels_only_in_posthoc_judge':True})
    print('FINAL_CANDIDATE_DIAGNOSTIC',case['case_id'],len(cache),flush=True)


def oracle(project,output,workers):
    p=verified(project,output)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(oracle_case,project,output,c) for c in p['pilot_cases']]
        for f in as_completed(futures): f.result()
    summarize(output)


def summarize(output):
    output=Path(output); p=read_json(output/'protocol.json'); rows=[]; calls=0
    for c in p['pilot_cases']:
        r=read_json(output/'oracle'/c['case_id']/'metrics.json'); rows+=r['rows']; calls+=r['counterfactual_scorer_calls']
    summary={}
    for name in AGENTS:
        rr=[r for r in rows if r['policy']==name]
        summary[name]={'mean_actual_AUROC8':float(np.mean([r['actual_AUROC8'] for r in rr])),
            'mean_proposal_oracle_AUROC8':float(np.mean([r['proposal_oracle_AUROC8'] for r in rr])),
            'mean_matched_random_oracle_AUROC8':float(np.mean([r['matched_random_oracle_mean_AUROC8'] for r in rr])),
            'mean_proposal_pool_size':float(np.mean([r['proposal_pool_size'] for r in rr])),
            **{key:paired_summary([r[key] for r in rr]) for key in ['oracle_minus_actual','oracle_minus_matched_random_oracle','score_focus_minus_actual']}}
    result={'scope':'Posthoc final-query oracle conditional on each actual7-group prefix, not an executable policy or eight-step oracle.',
            'summary':summary,'rows':rows,'new_local_scorer_calls':calls,'new_API_calls':0,'multi_agent_superiority_established':False}
    write_json(output/'oracle_results.json',result); print('FINAL_CANDIDATE_SUMMARY',{k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['freeze','budget','oracle'],required=True); parser.add_argument('--workers',type=int,default=2); a=parser.parse_args()
    if not 1<=a.workers<=2: raise ValueError('One or two local GPU workers')
    if a.stage=='freeze': freeze(a.project,a.output)
    else:
        with exclusive(a.output/f'{a.stage}.lock'):
            if a.stage=='oracle': oracle(a.project,a.output,a.workers)
            else: budget(a.project,a.output)
