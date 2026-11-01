"""Patient-isolated relative query-value training, no test-oracle adoption."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import joblib
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,check_groups,exclusive
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.environment import PhysicalAcquisition
from experiments.spes_physical.mapping import physical_input
from experiments.spes_agent_v14.audit_features import audit_table
from .tools import FEATURES,VIEWS,ALPHA,POLICIES,fit_models,predict,choose


def freeze(project,output):
    project,output=Path(project),Path(output); parent=read_json(project/'outputs/spes_agent_v14/protocol.json')
    if read_json(project/'outputs/spes_agent_v14/audit_full.json')['status']!='verified': raise ValueError('Audited physical counterfactual banks required')
    files=['experiments/spes_agent_v17/experiment.py','experiments/spes_agent_v17/tools.py','experiments/spes_agent_v14/tools.py',
           'experiments/spes_agent_v14/audit_features.py','experiments/spes_agent_v10/features.py','experiments/spes_physical/environment.py',
           'experiments/spes_physical/mapping.py','experiments/spes_agent_v4/neural.py','experiments/spes_agent_v5/neural.py','vendor/norris2026_hat/src/models.py']
    p={'study':'v17 within-state pairwise physical query ranking','scope':'Numerical-tool feasibility after v16; no new LLM experiment or clinical/collaboration claim.',
       'jobs':parent['jobs'],'cases':parent['cases'],'features':FEATURES,'views':VIEWS,'policies':POLICIES,'alpha':ALPHA,
       'primary':'pair_experts minus frozen v11 score_focus, mean patient SOZ AUROC at eight physical groups',
       'secondary':'Pairwise all-feature and pointwise all-feature controls, allpolicies versusallv11controls; no posthocprimaryswap.',
       'features_expansion':'Scale44observed-only descriptors on fitting banks. For each view retain itscolumns and all interactions of itsstate columns0..9 with itscandidate columns10..43.',
       'target':'For the same source-held-out patient/prefix, gain_i-gain_j equals after_AUROC_i-after_AUROC_j; the common before_AUROC cancels.',
       'loss_weighting':'All15unorderedpairs from sixcandidates; weight0.4each, so eachstate totals the same6weight as the pointwisecontrol. Everypatient hasfourstates and equal24totalweight.',
       'learners':'Four fixedalpha100 Ridge pairrankers, nointercept, geometry/neural/waveform/allfeatures; one fixedalpha100intercept point-regressor on identical expandedallfeatures. Threeexpert percentile ranks averaged with fixedequalweight.',
       'data':'Only audited v14 physical source banks. No v16test-oracle labels or choices, no oldcolumnbanks/learnedweights. Outer7absentfromsource neuraldevelopmentandqueryrankerfitting; source targetpatientabsentfromitsHAT fitting/earlystop.',
       'runtime':'Same commonfourrandom0physicalgroups plusfouradaptivegroups; originalunchangedouterHAT finalranking. Allthreecontrollerscomputeallfiveutilities fromsame44features.',
       'fast_feature_recipe':'Pin v14 independent cachedaudit featureformula, alreadyverifiedagainstoriginalobserved-onlyruntimeat1e-12; no v14 frozenruntime edit.',
       'next_tool_gate':'Primarygain>0.01 and positive two-sided95 patient-bootstrap lowerbounds versusbothscore_focusandpair_monolithic. NumericalgainstilldoesnotproveLLMcommunicationvalue.',
       'new_small_fits':25,'new_neural_fits':0,'new_API_calls':0,'measurement_budget':8,
       'source_sha256':{f:sha(project/f) for f in files},'parent_v14_protocol_sha256':sha(project/'outputs/spes_agent_v14/protocol.json'),
       'source_bank_sha256':{f'outer{j["outer"]}/{s}':sha(project/f'outputs/spes_agent_v14/banks/outer{j["outer"]}/{s}.npz') for j in parent['jobs'] for s in j['predict']},
       'source_HAT_sha256':parent['source_HAT_sha256'],'mapping_receipt_sha256':parent['mapping_receipt_sha256'],
       'model_sha256':parent['model_sha256'],'v11_results_sha256':sha(project/'outputs/spes_agent_v11/numeric_results.json'),
       'limits':['Pair differences are correlated algebraic combinations, not new patients.','Three views have an inductive bias and fewer total coefficients than the all-feature model; this is not an exact parameter-matched architecture proof.',
                 'Random-prefix source training and17-versus21patientneuralfits can create runtime shift.','Repeated cohort exploration; nominalintervals notadjustedforallresearchtrials.','Groups do not equatepulse/current/charge/time; SOZ nottreatment.']}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=p: raise ValueError('Frozen pair-ranking protocol differs')
    write_json(path,p); print('V17_PROTOCOL_FROZEN',sha(path),flush=True)


def verified(project,output):
    project,output=Path(project),Path(output); p=read_json(output/'protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()): raise ValueError('Frozen pair-ranking source differs')
    return p


def fitting_data(project,output,outer):
    project,output=Path(project),Path(output); p=verified(project,output); chunks=[]; gains=[]; groups=[]; subjects=[]; hashes={}
    jobs=[j for j in p['jobs'] if j['outer']==outer]
    for j in jobs:
        check_groups(j['train'],j['early_stop'],j['predict'],j['outer_test'])
        key=f'outer{outer}_inner{j["inner"]}'; source=project/'outputs/spes_agent_v8/crossfit'/key; source_receipt=read_json(source/'receipt.json')
        if sha(source/'checkpoint.pth')!=p['source_HAT_sha256'][key] or any(source_receipt[k]!=j[k] for k in ['train','early_stop','predict','outer_test']): raise ValueError('Source HAT split/hash differs')
        for subject in j['predict']:
            if subject in j['train']+j['early_stop']+j['outer_test']: raise ValueError('Source target patient not held out')
            path=project/f'outputs/spes_agent_v14/banks/outer{outer}/{subject}.npz'; receipt=read_json(path.with_suffix('.json'))
            if sha(path)!=p['source_bank_sha256'][f'outer{outer}/{subject}'] or receipt['file_sha256']!=sha(path) or receipt['source_action_kind']!='verified_anatomical_pairs' or receipt['source_HAT_sha256']!=p['source_HAT_sha256'][key]: raise ValueError('Wrong physical training bank')
            with np.load(path) as d:
                if len(d['gain'])!=24: raise ValueError('Incomplete four-state source bank')
                chunks.append(d['features']); gains.append(d['gain']); groups.extend([tuple([int(subject[-2:])]+list(map(int,prefix))) for prefix in d['prefix']])
            subjects.append(subject); hashes[subject]=sha(path)
    if len(subjects)!=28 or len(set(subjects))!=28 or set(subjects)&set(jobs[0]['outer_test']): raise ValueError('Pair-ranker patient isolation differs')
    x,y=np.vstack(chunks),np.concatenate(gains)
    if x.shape!=(672,44) or y.shape!=(672,): raise ValueError('Unexpected source training dimensions')
    return x,y,groups,subjects,hashes,jobs[0]['outer_test']


def fit(project,output):
    project,output=Path(project),Path(output); verified(project,output)
    for outer in range(5):
        path=output/f'learners/outer{outer}/models.joblib'
        if path.with_suffix('.json').exists(): continue
        x,y,groups,subjects,hashes,test=fitting_data(project,output,outer); model=fit_models(x,y,groups)
        path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,path)
        write_json(path.with_suffix('.json'),{'outer':outer,'fit_patients':subjects,'outer_test':test,'candidate_rows':672,'source_states':112,'pair_rows':1680,
            'sum_pair_weights':672.,'dimensions':{name:len(columns)+sum(k<10 for k in columns)*sum(k>=10 for k in columns) for name,columns in VIEWS.items()},
            'source_bank_sha256':hashes,'models_sha256':sha(path),'protocol_sha256':sha(output/'protocol.json')})
        print('V17_PAIR_MODELS_FIT',outer,flush=True)
    frozen={'protocol_sha256':sha(output/'protocol.json'),'models_sha256':{str(i):sha(output/f'learners/outer{i}/models.joblib') for i in range(5)}}; path=output/'evaluation_protocol.json'
    if path.exists() and read_json(path)!=frozen: raise ValueError('Frozen pair-ranker set differs')
    write_json(path,frozen)


def run_case(project,output,c,policy):
    import torch
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2); folder=output/'evaluation'/c['case_id']/policy
    if (folder/'metrics.json').exists(): return
    path=output/f'learners/outer{c["outer"]}/models.joblib'; receipt=read_json(path.with_suffix('.json')); frozen=read_json(output/'evaluation_protocol.json')
    if sha(path)!=frozen['models_sha256'][str(c['outer'])] or frozen['protocol_sha256']!=sha(output/'protocol.json') or receipt['protocol_sha256']!=sha(output/'protocol.json') or c['subject'] in receipt['fit_patients'] or c['subject'] not in receipt['outer_test']: raise ValueError('Pair-ranker model/provenance differs')
    if sha(project/'outputs/spes_physical/maps'/c['subject']/'receipt.json')!=p['mapping_receipt_sha256'][c['subject']]: raise ValueError('Physical map changed')
    for family in ['baseline','mte','hat']:
        if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{c["outer"]}/{family}/checkpoint.pth')!=p['model_sha256'][f'{c["outer"]}/{family}']: raise ValueError('Outer scorer changed')
    env=PhysicalAcquisition(project,c['outer'],c['subject']); raw=load_array(project/f'vendor/norris2026_hat/data/X_{c["subject"]}.npz')[:,:20]; distance=np.where(env.index>=0,env.distance,0.)
    model=joblib.load(path); prefix=fixed_queries(env.available,c['outer'],int(c['subject'][-2:]),0)[:4]
    for q in prefix: write_json(folder/f'{len(env.selected):02d}_observation.json',env.observe(q))
    for k in [4,5,6,7]:
        remaining=[q for q in env.available if q not in env.selected]; ids=[env.pairs[q]['id'] for q in remaining]
        x=audit_table(distance,physical_input(raw,env.index,env.selected),env.predictions['hat'],env.selected,remaining); values=predict(model,x); choice=choose(ids,values[policy])
        write_json(folder/f'{k:02d}_choice.json',{'ids':ids,'features':x.tolist(),'values':{name:v.tolist() for name,v in values.items()},'choice':choice,
            'selected':[env.pairs[q]['id'] for q in env.selected],'current_HAT_score':env.predictions['hat'].tolist(),'policy':policy,'no_labels_or_future_responses':True})
        write_json(folder/f'{k:02d}_observation.json',env.observe(int(choice[1:])))
    result={**env.final_judge(),'case_id':c['case_id'],'policy':policy,'physical_measurements':8,'new_API_calls':0,'evaluation_protocol_sha256':sha(output/'evaluation_protocol.json')}
    write_json(folder/'metrics.json',result); print('V17_PHYSICAL_RANKING',c['case_id'],policy,round(result['AUROC'],6),flush=True)


def evaluate(project,output,workers):
    project,output=Path(project),Path(output); p=verified(project,output)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_case,project,output,c,name) for c in p['cases'] for name in p['policies']]
        for f in as_completed(futures): f.result()
    ids=[c['case_id'] for c in p['cases']]; values={name:np.array([read_json(output/'evaluation'/c/name/'metrics.json')['AUROC'] for c in ids]) for name in POLICIES}
    old=read_json(project/'outputs/spes_agent_v11/numeric_results.json')
    if sha(project/'outputs/spes_agent_v11/numeric_results.json')!=p['v11_results_sha256']: raise ValueError('Numerical control results differ')
    controls={name:np.array([next(r['AUROC'] for r in old['rows'] if r['case_id']==c and r['policy']==name) for c in ids]) for name in old['policies']}
    controls['pair_monolithic']=values['pair_monolithic']; controls['point_monolithic']=values['point_monolithic']; summary={}
    for name,score in values.items():
        summary[name]={'mean_patient_AUROC':float(score.mean()),'comparisons':{}}
        for control,base in controls.items():
            delta=score-base; summary[name]['comparisons'][control]={'gain':float(delta.mean()),'paired_95CI':bootstrap_delta(delta)}
    gate=all(summary['pair_experts']['comparisons'][name]['gain']>.01 and summary['pair_experts']['comparisons'][name]['paired_95CI'][0]>0 for name in ['score_focus','pair_monolithic'])
    r={'primary':p['primary'],'summary':summary,'next_tool_gate':gate,'patients':35,'physical_trajectories':105,'new_API_calls':0,'multi_agent_superiority_established':False}
    write_json(output/'results.json',r); print('V17_RESULTS',r,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['freeze','fit','evaluate'],required=True); parser.add_argument('--workers',type=int,default=4); a=parser.parse_args()
    if not 1<=a.workers<=4: raise ValueError('One to four local GPU workers')
    if a.stage=='freeze': freeze(a.project,a.output)
    else:
        with exclusive(a.output/f'{a.stage}.lock'):
            if a.stage=='fit': fit(a.project,a.output)
            else: evaluate(a.project,a.output,a.workers)
