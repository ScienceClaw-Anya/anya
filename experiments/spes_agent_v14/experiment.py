"""Frozen physical query-value training, source-held-out targets and controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from experiments.spes_agent_v3.common import read_json,sha,check_groups,exclusive
from experiments.spes_agent_v4.neural import load_array,upstream_models
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v8.targets import score_subset
from experiments.spes_physical.mapping import physical_input
from experiments.spes_physical.environment import PhysicalAcquisition
from .tools import FEATURES,VIEWS,POLICIES,table,fit_models,predict,choose


def freeze(project,output):
    project,output=Path(project),Path(output); parent=read_json(project/'outputs/spes_agent_v12/protocol.json')
    sources=['experiments/spes_agent_v14/'+s for s in ['experiment.py','tools.py']]
    sources+=['experiments/spes_agent_v8/features.py','experiments/spes_agent_v8/targets.py','experiments/spes_agent_v10/features.py',
              'experiments/spes_agent_v13/tools.py','experiments/spes_physical/environment.py','experiments/spes_physical/mapping.py',
              'experiments/spes_agent_v4/neural.py','experiments/spes_agent_v5/neural.py']
    p={'study':'v14 physical counterfactual query-value experts','scope':'Numerical-tool and specialist feasibility, not evidence for LLM communication or clinical treatment.',
       'jobs':parent['jobs'],'cases':parent['cases'],'features':FEATURES,'views':VIEWS,'policies':POLICIES,
       'prefix_sizes':[4,5,6,7],'candidates_per_state':6,'primary':'expert_consensus minus frozen v11 score_focus, mean patient SOZ AUROC after eight physical groups',
       'secondary':'Monolithic and ridge value learners, all policies against five v11 controls; consensus versus monolithic. No posthoc primary swap.',
       'training_target':'One-query gain in source-held-out patient SOZ AUROC, from actual recorded physical-pair responses. Candidate features fixed before candidate response/labels are used.',
       'training_path':'Common random0 physical path. Six candidates include the next random0 query, maximum coverage, maximum raw-HAT rank-focus, and seeded random alternatives; no utility-based candidate filtering.',
       'experts':'Three Huber boosted regressors with fixed geometry, neural-state and observed-waveform feature views. All-feature boosted and ridge controls use same targets and patients. Equal patient weight.',
       'learner_settings':{'trees':150,'depth':2,'minimum_leaf':30,'learning_rate':.03,'loss':'huber','alpha':.9,'ridge_alpha':100.,'expert_disagreement_penalty':.5},
       'relative_score':'Unsupervised patient median/MAD logit transform for query features. Not a probability calibrator. No v13 learned threshold/parameters adopted.',
       'gate':'Primary gain>0.01 with positive two-sided95 patient-bootstrap lower bound versus score_focus and monolithic, and different expert top choices at>=25% of consensus steps. Required before a paid agent trial for this tool version.',
       'runtime':'Same initial four physical groups and four adaptive choices. Same unchanged outer HAT final ranking; no test SOZ or future signals available to policy.',
       'new_neural_fits':0,'new_small_fits':25,'new_API_calls':0,'target_rows_planned':3360,
       'data_reuse':'Only audited v12 source HAT scores for the identical physical random0 prefixes are reused. All alternative response targets are newly computed. No v8-v10 column bank, learner, response forecast or old posterior is adopted.',
       'limits':['Repeated cohort exploration; retrospective availability; SOZ proxy only.','Twenty source HATs fit17patients, final HAT fits21, so score/state distributions may shift.',
                 'Random-prefix training can differ from learned-policy runtime states.','Expert spread is a disagreement heuristic, not a calibrated confidence interval.'],
       'source_sha256':{f:sha(project/f) for f in sources},'mapping_receipt_sha256':parent['mapping_receipt_sha256'],
       'model_sha256':parent['model_sha256'],'source_HAT_sha256':parent['source_HAT_sha256'],
       'bank_sha256':{f'outer{j["outer"]}/{s}':sha(project/f'outputs/spes_agent_v12/banks/outer{j["outer"]}/{s}.npz') for j in parent['jobs'] for s in j['predict']},
       'v11_results_sha256':sha(project/'outputs/spes_agent_v11/numeric_results.json')}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=p: raise ValueError('Frozen query-value protocol differs')
    write_json(path,p); print('V14_PROTOCOL_FROZEN',sha(path),flush=True)


def verified(project,output):
    p=read_json(Path(output)/'protocol.json')
    if any(sha(Path(project)/f)!=h for f,h in p['source_sha256'].items()): raise ValueError('Frozen source differs')
    return p


def subject_banks(project,output,subject):
    import torch
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2)
    jobs=[j for j in p['jobs'] if subject in j['predict']]
    if all((output/f'banks/outer{j["outer"]}/{subject}.json').exists() for j in jobs): return
    folder=project/'outputs/spes_physical/maps'/subject; mapping=read_json(folder/'receipt.json'); index=load_array(folder/'index.npz')
    raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
    if sha(folder/'receipt.json')!=p['mapping_receipt_sha256'][subject] or sha(folder/'index.npz')!=mapping['index_sha256'] or sha(raw_path)!=mapping['raw_sha256']:
        raise ValueError('Physical mapping/source differs')
    raw=load_array(raw_path)[:,:20]; y=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
    coords=load_array(project/f'vendor/norris2026_hat/data/coords_{subject}.npz'); centers=np.array([r['center_mm'] for r in mapping['pair_rows']])
    distance=np.where(index>=0,cdist(coords,centers),0.); kernel=np.where(index>=0,np.exp(-distance/40),0.)
    available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
    mods=upstream_models(project)
    for j in jobs:
        path=output/f'banks/outer{j["outer"]}/{subject}.npz'
        if path.with_suffix('.json').exists(): continue
        check_groups(j['train'],j['early_stop'],j['predict'],j['outer_test'])
        if subject in j['train']+j['early_stop']+j['outer_test']: raise ValueError('Source patient not held out')
        key=f'outer{j["outer"]}_inner{j["inner"]}'; source=project/'outputs/spes_agent_v8/crossfit'/key; receipt=read_json(source/'receipt.json')
        if sha(source/'checkpoint.pth')!=p['source_HAT_sha256'][key] or any(receipt[k]!=j[k] for k in ['train','early_stop','predict','outer_test']):
            raise ValueError('Source model lineage differs')
        old=project/f'outputs/spes_agent_v12/banks/outer{j["outer"]}/{subject}.npz'
        if sha(old)!=p['bank_sha256'][f'outer{j["outer"]}/{subject}']: raise ValueError('Physical prefix scores differ')
        with np.load(old) as d: before={k:d['HAT_score'][d['state']==k] for k in [4,5,6,7,8]}
        model,_,_=mods.create_model('hat'); model.load_state_dict(torch.load(source/'checkpoint.pth',map_location='cpu',weights_only=True)['model_state_dict']); model=model.cuda().eval()
        sequence=fixed_queries(available,j['outer'],int(subject[-2:]),0); rows=[]
        for k in p['prefix_sizes']:
            selected=sequence[:k]; remaining=[q for q in available if q not in selected]; observed=physical_input(raw,index,selected)
            x=table(distance,observed,before[k],selected,remaining)
            current=kernel[:,selected].max(axis=1)
            from scipy.stats import rankdata
            rank=rankdata(before[k])/(len(y)+1)
            geometric=[np.maximum(kernel[:,q]-current,0.) for q in remaining]
            coverage=remaining[int(np.argmax([g.mean() for g in geometric]))]
            focus=remaining[int(np.argmax([np.average(g,weights=.25+rank) for g in geometric]))]
            candidates=list(dict.fromkeys([sequence[k],coverage,focus]))
            pool=[q for q in remaining if q not in candidates]
            rng=np.random.default_rng(140000+j['outer']*10000+int(subject[-2:])*100+k)
            candidates+=list(map(int,rng.choice(pool,6-len(candidates),replace=False)))
            # All candidate features and the slate are fixed before after-query scores.
            features=[x[remaining.index(q)].copy() for q in candidates]; base_auc=independent_auc(y,before[k])
            for q,feature in zip(candidates,features):
                after=before[k+1].copy() if q==sequence[k] else score_subset(model,physical_input(raw,index,selected+[q]),receipt['normalisation'],list(range(k+1)))
                rows.append({'features':feature,'gain':independent_auc(y,after)-base_auc,'before_scores':before[k].copy(),'after_scores':after,
                             'prefix':selected+[-1]*(7-k),'candidate':q})
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**{name:np.array([r[name] for r in rows]) for name in rows[0]},label=y)
        write_json(path.with_suffix('.json'),{'outer':j['outer'],'source_inner':j['inner'],'subject':subject,'rows':len(rows),
            'protocol_sha256':sha(output/'protocol.json'),'file_sha256':sha(path),'mapping_receipt_sha256':p['mapping_receipt_sha256'][subject],
            'source_HAT_sha256':p['source_HAT_sha256'][key],'raw_sha256':mapping['raw_sha256'],'physical_random0_sequence':sequence,
            'source_action_kind':'verified_anatomical_pairs','outer_test':j['outer_test']})
        del model
    print('V14_PHYSICAL_TARGETS',subject,len(jobs)*24,flush=True)


def banks(project,output,workers):
    p=verified(project,output); subjects=sorted({s for j in p['jobs'] for s in j['predict']})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(subject_banks,project,output,s) for s in subjects]
        for f in as_completed(futures): f.result()


def fit(project,output):
    project,output=Path(project),Path(output); p=verified(project,output)
    for outer in range(5):
        path=output/f'learners/outer{outer}/models.joblib'
        if path.with_suffix('.json').exists(): continue
        subjects=sorted({s for j in p['jobs'] if j['outer']==outer for s in j['predict']}); chunks=[]; targets=[]; patients=[]; hashes={}
        test=next(j['outer_test'] for j in p['jobs'] if j['outer']==outer)
        for subject in subjects:
            bank=output/f'banks/outer{outer}/{subject}.npz'; receipt=read_json(bank.with_suffix('.json'))
            if sha(bank)!=receipt['file_sha256'] or receipt['protocol_sha256']!=sha(output/'protocol.json') or subject in test or receipt['source_action_kind']!='verified_anatomical_pairs':
                raise ValueError('Value training bank lineage differs')
            with np.load(bank) as d: chunks.append(d['features']); targets.append(d['gain']); patients.append(np.full(len(d['gain']),int(subject[-2:])))
            hashes[subject]=sha(bank)
        x,y,patient=np.vstack(chunks),np.concatenate(targets),np.concatenate(patients)
        if len(subjects)!=28 or len(y)!=672: raise ValueError('Incomplete value training data')
        models=fit_models(x,y,patient,outer); path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(models,path)
        write_json(path.with_suffix('.json'),{'outer':outer,'fit_patients':subjects,'outer_test':test,'bank_sha256':hashes,'rows':len(y),
                   'models_sha256':sha(path),'protocol_sha256':sha(output/'protocol.json')}); print('V14_VALUE_MODELS_FIT',outer,flush=True)
    frozen={'protocol_sha256':sha(output/'protocol.json'),'models_sha256':{str(i):sha(output/f'learners/outer{i}/models.joblib') for i in range(5)}}
    path=output/'evaluation_protocol.json'
    if path.exists() and read_json(path)!=frozen: raise ValueError('Frozen value model set differs')
    write_json(path,frozen)


def run_case(project,output,c,policy):
    import torch
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2)
    folder=output/'evaluation'/c['case_id']/policy
    if (folder/'metrics.json').exists(): return
    path=output/f'learners/outer{c["outer"]}/models.joblib'; receipt=read_json(path.with_suffix('.json')); frozen=read_json(output/'evaluation_protocol.json')
    if sha(path)!=frozen['models_sha256'][str(c['outer'])] or receipt['protocol_sha256']!=sha(output/'protocol.json') or c['subject'] in receipt['fit_patients']:
        raise ValueError('Value learner changed/patient leak')
    if sha(project/'outputs/spes_physical/maps'/c['subject']/'receipt.json')!=p['mapping_receipt_sha256'][c['subject']]: raise ValueError('Physical map changed')
    for f in ['baseline','mte','hat']:
        if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{c["outer"]}/{f}/checkpoint.pth')!=p['model_sha256'][f'{c["outer"]}/{f}']: raise ValueError('Scorer changed')
    models=joblib.load(path); env=PhysicalAcquisition(project,c['outer'],c['subject']); raw=load_array(project/f'vendor/norris2026_hat/data/X_{c["subject"]}.npz')[:,:20]
    distance=np.where(env.index>=0,env.distance,0.); prefix=fixed_queries(env.available,c['outer'],int(c['subject'][-2:]),0)[:4]
    for q in prefix: write_json(folder/f'{len(env.selected):02d}_observation.json',env.observe(q))
    for k in [4,5,6,7]:
        remaining=[q for q in env.available if q not in env.selected]; ids=[env.pairs[q]['id'] for q in remaining]
        x=table(distance,physical_input(raw,env.index,env.selected),env.predictions['hat'],env.selected,remaining); values=predict(models,x)
        choice=choose(ids,values[policy]); write_json(folder/f'{k:02d}_choice.json',{'ids':ids,'features':x.tolist(),'values':{n:v.tolist() for n,v in values.items()},
            'choice':choice,'selected':[env.pairs[q]['id'] for q in env.selected],'current_HAT_score':env.predictions['hat'].tolist(),'policy':policy,'no_labels_or_future_waveforms':True})
        write_json(folder/f'{k:02d}_observation.json',env.observe(int(choice[1:])))
    result={**env.final_judge(),'case_id':c['case_id'],'policy':policy,'actual_physical_measurements':8,'new_API_calls':0,'evaluation_protocol_sha256':sha(output/'evaluation_protocol.json')}
    write_json(folder/'metrics.json',result); print('V14_QUERY_POLICY',c['case_id'],policy,round(result['AUROC'],6),flush=True)


def evaluate(project,output,workers):
    project,output=Path(project),Path(output); p=verified(project,output)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_case,project,output,c,name) for c in p['cases'] for name in p['policies']]
        for f in as_completed(futures): f.result()
    ids=[c['case_id'] for c in p['cases']]; parent=read_json(project/'outputs/spes_agent_v11/numeric_results.json')
    if sha(project/'outputs/spes_agent_v11/numeric_results.json')!=p['v11_results_sha256']: raise ValueError('Fixed controls differ')
    values={name:np.array([read_json(output/'evaluation'/c/name/'metrics.json')['AUROC'] for c in ids]) for name in p['policies']}
    controls={name:np.array([next(r['AUROC'] for r in parent['rows'] if r['policy']==name and r['case_id']==c) for c in ids]) for name in parent['policies']}
    controls['monolithic']=values['monolithic']; summary={}
    for name,scores in values.items():
        summary[name]={'mean_patient_AUROC':float(scores.mean()),'comparisons':{}}
        for control,base in controls.items():
            delta=scores-base; summary[name]['comparisons'][control]={'gain':float(delta.mean()),'paired_95CI':bootstrap_delta(delta)}
    diverse=0
    for c in ids:
        for k in [4,5,6,7]:
            trace=read_json(output/'evaluation'/c/'expert_consensus'/f'{k:02d}_choice.json')
            diverse+=len({choose(trace['ids'],trace['values'][n]) for n in ['geometry','neural','waveform']})>1
    primary=summary['expert_consensus']['comparisons']; gate=all(primary[n]['gain']>.01 and primary[n]['paired_95CI'][0]>0 for n in ['score_focus','monolithic']) and diverse>=35
    result={'primary':p['primary'],'summary':summary,'diverse_expert_steps':int(diverse),'consensus_steps':140,'next_agent_gate':gate,
            'physical_trajectories':105,'new_API_calls':0,'multi_agent_superiority_established':False}
    write_json(output/'results.json',result); print('V14_RESULT',result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['freeze','banks','fit','evaluate'],required=True); parser.add_argument('--workers',type=int,default=4); args=parser.parse_args()
    if not 1<=args.workers<=4: raise ValueError('One to four local workers')
    if args.stage=='freeze': freeze(args.project,args.output)
    else:
        with exclusive(args.output/f'{args.stage}.lock'):
            if args.stage in ['banks','evaluate']: globals()[args.stage](args.project,args.output,args.workers)
            else: fit(args.project,args.output)
