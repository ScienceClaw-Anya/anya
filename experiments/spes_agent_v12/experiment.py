"""Rebuild contact-posterior training states with true physical pair actions."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import rankdata
from experiments.spes_agent_v3.common import read_json,sha,check_groups,exclusive
from experiments.spes_agent_v4.neural import load_array,upstream_models
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v8.targets import score_subset
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v10.features import contact_features,FEATURES
from experiments.spes_agent_v10.experiment import fit as fit_recipe,SETTINGS
from experiments.spes_physical.mapping import physical_input
from experiments.spes_physical.environment import PhysicalAcquisition


def freeze(project,output):
    project,output=Path(project),Path(output);cross=read_json(project/'outputs/spes_agent_v8/crossfit_protocol.json');parent=read_json(project/'outputs/spes_agent_v11/numeric_protocol.json')
    files=['experiments/spes_agent_v12/experiment.py','experiments/spes_agent_v10/features.py','experiments/spes_agent_v10/experiment.py',
      'experiments/spes_physical/mapping.py','experiments/spes_physical/environment.py','experiments/spes_agent_v8/targets.py']
    p={'study':'v12 contact specialists trained on verified physical measurement states','scope':'Numerical-tool feasibility; not multi-agent, medical calibration or treatment evidence.',
      'jobs':cross['jobs'],'cases':parent['cases'],'settings':SETTINGS,'features':FEATURES,'states':[4,5,6,7,8],
      'primary':'logistic minus unchanged HAT at the same eight random0 physical queries, mean patient SOZ AUROC',
      'secondary':'Report HAT/logistic/boosting/fixed HAT-boost rankmean at four and eight physical queries; no posthoc primary swap.',
      'source_isolation':'Reuse twenty own v8 inner HATs. Each source target patient excluded from source fitting/earlystop; outer test group absent from all source jobs and fitted contact models.',
      'recipe_reuse':'Only pure v10 observed-feature and fixed fitting code is reused. NO old column-state banks, trained weights, predictions or query-utility targets are adopted.',
      'next_tool_gate':'Primary gain>0.01 and positive two-sided95 patient-bootstrap lower bound; gate supports tool development, not collaboration superiority.',
      'new_neural_fits':0,'new_small_fits':10,'new_API_calls':0,'source_sha256':{f:sha(project/f) for f in files},
      'mapping_receipt_sha256':parent['mapping_receipt_sha256'],'model_sha256':parent['model_sha256'],
      'source_HAT_sha256':{f'outer{j["outer"]}_inner{j["inner"]}':sha(project/f'outputs/spes_agent_v8/crossfit/outer{j["outer"]}_inner{j["inner"]}/checkpoint.pth') for j in cross['jobs']},
      'v11_numeric_protocol_sha256':sha(project/'outputs/spes_agent_v11/numeric_protocol.json')}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=p:raise ValueError('Frozen physical contact protocol differs')
    write_json(path,p)


def banks(project,output):
    import torch
    project,output=Path(project),Path(output);p=read_json(output/'protocol.json');mods=upstream_models(project);torch.set_num_threads(2)
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Frozen source differs')
    for subject in sorted({s for j in p['jobs'] for s in j['predict']}):
        jobs=[j for j in p['jobs'] if subject in j['predict']];paths=[output/f'banks/outer{j["outer"]}/{subject}.npz' for j in jobs]
        if all(path.exists() and path.with_suffix('.json').exists() for path in paths):continue
        folder=project/'outputs/spes_physical/maps'/subject;mapping=read_json(folder/'receipt.json')
        if sha(folder/'receipt.json')!=p['mapping_receipt_sha256'][subject] or sha(folder/'index.npz')!=mapping['index_sha256']:raise ValueError('Physical map changed')
        raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
        if sha(raw_path)!=mapping['raw_sha256']:raise ValueError('Raw source changed')
        raw=load_array(raw_path)[:,:20];index=load_array(folder/'index.npz');labels=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
        available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
        for job,path in zip(jobs,paths):
            check_groups(job['train'],job['early_stop'],job['predict'],job['outer_test'])
            if subject in job['outer_test']+job['train']+job['early_stop']:raise ValueError('Source patient not held out')
            key=f'outer{job["outer"]}_inner{job["inner"]}';source=project/'outputs/spes_agent_v8/crossfit'/key;receipt=read_json(source/'receipt.json')
            if sha(source/'checkpoint.pth')!=p['source_HAT_sha256'][key] or receipt['checkpoint_sha256']!=p['source_HAT_sha256'][key]:raise ValueError('Source HAT changed')
            if any(receipt[k]!=job[k] for k in ['train','early_stop','predict','outer_test']):raise ValueError('Source patient split differs')
            model,_,_=mods.create_model('hat');model.load_state_dict(torch.load(source/'checkpoint.pth',map_location='cpu',weights_only=True)['model_state_dict']);model=model.cuda().eval()
            sequence=fixed_queries(available,job['outer'],int(subject[-2:]),0);x=[];ys=[];states=[];scores=[]
            for k in p['states']:
                observed=physical_input(raw,index,sequence[:k]);hat=score_subset(model,observed,receipt['normalisation'],list(range(k)))
                x.append(contact_features(observed[...,-512:],observed[...,0],hat));ys.append(labels);states.append(np.full(len(labels),k));scores.append(hat)
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,features=np.vstack(x),label=np.concatenate(ys),state=np.concatenate(states),HAT_score=np.concatenate(scores))
            write_json(path.with_suffix('.json'),{'outer':job['outer'],'subject':subject,'source_inner':job['inner'],'rows':sum(len(z) for z in x),
              'protocol_sha256':sha(output/'protocol.json'),'file_sha256':sha(path),'source_checkpoint_sha256':p['source_HAT_sha256'][key],
              'mapping_receipt_sha256':p['mapping_receipt_sha256'][subject],'raw_sha256':mapping['raw_sha256'],'physical_queries':sequence,
              'source_action_kind':'verified_anatomical_pairs','outer_test':job['outer_test']})
            del model
        print('PHYSICAL_CONTACT_BANK',subject,flush=True)


def evaluate(project,output):
    import torch
    project,output=Path(project),Path(output);p=read_json(output/'protocol.json');torch.set_num_threads(2)
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Frozen source differs')
    frozen={'protocol_sha256':sha(output/'protocol.json'),'models_sha256':{str(i):sha(output/f'learners/outer{i}/models.joblib') for i in range(5)}}
    path=output/'evaluation_protocol.json'
    if path.exists() and read_json(path)!=frozen:raise ValueError('Frozen evaluation learners differ')
    write_json(path,frozen)
    for case in p['cases']:
        folder=output/'evaluation'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'metrics.json').exists():continue
        outer=case['outer'];learner_folder=output/f'learners/outer{outer}';r=read_json(learner_folder/'receipt.json')
        if r['protocol_sha256']!=sha(output/'protocol.json') or r['models_sha256']!=frozen['models_sha256'][str(outer)] or int(case['subject'][-2:]) in r['fit_patients']:raise ValueError('Learner provenance or patient exclusion differs')
        if sha(project/'outputs/spes_physical/maps'/case['subject']/'receipt.json')!=p['mapping_receipt_sha256'][case['subject']]:raise ValueError('Evaluation map changed')
        for family in ['baseline','mte','hat']:
            if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{outer}/{family}/checkpoint.pth')!=p['model_sha256'][f'{outer}/{family}']:raise ValueError('Outer scorer changed')
        learners=joblib.load(learner_folder/'models.joblib');env=PhysicalAcquisition(project,outer,case['subject']);sequence=fixed_queries(env.available,outer,int(case['subject'][-2:]),0)
        raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20];predictions={}
        for k,q in enumerate(sequence,1):
            env.observe(q)
            if k not in [4,8]:continue
            observed=physical_input(raw,env.index,sequence[:k]);hat=env.predictions['hat'];x=contact_features(observed[...,-512:],observed[...,0],hat)
            values={'hat':hat.copy(),**{name:m.predict_proba(x)[:,1] for name,m in learners.items() if name in ['logistic','boosting']}}
            values['rank_mean_boost']=(rankdata(hat)+rankdata(values['boosting']))/(2*(len(hat)+1))
            for name,value in values.items():predictions[f'{k}/{name}']=value
        old=read_json(project/'outputs/spes_agent_v11/numeric'/case['case_id']/'random_0/metrics.json')
        if [env.pairs[q]['id'] for q in sequence]!=old['queries']:raise ValueError('Physical evaluation sequence differs')
        np.testing.assert_allclose(predictions['8/hat'],old['prediction'],rtol=0,atol=1e-7)
        labels=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz');rows=[]
        for key,value in predictions.items():
            k,name=key.split('/');rows.append({'measurements':int(k),'scorer':name,'AUROC':independent_auc(labels,value),'Brier':None if name=='rank_mean_boost' else float(np.mean((labels-value)**2)),'case_id':case['case_id']})
        np.savez_compressed(folder/'predictions.npz',**predictions,label=labels);write_json(folder/'metrics.json',{'rows':rows,'queries':old['queries'],'protocol_sha256':sha(path),'prediction_sha256':sha(folder/'predictions.npz')});print('PHYSICAL_CONTACT_EVALUATED',case['case_id'],flush=True)
    rows=[r for c in p['cases'] for r in read_json(output/'evaluation'/c['case_id']/'metrics.json')['rows']];summary={}
    for k in [4,8]:
        base={r['case_id']:r['AUROC'] for r in rows if r['measurements']==k and r['scorer']=='hat'}
        for name in ['hat','logistic','boosting','rank_mean_boost']:
            selected=[r for r in rows if r['measurements']==k and r['scorer']==name];delta=np.array([r['AUROC']-base[r['case_id']] for r in selected]);summary[f'{k}/{name}']={'mean_patient_AUROC':float(np.mean([r['AUROC'] for r in selected])),'gain_over_hat':float(delta.mean()),'paired_95CI':bootstrap_delta(delta),'mean_patient_Brier':None if name=='rank_mean_boost' else float(np.mean([r['Brier'] for r in selected]))}
    primary=summary['8/logistic'];result={'primary':p['primary'],'results':summary,'rows':rows,'next_tool_eligible':primary['gain_over_hat']>.01 and primary['paired_95CI'][0]>0,'new_API_calls':0,'multi_agent_superiority_established':False,'scope':'Physically valid contact-posterior feasibility; not an agent comparison.'};write_json(output/'results.json',result);print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['freeze','banks','fit','evaluate'],required=True);a=p.parse_args()
    if a.stage=='freeze':freeze(a.project,a.output)
    else:
        with exclusive(a.output/f'{a.stage}.lock'):
            if a.stage=='fit':fit_recipe(a.project,a.output)
            else:globals()[a.stage](a.project,a.output)
