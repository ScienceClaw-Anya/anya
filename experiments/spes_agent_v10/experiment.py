"""Fixed patient-isolated contact-model feasibility test, no LLM requests."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from experiments.spes_agent_v3.common import read_json,sha,exclusive,check_groups
from experiments.spes_agent_v4.neural import load_array,upstream_models
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v8.features import distance_catalog
from experiments.spes_agent_v8.targets import score_subset
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v5.neural import fixed_queries
from .features import from_revealed,FEATURES

SETTINGS={'logistic_C':.1,'logistic_class_weight':None,'boost_trees':150,'boost_depth':2,'boost_min_leaf':50,'boost_learning_rate':.03,'prefix_sizes':[4,5,6,7,8],
 'weighting':'Each development patient has equal total fitting weight across contacts and all five query states; patient ID is never a feature.'}


def freeze(project,output):
    project,output=Path(project),Path(output);source=project/'outputs/spes_agent_v8';cross=read_json(source/'crossfit_protocol.json');active=read_json(project/'outputs/spes_agent_v4/active/active_protocol.json')
    files=['experiments/spes_agent_v10/features.py','experiments/spes_agent_v10/experiment.py','experiments/spes_agent_v8/targets.py','experiments/spes_agent_v5/neural.py','vendor/norris2026_hat/src/models.py']
    r={'study':'v10 complementary contact posterior feasibility','scope':'Exploratory numerical modelling, not LLM or multi-agent superiority.',
      'rationale':'Response forecasts alone do not identify which contact scores are wrong. Test complementary partial-waveform SOZ models before combining them with hypothetical-response tools.',
      'features':FEATURES,'feature_semantics':'Sample-index and FFT-bin summaries only; no unverified physiological time/frequency-band or extra128-feature meaning is assigned.',
      'settings':SETTINGS,'jobs':cross['jobs'],'cases':active['cases'],'new_neural_fits':0,'new_API_calls':0,
      'primary':'logistic minus HAT at the same fixed eight random_0 measurements; mean patient SOZ AUROC',
      'secondary':'All four scorers at four and eight measurements, including boosting and fixed HAT/boost rank mean; no post-hoc primary swap.',
      'next_tool_gate':'Primary mean gain >0.01 with positive two-sided95 patient-bootstrap lower bound. This only supports developing an expected-ranking-gain tool, not medical calibration.',
      'expected_gain_note':'The expected-rank helper uses ratio of expected independent-label pair counts, not exact expected AUROC. It is unit-tested but not used to select measurements in this experiment.',
      'source_sha256':{p:sha(project/p) for p in files},'parent_crossfit_sha256':sha(source/'crossfit_protocol.json'),
      'target_receipt_sha256':{f'outer{j["outer"]}_inner{j["inner"]}':sha(source/f'targets/phase0/outer{j["outer"]}_inner{j["inner"]}/receipt.json') for j in cross['jobs']},
      'outer_model_sha256':active['model_sha256'],'prefix_sha256':{c['case_id']:sha(project/'outputs/spes_agent_v4/active/trajectories'/c['case_id']/'random_0/metrics.json') for c in active['cases']}}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=r:raise ValueError('Frozen contact protocol changed')
    write_json(path,r);return r


def banks(project,output):
    project,output=Path(project),Path(output);p=read_json(output/'protocol.json');source=project/'outputs/spes_agent_v8'
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Frozen source differs')
    # Per-patient preprocessing has no population statistics. Loading one raw
    # patient for its separate outer-development banks does not join cohorts.
    subjects=sorted({s for j in p['jobs'] for s in j['predict']})
    for subject in subjects:
        jobs=[j for j in p['jobs'] if subject in j['predict']]
        paths=[output/f'banks/outer{j["outer"]}/{subject}.npz' for j in jobs]
        if all(path.exists() and path.with_suffix('.json').exists() for path in paths):continue
        raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz';digest=sha(raw_path);raw=load_array(raw_path)[:,:20];_,available=distance_catalog(raw[...,0])
        for j,path in zip(jobs,paths):
            check_groups(j['train'],j['early_stop'],j['predict'],j['outer_test'])
            if subject in j['outer_test']+j['train']+j['early_stop']:raise ValueError('Source label posterior target not held out')
            key=f'outer{j["outer"]}_inner{j["inner"]}';old_folder=source/f'targets/phase0/{key}';r=read_json(old_folder/'receipt.json')
            if sha(old_folder/'receipt.json')!=p['target_receipt_sha256'][key] or digest!=r['raw_input_sha256'][subject]:raise ValueError('Parent/raw inputs differ')
            old_path=old_folder/f'{subject}.npz'
            if sha(old_path)!=r['patient_file_sha256'][subject]:raise ValueError('Source scores differ')
            with np.load(old_path) as d:old={k:d[k] for k in d.files}
            sequence=fixed_queries(available,j['outer'],int(subject[-2:]),0);x=[];y=[];states=[]
            for k in SETTINGS['prefix_sizes']:
                if k<8:
                    rows=np.flatnonzero((old['prefix']==np.array(sequence[:k]+[-1]*(7-k))).all(axis=1));scores=old['before_scores'][rows[0]]
                else:
                    rows=np.flatnonzero((old['prefix']==np.array(sequence[:7])).all(axis=1)&(old['candidate']==sequence[7]));scores=old['after_scores'][rows[0]]
                x.append(from_revealed(raw,sequence[:k],scores));y.append(old['y']);states.append(np.full(len(scores),k))
            path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,features=np.vstack(x),label=np.concatenate(y),state=np.concatenate(states))
            write_json(path.with_suffix('.json'),{'outer':j['outer'],'subject':subject,'source_inner':j['inner'],'rows':sum(len(a) for a in x),'fit_excluded_outer_test':j['outer_test'],
              'protocol_sha256':sha(output/'protocol.json'),'raw_sha256':digest,'source_bank_sha256':sha(old_path),'file_sha256':sha(path)})
        print('CONTACT_BANK_PATIENT',subject,flush=True)


def fit(project,output):
    project,output=Path(project),Path(output);p=read_json(output/'protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Frozen source differs')
    for outer in range(5):
        jobs=[j for j in p['jobs'] if j['outer']==outer];subjects=sorted({s for j in jobs for s in j['predict']});chunks=[];labels=[];patients=[];inputs={}
        for subject in subjects:
            path=output/f'banks/outer{outer}/{subject}.npz';r=read_json(path.with_suffix('.json'))
            if sha(path)!=r['file_sha256'] or r['protocol_sha256']!=sha(output/'protocol.json') or subject in jobs[0]['outer_test']:raise ValueError('Contact bank provenance/isolation differs')
            with np.load(path) as d:x,y=d['features'],d['label']
            chunks.append(x);labels.append(y);patients.append(np.full(len(y),int(subject[-2:])));inputs[subject]=sha(path)
        x,y,patient=np.vstack(chunks),np.concatenate(labels),np.concatenate(patients);unique=sorted(set(patient))
        if len(unique)!=28 or set(unique)&{int(s[-2:]) for s in jobs[0]['outer_test']}:raise ValueError('Outer fitting leak')
        folder=output/f'learners/outer{outer}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():
            old=read_json(folder/'receipt.json')
            if old['input_sha256']!=inputs or old['models_sha256']!=sha(folder/'models.joblib') or old['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('Completed contact learner differs')
            continue
        weight=np.array([len(y)/(28*(patient==s).sum()) for s in patient]);logistic=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=1000))]);logistic.fit(x,y,scale__sample_weight=weight,model__sample_weight=weight)
        boosting=GradientBoostingClassifier(n_estimators=150,max_depth=2,min_samples_leaf=50,learning_rate=.03,random_state=10000+outer).fit(x,y,sample_weight=weight)
        joblib.dump({'logistic':logistic,'boosting':boosting,'fit_patients':list(map(int,unique)),'settings':SETTINGS,'features':FEATURES},folder/'models.joblib')
        write_json(folder/'receipt.json',{'outer':outer,'fit_patients':list(map(int,unique)),'outer_test':jobs[0]['outer_test'],'rows':len(y),'settings':SETTINGS,'input_sha256':inputs,'models_sha256':sha(folder/'models.joblib'),'protocol_sha256':sha(output/'protocol.json')});print('CONTACT_MODELS_FIT',outer,len(y),flush=True)


def evaluate(project,output):
    import torch
    project,output=Path(project),Path(output);p=read_json(output/'protocol.json');torch.set_num_threads(2);training=project/'outputs/spes_agent_v4/multitrial';mods=upstream_models(project)
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Frozen source differs')
    for case in p['cases']:
        folder=output/'evaluation'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'metrics.json').exists():continue
        outer=case['outer'];f=output/f'learners/outer{outer}';receipt=read_json(f/'receipt.json')
        if receipt['models_sha256']!=sha(f/'models.joblib') or receipt['protocol_sha256']!=sha(output/'protocol.json') or int(case['subject'][-2:]) in receipt['fit_patients']:raise ValueError('Outer learner leak/provenance differs')
        learners=joblib.load(f/'models.joblib');model,_,_=mods.create_model('hat');checkpoint=training/f'neural/outer{outer}/hat/checkpoint.pth'
        if sha(checkpoint)!=p['outer_model_sha256'][f'{outer}/hat']:raise ValueError('HAT changed')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']);model=model.cuda().eval();stats=read_json(checkpoint.parent/'receipt.json')['normalisation']
        prefix_path=project/'outputs/spes_agent_v4/active/trajectories'/case['case_id']/'random_0/metrics.json'
        if sha(prefix_path)!=p['prefix_sha256'][case['case_id']]:raise ValueError('Prefix differs')
        old=read_json(prefix_path);sequence=[int(q[1:]) for q in old['queries']];raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20];y=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz');metrics=[];predictions={}
        for k in [4,8]:
            hat=score_subset(model,raw,stats,sequence[:k]);x=from_revealed(raw,sequence[:k],hat);values={'hat':hat,**{n:m.predict_proba(x)[:,1] for n,m in learners.items() if n in ['logistic','boosting']}}
            values['rank_mean_boost']=(rankdata(hat)+rankdata(values['boosting']))/(2*(len(hat)+1))
            if k==8:np.testing.assert_allclose(hat,old['prediction'],rtol=0,atol=1e-7)
            for name,scores in values.items():metrics.append({'scorer':name,'measurements':k,'AUROC':independent_auc(y,scores),'Brier':None if name=='rank_mean_boost' else float(np.mean((y-scores)**2)),'patient':int(case['subject'][-2:]),'outer':outer,'case_id':case['case_id']});predictions[f'{k}/{name}']=scores
        np.savez_compressed(folder/'predictions.npz',**predictions,label=y);write_json(folder/'metrics.json',{'rows':metrics,'predictions_sha256':sha(folder/'predictions.npz'),'protocol_sha256':sha(output/'protocol.json')});print('CONTACT_EVALUATED',case['case_id'],flush=True)
    rows=[row for case in p['cases'] for row in read_json(output/'evaluation'/case['case_id']/'metrics.json')['rows']];summary={}
    for k in [4,8]:
        base={r['case_id']:r for r in rows if r['scorer']=='hat' and r['measurements']==k}
        for scorer in ['hat','logistic','boosting','rank_mean_boost']:
            selected=[r for r in rows if r['measurements']==k and r['scorer']==scorer];delta=np.array([r['AUROC']-base[r['case_id']]['AUROC'] for r in selected])
            summary[f'{k}/{scorer}']={'mean_patient_AUROC':float(np.mean([r['AUROC'] for r in selected])),'gain_over_hat':float(delta.mean()),'paired_95CI':bootstrap_delta(delta),'mean_patient_Brier':None if scorer=='rank_mean_boost' else float(np.mean([r['Brier'] for r in selected]))}
    primary=summary['8/logistic'];write_json(output/'results.json',{'primary':p['primary'],'results':summary,'rows':rows,'next_tool_eligible':primary['gain_over_hat']>.01 and primary['paired_95CI'][0]>0,'new_API_calls':0,'scope':'Contact ranking feasibility only; not acquisition or multi-agent benefit.'});print(summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['freeze','banks','fit','evaluate'],required=True);a=p.parse_args()
    if a.stage=='freeze':freeze(a.project,a.output)
    else:
        with exclusive(a.output/f'{a.stage}.lock'):globals()[a.stage](a.project,a.output)
