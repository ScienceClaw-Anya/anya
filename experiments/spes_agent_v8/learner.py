"""Fixed patient-balanced query-value regressors, no outer labels/features."""
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from experiments.spes_agent_v3.common import read_json,sha
from .crossfit import write_json
from .features import FEATURES

SETTINGS={'ridge_alpha':100.,'bags':5,'trees':150,'depth':2,'minimum_leaf_rows':30,'learning_rate':.03,'loss':'huber','huber_alpha':.9,
          'robust_disagreement_penalty':.5,'relative_bags':True,'relative_target':'Gain minus the sampled-candidate mean within the same patient/prefix/phase state',
          'weighting':'Each patient has equal total fitting weight; no patient ID is a feature.'}

def fit(project,output,phase):
    project,output=Path(project),Path(output);protocol=read_json(output/'query_protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['sources'].items()) or protocol['learner_settings']!=SETTINGS:raise ValueError('Frozen learner sources/settings changed')
    if sha(output/'crossfit_protocol.json')!=protocol['crossfit_protocol_sha256']:raise ValueError('Source split schedule changed')
    for outer in range(5):
        chunks=[];targets=[];relative_targets=[];patients=[];inputs={}
        for used_phase in range(phase+1):
            for job in [j for j in read_json(output/'crossfit_protocol.json')['jobs'] if j['outer']==outer]:
                folder=output/f'targets/phase{used_phase}/outer{outer}_inner{job["inner"]}';receipt=read_json(folder/'receipt.json')
                if receipt['outer_test']!=job['outer_test']:raise ValueError('Outer isolation differs')
                for subject,digest in receipt['patient_file_sha256'].items():
                    path=folder/f'{subject}.npz'
                    if sha(path)!=digest:raise ValueError('Target bank changed')
                    d=np.load(path);p=int(subject[-2:])
                    if subject in job['train']+job['early_stop']+job['outer_test'] or subject not in job['predict']:raise ValueError('Source patient was not inner-held-out')
                    relative=d['gain'].copy()
                    for prefix in np.unique(d['prefix'],axis=0):
                        state=(d['prefix']==prefix).all(axis=1);relative[state]-=relative[state].mean()
                    chunks.append(d['features']);targets.append(d['gain']);relative_targets.append(relative);patients.append(np.full(len(d['gain']),p));inputs[path.relative_to(output).as_posix()]=digest
        x,y,yr,patient=np.vstack(chunks),np.concatenate(targets),np.concatenate(relative_targets),np.concatenate(patients);unique=sorted(set(patient))
        if x.shape[1]!=len(FEATURES) or len(unique)!=28 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Incomplete/invalid query training bank')
        outer_test=next(j['outer_test'] for j in read_json(output/'crossfit_protocol.json')['jobs'] if j['outer']==outer)
        if set(unique)&{int(s[-2:]) for s in outer_test}:raise ValueError('Outer label leakage')
        weights=np.array([len(y)/(len(unique)*(patient==p).sum()) for p in patient])
        target=output/f'learners/phase{phase}/outer{outer}';target.mkdir(parents=True,exist_ok=True)
        if (target/'receipt.json').exists():
            saved=read_json(target/'receipt.json')
            if saved['input_sha256']!=inputs or saved['query_protocol_sha256']!=sha(output/'query_protocol.json') or saved['models_sha256']!=sha(target/'models.joblib'):raise ValueError('Completed learner inputs changed')
            print('LEARNERS_REUSED',phase,outer,flush=True);continue
        ridge=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=SETTINGS['ridge_alpha']))])
        ridge.fit(x,y,scale__sample_weight=weights,ridge__sample_weight=weights)
        permutation=np.random.default_rng(8920+outer).permutation(unique);blocks=np.array_split(permutation,5);bags=[];relative_bags=[];bag_groups=[]
        for i,held in enumerate(blocks):
            selected=~np.isin(patient,held)
            model=GradientBoostingRegressor(n_estimators=150,max_depth=2,min_samples_leaf=30,learning_rate=.03,
                 loss='huber',alpha=.9,random_state=8940+outer*10+i)
            model.fit(x[selected],y[selected],sample_weight=weights[selected]);bags.append(model)
            relative_model=GradientBoostingRegressor(n_estimators=150,max_depth=2,min_samples_leaf=30,learning_rate=.03,
                 loss='huber',alpha=.9,random_state=8940+outer*10+i)
            relative_model.fit(x[selected],yr[selected],sample_weight=weights[selected]);relative_bags.append(relative_model)
            bag_groups.append({'fit_patients':sorted(set(map(int,patient[selected]))),'excluded_patients':sorted(map(int,held))})
        artifact={'ridge':ridge,'bags':bags,'relative_bags':relative_bags,'features':FEATURES,'settings':SETTINGS,'fit_patients':list(map(int,unique)),'outer':outer,'phase':phase}
        joblib.dump(artifact,target/'models.joblib')
        write_json(target/'receipt.json',{'outer':outer,'phase':phase,'rows':len(y),'fit_patients':list(map(int,unique)),
           'outer_test':outer_test,'feature_names':FEATURES,'settings':SETTINGS,'target_definition':'Later minus current within-patient SOZ AUROC, from source-heldout patient replay',
           'target_mean':float(y.mean()),'target_std':float(y.std()),'bag_patient_groups':bag_groups,
           'training_resubstitution_MAE_ridge':float(np.average(np.abs(ridge.predict(x)-y),weights=weights)),
           'training_resubstitution_MAE_bag_mean':float(np.average(np.abs(np.mean([m.predict(x) for m in bags],axis=0)-y),weights=weights)),
           'input_sha256':inputs,'models_sha256':sha(target/'models.joblib'),'query_protocol_sha256':sha(output/'query_protocol.json'),
           'limits':['Training residuals are not validation performance. Bag disagreement is not a calibrated confidence interval.',
                     'Inner scorers fit 17 patients; final outer scorer fits 21. The model/input distribution can shift.',
                     'Outer-cohort evaluation remains exploratory after repeated research.']})
        print('LEARNERS_FIT',phase,outer,len(y),flush=True)

def load_model(output,phase,outer):
    folder=Path(output)/f'learners/phase{phase}/outer{outer}';receipt=read_json(folder/'receipt.json')
    if sha(folder/'models.joblib')!=receipt['models_sha256']:raise ValueError('Value learner changed')
    return joblib.load(folder/'models.joblib')

def predict(model,features):
    features=np.asarray(features);bag=np.vstack([m.predict(features) for m in model['bags']]);mean=bag.mean(axis=0);sd=bag.std(axis=0)
    relative=np.vstack([m.predict(features) for m in model['relative_bags']])
    return {'ridge':model['ridge'].predict(features),'bag_mean':mean,'bag_disagreement':sd,
            'relative_mean':relative.mean(axis=0),'relative_disagreement':relative.std(axis=0),'robust':mean-.5*sd}

if __name__=='__main__':
    import argparse
    from experiments.spes_agent_v3.common import exclusive
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',type=int,choices=[0,1],required=True);a=p.parse_args()
    with exclusive(a.output/f'learners_phase{a.phase}.lock'):fit(a.project,a.output,a.phase)
