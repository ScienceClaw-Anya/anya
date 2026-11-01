"""Patient-balanced fixed value models on hypothetical-response features."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v8.learner import SETTINGS,predict
from .features import FEATURES


def fit(project,output):
    project,output=Path(project),Path(output);protocol=read_json(output/'protocol.json');cross=read_json(project/'outputs/spes_agent_v8/crossfit_protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()):raise ValueError('Frozen source differs')
    for outer in range(5):
        chunks=[];gains=[];relative=[];patients=[];inputs={}
        jobs=[j for j in cross['jobs'] if j['outer']==outer]
        for j in jobs:
            folder=output/f'banks/outer{outer}_inner{j["inner"]}';r=read_json(folder/'receipt.json')
            if r['protocol_sha256']!=sha(output/'protocol.json') or set(r['patients'])!=set(j['predict']):raise ValueError('Bank lineage differs')
            for subject,h in r['bank_sha256'].items():
                path=folder/f'{subject}.npz'
                if sha(path)!=h or subject in j['train']+j['early_stop']+j['outer_test']:raise ValueError('Source bank/patient differs')
                with np.load(path) as d:x,y,prefix=d['features'],d['gain'],d['prefix']
                yr=y.copy()
                for state in np.unique(prefix,axis=0):
                    selected=(prefix==state).all(axis=1);yr[selected]-=yr[selected].mean()
                chunks.append(x);gains.append(y);relative.append(yr);patients.append(np.full(len(y),int(subject[-2:])));inputs[path.relative_to(output).as_posix()]=h
        x,y,yr,patient=np.vstack(chunks),np.concatenate(gains),np.concatenate(relative),np.concatenate(patients);unique=sorted(set(patient));test=jobs[0]['outer_test']
        if len(unique)!=28 or set(unique)&{int(s[-2:]) for s in test} or x.shape[1]!=len(FEATURES) or not np.isfinite(x).all():raise ValueError('Outer isolation/features differ')
        folder=output/f'learners/outer{outer}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():
            r=read_json(folder/'receipt.json')
            if r['input_sha256']!=inputs or r['protocol_sha256']!=sha(output/'protocol.json') or r['models_sha256']!=sha(folder/'models.joblib'):raise ValueError('Completed v9 learner changed')
            continue
        weight=np.array([len(y)/(len(unique)*(patient==p).sum()) for p in patient]);ridge=Pipeline([('scale',StandardScaler()),('ridge',Ridge(alpha=100))]);ridge.fit(x,y,scale__sample_weight=weight,ridge__sample_weight=weight)
        blocks=np.array_split(np.random.default_rng(8920+outer).permutation(unique),5);bags=[];relative_bags=[];groups=[]
        for i,held in enumerate(blocks):
            selected=~np.isin(patient,held)
            def model():return GradientBoostingRegressor(n_estimators=150,max_depth=2,min_samples_leaf=30,learning_rate=.03,loss='huber',alpha=.9,random_state=8940+outer*10+i)
            absolute=model().fit(x[selected],y[selected],sample_weight=weight[selected]);relative_model=model().fit(x[selected],yr[selected],sample_weight=weight[selected])
            bags.append(absolute);relative_bags.append(relative_model);groups.append({'fit_patients':sorted(set(map(int,patient[selected]))),'excluded_patients':list(map(int,held))})
        artifact={'ridge':ridge,'bags':bags,'relative_bags':relative_bags,'fit_patients':list(map(int,unique)),'features':FEATURES,'outer':outer,'settings':SETTINGS}
        joblib.dump(artifact,folder/'models.joblib');write_json(folder/'receipt.json',{'outer':outer,'rows':len(y),'fit_patients':list(map(int,unique)),'outer_test':test,'bag_patient_groups':groups,'settings':SETTINGS,'features':FEATURES,'models_sha256':sha(folder/'models.joblib'),'protocol_sha256':sha(output/'protocol.json'),'input_sha256':inputs})
        print('HYPOTHESIS_LEARNER',outer,len(y),flush=True)


def load_model(output,outer):
    folder=Path(output)/f'learners/outer{outer}';r=read_json(folder/'receipt.json')
    if sha(folder/'models.joblib')!=r['models_sha256']:raise ValueError('Value model changed')
    return joblib.load(folder/'models.joblib')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with exclusive(a.output/'learners.lock'):fit(a.project,a.output)
