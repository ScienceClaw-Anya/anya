"""Fixed, cheap numerical feasibility screen. Never calls an LLM.

Uses v3's honestly outer-separated base OOF predictions. Outer evaluation is
exploratory because this public cohort was repeatedly inspected previously.
No selected winner is represented as an untouched confirmatory result.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from experiments.spes_agent_v3.common import read_json, write_json, sha, SLUGS, WEIGHTS
from experiments.spes_agent_v3.selector import fit_selector, features, candidates

METHODS = ['transformer', 'conditional_ridge_no_global_veto', 'contact_logistic', 'patient_pairwise_ranker', 'pairwise_ranker_10pct']

def ranks(x, patient):
    out = np.empty_like(x, dtype=float)
    for p in np.unique(patient):
        m = patient == p
        out[m] = np.column_stack([rankdata(x[m,i], method='average')/(m.sum()+1) for i in range(x.shape[1])])
    return out

def fit_contact(x, y, patient):
    weight = np.zeros(len(y))
    for p in np.unique(patient):
        for label in [0,1]:
            m = (patient==p)&(y==label)
            if not m.any(): raise ValueError('Single-class patient')
            weight[m] = len(y)/(2*len(np.unique(patient))*m.sum())
    return LogisticRegression(C=.1, solver='lbfgs', max_iter=1000).fit(logit(np.clip(x,1e-5,1-1e-5)),y,sample_weight=weight)

def fit_pairwise(x,y,patient):
    x = ranks(x, patient); rows=[]; weights=[]
    for p in np.unique(patient):
        m = patient == p; local=x[m]; labels=y[m]
        delta = (local[labels==1,None,:]-local[None,labels==0,:]).reshape(-1,x.shape[1])
        if not len(delta): raise ValueError('Single-class patient')
        rows.append(delta); weights.append(np.full(len(delta),1/len(delta)))
    pairs=np.vstack(rows); weight=np.concatenate(weights)
    weight *= len(pairs)/weight.sum()
    return LogisticRegression(C=.1,fit_intercept=False,solver='lbfgs',max_iter=1000).fit(
        np.vstack([pairs,-pairs]),np.r_[np.ones(len(pairs)),np.zeros(len(pairs))],sample_weight=np.r_[weight,weight])

def bootstrap_delta(delta, seed=49103):
    delta=np.array(delta);rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws,[.025,.975])]

def run(project, output):
    project, output=Path(project),Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol={'study':'v4 numerical feasibility screen','scope':'Exploratory reuse of an inspected 35-patient cohort; not confirmatory',
      'methods':METHODS,'fixed_C':.1,'primary':'patient_pairwise_ranker','ridge_min_gain':.005,
      'no_hyperparameter_search':True,'new_neural_fits':0,'new_API_calls':0,
      'input':'v3 inner OOF; original first-repeat outer predictions','source_sha256':sha(Path(__file__))}
    path=output/'screen_protocol.json'
    if path.exists() and read_json(path)!=protocol:raise ValueError('Frozen protocol changed')
    write_json(path,protocol)
    records=[];receipts=[]
    for outer in range(5):
        oof_path=project/f'outputs/spes_agent_v3/selectors/outer{outer}/OOF.npz';d=np.load(oof_path)
        x,y,patient=d['predictions'],d['y'],d['patient']
        selector=fit_selector(x,y,patient);contact=fit_contact(x,y,patient);pair=fit_pairwise(x,y,patient)
        training=project/'outputs/reproduction/norris2024/training';split=read_json(training/f'split_seed1_fold{outer}.json')
        if set(patient)&{int(p[-2:]) for p in split['test']}:raise ValueError('Outer leakage')
        values=[np.load(training/f'{slug}_seed1_fold{outer}/predictions.npz') for slug in SLUGS]
        for value in values[1:]:
            if not np.array_equal(value['patient'],values[0]['patient']) or not np.array_equal(value['y'],values[0]['y']):raise ValueError('Row alignment')
        test_x=np.column_stack([v['prediction'] for v in values]);test_p=values[0]['patient'];truth=values[0]['y']
        outputs={'transformer':test_x[:,2],
          'contact_logistic':contact.predict_proba(logit(np.clip(test_x,1e-5,1-1e-5)))[:,1],
          'patient_pairwise_ranker':pair.decision_function(ranks(test_x,test_p))}
        outputs['pairwise_ranker_10pct']=.9*ranks(test_x,test_p)[:,2]+.1*ranks(outputs['patient_pairwise_ranker'][:,None],test_p)[:,0]
        outputs['conditional_ridge_no_global_veto']=test_x[:,2].copy()
        actions={}
        for p in sorted(set(test_p)):
            m=test_p==p;pred=selector['model'].predict(features(test_x[m])[None])[0]
            name=selector['alternatives'][int(pred.argmax())] if pred.max()>.005 else 'transformer'
            outputs['conditional_ridge_no_global_veto'][m]=candidates(test_x[m])[name];actions[int(p)]=name
        np.savez_compressed(output/f'screen_outer{outer}.npz',patient=test_p,y=truth,**outputs)
        receipts.append({'outer':outer,'OOF_sha256':sha(oof_path),'contact_coefficients':contact.coef_.tolist(),
            'pairwise_coefficients':pair.coef_.tolist(),'conditional_actions':actions})
        for p in sorted(set(test_p)):
            m=test_p==p
            records.append({'patient':int(p),'outer':outer,**{n:float(roc_auc_score(truth[m],outputs[n][m])) for n in METHODS}})
    baseline=np.array([r['transformer'] for r in records]);summary={}
    for name in METHODS:
        scores=np.array([r[name] for r in records]);delta=scores-baseline
        summary[name]={'mean_patient_AUROC':float(scores.mean()),'mean_gain':float(delta.mean()),
            'patient_bootstrap_95CI':bootstrap_delta(delta),'patients_better':int((delta>1e-12).sum()),'patients_worse':int((delta < -1e-12).sum())}
    result={'protocol':protocol,'patients':35,'methods':summary,'rows':records,'receipts':receipts,
      'interpretation':'Screen only. Numerical gains do not demonstrate multi-agent collaboration; repeated public cohort use precludes an untouched-test claim.'}
    write_json(output/'screen_results.json',result);print(summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.project,args.output)
