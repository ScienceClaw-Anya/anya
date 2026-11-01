"""Evaluate the complete frozen neural stage and fixed numerical controls."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.spes_agent_v3.common import read_json,write_json,sha
from .neural import FAMILIES
from .screen import fit_pairwise,fit_contact,ranks,bootstrap_delta
from scipy.special import logit

NAMES=FAMILIES+['rank_mean','validation_best','pairwise_from_validation','pairwise_10pct']

def freeze_evaluation(output):
    output=Path(output)
    value={'scope':'Exploratory, newly released processed cohort; all old study failures retained',
        'primary':'hat versus baseline','numerical_controls':NAMES,'contact_model':'C=0.1; patient/class balanced',
        'pairwise_model':'C=0.1, patient-normalized positive-negative differences of within-patient score ranks',
        'selection':'Validation-best fixed model picked using seven validation patients only; optional pairwise trained on those same seven validation patients.',
        'primary_endpoint':'Mean patient AUROC; two-sided 95% patient bootstrap interval on paired gains',
        'not_collaboration_evidence':True,'no_hyperparameter_search':True,'source_sha256':sha(Path(__file__))}
    path=output/'evaluation_protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen evaluation changed')
    write_json(path,value);return value

def run(output):
    output=Path(output);protocol=freeze_evaluation(output);rows=[];fit_rows=[]
    neural=read_json(output/'neural_protocol.json')
    for outer in range(5):
        for family in FAMILIES:
            folder=output/f'neural/outer{outer}/{family}';receipt=read_json(folder/'receipt.json')
            if receipt['protocol_sha256']!=sha(output/'neural_protocol.json'):raise ValueError('Protocol differs')
            if sha(folder/'predictions.npz')!=receipt['prediction_sha256']:raise ValueError('Prediction digest differs')
            if set(receipt['train']+receipt['early_stop'])&set(receipt['predict']):raise ValueError('Patient leakage')
    for outer in range(5):
        folder=output/f'neural/outer{outer}'
        val=[np.load(folder/f/'validation.npz') for f in FAMILIES];test=[np.load(folder/f/'predictions.npz') for f in FAMILIES]
        for group in [val,test]:
            for d in group[1:]:
                if any(not np.array_equal(d[k],group[0][k]) for k in ['y','patient','contact']):raise ValueError('Misaligned scores')
        vx=np.column_stack([v['prediction'] for v in val]);vy=val[0]['y'];vp=val[0]['patient']
        validation_auc=[np.mean([roc_auc_score(vy[vp==p],vx[vp==p,i]) for p in np.unique(vp)]) for i in range(3)]
        best=int(np.argmax(validation_auc));pair=fit_pairwise(vx,vy,vp)
        x=np.column_stack([v['prediction'] for v in test]);p=test[0]['patient'];y=test[0]['y'];ranked=ranks(x,p)
        predictions={n:x[:,i] for i,n in enumerate(FAMILIES)}
        predictions['rank_mean']=ranked.mean(axis=1);predictions['validation_best']=x[:,best]
        predictions['pairwise_from_validation']=pair.decision_function(ranked)
        predictions['pairwise_10pct']=.9*ranked[:,2]+.1*ranks(predictions['pairwise_from_validation'][:,None],p)[:,0]
        np.savez_compressed(output/f'numerical_outer{outer}.npz',y=y,patient=p,contact=test[0]['contact'],**predictions)
        fit_rows.append({'outer':outer,'validation_AUROC':dict(zip(FAMILIES,map(float,validation_auc))),
          'validation_choice':FAMILIES[best],'pairwise_coefficients':pair.coef_.tolist()})
        for patient in np.unique(p):
            m=p==patient;values={n:float(roc_auc_score(y[m],predictions[n][m])) for n in NAMES}
            rows.append({'patient':int(patient),'outer':outer,'contacts':int(m.sum()),'SOZ':int(y[m].sum()),**values})
    summary={};base=np.array([r['baseline'] for r in rows]);hat=np.array([r['hat'] for r in rows])
    for name in NAMES:
        score=np.array([r[name] for r in rows]);delta=score-hat
        summary[name]={'mean_patient_AUROC':float(score.mean()),'gain_over_hat':float(delta.mean()),
            'paired_95CI_over_hat':bootstrap_delta(delta),'patients_better_than_hat':int((delta>1e-12).sum()),'patients_worse_than_hat':int((delta < -1e-12).sum())}
    result={'protocol':protocol,'patients':35,'contacts':sum(r['contacts'] for r in rows),'SOZ':sum(r['SOZ'] for r in rows),
        'methods':summary,'HAT_vs_baseline':{'mean_gain':float((hat-base).mean()),'paired_95CI':bootstrap_delta(hat-base)},
        'rows':rows,'validation_fits':fit_rows,'caveat':'No multi-agent claim. One repeat, adapted runner, previously inspected patient cohort. Counts differ from v1-v3 and published papers.'}
    write_json(output/'numerical_results.json',result);print({k:round(v['mean_patient_AUROC'],6) for k,v in summary.items()},flush=True)
    print('HAT vs baseline',result['HAT_vs_baseline'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.freeze_only:freeze_evaluation(a.output);print('Evaluation frozen; no outer outcomes read.')
    else:run(a.output)
