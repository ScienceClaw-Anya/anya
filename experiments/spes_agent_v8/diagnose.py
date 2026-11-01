"""Post-hoc development diagnostics; never a new policy or outer tuning gate."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from experiments.spes_agent_v3.common import read_json
from .learner import load_model,predict
from .features import FEATURES
from .crossfit import write_json


def diagnose(output):
    output=Path(output);source=read_json(output/'crossfit_protocol.json');results={}
    for phase in [0,1]:
        states=[];importance=[]
        for outer in range(5):
            learner=load_model(output,phase,outer);receipt=read_json(output/f'learners/phase{phase}/outer{outer}/receipt.json')
            importance.append(np.mean([m.feature_importances_ for m in learner['bags']],axis=0))
            for used_phase in range(phase+1):
                for job in [j for j in source['jobs'] if j['outer']==outer]:
                    folder=output/f'targets/phase{used_phase}/outer{outer}_inner{job["inner"]}'
                    for subject in job['predict']:
                        d=np.load(folder/f'{subject}.npz');p=int(subject[-2:]);excluded=[i for i,b in enumerate(receipt['bag_patient_groups']) if p in b['excluded_patients']]
                        if len(excluded)!=1:raise ValueError('Exactly one bag must exclude this patient')
                        oob=learner['bags'][excluded[0]].predict(d['features']);fitted=predict(learner,d['features'])['bag_mean']
                        for prefix in np.unique(d['prefix'],axis=0):
                            ix=np.flatnonzero((d['prefix']==prefix).all(axis=1));gain=d['gain'][ix]
                            def corr(a,b):return float(spearmanr(a,b).statistic) if np.std(a)>1e-12 and np.std(b)>1e-12 else None
                            states.append({'outer':outer,'patient':p,'phase':used_phase,'mean_gain':float(gain.mean()),'gain_sd':float(gain.std()),
                              'oracle_minus_average':float(gain.max()-gain.mean()),'oob_pick_minus_average':float(gain[np.argmax(oob[ix])]-gain.mean()),
                              'fitted_pick_minus_average':float(gain[np.argmax(fitted[ix])]-gain.mean()),'oob_candidate_spearman':corr(oob[ix],gain)})
        # Every patient contributes equally within each outer/phase, then aggregate
        # patient-fold cases; repeated appearances are not independent subjects.
        groups=sorted({(s['outer'],s['patient']) for s in states});fields=['mean_gain','gain_sd','oracle_minus_average','oob_pick_minus_average','fitted_pick_minus_average']
        summary={k:float(np.mean([np.mean([s[k] for s in states if (s['outer'],s['patient'])==g]) for g in groups])) for k in fields}
        valid=[s['oob_candidate_spearman'] for s in states if s['oob_candidate_spearman'] is not None]
        summary['descriptive_mean_within_state_oob_spearman']=float(np.mean(valid))
        summary['feature_importance']={name:float(v) for name,v in zip(FEATURES,np.mean(importance,axis=0))}
        summary['global_state_feature_importance_fraction']=float(np.mean(importance,axis=0)[:10].sum())
        results[f'phase{phase}']={'summary':summary,'states':states}
    value={'status':'posthoc_development_diagnostic_not_policy_validation','results':results,
      'limits':['The excluded value bag did not fit this patient, but upstream source models for other target patients may have used it. This is not fully nested validation.',
        'Fitted-policy picks are resubstitution and optimistically biased.','The best of six sampled candidates uses hidden labels retrospectively; it is not an executable policy.',
        'No outer result is used to refit a policy here; all v8 methods and primary outcome remain unchanged.']}
    write_json(output/'development_diagnostic.json',value)
    for phase,row in results.items():print(phase,row['summary'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();diagnose(a.output)
