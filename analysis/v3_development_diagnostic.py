"""Cheap post-hoc coefficient/development replay, with no neural or API calls."""
import argparse
from pathlib import Path
import sys
import numpy as np
import joblib

p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();project,output=a.project.resolve(),a.output.resolve();sys.path.insert(0,str(project))
from experiments.spes_agent_v3.common import SELECTOR,WEIGHTS,read_json,write_json,sha
from experiments.spes_agent_v2.verify import auc

headroom=[];conditional=[];margin=SELECTOR['minimum_gain'];prepared=read_json(output/'prepared.json')
for outer in range(5):
    folder=output/f'selectors/outer{outer}';data=np.load(folder/'OOF.npz');fitted=joblib.load(folder/'selector.joblib')
    gains=[]
    for patient in sorted(set(data['patient'])):
        mask=data['patient']==patient;x=data['predictions'][mask].astype(float);y=data['y'][mask]
        scores={n:auc(y,x@np.array(w)) for n,w in WEIGHTS.items()}
        gains.append(max(scores.values())-scores['transformer'])
    headroom.append({'outer':outer,'development_patients':28,'hindsight_best_candidate_mean_gain':float(np.mean(gains)),
        'patients_with_any_candidate_gain':int(sum(g>1e-12 for g in gains))})
    # Reuse saved linear coefficients. No model is refitted; this is training resubstitution.
    z=(fitted['training_features']-fitted['model'].named_steps['scale'].mean_)/fitted['model'].named_steps['scale'].scale_
    predicted=z@fitted['model'].named_steps['ridge'].coef_.T+fitted['model'].named_steps['ridge'].intercept_
    best=predicted.argmax(axis=1);active=predicted.max(axis=1)>margin
    gain=np.where(active,fitted['training_targets'][np.arange(len(best)),best],0.)
    outer_values=[]
    for row in prepared['cases']:
        if row['outer']!=outer:continue
        public=read_json(output/'cases'/row['case_id']/'public.json')
        outer_values.append(max(public['evidence']['selector_decision']['predicted_gain'].values()))
    conditional.append({'outer':outer,'ungated_training_activations':int(active.sum()),'training_resubstitution_mean_gain':float(gain.mean()),
        'unlabeled_outer_cases':len(outer_values),'ungated_unlabeled_outer_activations':int(sum(v>margin for v in outer_values)),
        'largest_predicted_outer_gain':float(max(outer_values))})
shared={'source_sha256':sha(Path(__file__)),'scope':'Exploratory development-only arithmetic; no fitting, neural inference, API calls or outer label use'}
write_json(output/'development_headroom.json',{**shared,'folds':headroom,
    'mean_of_fold_hindsight_gains':float(np.mean([r['hindsight_best_candidate_mean_gain'] for r in headroom])),
    'caveat':'Hindsight chooses with development labels. It is optimistic, not deployable, not an arbitrary-mixture bound and not proof of learnable benefit; overlapping development pools are not independent cohorts.'})
write_json(output/'conditional_selector_diagnostic.json',{**shared,'margin_unchanged':margin,'folds':conditional,
    'ungated_unlabeled_outer_activations_total':sum(r['ungated_unlabeled_outer_activations'] for r in conditional),
    'caveat':'Training gains are resubstitution, not validation. Ignoring the global veto is a diagnostic only, not an approved improved policy or a new held-out performance claim.'})
print('Development-only diagnostics saved; no API or neural jobs started.')
