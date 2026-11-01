"""Retrospective hypothesis-quality diagnostic, never an executable policy."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json
from .features import METHODS


def diagnose(project,output):
    project,output=Path(project),Path(output);source=project/'outputs/spes_agent_v8';cross=read_json(source/'crossfit_protocol.json');states=[]
    for j in cross['jobs']:
        folder=output/f'banks/outer{j["outer"]}_inner{j["inner"]}';receipt=read_json(folder/'receipt.json')
        for subject,h in receipt['bank_sha256'].items():
            path=folder/f'{subject}.npz';old_path=source/f'targets/phase0/outer{j["outer"]}_inner{j["inner"]}/{subject}.npz'
            if sha(path)!=h or sha(old_path)!=receipt['v8_target_sha256'][subject]:raise ValueError('Diagnostic inputs changed')
            with np.load(path) as d:bank={k:d[k] for k in d.files}
            with np.load(old_path) as d:old={k:d[k] for k in d.files}
            for prefix in np.unique(bank['prefix'],axis=0):
                ix=np.flatnonzero((bank['prefix']==prefix).all(axis=1));actual=old['gain'][ix]
                for m,method in enumerate(METHODS):
                    imagined=np.array([independent_auc(old['y'],bank['hypothetical_scores'][i,m])-old['before_auc'][i] for i in ix])
                    corr=float(spearmanr(imagined,actual).statistic) if np.std(imagined)>1e-12 and np.std(actual)>1e-12 else None
                    cosine=[]
                    for i in ix:
                        pred=bank['hypothetical_scores'][i,m]-old['before_scores'][i];truth=old['after_scores'][i]-old['before_scores'][i]
                        denom=np.linalg.norm(pred)*np.linalg.norm(truth)
                        if denom>1e-12:cosine.append(float(pred@truth/denom))
                    states.append({'outer':j['outer'],'patient':int(subject[-2:]),'method':method,'within_state_AUROC_change_spearman':corr,
                      'score_change_vector_cosine':float(np.mean(cosine)) if cosine else None,
                      'hindsight_label_assisted_pick_minus_average':float(actual[np.argmax(imagined)]-actual.mean())})
    summary={}
    for method in METHODS:
        rows=[r for r in states if r['method']==method];summary[method]={}
        for key in ['within_state_AUROC_change_spearman','score_change_vector_cosine','hindsight_label_assisted_pick_minus_average']:
            values=[r[key] for r in rows if r[key] is not None];summary[method][key]=float(np.mean(values)) if values else None
    result={'status':'retrospective_development_hypothesis_diagnostic','patient_fold_cases':140,'states_per_hypothesis':560,'summary':summary,
      'limits':['Clinical labels are used only to judge hypothetical predictions retrospectively. Label-assisted picks are not executable policies.',
        'No outer patient enters this diagnostic or the source fitting; patient-fold repetitions are correlated.',
        'This checks the frozen hypothesis mechanism, not multi-agent collaboration or an untouched external cohort.']}
    write_json(output/'hypothesis_diagnostic.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();diagnose(a.project,a.output)
