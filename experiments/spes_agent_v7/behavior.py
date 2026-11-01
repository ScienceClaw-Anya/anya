"""Describe executed choices without attributing untested causal mechanisms."""
from collections import Counter
from pathlib import Path
import argparse
import numpy as np
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v4.acquisition import numeric_choice
from .experiment import write_json,LLM

def run(project):
    root=Path(project)/'outputs/spes_agent_v7';choices=Counter();overrides=Counter();finals=[]
    for path in root.glob('trajectories/*/*/metrics.json'):
        metric=read_json(path)
        if metric['policy'] not in LLM:continue
        folder=path.parent;coverage=read_json(folder.parent/'coverage/metrics.json')
        finals.append({'case_id':metric['case_id'],'policy':metric['policy'],
          'same_query_set_as_coverage':set(metric['queries'])==set(coverage['queries']),
          'same_AUROC_as_coverage':abs(metric['AUROC']-coverage['AUROC'])<1e-12,
          'same_ordered_ranking_as_coverage':bool(np.array_equal(np.argsort(metric['prediction'],kind='stable'),np.argsort(coverage['prediction'],kind='stable'))),
          'maximum_score_difference':float(np.max(np.abs(np.array(metric['prediction'])-coverage['prediction'])))})
        for step in range(8):
            decision=read_json(folder/f'{step:02d}_decision.json');public=read_json(folder/f'{step:02d}_public.json')
            is_coverage=decision['next_query']==numeric_choice(public,'coverage');choices[f'{decision["stage"]}_coverage_{is_coverage}']+=1
            if step%2:
                proposal=read_json(folder/f'{step-1:02d}_decision.json');changed=decision['next_query']!=proposal['next_query']
                overrides['changed_proposal' if changed else 'kept_proposal']+=1
                if changed and is_coverage:overrides['changed_to_maximum_geometric_coverage']+=1
    result={'scope':'Post-hoc behavioral description; no causal attribution from five patients',
      'choice_counts':dict(choices),'review_counts':dict(overrides),'completed_LLM_trajectories':len(finals),
      'same_query_set_as_coverage':sum(r['same_query_set_as_coverage'] for r in finals),
      'same_AUROC_as_coverage':sum(r['same_AUROC_as_coverage'] for r in finals),
      'same_ranking_as_coverage':sum(r['same_ordered_ranking_as_coverage'] for r in finals),'rows':finals,
      'interpretation':'Roles repeatedly selected a shared geometric heuristic. This does not demonstrate added localization information or clinical efficacy. A trained query-value tool needs separate patient-isolated evaluation.'}
    write_json(root/'behavior.json',result);print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();run(a.project)
