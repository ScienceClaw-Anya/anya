"""Descriptive collaboration behavior; never changes a frozen decision."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def describe(output):
    output=Path(output); p=read_json(output/'protocol.json'); summary={}
    for policy in p['policies']:
        rounds=0; agree=0; overlap=[]; chosen_first=0; chosen_neither=0; leaders={'coverage':0,'score_focus':0,'uncertainty':0}; proposals_by_slot={str(i):{n:0 for n in leaders} for i in [0,1]}; selected_conditions={}; carry=total_carry=0
        for c in [c for c in p['cases'] if c['pilot']]:
            folder=output/'trajectories'/c['case_id']/policy
            if not (folder/'metrics.json').exists(): raise ValueError('Only a complete pilot can be described')
            previous=None
            for r in range(4):
                public=read_json(folder/f'{r:02d}_public.json'); rows=public['evidence']['geometry']['candidates']; joint=read_json(folder/f'{r:02d}_merge.json')
                proposals=[read_json(folder/f'{r:02d}_{slot}_decision.json') for slot in [0,1]]; a,b=[row['ranked_queries'] for row in proposals]; rounds+=1
                agree+=a[0]==b[0]; overlap.append(len(set(a)&set(b))/len(set(a)|set(b)))
                chosen_first+=joint['choice'] in [a[0],b[0]]; chosen_neither+=joint['choice'] not in [a[0],b[0]]
                for name,key in [('coverage','coverage_gain'),('score_focus','score_focus_gain'),('uncertainty','uncertainty_focus_gain')]:
                    q=min(rows,key=lambda row:(-row[key],row['id']))['id']; leaders[name]+=joint['choice']==q
                    for slot in [0,1]: proposals_by_slot[str(slot)][name]+=proposals[slot]['ranked_queries'][0]==q
                selected=next(row for row in rows if row['id']==joint['choice']); condition=public['evidence']['geometry']['recorded_conditions'][selected['recorded_condition_id']]
                key=str(condition); selected_conditions[key]=selected_conditions.get(key,0)+1
                if previous is not None:
                    available={row['id'] for row in rows}
                    for slot in [0,1]:
                        prior=set(previous[slot]['ranked_queries'])&available
                        if prior: total_carry+=1; carry+=proposals[slot]['ranked_queries'][0] in prior
                previous=proposals
        summary[policy]={'adaptive_rounds':rounds,'same_independent_top_choice':int(agree),'mean_top3_Jaccard':float(np.mean(overlap)),
           'merged_choice_is_a_top1':int(chosen_first),'merged_choice_is_neither_top1':int(chosen_neither),
           'merged_choice_matches_numeric_leader':{k:int(v) for k,v in leaders.items()},
           'proposal_top1_matches_numeric_leader_by_slot':{s:{k:int(v) for k,v in row.items()} for s,row in proposals_by_slot.items()},
           'next_top1_carries_unexecuted_previous_top3':int(carry),'eligible_carry_comparisons':int(total_carry),
           'adaptive_recorded_conditions':selected_conditions}
    result={'scope':'Posthoc descriptive pilot behavior; no new model call or policy selection.','policies':summary,
            'interpretation_limits':['Disagreement, repeated choices or rationale changes do not establish useful reflection.','Numeric-leader matching is descriptive, not proof a heuristic is optimal.',
                                     'Same response-group count does not equate current/charge; recorded conditions are reported separately.']}
    write_json(output/'behavior.json',result); print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); a=parser.parse_args(); describe(a.output)
