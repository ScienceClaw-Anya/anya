"""Two verified physical-pair observations, without a localization-effect claim."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v8.crossfit import write_json
from .environment import PhysicalAcquisition


def smoke(project,output):
    import torch
    project,output=Path(project),Path(output);torch.set_num_threads(2);case=read_json(project/'outputs/spes_agent_v4/active/active_protocol.json')['cases'][0]
    path=output/'two_query_smoke.json'
    if path.exists():raise ValueError('Smoke already recorded; do not overwrite')
    write_json(output/'smoke_protocol.json',{'case':case,'measurements':2,'new_API_calls':0,'score':'No efficacy/localization score after two measurements.',
      'sources':{f:sha(project/f) for f in ['experiments/spes_physical/environment.py','experiments/spes_physical/mapping.py','experiments/spes_physical/smoke.py']},
      'mapping_sha256':sha(project/'outputs/spes_physical/maps'/case['subject']/'receipt.json')})
    env=PhysicalAcquisition(project,case['outer'],case['subject']);before=env.public();chosen=env.available[:2];observations=[env.observe(q) for q in chosen]
    try:env.final_judge()
    except ValueError:score_guard=True
    else:raise ValueError('Judge exposed scores before full budget')
    result={'status':'two_physical_queries_verified','case':case,'observations':observations,'actual_anatomical_pairs':len(chosen),
      'distinct_anatomical_contact_pairs':len({tuple(env.pairs[q]['contacts']) for q in chosen}),
      'different_source_columns_per_query':[len(set(map(int,env.index[env.index[:,q]>=0,q]))) for q in chosen],
      'localization_score_before_eight_queries_blocked':score_guard,'new_API_calls':0,'no_treatment_claim':True,
      'public_roots':sorted(before['evidence'])}
    write_json(path,result);write_json(project/'examples/spes_physical_smoke_summary.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();smoke(a.project,a.output)
