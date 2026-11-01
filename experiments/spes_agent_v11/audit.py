"""Independent physical-action identity, decision, budget and outcome checks."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v4.acquisition import numeric_choice
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json


def audit(project,output):
    project,output=Path(project),Path(output);p=read_json(output/'numeric_protocol.json');count=0;different_columns=0
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()):raise ValueError('Frozen source differs')
    for case in p['cases']:
        neural=read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json')['jobs'][case['outer']]
        if case['subject'] not in neural['predict'] or case['subject'] in neural['train']+neural['early_stop']:raise ValueError('Scorer patient isolation differs')
        for family in ['baseline','mte','hat']:
            path=project/f'outputs/spes_agent_v4/multitrial/neural/outer{case["outer"]}/{family}/checkpoint.pth'
            if sha(path)!=p['model_sha256'][f'{case["outer"]}/{family}']:raise ValueError('Scorer weights differ')
        folder=project/'outputs/spes_physical/maps'/case['subject'];mapping=read_json(folder/'receipt.json');index=load_array(folder/'index.npz');patient=int(case['subject'][-2:])
        if sha(folder/'receipt.json')!=p['mapping_receipt_sha256'][case['subject']] or sha(folder/'index.npz')!=mapping['index_sha256']:raise ValueError('Physical mapping changed')
        available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1];prefix=fixed_queries(available,case['outer'],patient,0)[:4]
        labels=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz')
        for policy in p['policies']:
            path=output/'numeric'/case['case_id']/policy;row=read_json(path/'metrics.json');queries=[int(q[1:]) for q in row['queries']]
            if len(queries)!=8 or len(set(queries))!=8 or queries[:4]!=prefix or any(q not in available for q in queries) or row['actual_physical_measurements']!=8 or row['new_API_calls']!=0:raise ValueError('Physical-query budget/availability/prefix differs')
            if row['protocol_sha256']!=sha(output/'numeric_protocol.json') or abs(independent_auc(labels,row['prediction'])-row['AUROC'])>1e-12:raise ValueError('Outcome/protocol differs')
            seed=6190+case['outer']*100+patient+(10000 if policy=='random_1' else 0)
            for step,q in enumerate(queries):
                pair=mapping['pair_rows'][q];obs=read_json(path/f'{step:02d}_observation.json')
                if row['query_contacts'][step]!=pair['contacts'] or obs['contacts']!=pair['contacts'] or obs['anatomical_pair']!=pair['id'] or obs['measurement']!=step+1:raise ValueError('Recorded physical identity differs')
                if obs['revealed_recording_contacts']!=int((index[:,q]>=0).sum()):raise ValueError('Response-map coverage differs')
                different_columns+=len(set(map(int,index[index[:,q]>=0,q])))>1
                if step>=4:
                    trace=read_json(path/f'{step:02d}_choice.json');public={'evidence':{'geometry':trace['candidate_geometry']}}
                    expected=numeric_choice(public,policy,np.random.default_rng(seed+step*100003))
                    if trace['choice']!=pair['id'] or expected!=pair['id']:raise ValueError('Fixed numerical policy decision differs')
                    if trace['used_physical_pairs']!=row['queries'][:step]:raise ValueError('Decision history differs')
            count+=1
    r={'status':'verified','patients':35,'physical_trajectories':count,'actual_physical_queries':count*8,
      'queries_requiring_different_compact_columns_across_contacts':different_columns,'new_API_calls':0,
      'checks':['Frozen source/model mapping lineage; anatomical-pair identity; unambiguous recorded availability; common four-query prefixes.',
        'Eight unique real pair IDs per trajectory, exact fixed numerical choices and observation identities.',
        'Every final AUROC independently recomputed by pairwise comparison.'],
      'limits':['GPU outputs are audited from recorded predictions, not all re-inferred.','First20trials; retrospective preprocessing availability; no physical treatment or prospective safety.']}
    write_json(output/'audit.json',r);print(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.project,a.output)
