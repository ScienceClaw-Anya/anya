"""Reconstruct physical query-value features, labels, decisions and metrics."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from experiments.spes_agent_v3.common import read_json,sha,check_groups
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.mapping import physical_input
from .tools import predict,choose
from .audit_features import audit_table as table


def audit(project,output,stage):
    project,output=Path(project),Path(output); p=read_json(output/'protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()): raise ValueError('Frozen source differs')
    row_count=0; state_count=0; trajectory_count=0; decision_count=0
    outer_jobs=read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json')['jobs']
    models={}; evaluation=read_json(output/'evaluation_protocol.json') if stage=='full' else None
    if evaluation:
        if evaluation['protocol_sha256']!=sha(output/'protocol.json'): raise ValueError('Evaluation protocol differs')
        for outer in range(5):
            path=output/f'learners/outer{outer}/models.joblib'; receipt=read_json(path.with_suffix('.json'))
            expected={s for j in p['jobs'] if j['outer']==outer for s in j['predict']}
            if set(receipt['fit_patients'])!=expected or len(expected)!=28 or expected&set(receipt['outer_test']): raise ValueError('Value learner patient isolation differs')
            if sha(path)!=receipt['models_sha256'] or receipt['models_sha256']!=evaluation['models_sha256'][str(outer)] or receipt['protocol_sha256']!=sha(output/'protocol.json'):
                raise ValueError('Value model provenance differs')
            for s,h in receipt['bank_sha256'].items():
                if sha(output/f'banks/outer{outer}/{s}.npz')!=h: raise ValueError('Value model bank changed')
            models[outer]=joblib.load(path)
    for subject in sorted({s for j in p['jobs'] for s in j['predict']}):
        source=project/'outputs/spes_physical/maps'/subject; mapping=read_json(source/'receipt.json'); index=load_array(source/'index.npz')
        raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
        if sha(source/'receipt.json')!=p['mapping_receipt_sha256'][subject] or sha(source/'index.npz')!=mapping['index_sha256'] or sha(raw_path)!=mapping['raw_sha256']:
            raise ValueError('Physical map/source differs')
        raw=load_array(raw_path)[:,:20]; labels=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
        coords=load_array(project/f'vendor/norris2026_hat/data/coords_{subject}.npz'); centers=np.array([r['center_mm'] for r in mapping['pair_rows']])
        distance=np.where(index>=0,cdist(coords,centers),0.); kernel=np.where(index>=0,np.exp(-distance/40),0.)
        available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
        for j in [j for j in p['jobs'] if subject in j['predict']]:
            check_groups(j['train'],j['early_stop'],j['predict'],j['outer_test'])
            key=f'outer{j["outer"]}_inner{j["inner"]}'; source_model=project/'outputs/spes_agent_v8/crossfit'/key; model_receipt=read_json(source_model/'receipt.json')
            if sha(source_model/'checkpoint.pth')!=p['source_HAT_sha256'][key] or any(model_receipt[k]!=j[k] for k in ['train','early_stop','predict','outer_test']): raise ValueError('Source HAT provenance differs')
            path=output/f'banks/outer{j["outer"]}/{subject}.npz'; receipt=read_json(path.with_suffix('.json'))
            if sha(path)!=receipt['file_sha256'] or receipt['protocol_sha256']!=sha(output/'protocol.json') or receipt['source_action_kind']!='verified_anatomical_pairs': raise ValueError('Wrong physical value bank')
            with np.load(path) as d: bank={k:d[k] for k in d.files}
            np.testing.assert_array_equal(bank['label'],labels)
            sequence=fixed_queries(available,j['outer'],int(subject[-2:]),0)
            if sequence!=receipt['physical_random0_sequence'] or len(bank['gain'])!=24: raise ValueError('Source sequence differs')
            old=project/f'outputs/spes_agent_v12/banks/outer{j["outer"]}/{subject}.npz'
            if sha(old)!=p['bank_sha256'][f'outer{j["outer"]}/{subject}']: raise ValueError('Audited source scores differ')
            with np.load(old) as d: before={k:d['HAT_score'][d['state']==k] for k in [4,5,6,7,8]}
            for k in p['prefix_sizes']:
                selected=sequence[:k]; prefix=selected+[-1]*(7-k); rows=np.flatnonzero((bank['prefix']==np.array(prefix)).all(axis=1))
                if len(rows)!=6: raise ValueError('Candidate state row count differs')
                candidates=list(map(int,bank['candidate'][rows])); remaining=[q for q in available if q not in selected]
                current=kernel[:,selected].max(axis=1); rank=rankdata(before[k])/(len(labels)+1); gains=[np.maximum(kernel[:,q]-current,0) for q in remaining]
                coverage=remaining[int(np.argmax([g.mean() for g in gains]))]; focus=remaining[int(np.argmax([np.average(g,weights=.25+rank) for g in gains]))]
                expected=list(dict.fromkeys([sequence[k],coverage,focus])); pool=[q for q in remaining if q not in expected]
                rng=np.random.default_rng(140000+j['outer']*10000+int(subject[-2:])*100+k)
                expected+=list(map(int,rng.choice(pool,6-len(expected),replace=False)))
                if expected!=candidates: raise ValueError('Training slate selected using unexpected evidence')
                observed=physical_input(raw,index,selected); features=table(distance,observed,before[k],selected,candidates)
                np.testing.assert_allclose(features,bank['features'][rows],rtol=0,atol=1e-12)
                for row in rows:
                    np.testing.assert_array_equal(bank['before_scores'][row],before[k])
                    if bank['candidate'][row]==sequence[k]: np.testing.assert_array_equal(bank['after_scores'][row],before[k+1])
                    gain=roc_auc_score(labels,bank['after_scores'][row])-roc_auc_score(labels,bank['before_scores'][row])
                    np.testing.assert_allclose(gain,bank['gain'][row],rtol=0,atol=1e-12); row_count+=1
                state_count+=1
        if not evaluation: continue
        case=next(c for c in p['cases'] if c['subject']==subject); outer=case['outer']
        if subject not in outer_jobs[outer]['predict'] or subject in outer_jobs[outer]['train']+outer_jobs[outer]['early_stop']: raise ValueError('Outer scorer patient leak')
        for name in ['baseline','mte','hat']:
            if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{outer}/{name}/checkpoint.pth')!=p['model_sha256'][f'{outer}/{name}']: raise ValueError('Outer HAT changed')
        prefix=fixed_queries(available,outer,int(subject[-2:]),0)[:4]
        for policy in p['policies']:
            folder=output/'evaluation'/case['case_id']/policy; result=read_json(folder/'metrics.json'); queries=[int(q[1:]) for q in result['queries']]
            if result['evaluation_protocol_sha256']!=sha(output/'evaluation_protocol.json') or len(set(queries))!=8 or queries[:4]!=prefix or any(q not in available for q in queries): raise ValueError('Acquisition provenance/budget differs')
            np.testing.assert_allclose(roc_auc_score(labels,result['prediction']),result['AUROC'],rtol=0,atol=1e-12)
            for k,q in enumerate(queries):
                pair=mapping['pair_rows'][q]; obs=read_json(folder/f'{k:02d}_observation.json')
                if result['query_contacts'][k]!=pair['contacts'] or obs['anatomical_pair']!=pair['id'] or obs['contacts']!=pair['contacts']:
                    raise ValueError('Physical stimulation pair identity differs')
                if obs['revealed_recording_contacts']!=int((index[:,q]>=0).sum()): raise ValueError('Recording availability differs')
                if k<4: continue
                trace=read_json(folder/f'{k:02d}_choice.json'); remaining=[v for v in available if v not in queries[:k]]; ids=[mapping['pair_rows'][v]['id'] for v in remaining]
                if trace['ids']!=ids or trace['selected']!=result['queries'][:k]: raise ValueError('Policy exposed candidate/history differs')
                observed=physical_input(raw,index,queries[:k]); x=table(distance,observed,np.array(trace['current_HAT_score']),queries[:k],remaining)
                np.testing.assert_allclose(x,trace['features'],rtol=0,atol=1e-12); values=predict(models[outer],x)
                for n,v in values.items(): np.testing.assert_allclose(v,trace['values'][n],rtol=0,atol=1e-12)
                if trace['choice']!=choose(ids,values[policy]) or trace['choice']!=pair['id']: raise ValueError('Value policy decision differs')
                decision_count+=1
            trajectory_count+=1
        print('V14_AUDIT_PATIENT',subject,flush=True)
    value={'status':'verified','stage':stage,'training_rows_reconstructed':row_count,'source_states':state_count,
           'physical_trajectories':trajectory_count,'adaptive_choices_replayed':decision_count,'new_API_calls':0,
           'checks':['Patient/model/map/raw/bank/protocol hashes and source/outer patient exclusion.',
                     'Every training candidate slate and observed-only feature row reconstructed.',
                     'Every gain recomputed using independent sklearn AUROC; random0 next scores match audited physical banks.',
                     'For full audit, every runtime feature, expert prediction, decision, identity and final AUROC replayed.'],
           'limits':['New alternative-response and runtime neural scores are audited from stored predictions, not all re-inferred.',
                     'Pure feature/model prediction recipes are replayed; this does not establish causal or external validity.']}
    write_json(output/f'audit_{stage}.json',value); print('V14_AUDIT',value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['banks','full'],default='full'); a=parser.parse_args(); audit(a.project,a.output,a.stage)
