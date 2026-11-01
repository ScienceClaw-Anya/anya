"""Patient-isolation and exact feature/prediction replay for physical tools."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,check_groups
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v10.features import contact_features
from experiments.spes_physical.mapping import physical_input


def audit(project,output):
    project,output=Path(project),Path(output);p=read_json(output/'protocol.json');evaluation=read_json(output/'evaluation_protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()) or evaluation['protocol_sha256']!=sha(output/'protocol.json'):raise ValueError('Frozen source/protocol differs')
    learners={};banks={};feature_rows=0
    for outer in range(5):
        folder=output/f'learners/outer{outer}';r=read_json(folder/'receipt.json');jobs=[j for j in p['jobs'] if j['outer']==outer];expected={int(s[-2:]) for j in jobs for s in j['predict']}
        if set(r['fit_patients'])!=expected or len(expected)!=28 or expected&{int(s[-2:]) for s in jobs[0]['outer_test']}:raise ValueError('Contact learner patient leak')
        if r['protocol_sha256']!=sha(output/'protocol.json') or sha(folder/'models.joblib')!=r['models_sha256'] or r['models_sha256']!=evaluation['models_sha256'][str(outer)]:raise ValueError('Learner provenance differs')
        for subject,h in r['input_sha256'].items():
            path=output/f'banks/outer{outer}/{subject}.npz'
            if sha(path)!=h:raise ValueError('Learner input changed')
        learners[outer]=joblib.load(folder/'models.joblib')
        for j in jobs:
            check_groups(j['train'],j['early_stop'],j['predict'],j['outer_test']);key=f'outer{outer}_inner{j["inner"]}';source=project/'outputs/spes_agent_v8/crossfit'/key;r=read_json(source/'receipt.json')
            if any(r[k]!=j[k] for k in ['train','early_stop','predict','outer_test']) or sha(source/'checkpoint.pth')!=p['source_HAT_sha256'][key]:raise ValueError('Source model lineage differs')
            for subject in j['predict']:banks.setdefault(subject,[]).append(j)
    cases={c['subject']:c for c in p['cases']};test_rows=0
    for subject,jobs in sorted(banks.items()):
        folder=project/'outputs/spes_physical/maps'/subject;mapping=read_json(folder/'receipt.json');index=load_array(folder/'index.npz');raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
        if sha(folder/'receipt.json')!=p['mapping_receipt_sha256'][subject] or sha(raw_path)!=mapping['raw_sha256'] or sha(folder/'index.npz')!=mapping['index_sha256']:raise ValueError('Mapping or waveform source differs')
        raw=load_array(raw_path)[:,:20];y=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz');available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
        for j in jobs:
            path=output/f'banks/outer{j["outer"]}/{subject}.npz';r=read_json(path.with_suffix('.json'))
            if r['source_action_kind']!='verified_anatomical_pairs' or r['protocol_sha256']!=sha(output/'protocol.json') or sha(path)!=r['file_sha256']:raise ValueError('Wrong action-grain bank')
            sequence=fixed_queries(available,j['outer'],int(subject[-2:]),0)
            if r['physical_queries']!=sequence:raise ValueError('Source physical query sequence differs')
            with np.load(path) as d:x,label,state,score=d['features'],d['label'],d['state'],d['HAT_score']
            for k in p['states']:
                mask=state==k;np.testing.assert_array_equal(label[mask],y);observed=physical_input(raw,index,sequence[:k])
                expected=contact_features(observed[...,-512:],observed[...,0],score[mask]);np.testing.assert_allclose(expected,x[mask],rtol=0,atol=1e-12);feature_rows+=len(expected)
        case=cases[subject];f=output/'evaluation'/case['case_id'];r=read_json(f/'metrics.json')
        if r['protocol_sha256']!=sha(output/'evaluation_protocol.json') or sha(f/'predictions.npz')!=r['prediction_sha256']:raise ValueError('Test prediction provenance differs')
        old=read_json(project/'outputs/spes_agent_v11/numeric'/case['case_id']/'random_0/metrics.json');sequence=[int(q[1:]) for q in old['queries']]
        with np.load(f/'predictions.npz') as d:prediction={k:d[k] for k in d.files}
        np.testing.assert_array_equal(prediction['label'],y);np.testing.assert_allclose(prediction['8/hat'],old['prediction'],rtol=0,atol=1e-7)
        for k in [4,8]:
            observed=physical_input(raw,index,sequence[:k]);x=contact_features(observed[...,-512:],observed[...,0],prediction[f'{k}/hat'])
            for name in ['logistic','boosting']:np.testing.assert_allclose(learners[case['outer']][name].predict_proba(x)[:,1],prediction[f'{k}/{name}'],rtol=0,atol=1e-12)
        for row in r['rows']:
            scores=prediction[f'{row["measurements"]}/{row["scorer"]}']
            if abs(independent_auc(y,scores)-row['AUROC'])>1e-12:raise ValueError('Independent AUROC differs')
            if row['Brier'] is not None and abs(float(np.mean((y-scores)**2))-row['Brier'])>1e-12:raise ValueError('Independent Brier differs')
            test_rows+=1
        print('PHYSICAL_POSTERIOR_AUDIT',subject,flush=True)
    result={'status':'verified','source_models_reused':20,'new_small_models':10,'training_feature_rows_reconstructed':feature_rows,'outer_patients':35,'scorer_state_results':test_rows,'new_API_calls':0,
      'checks':['Patient/source-model isolation and all map/model/bank/protocol hashes.','All physical-state training features reconstructed, labels checked against released references.',
        'Both fitted contact predictors replayed at four/eight physical measurements; every AUROC/Brier independently recomputed; HAT matches v11.'],
      'limits':['Recorded source/outer HAT outputs were not all re-inferred in this audit.','Repeated-cohort exploration and correlated state rows do not provide independent clinical validation.']}
    write_json(output/'audit.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.project,a.output)
