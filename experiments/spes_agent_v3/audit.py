"""Independent replay of cross-fit provenance, fitted coefficients and policy budgets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from .common import SLUGS, SELECTOR, WEIGHTS, check_groups, read_json, sha, source_hashes, write_json
from .crossfit import TRAIN_SOURCES
from experiments.spes_agent_v2.verify import auc, read_native

def independent_features(x):
    x=np.asarray(x,dtype=float)
    k=min(10,len(x));t=np.argsort(x[:,2],kind='stable')[-k:];c=np.argsort(x[:,1],kind='stable')[-k:]
    return np.array([np.std(x[:,2]),np.mean(np.abs(x[:,1]-x[:,2])),np.mean(np.abs(x[:,0]-x[:,2])),len(set(t)&set(c))/k])

def audit_preparation(project,output):
    project,output=Path(project),Path(output);protocol=read_json(output/'training_protocol.json');prepared=read_json(output/'prepared.json')
    assert source_hashes(project,TRAIN_SOURCES)==protocol['sources']
    original_protocol=read_json(project/'outputs/reproduction/norris2024/training/protocol.json')
    assert protocol['upstream_sources']==original_protocol['source_sha256']
    for name,digest in protocol['upstream_sources'].items():assert sha(project/'vendor/norris2024/src'/name)==digest
    selector_protocol=read_json(output/'selector_protocol.json')
    assert source_hashes(project,list(selector_protocol['sources']))==selector_protocol['sources']
    fits=0;distinct=0;gate_counts={};records=[]
    for outer in range(5):
        chunks=[];truth=[];people=[]
        for row in [r for r in protocol['jobs'] if r['outer']==outer]:
            check_groups(row['train'],row['early_stop'],row['predict'],row['outer_test'])
            folder=output/f"crossfit/outer{outer}_inner{row['inner']}";values=[]
            for slug in SLUGS:
                receipt=read_json(folder/slug/'receipt.json')
                assert receipt['outer']==outer and receipt['inner']==row['inner']
                for key in ['train','early_stop','predict','outer_test']:assert receipt[key]==row[key]
                assert sha(folder/slug/'predictions.npz')==receipt['prediction_sha256']
                assert sha(folder/slug/'checkpoint.pth')==receipt['checkpoint_sha256']
                data=np.load(folder/slug/'predictions.npz');values.append(data);fits+=1
                assert set(data['patient'])=={int(p[-2:]) for p in row['predict']}
                for patient in set(data['patient']):
                    y=np.load(project/f'vendor/norris2024/data/main/y_ccepAgeUMCU{int(patient):02d}.npy')
                    assert np.array_equal(y,data['y'][data['patient']==patient])
            for value in values[1:]:
                assert np.array_equal(value['patient'],values[0]['patient']) and np.array_equal(value['y'],values[0]['y'])
            chunks.append(np.column_stack([v['prediction'] for v in values]));truth.append(values[0]['y']);people.append(values[0]['patient'])
        x,y,p=np.vstack(chunks),np.concatenate(truth),np.concatenate(people)
        assert len(set(p))==28
        target=output/f'selectors/outer{outer}';receipt=read_json(target/'receipt.json');fitted=joblib.load(target/'selector.joblib')
        assert sha(target/'selector.joblib')==receipt['model_sha256'] and sha(target/'OOF.npz')==receipt['OOF_sha256']
        assert set(fitted['training_patients'])==set(p) and not set(p)&{int(n[-2:]) for n in receipt['outer_test']}
        saved=np.load(target/'OOF.npz')
        assert np.array_equal(saved['predictions'],x) and np.array_equal(saved['y'],y) and np.array_equal(saved['patient'],p)
        z=[];rewards=[];alternative=list(WEIGHTS)[1:]
        for patient in sorted(set(p)):
            mask=p==patient;sample=x[mask];z.append(independent_features(sample));baseline=auc(y[mask],sample[:,2])
            rewards.append([auc(y[mask],sample@np.array(WEIGHTS[name]))-baseline for name in alternative])
        z,rewards=np.array(z),np.array(rewards);mean=z.mean(axis=0);scale=z.std(axis=0);scale[scale==0]=1
        zscale=(z-mean)/scale;intercept=rewards.mean(axis=0)
        coefficient=np.linalg.solve(zscale.T@zscale+SELECTOR['ridge_alpha']*np.eye(4),zscale.T@(rewards-intercept))
        assert np.allclose(fitted['model'].named_steps['scale'].mean_,mean,atol=1e-12)
        assert np.allclose(fitted['model'].named_steps['scale'].scale_,scale,atol=1e-12)
        assert np.allclose(fitted['model'].named_steps['ridge'].coef_,coefficient.T,atol=1e-10)
        assert np.allclose(fitted['model'].named_steps['ridge'].intercept_,intercept,atol=1e-10)
        draw=np.random.default_rng(SELECTOR['bootstrap_seed']).integers(0,28,(SELECTOR['bootstrap_draws'],28))
        lower=np.quantile(rewards[draw].mean(axis=1),SELECTOR['family_alpha']/3,axis=0)
        eligible={name:bool(lower[i]>SELECTOR['minimum_gain']) for i,name in enumerate(alternative)}
        for i,name in enumerate(alternative):
            assert abs(lower[i]-fitted['evidence'][name]['bootstrap_lower_bound'])<1e-12
            assert eligible[name]==fitted['evidence'][name]['eligible']
        gate_counts[outer]=eligible
        old=project/'outputs/reproduction/norris2024/training'
        assert sha(old/f'split_seed1_fold{outer}.json')==receipt['outer_split_sha256']
        split=read_json(old/f'split_seed1_fold{outer}.json');assert set(split['train']+split['val'])=={f'ccepAgeUMCU{n:02d}' for n in p}
        originals=[]
        for slug in SLUGS:
            assert sha(old/f'{slug}_seed1_fold{outer}/predictions.npz')==receipt['outer_prediction_sha256'][slug]
            assert sha(old/f'{slug}_seed1_fold{outer}/checkpoint.pth')==receipt['outer_checkpoint_sha256'][slug]
            originals.append(np.load(old/f'{slug}_seed1_fold{outer}/predictions.npz'))
        base=originals[0];pred=np.column_stack([v['prediction'] for v in originals])
        for row in [r for r in prepared['cases'] if r['outer']==outer]:
            mask=base['patient']==row['patient'];sample=pred[mask];feat=independent_features(sample)
            expected=(feat-mean)/scale@coefficient+intercept
            choices=[(float(expected[i]),name) for i,name in enumerate(alternative) if eligible[name] and expected[i]>SELECTOR['minimum_gain']]
            selected=max(choices)[1] if choices else 'transformer'
            assert selected==row['selector_candidate']
            case=output/'cases'/row['case_id'];bank=np.load(case/'scores.npz');scores=bank['scores'];assert bank.files==['scores']
            assert np.array_equal(scores[:,0],sample[:,2])
            assert np.allclose(scores[:,1],sample@np.array(WEIGHTS[selected]),atol=1e-14)
            public=read_json(case/'public.json');judge=read_json(case/'judge.json');tools=read_json(case/'tools.json')
            assert public['default_action']=='transformer' and public['allowed_actions']==['transformer','conservative_selector']
            assert set(tools)=={'baseline','selector','validation','distribution'}
            assert np.allclose([c['scorers'] for c in public['contacts']],np.round(scores,6),atol=1e-14)
            assert judge['y']==base['y'][mask].astype(int).tolist()
            for name in ['patient','y','test_accuracy','prevalence']:assert '"'+name+'"' not in json.dumps(public)
            distinct+=int(np.max(np.abs(scores[:,1]-scores[:,0]))>1e-12)
            records.append({'case_id':row['case_id'],'patient':row['patient'],'transformer_AUROC':auc(judge['y'],scores[:,0]),
                'selector_AUROC':auc(judge['y'],scores[:,1])})
    result={'verified':True,'patients':35,'new_inner_fits':fits,'selector_fits':5,'distinct_action_patients':distinct,
        'gates':gate_counts,'records':records,
        'verification_scope':['All inner training/early-stop/OOF/outer-test disjointness and checkpoint/prediction hashes',
            'Released-label row alignment and actual patient IDs','Manual ridge normal-equation coefficient replay',
            'Independent pairwise AUROC and eligibility bootstrap','Exact full-precision scorer execution and public observation gate'],
        'limitations':['Neural predictions are frozen captured outputs; CNN evaluation includes stochastic channel sampling, and this audit does not claim bit-exact new neural inference replay']}
    write_json(output/'preparation_audit.json',result)
    return {k:v for k,v in result.items() if k!='records'}

def audit_policies(project,output,allow_incomplete=False):
    from .engine import POLICIES, visible_indices
    project,output=Path(project),Path(output);protocol=read_json(output/'agent_protocol.json')
    assert source_hashes(project,list(protocol['source_sha256_LF']))==protocol['source_sha256_LF']
    for path,digest in protocol['input_sha256'].items():assert sha(output/path)==digest
    assert sha(output/'selector_protocol.json')==protocol['selector_protocol_sha256'] and sha(output/'prepared.json')==protocol['prepared_sha256']
    completed=list((output/'policies').glob('*/*/metrics.json'))
    if not allow_incomplete and len(completed)!=210:raise ValueError('Full policy cohort incomplete')
    api=accepted=tools_count=artifacts=0
    for path in completed:
        folder=path.parent;case_id=folder.parent.name;policy=folder.name;case=output/'cases'/case_id
        public=read_json(case/'public.json');judge=read_json(case/'judge.json');matrix=np.load(case/'scores.npz')['scores']
        native=read_native(folder/'scienceclaw');ids=read_json(folder/'native_ids.json')
        assert len(native)==17 and len(ids['states'])==8 and len(ids['tools'])==8
        assert native[ids['observation']]['payload']==public
        states=[read_json(folder/f'state_{i:02d}.json') for i in range(8)]
        numeric=policy.startswith('numeric_')
        for i,state in enumerate(states):
            assert state['action'] in public['allowed_actions']
            assert native[ids['states'][i]]['payload']==state
            expected=[ids['observation'],ids['tools'][i]]+[ids['states'][j] for j in visible_indices(policy,i)]
            assert native[ids['states'][i]]['parent_artifact_ids']==expected
            tool=read_json(folder/f'tool_{i:02d}.json');assert tool['source_sha256']==sha(case/'tools.json')
            assert native[ids['tools'][i]]['payload']==tool;tools_count+=1
            if numeric:
                assert not (folder/f'usage_{i:02d}.json').exists()
                assert state['action']==('transformer' if policy=='numeric_transformer' else 'conservative_selector')
            else:
                prompt=read_json(folder/f'request_{i:02d}.json');usage=read_json(folder/f'usage_{i:02d}.json')
                assert {k:prompt[k] for k in public}==public
                assert [r['step'] for r in prompt['shared_board']]==[j+1 for j in visible_indices(policy,i)]
                assert usage['prompt_sha256']==hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()
                api+=1;accepted+=state['accepted_output']
                if not state['accepted_output']:assert state['action']=='transformer'
        prediction=np.mean([matrix[:,0 if s['action']=='transformer' else 1] for s in states[-4:]],axis=0)
        metric=read_json(path)
        assert np.array_equal(prediction,metric['prediction']) and abs(auc(judge['y'],prediction)-metric['AUROC'])<1e-12
        artifacts+=17
    result={'verified':True,'completed_policy_cases':len(completed),'expected_policy_cases':210,'full_cohort_complete':len(completed)==210,
        'model_attempts_completed_cases':api,'accepted_model_outputs_completed_cases':accepted,'native_tool_queries':tools_count,'native_artifacts':artifacts}
    write_json(output/'policy_audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policies',action='store_true');p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    result=audit_policies(a.project,a.output,a.allow_incomplete) if a.policies else audit_preparation(a.project,a.output)
    print(json.dumps(result,indent=2))
