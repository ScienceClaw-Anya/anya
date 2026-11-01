"""Independent provenance, target arithmetic and outcome audit (no API calls)."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,check_groups
from experiments.spes_agent_v4.neural import load_array
from .crossfit import write_json
from .features import distance_catalog,feature_vector


def independent_auc(y,s):
    y=np.asarray(y);s=np.asarray(s)
    positive=s[y==1,None];negative=s[y==0][None,:]
    return float(((positive>negative)+.5*(positive==negative)).mean())


def audit(project,output):
    project,output=Path(project),Path(output)
    source=read_json(output/'crossfit_protocol.json');query=read_json(output/'query_protocol.json');numeric=read_json(output/'numeric_protocol.json')
    for manifest,key in [(source,'sources'),(query,'sources'),(numeric,'source_sha256')]:
        if any(sha(project/p)!=h for p,h in manifest[key].items()):raise ValueError('Frozen source hash differs')
    if query['crossfit_protocol_sha256']!=sha(output/'crossfit_protocol.json') or numeric['query_protocol_sha256']!=sha(output/'query_protocol.json'):raise ValueError('Protocol lineage changed')
    target_rows=feature_checks=0
    for job in source['jobs']:
        check_groups(job['train'],job['early_stop'],job['predict'],job['outer_test'])
        folder=output/f'crossfit/outer{job["outer"]}_inner{job["inner"]}'
        r=read_json(folder/'receipt.json')
        if r['protocol_sha256']!=sha(output/'crossfit_protocol.json') or sha(folder/'checkpoint.pth')!=r['checkpoint_sha256'] or sha(folder/'OOF_full_response.npz')!=r['OOF_sha256']:raise ValueError('Inner model provenance differs')
        if set(r['loaded_patients'])!=set(job['development']):raise ValueError('Unexpected source patient loaded')
        for phase in [0,1]:
            bank_folder=output/f'targets/phase{phase}/outer{job["outer"]}_inner{job["inner"]}';receipt=read_json(bank_folder/'receipt.json')
            if receipt['source_receipt_sha256']!=sha(folder/'receipt.json') or receipt['source_checkpoint_sha256']!=sha(folder/'checkpoint.pth') or receipt['query_protocol_sha256']!=sha(output/'query_protocol.json'):raise ValueError('Bank source differs')
            if phase and receipt['phase0_learner_sha256']!=sha(output/f'learners/phase0/outer{job["outer"]}/models.joblib'):raise ValueError('Phase-one rollout policy differs')
            if set(receipt['patient_file_sha256'])!=set(job['predict']):raise ValueError('Incomplete target patients')
            for subject,digest in receipt['patient_file_sha256'].items():
                path=bank_folder/f'{subject}.npz';raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
                if sha(path)!=digest or sha(raw_path)!=receipt['raw_input_sha256'][subject]:raise ValueError('Bank/raw waveform digest differs')
                d=np.load(path);labels=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
                np.testing.assert_array_equal(labels,d['y'])
                for i in range(len(d['gain'])):
                    before=independent_auc(labels,d['before_scores'][i]);after=independent_auc(labels,d['after_scores'][i])
                    np.testing.assert_allclose([before,after,after-before],[d['before_auc'][i],d['after_auc'][i],d['gain'][i]],rtol=0,atol=1e-12)
                # Reconstruct every feature from observed data only. Cache per state,
                # using RMS sufficient statistics independently of numeric.py.
                raw=load_array(raw_path)[:,:20];distance,_=distance_catalog(raw[...,0])
                for prefix in np.unique(d['prefix'],axis=0):
                    rows=np.flatnonzero((d['prefix']==prefix).all(axis=1));selected=list(map(int,prefix[prefix>=0]));scores=d['before_scores'][rows[0]]
                    signal=np.asarray(raw[:,:,selected,-512:],dtype=float);rms=np.sqrt(np.square(signal).mean(axis=-1))[...,None];valid=np.abs(raw[:,:,selected,0])>1e-12
                    for i in rows:
                        if int(d['candidate'][i]) in selected:raise ValueError('Repeated target query')
                        np.testing.assert_array_equal(scores,d['before_scores'][i])
                        expected=feature_vector(distance,rms,valid,scores,selected,int(d['candidate'][i]))
                        np.testing.assert_allclose(expected,d['features'][i],rtol=0,atol=1e-12);feature_checks+=1
                target_rows+=len(d['gain'])
    for outer in range(5):
        jobs=[j for j in source['jobs'] if j['outer']==outer]
        held=[s for j in jobs for s in j['predict']]
        if len(held)!=28 or len(set(held))!=28:raise ValueError('OOF coverage is not exactly once per development patient')
        for phase in [0,1]:
            folder=output/f'learners/phase{phase}/outer{outer}';r=read_json(folder/'receipt.json')
            if sha(folder/'models.joblib')!=r['models_sha256'] or r['query_protocol_sha256']!=sha(output/'query_protocol.json'):raise ValueError('Learner changed')
            if set(r['fit_patients'])!={int(s[-2:]) for s in held} or set(r['fit_patients'])&{int(s[-2:]) for s in jobs[0]['outer_test']}:raise ValueError('Learner patient isolation failed')
            for p,h in r['input_sha256'].items():
                if sha(output/p)!=h:raise ValueError('Learner input bank changed')
            for bag in r['bag_patient_groups']:
                if set(bag['fit_patients'])&set(bag['excluded_patients']) or set(bag['fit_patients'])|set(bag['excluded_patients'])!=set(r['fit_patients']):raise ValueError('Bag patient split differs')
    count=0
    for c in numeric['cases']:
        labels=load_array(project/f'vendor/norris2026_hat/data/y_{c["subject"]}.npz')
        old_path=project/'outputs/spes_agent_v4/active/trajectories'/c['case_id']/'random_0/metrics.json'
        old=read_json(old_path)
        for policy in numeric['policies']:
            folder=output/'numeric'/c['case_id']/policy;r=read_json(folder/'metrics.json')
            if r['actual_measurements']!=8 or len(r['queries'])!=8 or len(set(r['queries']))!=8 or r['new_LLM_calls']!=0 or r['queries'][:4]!=old['queries'][:4]:raise ValueError('Measurement, prefix or call budget differs')
            if abs(independent_auc(labels,r['prediction'])-r['AUROC'])>1e-12:raise ValueError('Independent AUROC differs')
            if policy=='random_0':np.testing.assert_allclose(r['prediction'],old['prediction'],rtol=0,atol=1e-7)
            for step in range(4):
                choice=read_json(folder/f'{step:02d}_choice.json')
                if choice['choice']!=r['queries'][4+step]:raise ValueError('Trace choice differs')
                if 'predicted_values' in choice:
                    key=policy.split('_',1)[1];expected=choice['candidates'][int(np.argmax(choice['predicted_values'][key]))]
                    if choice['choice']!=f's{expected:03d}':raise ValueError('Learned argmax trace differs')
            count+=1
    result={'status':'verified','source_neural_fits':20,'phases':2,'target_rows':target_rows,'feature_rows_reconstructed':feature_checks,
      'development_patients_per_outer':28,'outer_patients':35,'small_regressors':110,'numeric_trajectories':count,'new_API_calls':0,
      'independent_checks':['Disjoint source fitting/early-stop/OOF/outer patients; no outer loading in source jobs.',
        'All target score differences recomputed using pairwise AUROC, all feature rows reconstructed from revealed data.',
        'Raw bank/model/protocol source hashes; bag patient splits; all eight-query trajectories and independent final AUROC.',
        'Exact original random_0 reproduction and recorded learned-policy argmax choices.'],
      'limits':['No new external cohort; repeatedly inspected outer cohort remains exploratory.','Audit replays predictions, not GPU training bit-for-bit.']}
    write_json(output/'audit.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.project,a.output)
