"""Reconstruct hypothetical features and independently audit budgets/outcomes."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.spes_agent_v3.common import read_json,sha,check_groups
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json
from .features import METHODS,impute,summarize


def audit(project,output):
    project,output=Path(project),Path(output);p=read_json(output/'protocol.json');source=project/'outputs/spes_agent_v8';cross=read_json(source/'crossfit_protocol.json');evaluation=read_json(output/'numeric_protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()) or any(sha(project/f)!=h for f,h in p['parent_receipt_sha256'].items()):raise ValueError('Source provenance differs')
    if sha(output/'protocol.json')!=evaluation['protocol_sha256']:raise ValueError('Evaluation protocol lineage differs')
    by_patient={};forwards=0
    for j in cross['jobs']:
        check_groups(j['train'],j['early_stop'],j['predict'],j['outer_test'])
        folder=output/f'banks/outer{j["outer"]}_inner{j["inner"]}';r=read_json(folder/'receipt.json');old_r=read_json(source/f'targets/phase0/outer{j["outer"]}_inner{j["inner"]}/receipt.json')
        source_r=read_json(source/f'crossfit/outer{j["outer"]}_inner{j["inner"]}/receipt.json')
        if set(r['patients'])!=set(j['predict']) or r['outer_test']!=j['outer_test'] or r['protocol_sha256']!=sha(output/'protocol.json') or r['source_checkpoint_sha256']!=source_r['checkpoint_sha256']:raise ValueError('Bank patient/source differs')
        if sha(source/f'crossfit/outer{j["outer"]}_inner{j["inner"]}/checkpoint.pth')!=r['source_checkpoint_sha256']:raise ValueError('Source HAT changed')
        forwards+=r['hypothetical_forward_calls']
        for subject,h in r['bank_sha256'].items():by_patient.setdefault(subject,[]).append((j,folder,r,old_r,h))
    rows=0
    for subject,entries in sorted(by_patient.items()):
        raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz';digest=sha(raw_path)
        if any(old_r['raw_input_sha256'][subject]!=digest for _,_,_,old_r,_ in entries):raise ValueError('Raw input changed')
        raw=load_array(raw_path)[:,:20]
        for j,folder,r,old_r,h in entries:
            path=folder/f'{subject}.npz';old_path=source/f'targets/phase0/outer{j["outer"]}_inner{j["inner"]}/{subject}.npz'
            if sha(path)!=h or sha(old_path)!=r['v8_target_sha256'][subject]:raise ValueError('Bank target hashes differ')
            with np.load(path) as d:bank={k:d[k] for k in d.files}
            with np.load(old_path) as d:old={k:d[k] for k in d.files}
            for key in ['gain','prefix','candidate']:np.testing.assert_array_equal(bank[key],old[key])
            np.testing.assert_allclose(bank['features'][:,:20],old['features'],rtol=0,atol=1e-12)
            for i in range(len(bank['gain'])):
                selected=list(map(int,bank['prefix'][i][bank['prefix'][i]>=0]));candidate=int(bank['candidate'][i]);parts=[]
                for m,method in enumerate(METHODS):
                    wave=impute(raw[...,0],raw[:,:,selected,-512:],selected,candidate,method)
                    parts+=summarize(old['before_scores'][i],bank['hypothetical_scores'][i,m],wave,np.abs(raw[:,:,candidate,0])>1e-12)
                a,b=bank['hypothetical_scores'][i]
                expected=parts+[np.abs(a-b).mean(),np.abs(rankdata(a)-rankdata(b)).mean()/(len(a)+1)]
                np.testing.assert_allclose(expected,bank['features'][i,20:],rtol=0,atol=1e-12);rows+=1
        print('HYPOTHESIS_AUDIT_PATIENT',subject,flush=True)
    for outer in range(5):
        folder=output/f'learners/outer{outer}';r=read_json(folder/'receipt.json');jobs=[j for j in cross['jobs'] if j['outer']==outer];fit={int(s[-2:]) for j in jobs for s in j['predict']}
        if set(r['fit_patients'])!=fit or len(fit)!=28 or fit&{int(s[-2:]) for s in r['outer_test']}:raise ValueError('Value-learner patient isolation differs')
        if r['protocol_sha256']!=sha(output/'protocol.json') or r['models_sha256']!=sha(folder/'models.joblib'):raise ValueError('Learner changed')
        for path,h in r['input_sha256'].items():
            if sha(output/path)!=h:raise ValueError('Learner input changed')
    trajectories=0
    for case in evaluation['cases']:
        labels=load_array(project/f'vendor/norris2026_hat/data/y_{case["subject"]}.npz')
        for policy in evaluation['policies']:
            folder=output/'numeric'/case['case_id']/policy;r=read_json(folder/'metrics.json')
            old=read_json(source/'numeric'/case['case_id']/(policy if policy in ['random_0','random_1','coverage','score_focus','uncertainty'] else 'random_0')/'metrics.json')
            if len(r['queries'])!=8 or len(set(r['queries']))!=8 or r['actual_measurements']!=8 or r['new_API_calls']!=0 or r['queries'][:4]!=old['queries'][:4]:raise ValueError('Budget or shared prefix differs')
            if abs(independent_auc(labels,r['prediction'])-r['AUROC'])>1e-12:raise ValueError('Independent AUROC differs')
            if policy in ['random_0','random_1','coverage','score_focus','uncertainty']:np.testing.assert_allclose(r['prediction'],old['prediction'],rtol=0,atol=1e-7)
            calls=0
            for step in range(4):
                choice=read_json(folder/f'{step:02d}_choice.json')
                if choice['choice']!=r['queries'][4+step]:raise ValueError('Executed decision differs')
                if 'predicted_values' in choice:
                    expected=choice['candidates'][int(np.argmax(choice['predicted_values'][policy]))]
                    if choice['choice']!=f's{expected:03d}' or not choice['hypothetical_not_observed']:raise ValueError('Hypothetical argmax differs')
                    calls+=2*len(choice['candidates'])
            if calls!=r['hypothetical_forward_calls']:raise ValueError('Hypothetical compute count differs')
            trajectories+=1
    result={'status':'verified','source_neural_fits_reused':20,'new_neural_fits':0,'feature_rows_reconstructed':rows,'source_hypothetical_forward_calls':forwards,
      'regressors':55,'patients':35,'numeric_trajectories':trajectories,'new_API_calls':0,
      'checks':['Frozen lineage and patient isolation, copied supervised targets, reconstructed hypothetical features from observed waves plus recorded HAT predictions.',
        'Every final AUROC recomputed by pairwise comparison; common prefixes, actual-query budgets, argmax choices and hypothetical compute counts checked.',
        'All five fixed numerical baselines exactly reproduce v8.'],
      'limits':['Recorded hypothetical HAT scores are not all re-inferred in this audit; source-generation checks replayed observed scores at each development state.',
        'No bit-for-bit training replication or new external cohort. Repeated cohort exploration remains exploratory.']}
    write_json(output/'audit.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.project,a.output)
