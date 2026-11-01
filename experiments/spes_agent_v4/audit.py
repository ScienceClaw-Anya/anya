"""Independent arithmetic, budget, public-gate and native-lineage audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,write_json,sha
from experiments.spes_agent_v2.verify import auc,read_native
from .neural import FAMILIES,load_array

def audit(project):
    project=Path(project);root=project/'outputs/spes_agent_v4';training=root/'multitrial';active=root/'active'
    protocol=read_json(training/'neural_protocol.json');fits=0
    for row in protocol['jobs']:
        groups=[set(row[k]) for k in ['train','early_stop','predict']]
        if [len(g) for g in groups]!=[21,7,7] or any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Patient isolation')
        for family in FAMILIES:
            folder=training/f'neural/outer{row["outer"]}/{family}';receipt=read_json(folder/'receipt.json')
            for key in ['train','early_stop','predict']:
                if receipt[key]!=row[key]:raise ValueError('Fit provenance')
            for file,key in [('checkpoint.pth','checkpoint_sha256'),('predictions.npz','prediction_sha256'),('validation.npz','validation_sha256')]:
                if sha(folder/file)!=receipt[key]:raise ValueError('Fit artifact changed')
            if receipt['best_epoch']!=max(receipt['training_trace'],key=lambda r:r['validation_mean_patient_AUROC'])['epoch']:raise ValueError('Best epoch selection')
            for file,key in [('predictions.npz','predict'),('validation.npz','early_stop')]:
                d=np.load(folder/file)
                if set(map(int,d['patient']))!={int(s[-2:]) for s in row[key]}:raise ValueError('Prediction patient provenance')
                for subject in row[key]:
                    p=int(subject[-2:]);m=d['patient']==p;y=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
                    if not np.array_equal(d['y'][m],y) or not np.array_equal(d['contact'][m],np.arange(len(y))):raise ValueError('Contact-label alignment')
            fits+=1
    numerical=read_json(training/'numerical_results.json')
    for row in numerical['rows']:
        data=np.load(training/f'numerical_outer{row["outer"]}.npz');m=data['patient']==row['patient']
        for method in numerical['methods']:
            if abs(auc(data['y'][m],data[method][m])-row[method])>1e-12:raise ValueError('Independent numerical AUROC replay')
    trajectories=attempts=accepted=artifacts=repairs=0
    if (active/'active_protocol.json').exists():
        ap=read_json(active/'active_protocol.json');lookup={c['case_id']:c for c in ap['cases']}
        for path in sorted(active.glob('trajectories/*/*/metrics.json')):
            metrics=read_json(path);folder=path.parent;c=lookup[metrics['case_id']];native=read_native(folder/'scienceclaw');ids=read_json(folder/'artifact_ids.json')
            if len(native)!=24 or len(ids)!=24:raise ValueError('Native artifact budget')
            usages=list(folder.glob('*_usage.json'));numeric=metrics['policy'] in ['random_0','random_1','coverage','score_focus','uncertainty']
            if len(usages)!=(0 if numeric else 8):raise ValueError('Actual LLM attempt budget')
            if metrics['measurements']!=8 or len(set(metrics['queries']))!=8 or metrics['native_public_queries']!=8:raise ValueError('Measurement budget')
            y=load_array(project/f'vendor/norris2026_hat/data/y_{c["subject"]}.npz')
            if abs(auc(y,metrics['prediction'])-metrics['AUROC'])>1e-12:raise ValueError('Independent active AUROC replay')
            expected_prior=0
            for step in range(8):
                public=read_json(folder/f'{step:02d}_public.json');tool=read_json(folder/f'{step:02d}_tool.json');decision=read_json(folder/f'{step:02d}_decision.json');obs=read_json(folder/f'{step:02d}_observation.json')
                if tool!=public or native[ids[f'tool_{step}']]['payload']!=tool or native[ids[f'decision_{step}']]['payload']!=decision or native[ids[f'observation_{step}']]['payload']!=obs:raise ValueError('Native payload differs')
                expected_prior=step if numeric or metrics['policy']=='single' else (0 if step<4 else 4)
                if public['budget']['used']!=expected_prior or len(public['evidence']['observed']['queries'])!=expected_prior:raise ValueError('Future observation leak')
                if obs['stimulus']!=metrics['queries'][step]:raise ValueError('Query execution')
                repairs+=int(obs['duplicate_repair'])
                if not numeric:
                    request=read_json(folder/f'{step:02d}_request.json');usage=read_json(folder/f'{step:02d}_usage.json')
                    digest=hashlib.sha256(json.dumps(request,separators=(',',':')).encode()).hexdigest()
                    if usage['prompt_sha256']!=digest or usage['model']!='gemini-4-argon' or usage['retries']!=0 or usage['reasoning']!='medium':raise ValueError('Provider protocol')
                    expected_board=0 if metrics['policy']=='multi_no_exchange' or metrics['policy']=='multi' and step<4 else expected_prior
                    if len(request['shared_decisions'])!=expected_board:raise ValueError('Rationale visibility')
                    if decision['accepted_output']:
                        if usage.get('response_model')!='gemini-4-argon' or usage.get('finish_reason')!='stop':raise ValueError('Accepted incomplete generation')
                        accepted+=1
            trajectories+=1;attempts+=len(usages);artifacts+=len(native)
    reserved=len(list(active.glob('trajectories/*/*/*_usage.json'))) if active.exists() else 0
    result={'verified':True,'new_neural_fits':fits,'completed_trajectories':trajectories,'attempts_in_completed_trajectories':attempts,
       'all_reserved_attempts_including_incomplete':reserved,'accepted_model_outputs_in_completed':accepted,'native_artifacts_in_completed':artifacts,
       'duplicate_repairs_in_completed':repairs,'checks':['Disjoint fitting/early-stop/test patients','Exact contact-label alignment',
          'Checkpoint and prediction digests','First maximum validation epoch','Independent pairwise AUROC',
          'Actual API attempt files, no hidden retries','Independent-first/feedback visibility','Eight real measurements and native public queries','Native artifact hashes and backward lineage'],
       'limits':'No bitwise neural re-inference or clinical validation claimed. Incomplete trajectories remain preserved, never imputed as completed.'}
    write_json(root/'audit.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();audit(a.project)
