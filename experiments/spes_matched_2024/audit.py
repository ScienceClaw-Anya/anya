"""Independent normalization, inference and patient-level metric verification."""
import argparse
from pathlib import Path
import re
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from .experiment import read,write,sha,setup,model,loaders,tensor_hash


def normalizer(arrays,stats=None):
    """Reconstruct legacy representation without overlapping assignment.

    The historical implementation aliases its response view and overwrites the
    first response sample with distance before shifting. Explicitly model that
    quirk here, rather than repeating the original in-place operations.
    """
    maximum=max(a.shape[1] for a in arrays)
    joined=np.concatenate([np.pad(a,((0,0),(0,maximum-a.shape[1]),(0,0))) for a in arrays])
    d=joined[...,-1].copy(); waves=joined[...,:-1].copy()
    if stats is None:
        raw=[a[...,:-1] for a in arrays]
        md=d[d>13].mean(); sd=d[d>13].std()
        mt=np.mean([a[a.std(axis=2)>0].mean() for a in raw])
        st=np.mean([a.std(axis=2)[a.std(axis=2)>0].mean() for a in raw])
        stats=(md,mt,sd,st)
    md,mt,sd,st=stats; valid=d>13; d[valid]=(d[valid]-md)/sd
    valid=waves.std(axis=2)!=0; waves[valid]=(waves[valid]-mt)/st
    result=np.zeros_like(joined); result[...,0]=d; result[...,1]=d; result[...,2:]=waves[...,1:]
    return result[:,None],stats


def infer(net,loader):
    import torch
    values=[]
    net.eval()
    with torch.no_grad():
        for batch in loader:
            logits=net([batch[0].cuda(),batch[1].cuda()])[:,0].cpu().numpy()
            values.append(expit(logits))
    return np.concatenate(values)


def audit(project,output):
    import torch
    project,output=Path(project),Path(output); p=read(output/'protocol.json'); result=read(output/'results.json')
    if (output/'run.lock').exists(): raise ValueError('Wait for training and evaluation to finish')
    for f,h in {**p['sources'],**p['data']}.items():
        if sha(project/f)!=h: raise ValueError('Source or data changed: '+f)
    old,new,dataset=setup(project); torch.set_num_threads(4)
    all_rows=[]; input_checks=0; inference_checks=0; max_diff=0.; repeated_contact_counts={}
    old_training=project/'outputs/reproduction/norris2024/training'
    old_verified=read(project/'outputs/reproduction/norris2024/verification.json')
    if old_verified['verified_primary_runs']!=75 or not old_verified['source_hashes_verified'] or not old_verified['patient_splits_verified']: raise ValueError('Original reproduction audit missing')
    for index,job in enumerate(p['jobs']):
        seed,fold=job['seed'],job['fold']; folder=output/f'fits/seed{seed}_fold{fold}'; receipt=read(folder/'receipt.json')
        if receipt['job']!=job or receipt['protocol_sha256']!=sha(output/'protocol.json') or receipt['test_predictions_created']: raise ValueError('Training isolation receipt differs')
        baseline=old_training/f'Transformer_base_seed{seed}_fold{fold}'
        for name,expected in [('predictions.npz',job['baseline_predictions_sha256']),('checkpoint.pth',job['baseline_checkpoint_sha256']),('metrics.json',job['baseline_metrics_sha256'])]:
            if sha(baseline/name)!=expected: raise ValueError('Frozen baseline differs')
        if sha(old_training/f'split_seed{seed}_fold{fold}.json')!=job['split_sha256']: raise ValueError('Original split differs')
        if sha(folder/'checkpoint.pth')!=receipt['checkpoint_sha256'] or sha(folder/'validation.npz')!=receipt['validation_sha256']: raise ValueError('Candidate fit changed')
        training,validation,test,_=loaders(project,p,job,dataset)
        for name,loader in [('train',training),('val',validation),('test',test)]:
            if [tensor_hash(t) for t in loader.dataset.tensors]!=receipt['input_tensor_sha256'][name]: raise ValueError('Normalized tensor changed')
        # Validate all four mean/SD, stimulation/recording representations. Only
        # convergent mean enters either tested classifier, but no loader drift
        # should be hidden by slicing out unused channels.
        for mode,position in [('stim',0),('recording',1)]:
            for directory,channel in [('mean',0),('std',1)]:
                stats=None
                for name,loader in [('train',training),('val',validation),('test',test)]:
                    arrays=[np.load(project/f'vendor/norris2024/data/{directory}/X_{mode}_{s}.npy') for s in job['split'][name]]
                    expected,got_stats=normalizer(arrays,stats)
                    if name=='train': stats=got_stats
                    actual=loader.dataset.tensors[position][:,channel:channel+1].numpy()
                    np.testing.assert_array_equal(actual,expected); input_checks+=1
        ck=torch.load(folder/'checkpoint.pth',map_location='cpu'); net=model(old,new,p).cuda(); net.load_state_dict(ck['model_state_dict'])
        if sum(v.numel() for v in net.parameters())!=receipt['parameters']: raise ValueError('Architecture differs')
        log=(output/f'logs/fit_{index:02d}.log').read_text(encoding='utf8')
        trace=[(int(e),float(v)) for e,v in re.findall(r'Epoch (\d+)/1000, Valid AUROC: ([0-9.]+)',log)]
        if not trace or ck['epoch']!=receipt['best_epoch'] or ck['epoch'] not in [e for e,_ in trace]: raise ValueError('Training trace incomplete')
        # Text logging rounds to four decimals. Check the saved epoch is among
        # rounded maxima, then re-infer the exact saved validation prediction.
        saved_round=next(v for e,v in trace if e==ck['epoch'])
        if saved_round+0.00011<max(v for _,v in trace): raise ValueError('Saved checkpoint not best validation epoch')
        if len(trace)<1000 and len(trace)-ck['epoch']!=10: raise ValueError('Early-stopping patience differs')
        for name,loader in [('validation',validation),('predictions',test)]:
            computed=infer(net,loader)
            with np.load(folder/f'{name}.npz',allow_pickle=False) as saved:
                difference=float(np.max(np.abs(computed-saved['prediction']))); max_diff=max(max_diff,difference)
                np.testing.assert_allclose(computed,saved['prediction'],rtol=1e-5,atol=1e-6); inference_checks+=1
        evaluation=read(folder/'evaluation.json')
        if evaluation['prediction_sha256']!=sha(folder/'predictions.npz') or evaluation['receipt_sha256']!=sha(folder/'receipt.json'): raise ValueError('Evaluation changed')
        with np.load(folder/'predictions.npz',allow_pickle=False) as candidate,np.load(baseline/'predictions.npz',allow_pickle=False) as base:
            for key in ['y','patient','coords','lobes']: np.testing.assert_array_equal(candidate[key],base[key])
            y,patient=candidate['y'],candidate['patient']
            if set(np.unique(patient))!={int(s[-2:]) for s in job['split']['test']}: raise ValueError('Held-out patients differ')
            for subject in np.unique(patient):
                mask=patient==subject
                all_rows.append({'seed':seed,'fold':fold,'patient':int(subject),
                                 'baseline':float(roc_auc_score(y[mask],base['prediction'][mask])),
                                 'candidate':float(roc_auc_score(y[mask],candidate['prediction'][mask]))})
                repeated_contact_counts.setdefault(int(subject),[]).append((int(mask.sum()),int(y[mask].sum())))
        del net,training,validation,test; torch.cuda.empty_cache(); print('MATCHED_AUDIT_FOLD',seed,fold,flush=True)
    ids=sorted(repeated_contact_counts)
    if len(ids)!=35 or any(len(v)!=5 or len(set(v))!=1 for v in repeated_contact_counts.values()): raise ValueError('Cohort incomplete across repeats')
    if (sum(v[0][0] for v in repeated_contact_counts.values()),sum(v[0][1] for v in repeated_contact_counts.values()))!=(2145,304): raise ValueError('Wrong scored contacts')
    baseline=np.array([np.mean([r['baseline'] for r in all_rows if r['patient']==s]) for s in ids])
    candidate=np.array([np.mean([r['candidate'] for r in all_rows if r['patient']==s]) for s in ids]); delta=candidate-baseline
    rng=np.random.default_rng(20241101); indices=rng.integers(0,35,size=(20000,35)); interval=np.quantile(delta[indices].mean(axis=1),[.025,.975])
    for key,value in [('baseline_mean_patient_AUROC',baseline.mean()),('candidate_mean_patient_AUROC',candidate.mean()),('paired_gain',delta.mean())]:
        np.testing.assert_allclose(result[key],value,rtol=0,atol=1e-14)
    np.testing.assert_allclose(result['paired_nominal_95CI'],interval,rtol=0,atol=1e-14)
    for r in all_rows:
        target=next(v for v in result['cases'] if (v['seed'],v['fold'],v['patient'])==(r['seed'],r['fold'],r['patient']))
        for name in ['baseline','candidate']: np.testing.assert_allclose(target[name+'_AUROC'],r[name],rtol=0,atol=1e-14)
    audit_result={'status':'verified','fits':25,'normalized_array_comparisons':input_checks,'checkpoint_prediction_reinferences':inference_checks,
                  'maximum_prediction_difference':max_diff,'patient_repeat_cases':len(all_rows),'bootstrap_units':35,'contacts':2145,'SOZ':304,
                  'sources_and_data_files_hashed':len(p['sources'])+len(p['data']),'baseline_fits_reused':25,'baseline_training_rerun':False,
                  'legacy_normalizer_alias_quirk_preserved_for_both':True,'new_API_calls':0,
                  'checks':['All 25 original patient splits and baseline checkpoint/prediction/metric hashes match the completed 75-run reproduction audit.',
                            'All normalized mean/SD divergent/convergent arrays independently reconstructed from fitting-patient statistics, including historical aliasing behavior.',
                            'All candidate validation/test predictions re-inferred from saved best checkpoints; original stopping trace and patient/contact identities verified.',
                            'All 175 per-patient repeat AUROCs, 35 repeat-averaged patient pairs and bootstrap interval independently recomputed.'],
                  'limits':['This mean-only HAT adaptation cannot reproduce cross-trial attention or physical query selection.',
                            'Historical baseline training is reused, not independently retrained again.',
                            'Already examined single-centre cohort; nominal uncertainty does not adjust for all exploratory trials.']}
    write(output/'audit.json',audit_result); print(audit_result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); audit(a.project,a.output)
