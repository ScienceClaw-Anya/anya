"""Frozen 25-fit, same-input comparison with the original 0.7201 reconstruction.

The legacy arrays contain one mean response and a separate SD, not trials.
This experiment adapts the HAT architecture to ONE mean token per channel.
It does not reproduce the newer paper's cross-trial model or query policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8'); tmp.replace(path)


def disjoint(split):
    groups=[set(split[k]) for k in ['train','val','test']]
    if [len(s) for s in groups]!=[21,7,7] or len(set.union(*groups))!=35: raise ValueError('Patient overlap or incomplete cohort')


def tensor_hash(tensor):
    a=np.ascontiguousarray(tensor.detach().cpu().numpy()); h=hashlib.sha256()
    h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes()); return h.hexdigest()


def setup(project):
    import importlib.util
    project=Path(project)
    sys.path.insert(0,str(project/'reproduction/norris2024'))
    from runtime import load_upstream
    old=load_upstream(project/'vendor/norris2024')
    import dataset
    original=dataset.glob.glob
    dataset.glob.glob=lambda pattern:[Path(p).as_posix() for p in sorted(original(pattern))]
    spec=importlib.util.spec_from_file_location('matched_hat_source',project/'vendor/norris2026_hat/src/models.py')
    new=importlib.util.module_from_spec(spec); sys.modules[spec.name]=new; spec.loader.exec_module(new)
    os.chdir(project/'vendor/norris2024/src')
    return old,new,dataset


def model(old,new,p):
    # Preserve old augmentation, including its RandomNoise implementation and
    # two-channel RNG shape, then consume only the mean just like the baseline.
    class MeanHAT(new.SPES_Transformer_HAT):
        def forward(self,inputs):
            x=inputs[1].clone()
            if self.training: x=old.SPESResponseEncoder.apply_noise_and_zero_channels(self,x)
            x=x[:,:1]; distance=x[...,0].clone().detach()
            embedding=self.embed(x); embedding[distance.abs()<1e-12]=0
            return self.fc(self.attention(embedding,distance))
    a=p['architecture']
    attention=new.Interleaved_HAT(d_embedding=16,dropout=a['dropout'],num_layers=8,n_ct_head=2,n_cc_head=8,pretrain=False)
    net=MeanHAT(input_dim=509,d_embedding=16,dropout=a['dropout'],attention=attention)
    net.noise=old.RandomNoise(std=0.1)
    return net


def freeze(project,output):
    project,output=Path(project),Path(output); original=project/'outputs/reproduction/norris2024/training'
    old_protocol=read(original/'protocol.json'); jobs=[]
    for seed in range(1,6):
        for fold in range(5):
            split=read(original/f'split_seed{seed}_fold{fold}.json'); disjoint(split)
            baseline=original/f'Transformer_base_seed{seed}_fold{fold}'
            jobs.append({'seed':seed,'fold':fold,'split':split,'split_sha256':sha(original/f'split_seed{seed}_fold{fold}.json'),
                         'baseline_predictions_sha256':sha(baseline/'predictions.npz'),'baseline_checkpoint_sha256':sha(baseline/'checkpoint.pth'),
                         'baseline_metrics_sha256':sha(baseline/'metrics.json')})
    source_names=['experiments/spes_matched_2024/experiment.py','reproduction/norris2024/runtime.py',
                  'vendor/norris2026_hat/src/models.py']+[f'vendor/norris2024/src/{k}' for k in old_protocol['source_sha256']]
    sources={k:sha(project/k) for k in source_names}
    for k,h in old_protocol['source_sha256'].items():
        if sources[f'vendor/norris2024/src/{k}']!=h: raise ValueError('Original frozen source changed')
    files=sorted((project/'vendor/norris2024/data').glob('mean/*.npy'))+sorted((project/'vendor/norris2024/data').glob('std/*.npy'))+sorted((project/'vendor/norris2024/data').glob('main/*.npy'))
    files=[f for f in files if f.name.startswith(('X_stim','X_recording','y_','coords_','lobes_'))]
    data={f.relative_to(project).as_posix():sha(f) for f in files}
    labels=[np.load(f) for f in files if f.name.startswith('y_')]
    if (len(labels),sum(len(y) for y in labels),sum(int(y.sum()) for y in labels))!=(35,2145,304): raise ValueError('Wrong legacy cohort')
    p={'experiment':'Legacy complete-cohort mean-input HAT comparison','cohort':{'patients':35,'contacts':2145,'SOZ':304},
       'scope':'Exploratory architecture transfer on an already examined cohort; no new external validation, LLM experiment or clinical efficacy claim.',
       'primary':'HAT mean-input minus original Transformer base, mean within-patient AUROC across five repeats. Bootstrap 35 patient means, not 175 independent cases.',
       'input':'Identical old convergent mean, 508 waveform samples plus distance. SD is loaded by the original loader but never used as a feature or trial. One mean token per response channel.',
       'architecture':{'embedding':16,'layers':8,'trial_heads':2,'channel_heads':8,'dropout':old_protocol['models']['Transformer (base)']['dropout_rate'],'input_dim':509,'trial_tokens':1},
       'training':{'lr':old_protocol['models']['Transformer (base)']['lr'],'weight_decay':0.01,'epochs':1000,'patience':10,'batch_size':8,
                   'rule':'Reuse unchanged old train_model, AdamW, BCE positive weight, shuffle sampler, training-only normalization, mean-patient validation stopping and old noise/channel-drop augmentation. Architecture differs. RNG seed follows original base rule.'},
       'baseline':'Reuse all 25 previously audited original complete-cohort Transformer fits; pin predictions/checkpoints/metrics and splits. No selective fold replacement.',
       'adaptation_limits':['Legacy summaries cannot supply true cross-trial attention. This is a mean-input HAT adaptation, not a full-trial HAT replication.',
                            'This classification task uses all responses and cannot directly establish the newer eight-position policy or agent benefit.',
                            'Architectures have different parameter counts; equal training stopping rules do not imply equal FLOPs.',
                            'No hyperparameter search; old baseline hyperparameters and augmentation held fixed, HAT depth and heads fixed from published architecture.'],
       'test_seal':'No new test predictions or AUROC until all 25 fits complete. No adaptive model selection from new test results.',
       'sources':sources,'data':data,'jobs':jobs,'old_protocol_sha256':sha(original/'protocol.json'),'new_API_calls':0}
    target=output/'protocol.json'
    if target.exists() and read(target)!=p: raise ValueError('Frozen contract differs; use new output')
    write(target,p); print('FROZEN',sha(target),len(data),'data files',flush=True); return p


def check_sources(project,p):
    for f,h in p['sources'].items():
        if sha(Path(project)/f)!=h: raise ValueError('Frozen source changed: '+f)


def loaders(project,p,job,dataset):
    paths=dataset.glob.glob('../data/mean/X_stim*.npy')
    generated=dataset.get_splits(paths,seed=job['seed'])[job['fold']]
    names={k:[Path(paths[i]).stem.removeprefix('X_stim_') for i in v] for k,v in generated.items()}
    if names!=job['split']: raise ValueError('Original patient split/order changed')
    return dataset.create_dataset('data/mean','data/std',job['fold'],job['seed'],batch_size=8)


def train(project,output,index):
    import torch
    project,output=Path(project),Path(output); p=read(output/'protocol.json'); check_sources(project,p); job=p['jobs'][index]
    folder=output/f"fits/seed{job['seed']}_fold{job['fold']}"; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists(): return
    old,new,dataset=setup(project)
    from train import train_model
    from evaluate import get_preds
    torch.set_num_threads(4)
    train_loader,val_loader,test_loader,weight=loaders(project,p,job,dataset)
    hashes={name:[tensor_hash(t) for t in loader.dataset.tensors] for name,loader in [('train',train_loader),('val',val_loader),('test',test_loader)]}
    seed=100000*job['seed']+100*job['fold']+2
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net=model(old,new,p).cuda(); parameters=sum(t.numel() for t in net.parameters())
    optimizer=torch.optim.AdamW(net.parameters(),lr=p['training']['lr'])
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight),device='cuda'))
    start=time.perf_counter(); print('TRAIN_MATCHED',index,parameters,flush=True)
    net=train_model(net,train_loader,val_loader,optimizer,criterion,save_path=str(folder/'checkpoint.pth'),device='cuda',num_epochs=1000,patience=10)
    y,pred,patient=get_preds(net,val_loader,'cuda',mode='validation')
    np.savez_compressed(folder/'validation.npz',y=y,prediction=pred,patient=patient)
    ck=torch.load(folder/'checkpoint.pth',map_location='cpu')
    write(folder/'receipt.json',{'job':job,'rng_seed':seed,'input_tensor_sha256':hashes,'parameters':parameters,'best_epoch':ck.get('epoch'),
          'seconds':time.perf_counter()-start,'checkpoint_sha256':sha(folder/'checkpoint.pth'),'validation_sha256':sha(folder/'validation.npz'),
          'protocol_sha256':sha(output/'protocol.json'),'test_predictions_created':False})
    print('FIT_COMPLETE',index,'seconds',time.perf_counter()-start,flush=True)


def metrics(y,pred,patient):
    result=[]
    for s in np.unique(patient):
        truth=y[patient==s]; values=pred[patient==s]; precision,recall,_=precision_recall_curve(truth,values)
        result.append({'patient':int(s),'n':len(truth),'positive':int(truth.sum()),'AUROC':float(roc_auc_score(truth,values)),'AUPRC':float(auc(recall,precision))})
    return result


def bootstrap(values):
    values=np.array(values,dtype=float)
    draws=np.random.default_rng(20241101).choice(values,size=(20000,len(values)),replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(draws,[.025,.975])]


def aggregate_cases(cases):
    ids=sorted({r['patient'] for r in cases}); patient=[]
    for s in ids:
        rows=[r for r in cases if r['patient']==s]
        if len(rows)!=5 or {r['seed'] for r in rows}!={1,2,3,4,5}: raise ValueError('Incomplete repeats')
        patient.append({'patient':s,'baseline_AUROC':float(np.mean([r['baseline_AUROC'] for r in rows])),
                        'candidate_AUROC':float(np.mean([r['candidate_AUROC'] for r in rows]))})
    differences=[r['candidate_AUROC']-r['baseline_AUROC'] for r in patient]
    interval=bootstrap(differences)
    return {'patients':len(ids),'patient_repeat_cases':len(cases),'baseline_mean_patient_AUROC':float(np.mean([r['baseline_AUROC'] for r in patient])),
            'candidate_mean_patient_AUROC':float(np.mean([r['candidate_AUROC'] for r in patient])),
            'paired_gain':float(np.mean(differences)),'paired_nominal_95CI':interval,
            'patients_improved':sum(v>0 for v in differences),'patients_worsened':sum(v<0 for v in differences),
            'positive_lower_bound':interval[0]>0,'patient_means':patient}


def evaluate(project,output):
    import torch
    project,output=Path(project),Path(output); p=read(output/'protocol.json'); check_sources(project,p)
    folders=[output/f"fits/seed{j['seed']}_fold{j['fold']}" for j in p['jobs']]
    if not all((f/'receipt.json').exists() for f in folders): raise ValueError('All 25 fits must complete before opening new test outcomes')
    old,new,dataset=setup(project)
    from evaluate import get_preds
    torch.set_num_threads(4); cases=[]
    for job,folder in zip(p['jobs'],folders):
        receipt=read(folder/'receipt.json')
        if receipt['protocol_sha256']!=sha(output/'protocol.json') or receipt['checkpoint_sha256']!=sha(folder/'checkpoint.pth'): raise ValueError('Fit receipt mismatch')
        _,_,loader,_=loaders(project,p,job,dataset)
        if [tensor_hash(t) for t in loader.dataset.tensors]!=receipt['input_tensor_sha256']['test']: raise ValueError('Test input changed')
        net=model(old,new,p).cuda(); ck=torch.load(folder/'checkpoint.pth',map_location='cpu'); net.load_state_dict(ck['model_state_dict'])
        y,pred,coords,lobes,patient=get_preds(net,loader,'cuda',mode='test')
        path=folder/'predictions.npz'; np.savez_compressed(path,y=y,prediction=pred,patient=patient,coords=coords,lobes=lobes)
        baseline_path=project/f"outputs/reproduction/norris2024/training/Transformer_base_seed{job['seed']}_fold{job['fold']}/predictions.npz"
        if sha(baseline_path)!=job['baseline_predictions_sha256']: raise ValueError('Baseline changed')
        with np.load(baseline_path,allow_pickle=False) as b:
            for key,a in [('y',y),('patient',patient),('coords',coords),('lobes',lobes)]: np.testing.assert_array_equal(b[key],a)
            base=metrics(b['y'],b['prediction'],b['patient'])
        candidate=metrics(y,pred,patient)
        for a,b in zip(base,candidate):
            if (a['patient'],a['n'],a['positive'])!=(b['patient'],b['n'],b['positive']): raise ValueError('Scored contacts differ')
            cases.append({'seed':job['seed'],'fold':job['fold'],'patient':a['patient'],'n':a['n'],'positive':a['positive'],
                          'baseline_AUROC':a['AUROC'],'candidate_AUROC':b['AUROC'],'baseline_AUPRC':a['AUPRC'],'candidate_AUPRC':b['AUPRC']})
        write(folder/'evaluation.json',{'prediction_sha256':sha(path),'receipt_sha256':sha(folder/'receipt.json'),'per_patient':candidate})
        print('SCORED_MATCHED_FOLD',job['seed'],job['fold'],flush=True)
        del net; torch.cuda.empty_cache()
    result=aggregate_cases(cases)
    if result['patients']!=35 or result['patient_repeat_cases']!=175: raise ValueError('Incomplete evaluation')
    result.update({'cohort':p['cohort'],'candidate':'HAT architecture adapted to legacy mean-only input','new_API_calls':0,'multi_agent_superiority_established':False,
                   'scope':p['scope'],'adaptation_limits':p['adaptation_limits'],'cases':cases,'protocol_sha256':sha(output/'protocol.json')})
    write(output/'results.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ['cases','patient_means']},indent=2),flush=True)


def run(project,output,workers):
    p=freeze(project,output); pending=list(range(25)); active=[]; logs=output/'logs'; logs.mkdir(parents=True,exist_ok=True)
    lock=output/'run.lock'
    with lock.open('x',encoding='utf8') as f: f.write(str(os.getpid()))
    success=False
    try:
        while pending or active:
            while pending and len(active)<workers:
                i=pending.pop(0); j=p['jobs'][i]
                if (output/f"fits/seed{j['seed']}_fold{j['fold']}/receipt.json").exists(): continue
                log=(logs/f'fit_{i:02d}.log').open('a',encoding='utf8')
                cmd=[sys.executable,'-u','-X','utf8','-m','experiments.spes_matched_2024.experiment','--project',str(project),'--output',str(output),'--stage','train','--index',str(i)]
                proc=subprocess.Popen(cmd,cwd=project,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
                active.append((proc,log,i)); print('START_MATCHED_FIT',i,flush=True)
            failed=[]
            for proc,log,i in list(active):
                if proc.poll() is None: continue
                log.close(); active.remove((proc,log,i)); print('FINISHED_MATCHED_FIT',i,proc.returncode,flush=True)
                if proc.returncode: failed.append(i)
            if failed:
                for proc,log,_ in active: proc.wait(); log.close()
                raise RuntimeError(f'Failed local fits {failed}; preserve logs and inspect before resuming')
            if pending or active: time.sleep(2)
        evaluate(project,output); success=True
    finally:
        if success: lock.unlink()


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--project',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--stage',choices=['freeze','train','evaluate','run'],required=True); a.add_argument('--index',type=int); a.add_argument('--workers',type=int,choices=[1,2],default=2); a=a.parse_args()
    project=a.project.resolve(); output=a.output.resolve()
    if a.stage=='freeze': freeze(project,output)
    elif a.stage=='train': train(project,output,a.index)
    elif a.stage=='evaluate': evaluate(project,output)
    else: run(project,output,a.workers)
