"""Retrain complementary published SPES models with an auditable local runner.

This is a new exploratory experiment, NOT a resumption of v3 or a claim of an
exact reproduction. Best weights are cloned (the upstream runner's state_dict
reference can otherwise change after the best epoch). No upstream checkpoint
pickle is loaded. All new checkpoints contain tensors and plain metadata only.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import copy
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from experiments.spes_agent_v3.common import read_json, write_json, sha, exclusive

FAMILIES=['baseline','mte','hat']
UPSTREAM_COMMIT='11f65182a2114440f1425656f91d0ebec51ee726'

def upstream_models(project):
    path=Path(project)/'vendor/norris2026_hat/src/models.py'
    spec=importlib.util.spec_from_file_location('norris2026_models',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def load_array(path):
    with np.load(path,allow_pickle=False) as data:
        return np.array(data['data'] if 'data' in data else data[data.files[0]])

def header(path):
    from numpy.lib.format import read_magic,_read_array_header
    with zipfile.ZipFile(path) as archive:
        with archive.open(archive.namelist()[0]) as file:
            shape,order,dtype=_read_array_header(file,read_magic(file))
    return list(shape),str(dtype)

def inventory(project):
    root=Path(project)/'vendor/norris2026_hat/data'; records=[]
    for path in sorted(root.glob('X_*.npz')):
        subject=path.stem[2:];shape,dtype=header(path);y=load_array(root/f'y_{subject}.npz')
        if len(y)!=shape[0] or set(np.unique(y))!={0,1}:raise ValueError('Bad patient labels')
        records.append({'subject':subject,'patient':int(subject[-2:]),'shape':shape,'dtype':dtype,
            'contacts':len(y),'SOZ':int(y.sum()),'compressed_bytes':path.stat().st_size,
            'uncompressed_bytes':math.prod(shape)*np.dtype(dtype).itemsize})
    if len(records)!=35:raise ValueError('Expected 35-patient released multi-trial cohort')
    return records

def freeze(project,output):
    project,output=Path(project),Path(output); output.mkdir(parents=True,exist_ok=True)
    records=inventory(project);mods=upstream_models(project);jobs=[]
    folds=[held.tolist() for _,held in KFold(5,shuffle=True,random_state=0).split(records)]
    for outer in range(5):
        test=folds[outer];stop=folds[(outer+1)%5]
        train=sorted(set(range(35))-set(test)-set(stop))
        jobs.append({'outer':outer,'train':[records[i]['subject'] for i in train],
          'early_stop':[records[i]['subject'] for i in stop],'predict':[records[i]['subject'] for i in test]})
    sources=['experiments/spes_agent_v4/neural.py','vendor/norris2026_hat/src/models.py']
    result={'experiment':'v4 complementary multi-trial experts', 'upstream':{'repository':'https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT',
       'commit':UPSTREAM_COMMIT,'license':'MIT','paper':'https://doi.org/10.64898/2026.07.27.26358819','publication_status':'Preprint only'},
       'scope':'Exploratory training on a previously examined single-centre patient cohort, newly released processed arrays. Not untouched external validation.',
       'primary_metric':'Mean within-patient SOZ AUROC; no clinical outcome claim',
       'max_trials':20,'max_epochs':25,'patience':5,'batch_size':8,'families':FAMILIES,
       'training':'AdamW; published fixed model presets; original inverse-class and inverse-patient weighted sampling plus BCE positive weight. Standardisation only from fitting patients.',
       'changes_from_upstream_runner':['Clone best weights rather than retaining live state_dict references',
          'Deterministically seed Python, NumPy, Torch and sampler per family/fold',
          'Batch padding and streamed training-only standardisation reduce memory',
          'Max 25 epochs/patience 5 from paper rather than CLI 50/7 defaults',
          'Keep outer predictions sealed until all 15 fits complete'],
       'presets':{n:asdict(mods.PUBLICATION_PRESETS[n]) for n in FAMILIES},
       'sources':{s:sha(project/s) for s in sources},'records':records,'jobs':jobs,
       'agents':'No LLM requests during this feasibility/training stage. A numerical improvement alone is not evidence for collaboration.'}
    path=output/'neural_protocol.json'
    if path.exists() and read_json(path)!=result:raise ValueError('Frozen neural protocol changed')
    write_json(path,result);return result

def prepare_patients(project,protocol,row):
    """Load only capped trials; preserve zero padding exactly during scaling."""
    import torch
    root=Path(project)/'vendor/norris2026_hat/data'; patients={}
    for r in protocol['records']:
        subject=r['subject']; x=load_array(root/f'X_{subject}.npz')[:,:protocol['max_trials']].copy()
        if not np.isfinite(x).all():raise ValueError(f'Nonfinite raw input {subject}')
        patients[subject]={'x':x,'y':load_array(root/f'y_{subject}.npz').astype(np.float32),
            'coords':load_array(root/f'coords_{subject}.npz'),'patient':r['patient']}
    slices=[slice(0,1),slice(1,129),slice(129,None)];stats=[]
    for segment in slices:
        count=0;total=0.;total2=0.
        for subject in row['train']:
            v=patients[subject]['x'][...,segment]; v=v[np.abs(v)>1e-12].astype(np.float64)
            count+=v.size;total+=float(v.sum());total2+=float(np.square(v).sum())
        if not count:raise ValueError('Empty training feature segment')
        mean=total/count;std=max(math.sqrt(max(total2/count-mean*mean,0)),1e-12)
        stats.append({'mean':mean,'std':std,'nonzero_values':count})
        for obj in patients.values():
            view=obj['x'][...,segment];mask=np.abs(view)>1e-12;view[mask]=(view[mask]-mean)/std
    for obj in patients.values():obj['x']=torch.from_numpy(obj['x']);obj['y']=torch.from_numpy(obj['y'])
    return patients,stats

def loaders(patients,row,batch_size,seed):
    import torch
    from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
    class Contacts(Dataset):
        def __init__(self,subjects):self.entries=[(s,i) for s in subjects for i in range(len(patients[s]['y']))]
        def __len__(self):return len(self.entries)
        def __getitem__(self,i):
            s,k=self.entries[i];p=patients[s]
            return p['x'][k],p['y'][k],p['patient'],k
    def collate(batch):
        t=max(x.shape[0] for x,*_ in batch);c=max(x.shape[1] for x,*_ in batch);f=batch[0][0].shape[-1]
        x=torch.zeros(len(batch),t,c,f)
        for i,(v,*_) in enumerate(batch):x[i,:v.shape[0],:v.shape[1]]=v
        return x,torch.stack([b[1] for b in batch]),torch.tensor([b[2] for b in batch]),torch.tensor([b[3] for b in batch])
    datasets={k:Contacts(row[k]) for k in ['train','early_stop','predict']}
    labels=np.array([float(patients[s]['y'][i]) for s,i in datasets['train'].entries]);counts=np.bincount(labels.astype(int),minlength=2)
    weights=[1/(len(patients[s]['y'])*counts[int(patients[s]['y'][i])]) for s,i in datasets['train'].entries]
    generator=torch.Generator().manual_seed(seed)
    sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=generator)
    return {k:DataLoader(v,batch_size=batch_size,sampler=sampler if k=='train' else None,shuffle=False,
          collate_fn=collate,num_workers=0) for k,v in datasets.items()},float(counts[0]/counts[1])

def infer(model,loader,device,keep_stim=None):
    import torch
    model.eval();pred=[];ys=[];ps=[];cs=[]
    with torch.no_grad():
        for x,y,p,c in loader:
            if keep_stim is not None:
                # Sensitivity diagnostic only: same known channel indices across
                # all recordings. No future full-response scorer is exposed.
                x=x[:,:,:keep_stim].clone()
            v=torch.sigmoid(model(x.to(device))[:,0]).cpu().numpy()
            if not np.isfinite(v).all():raise ValueError('Nonfinite model output')
            pred.append(v);ys.append(y.numpy());ps.append(p.numpy());cs.append(c.numpy())
    return {'prediction':np.concatenate(pred),'y':np.concatenate(ys),'patient':np.concatenate(ps),'contact':np.concatenate(cs)}

def mean_auc(result):
    return float(np.mean([roc_auc_score(result['y'][result['patient']==p],result['prediction'][result['patient']==p]) for p in np.unique(result['patient'])]))

def train_fold(project,output,outer):
    import torch
    project,output=Path(project),Path(output);protocol=read_json(output/'neural_protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['sources'].items()):raise ValueError('Frozen source changed')
    row=protocol['jobs'][outer]
    sets=[set(row[k]) for k in ['train','early_stop','predict']]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Patient overlap')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    print('LOAD',outer,flush=True);begin=time.perf_counter();patients,stats=prepare_patients(project,protocol,row)
    print('LOADED',outer,round(time.perf_counter()-begin,1),flush=True);mods=upstream_models(project)
    for family in FAMILIES:
        folder=output/f'neural/outer{outer}/{family}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():continue
        seed=492100+100*outer+FAMILIES.index(family);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        loader,pos_weight=loaders(patients,row,protocol['batch_size'],seed)
        model,_,preset=mods.create_model(family);model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=preset.learning_rate,weight_decay=preset.weight_decay)
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device='cuda'))
        best=-float('inf');stale=0;best_weights=None;trace=[];begin=time.perf_counter()
        for epoch in range(1,protocol['max_epochs']+1):
            model.train();losses=[]
            for x,y,*_ in loader['train']:
                optimizer.zero_grad(set_to_none=True);logits=model(x.cuda())[:,0];loss=criterion(logits,y.cuda())
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
            val=infer(model,loader['early_stop'],'cuda');score=mean_auc(val)
            trace.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation_mean_patient_AUROC':score})
            print('EPOCH',outer,family,epoch,round(score,6),flush=True)
            if score>best:
                best=score;stale=0;best_epoch=epoch;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=protocol['patience']:break
        if best_weights is None:raise ValueError('No checkpoint')
        model.load_state_dict(best_weights);torch.save({'model_state_dict':best_weights,'family':family,'seed':seed,'best_epoch':best_epoch},folder/'checkpoint.pth')
        val=infer(model,loader['early_stop'],'cuda');test=infer(model,loader['predict'],'cuda')
        np.savez_compressed(folder/'validation.npz',**val);np.savez_compressed(folder/'predictions.npz',**test)
        write_json(folder/'receipt.json',{'outer':outer,'family':family,'seed':seed,'train':row['train'],'early_stop':row['early_stop'],'predict':row['predict'],
            'normalisation':stats,'training_trace':trace,'best_epoch':best_epoch,'best_validation_AUROC':best,'seconds':time.perf_counter()-begin,
            'checkpoint_sha256':sha(folder/'checkpoint.pth'),'prediction_sha256':sha(folder/'predictions.npz'),'validation_sha256':sha(folder/'validation.npz'),
            'protocol_sha256':sha(output/'neural_protocol.json')})
        print('FIT_DONE',outer,family,'seconds',round(time.perf_counter()-begin,1),flush=True)
        del model,optimizer,best_weights;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--outer',type=int);p.add_argument('--workers',type=int,default=2);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.outer is not None:train_fold(a.project,a.output,a.outer);return
    if not 1<=a.workers<=2:raise ValueError('One or two local GPU workers')
    with exclusive(a.output/'neural.lock'):
        protocol=freeze(a.project,a.output)
        if a.freeze_only:print('Frozen 15-fit training protocol');return
        queue=list(range(5));active=[];logs=a.output/'logs';logs.mkdir(parents=True,exist_ok=True)
        while queue or active:
            while queue and len(active)<a.workers:
                outer=queue.pop(0)
                if all((a.output/f'neural/outer{outer}/{f}/receipt.json').exists() for f in FAMILIES):continue
                log=(logs/f'neural_outer{outer}.log').open('a',encoding='utf-8')
                command=[sys.executable,'-u','-X','utf8','-m','experiments.spes_agent_v4.neural','--project',str(a.project),'--output',str(a.output),'--outer',str(outer)]
                process=subprocess.Popen(command,cwd=a.project,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONHASHSEED='0'))
                active.append((process,log,outer));print('START_OUTER',outer,flush=True)
            fail=False
            for process,log,outer in list(active):
                if process.poll() is None:continue
                log.close();active.remove((process,log,outer));print('DONE_OUTER',outer,process.returncode,flush=True);fail|=process.returncode!=0
            if fail:
                for process,log,outer in active:process.wait();log.close()
                raise RuntimeError('A training worker failed. Inspect receipts/logs before resuming.')
            if queue or active:time.sleep(2)
        print('ALL_15_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
