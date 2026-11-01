"""Train sparse-observation experts; do not alter frozen v4 sources/results."""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from experiments.spes_agent_v3.common import read_json,write_json,sha,exclusive
from experiments.spes_agent_v4.neural import prepare_patients,loaders,upstream_models,mean_auc

FAMILIES=['baseline','hat']

def fixed_queries(available,outer,patient,rep=0):
    remaining=list(map(int,available));out=[];seed=6190+outer*100+patient+rep*10000
    for step in range(8):out.append(remaining.pop(int(np.random.default_rng(seed+step*100003).integers(len(remaining)))))
    return out

def sparse_batch(x,p,available,rng=None,fixed=None):
    import torch
    k=int(rng.integers(1,9)) if fixed is None else 8
    out=torch.zeros(x.shape[0],x.shape[1],k,x.shape[-1],dtype=x.dtype)
    choices={int(pid):(list(map(int,rng.choice(available[int(pid)],k,replace=False))) if fixed is None else fixed[int(pid)]) for pid in set(p.tolist())}
    for i,pid in enumerate(p.tolist()):out[i]=x[i,:,choices[int(pid)],:]
    return out

def infer_sparse(model,loader,available,fixed):
    import torch
    model.eval();values={k:[] for k in ['prediction','y','patient','contact']}
    with torch.no_grad():
        for x,y,p,c in loader:
            x=sparse_batch(x,p,available,fixed=fixed)
            values['prediction'].append(torch.sigmoid(model(x.cuda())[:,0]).cpu().numpy())
            values['y'].append(y.numpy());values['patient'].append(p.numpy());values['contact'].append(c.numpy())
    return {k:np.concatenate(v) for k,v in values.items()}

def freeze(project,output):
    project,output=Path(project),Path(output);old=project/'outputs/spes_agent_v4/multitrial/neural_protocol.json';base=read_json(old)
    sources=['experiments/spes_agent_v5/neural.py','experiments/spes_agent_v4/neural.py','vendor/norris2026_hat/src/models.py']
    value={'study':'v5 sparse-observation expert training','scope':'Adaptive exploratory follow-up to negative v4 agent pilot; no claim of fresh external validation',
      'question':'Does training with whole stimulation columns absent improve eight-measurement ranking relative to full-response-trained experts?',
      'jobs':base['jobs'],'records':base['records'],'max_trials':20,'max_epochs':25,'patience':5,'batch_size':8,
      'families':FAMILIES,'presets':{k:base['presets'][k] for k in FAMILIES},
      'training':'Same published fixed architectures/optimizer and patient/class sampler as v4, but each batch uses 1-8 real stimulus columns sampled per patient before model processing.',
      'early_stop':'Mean patient AUROC averaged across two fixed, patient-specific eight-column random subsets of the seven validation patients. No outer labels.',
      'outer_evaluation':'All five v4 numerical-policy measurement sequences are frozen inputs. Test each new expert on those exact eight-position sets. Do not claim that this replay retrains or evaluates the acquisition controller.',
      'primary':'Sparse HAT minus original full-response HAT on the fixed random_0 sequence, mean patient AUROC; also report every other sequence.',
      'new_API_calls':0,'new_neural_fits':10,'sources':{s:sha(project/s) for s in sources},'v4_protocol_sha256':sha(old)}
    path=output/'sparse_protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen sparse-training protocol changed')
    write_json(path,value);return value

def train_fold(project,output,outer):
    import torch
    project,output=Path(project),Path(output);protocol=read_json(output/'sparse_protocol.json')
    if any(sha(project/p)!=v for p,v in protocol['sources'].items()):raise ValueError('Frozen source differs')
    row=protocol['jobs'][outer];torch.set_num_threads(4)
    print('LOAD',outer,flush=True);patients,stats=prepare_patients(project,protocol,row)
    available={p['patient']:np.flatnonzero(p['x'][...,0].abs().sum(dim=(0,1)).numpy()>0).tolist() for p in patients.values()}
    if any(len(v)<8 for v in available.values()):raise ValueError('Sparse acquisition needs at least 8 columns')
    fixed=[{p:fixed_queries(a,outer,p,r) for p,a in available.items()} for r in range(2)]
    mods=upstream_models(project)
    for family in FAMILIES:
        folder=output/f'neural/outer{outer}/{family}';folder.mkdir(parents=True,exist_ok=True)
        if (folder/'receipt.json').exists():continue
        seed=593100+100*outer+FAMILIES.index(family);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed);loader,weight=loaders(patients,row,8,seed)
        model,_,preset=mods.create_model(family);model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=preset.learning_rate,weight_decay=preset.weight_decay)
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device='cuda'));best=-float('inf');stale=0;trace=[];best_state=None;start=time.perf_counter()
        for epoch in range(1,26):
            model.train();losses=[]
            for x,y,p,_ in loader['train']:
                x=sparse_batch(x,p,available,rng=rng);optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x.cuda())[:,0],y.cuda())
                if not torch.isfinite(loss):raise ValueError('Nonfinite sparse training loss')
                loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
            validations=[infer_sparse(model,loader['early_stop'],available,f) for f in fixed];score=float(np.mean([mean_auc(v) for v in validations]))
            trace.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation_two_subset_mean_AUROC':score})
            print('EPOCH',outer,family,epoch,round(score,6),flush=True)
            if score>best:best=score;stale=0;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=5:break
        model.load_state_dict(best_state);torch.save({'model_state_dict':best_state,'family':family,'seed':seed,'best_epoch':best_epoch},folder/'checkpoint.pth')
        for rep in range(2):np.savez_compressed(folder/f'validation_{rep}.npz',**infer_sparse(model,loader['early_stop'],available,fixed[rep]))
        # No outer prediction/accuracy is inspected during fitting. Exact old
        # numerical measurement sets are evaluated after the full ten-fit stage.
        write_json(folder/'receipt.json',{'outer':outer,'family':family,'seed':seed,'train':row['train'],'early_stop':row['early_stop'],'outer_test':row['predict'],
          'normalisation':stats,'best_epoch':best_epoch,'best_validation_AUROC':best,'trace':trace,'seconds':time.perf_counter()-start,
          'checkpoint_sha256':sha(folder/'checkpoint.pth'),'protocol_sha256':sha(output/'sparse_protocol.json'),
          'validation_queries':[{str(p):v for p,v in f.items() if f'ccepAgeUMCU{p:02d}' in row['early_stop']} for f in fixed]})
        print('FIT_DONE',outer,family,round(time.perf_counter()-start,1),flush=True);del model,optimizer,best_state;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--outer',type=int);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.outer is not None:train_fold(a.project,a.output,a.outer);return
    if not 1<=a.workers<=2:raise ValueError('One or two GPU workers')
    with exclusive(a.output/'training.lock'):
        freeze(a.project,a.output);queue=list(range(5));active=[];(a.output/'logs').mkdir(parents=True,exist_ok=True)
        while queue or active:
            while queue and len(active)<a.workers:
                outer=queue.pop(0)
                if all((a.output/f'neural/outer{outer}/{f}/receipt.json').exists() for f in FAMILIES):continue
                log=(a.output/f'logs/outer{outer}.log').open('a',encoding='utf8')
                process=subprocess.Popen([sys.executable,'-u','-X','utf8','-m','experiments.spes_agent_v5.neural','--project',str(a.project),'--output',str(a.output),'--outer',str(outer)],
                     cwd=a.project,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONHASHSEED='0'))
                active.append((process,log,outer));print('START',outer,flush=True)
            failure=False
            for process,log,outer in list(active):
                if process.poll() is None:continue
                log.close();active.remove((process,log,outer));print('DONE',outer,process.returncode,flush=True);failure|=process.returncode!=0
            if failure:
                for process,log,outer in active:process.wait();log.close()
                raise RuntimeError('Sparse training failed; preserve traces and inspect before resume')
            if queue or active:time.sleep(2)
        print('ALL_TEN_SPARSE_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
