"""Twenty inner HAT fits for learning query value without in-sample targets.

Outer test patients are excluded from every inner fit, early-stop set and OOF
prediction set. Their raw waveforms and labels are not loaded by these jobs.
No language-model client is imported or constructed.
"""
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
from sklearn.model_selection import KFold
from experiments.spes_agent_v3.common import read_json,write_json as _atomic_write,sha,exclusive,check_groups
from experiments.spes_agent_v4.neural import prepare_patients,loaders,upstream_models,infer,mean_auc

def write_json(path,value):
    for attempt in range(10):
        try:return _atomic_write(path,value)
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05)

def freeze(project,output):
    project,output=Path(project),Path(output);base_path=project/'outputs/spes_agent_v4/multitrial/neural_protocol.json';base=read_json(base_path);jobs=[]
    for old in base['jobs']:
        outer=old['outer'];dev=sorted(old['train']+old['early_stop'])
        for inner,(fit,predict) in enumerate(KFold(4,shuffle=True,random_state=8300+outer).split(dev)):
            shuffled=np.random.default_rng(9100+outer*10+inner).permutation(fit);stop=sorted(shuffled[:4]);train=sorted(shuffled[4:])
            jobs.append({'outer':outer,'inner':inner,'train':[dev[i] for i in train],'early_stop':[dev[i] for i in stop],
              'predict':[dev[i] for i in sorted(predict)],'outer_test':old['predict'],'development':dev})
    sources=['experiments/spes_agent_v8/crossfit.py','experiments/spes_agent_v4/neural.py','vendor/norris2026_hat/src/models.py']
    result={'study':'v8 inner HAT models for query-value learning','scope':'Exploratory task-specific numeric learning, not evidence of collaboration benefit',
       'new_fits':20,'family':'hat','max_trials':20,'batch_size':8,'max_epochs':25,'patience':5,'preset':base['presets']['hat'],
       'training_rule':'Fixed published HAT; same sampler/loss as v4; seed per job; clone first best validation weights',
       'jobs':jobs,'records':[{k:r[k] for k in ['subject','patient','shape','dtype']} for r in base['records']],
       'sources':{p:sha(project/p) for p in sources},'outer_protocol_sha256':sha(base_path),
       'next_stage':'Generate query-value targets on each inner held-out patient. Candidate features must exclude future waveforms and labels; target is the later scorer AUROC change. No LLM experiments until numeric value learning is evaluated.',
       'new_API_calls':0}
    path=output/'crossfit_protocol.json'
    if path.exists() and read_json(path)!=result:raise ValueError('Frozen crossfit protocol changed')
    write_json(path,result);return result

def train_job(project,output,outer,inner):
    import torch
    project,output=Path(project),Path(output);protocol=read_json(output/'crossfit_protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['sources'].items()):raise ValueError('Frozen source changed')
    row=next(j for j in protocol['jobs'] if j['outer']==outer and j['inner']==inner)
    check_groups(row['train'],row['early_stop'],row['predict'],row['outer_test'])
    folder=output/f'crossfit/outer{outer}_inner{inner}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():return
    torch.set_num_threads(4);seed=782300+outer*1000+inner;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    local={**protocol,'records':[r for r in protocol['records'] if r['subject'] in row['development']]}
    print('LOAD',outer,inner,flush=True);patients,stats=prepare_patients(project,local,row)
    if set(patients)!=set(row['development']) or set(patients)&set(row['outer_test']):raise ValueError('An outer test patient was loaded')
    loader,weight=loaders(patients,row,8,seed);mods=upstream_models(project);model,_,preset=mods.create_model('hat');model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=preset.learning_rate,weight_decay=preset.weight_decay)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,device='cuda'));best=-float('inf');stale=0;trace=[];best_state=None;start=time.perf_counter()
    for epoch in range(1,26):
        model.train();losses=[]
        for x,y,*_ in loader['train']:
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(x.cuda())[:,0],y.cuda())
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
        validation=infer(model,loader['early_stop'],'cuda');score=mean_auc(validation)
        trace.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation_AUROC':score});print('EPOCH',outer,inner,epoch,round(score,6),flush=True)
        if score>best:best=score;stale=0;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if stale>=5:break
    model.load_state_dict(best_state);torch.save({'model_state_dict':best_state,'seed':seed,'best_epoch':best_epoch},folder/'checkpoint.pth')
    predicted=infer(model,loader['predict'],'cuda');np.savez_compressed(folder/'OOF_full_response.npz',**predicted)
    write_json(folder/'receipt.json',{'outer':outer,'inner':inner,'seed':seed,**{k:row[k] for k in ['train','early_stop','predict','outer_test']},
      'loaded_patients':sorted(patients),'normalisation':stats,'trace':trace,'best_epoch':best_epoch,'best_validation_AUROC':best,
      'seconds':time.perf_counter()-start,'checkpoint_sha256':sha(folder/'checkpoint.pth'),'OOF_sha256':sha(folder/'OOF_full_response.npz'),
      'protocol_sha256':sha(output/'crossfit_protocol.json')})
    print('FIT_DONE',outer,inner,round(time.perf_counter()-start,1),flush=True);del model,optimizer,best_state;gc.collect();torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--job');p.add_argument('--workers',type=int,default=2);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.job:train_job(a.project,a.output,*map(int,a.job.split(':')));return
    if not 1<=a.workers<=2:raise ValueError('One or two GPU workers')
    with exclusive(a.output/'crossfit.lock'):
        protocol=freeze(a.project,a.output)
        if a.freeze_only:print('Twenty-job HAT protocol frozen');return
        queue=list(protocol['jobs']);active=[];(a.output/'logs').mkdir(parents=True,exist_ok=True)
        while queue or active:
            while queue and len(active)<a.workers:
                row=queue.pop(0);key=f'{row["outer"]}:{row["inner"]}'
                if (a.output/f'crossfit/outer{row["outer"]}_inner{row["inner"]}/receipt.json').exists():continue
                log=(a.output/f'logs/outer{row["outer"]}_inner{row["inner"]}.log').open('a',encoding='utf8')
                process=subprocess.Popen([sys.executable,'-u','-X','utf8','-m','experiments.spes_agent_v8.crossfit','--project',str(a.project),'--output',str(a.output),'--job',key],
                  cwd=a.project,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONHASHSEED='0'))
                active.append((process,log,key));print('START',key,flush=True)
            failure=False
            for process,log,key in list(active):
                if process.poll() is None:continue
                log.close();active.remove((process,log,key));print('DONE',key,process.returncode,flush=True);failure|=process.returncode!=0
            if failure:
                for process,log,key in active:process.wait();log.close()
                raise RuntimeError('Inner training failed; inspect preserved logs before resume')
            if queue or active:time.sleep(2)
        print('ALL_TWENTY_INNER_HAT_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
