"""New inner models exclude both the outer test and their own OOF patients."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from sklearn.model_selection import KFold

from .common import SLUGS, MODEL_NAMES, SELECTOR, WEIGHTS, check_groups, exclusive, read_json, sha, source_hashes, write_json

TRAIN_SOURCES = ['experiments/spes_agent_v3/common.py', 'experiments/spes_agent_v3/crossfit.py',
                 'reproduction/norris2024/runtime.py', 'reproduction/norris2024/train_reproduction.py']

def schedule(project):
    rows = []
    for outer in range(5):
        split = read_json(Path(project) / f'outputs/reproduction/norris2024/training/split_seed1_fold{outer}.json')
        development = sorted(split['train'] + split['val'])
        if len(set(development)) != 28 or set(development) & set(split['test']): raise ValueError('Invalid outer split')
        for inner, (fit, held) in enumerate(KFold(4, shuffle=True, random_state=7300+outer).split(development)):
            perm = np.random.default_rng(8100+outer*10+inner).permutation(fit)
            stop, train = sorted(perm[:4]), sorted(perm[4:])
            row = {'outer': outer, 'inner': inner, 'development': development,
                'train': [development[i] for i in train], 'early_stop': [development[i] for i in stop],
                'predict': [development[i] for i in sorted(held)], 'outer_test': sorted(split['test'])}
            check_groups(row['train'], row['early_stop'], row['predict'], row['outer_test'])
            rows.append(row)
    return rows

def freeze(project, output):
    upstream = Path(project) / 'vendor/norris2024/src'
    protocol = {'version': 3, 'outer_repeat': 1, 'outer_folds': 5, 'patients': 35,
        'inner_folds': 4, 'new_base_fits': 60, 'inner_sizes': {'train':17, 'early_stop':4, 'predict':7},
        'scope': 'Exploratory development on the already inspected public-code cohort; not an untouched external validation',
        'selector': SELECTOR, 'candidate_weights': WEIGHTS, 'training_rule': 'Original fixed hyperparameters, patience 10, ceiling 1000 epochs; no new hyperparameter search',
        'sources': source_hashes(project, TRAIN_SOURCES),
        'upstream_sources': {p.name: sha(p) for p in sorted(upstream.glob('*.py'))}, 'jobs': schedule(project)}
    path = Path(output) / 'training_protocol.json'
    if path.exists():
        if read_json(path) != protocol: raise ValueError('Frozen inner-training protocol changed')
    else: write_json(path, protocol)
    return protocol

def train_job(project, output, outer, inner):
    project, output = Path(project).resolve(), Path(output).resolve()
    protocol = read_json(output / 'training_protocol.json')
    if source_hashes(project, TRAIN_SOURCES) != protocol['sources']: raise ValueError('Inner-training source changed')
    row = next(r for r in protocol['jobs'] if r['outer'] == outer and r['inner'] == inner)
    folder = output / f'crossfit/outer{outer}_inner{inner}'
    if all((folder / slug / 'receipt.json').exists() for slug in SLUGS): return
    sys.path.insert(0, str(project / 'reproduction/norris2024'))
    from runtime import load_upstream
    from train_reproduction import MODELS
    upstream = project / 'vendor/norris2024'
    models = load_upstream(upstream)
    import torch
    import dataset
    from train import train_model
    from evaluate import get_preds
    torch.set_num_threads(4)
    original = dataset.glob.glob
    def filtered(pattern, *args, **kwargs):
        paths = original(pattern, *args, **kwargs)
        if 'X_stim' in str(pattern):
            paths = [p for p in paths if Path(p).stem.removeprefix('X_stim_') in set(row['development'])]
        return [Path(p).as_posix() for p in sorted(paths)]
    names = row['development']
    indices = {'train': np.array([names.index(p) for p in row['train']]),
               'val': np.array([names.index(p) for p in row['early_stop']]),
               'test': np.array([names.index(p) for p in row['predict']])}
    dataset.glob.glob = filtered
    dataset.get_splits = lambda *args, **kwargs: [indices]
    os.chdir(upstream / 'src')
    # The unchanged upstream loader normalizes from the 17 fitting patients only.
    train, stop, predict, weight = dataset.create_dataset('data/mean', 'data/std', 0, 0, batch_size=8)
    for index, (slug, name) in enumerate(zip(SLUGS, MODEL_NAMES)):
        target = folder / slug; target.mkdir(parents=True, exist_ok=True)
        if (target / 'receipt.json').exists(): continue
        seed = 3100000 + outer*10000 + inner*100 + index
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        cfg = dict(MODELS[name]); lr = cfg.pop('lr')
        net = models.get_model_instance(name, **cfg).to('cuda:0')
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight), device='cuda:0'))
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        begin = time.perf_counter()
        print('FIT', outer, inner, slug, flush=True)
        net = train_model(net, train, stop, optimizer, criterion, str(target/'checkpoint.pth'), device='cuda:0', num_epochs=1000, patience=10)
        y, p, coords, lobes, patient = get_preds(net, predict, 'cuda:0', mode='test')
        expected = {int(p[-2:]) for p in row['predict']}
        if set(patient) != expected: raise ValueError('Inner OOF prediction patients differ from frozen split')
        np.savez_compressed(target/'predictions.npz', y=y, prediction=p, patient=patient, coords=coords)
        write_json(target/'receipt.json', {'outer':outer, 'inner':inner, 'model':slug, 'seed':seed,
            'train':row['train'], 'early_stop':row['early_stop'], 'predict':row['predict'], 'outer_test':row['outer_test'],
            'seconds':time.perf_counter()-begin, 'checkpoint_sha256':sha(target/'checkpoint.pth'), 'prediction_sha256':sha(target/'predictions.npz')})
        print('FIT_DONE', outer, inner, slug, flush=True)
        del net, optimizer; gc.collect(); torch.cuda.empty_cache()

def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2); p.add_argument('--job'); args=p.parse_args()
    project, output=args.project.resolve(),args.output.resolve()
    if args.job:
        outer,inner=map(int,args.job.split(':'));train_job(project,output,outer,inner);return
    if not 1 <= args.workers <= 2: raise ValueError('The local training protocol permits one or two GPU workers')
    with exclusive(output/'training.lock'):
        protocol=freeze(project,output);queue=list(protocol['jobs']);active=[]
        logs=output/'logs';logs.mkdir(parents=True,exist_ok=True)
        while queue or active:
            while queue and len(active)<args.workers:
                row=queue.pop(0); key=f"{row['outer']}:{row['inner']}"
                if all((output/f"crossfit/outer{row['outer']}_inner{row['inner']}"/s/'receipt.json').exists() for s in SLUGS):continue
                log=(logs/f"outer{row['outer']}_inner{row['inner']}.log").open('a',encoding='utf-8')
                command=[sys.executable,'-u','-X','utf8','-m','experiments.spes_agent_v3.crossfit','--project',str(project),'--output',str(output),'--job',key]
                process=subprocess.Popen(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
                active.append((process,log,key));print('INNER_START',key,flush=True)
            failure=False
            for process,log,key in list(active):
                if process.poll() is None:continue
                log.close();active.remove((process,log,key))
                print('INNER_DONE',key,'exit',process.returncode,flush=True)
                failure |= process.returncode != 0
            if failure:
                for process,log,key in active:process.wait();log.close()
                raise RuntimeError('Inner training failed; inspect logs before resume')
            if queue or active:time.sleep(2)
        print('ALL 60 INNER MODEL FITS COMPLETE',flush=True)

if __name__=='__main__':main()
