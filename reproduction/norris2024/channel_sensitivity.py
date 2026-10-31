"""Re-evaluate saved Transformer models with the paper's channel-count ablation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from runtime import load_upstream
from train_reproduction import MODELS, score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.upstream=args.upstream.resolve();args.output=args.output.resolve()
    summary=json.loads((args.output/'table1_reproduction.json').read_text())
    if summary['Transformer (base)']['finished_folds'] != 25:
        raise RuntimeError('Complete the 25 primary Transformer folds first')
    os.chdir(args.upstream/'src')
    models=load_upstream(args.upstream)
    import torch,dataset
    from evaluate import get_preds
    torch.set_num_threads(4)
    original=dataset.glob.glob
    dataset.glob.glob=lambda pattern:[Path(f).as_posix() for f in sorted(original(pattern))]
    folder=args.output/'sensitivity';folder.mkdir(exist_ok=True)
    rows=[]
    for seed in [1,2,3,4,5]:
        for fold in range(5):
            path=folder/f'seed{seed}_fold{fold}.json'
            if path.exists():
                rows.extend(json.loads(path.read_text()));continue
            train,val,test,weight=dataset.create_dataset('data/mean','data/std',fold,seed,8)
            del train,val
            cfg=dict(MODELS['Transformer (base)']);cfg.pop('lr')
            model=models.get_model_instance('Transformer (base)',**cfg).cuda()
            checkpoint=torch.load(args.output/'training'/f'Transformer_base_seed{seed}_fold{fold}'/'checkpoint.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            batch=[]
            for count in [1,2,4,9,18,37,74]:
                rng=1000000+seed*10000+fold*100+count
                random.seed(rng);np.random.seed(rng);torch.manual_seed(rng);torch.cuda.manual_seed_all(rng)
                model.eegnets[0].random_channels=count
                y,prediction,_,_,patient=get_preds(model,test,'cuda:0',mode='test')
                batch.extend({'seed':seed,'fold':fold,'channels':count,**row} for row in score(y,prediction,patient))
            path.write_text(json.dumps(batch,indent=2));rows.extend(batch)
            print('SENSITIVITY DONE',seed,fold,flush=True)
            del model,test;torch.cuda.empty_cache()
    rng=np.random.default_rng(42)
    result=[]
    for count in [1,2,4,9,18,37,74]:
        subset=[r for r in rows if r['channels']==count]
        subjects=sorted(set(r['patient'] for r in subset))
        means=np.array([np.mean([r['AUROC'] for r in subset if r['patient']==subject]) for subject in subjects])
        bootstrap=means[rng.integers(0,len(means),(10000,len(means)))].mean(axis=1)
        result.append({'channels':count,'mean_patient_AUROC':float(means.mean()),
                       'patient_cluster_bootstrap_95CI':np.quantile(bootstrap,[.025,.975]).tolist()})
    (args.output/'channel_sensitivity.json').write_text(json.dumps({'rows':rows,'summary':result,
        'resampling_unit':'patient, averaging its repeated held-out evaluations first',
        'caveat':'Original inference channel RNG and bootstrap implementation are not published; ours are explicitly frozen.'},indent=2))


if __name__=='__main__':main()
