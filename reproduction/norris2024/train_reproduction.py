"""Reproduce all three Table 1 models; save held-out predictions and both threshold rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import subprocess
import time
from pathlib import Path

import numpy as np

from runtime import load_upstream

MODELS = {
    'CNN (divergent)': {'lr': 0.003962831229235175, 'dropout_rate': 0.21763415739071962, 'input_channels': 49},
    'CNN (convergent)': {'lr': 0.001289042623854371, 'dropout_rate': 0.44374819954858546, 'input_channels': 37},
    'Transformer (base)': {'lr': 0.00014782432202142655, 'dropout_rate': 0.4610858126530446, 'embedding_dim': 16, 'num_layers': 2},
}
PAPER = {'CNN (divergent)': {'auroc': .574, 'sd': .065},
         'CNN (convergent)': {'auroc': .666, 'sd': .067},
         'Transformer (base)': {'auroc': .730, 'sd': .082}}


def aggregate(args):
    rows = [json.loads(path.read_text()) for path in sorted((args.output/'training').glob('*/metrics.json'))]
    summary = {}
    for name in MODELS:
        selected = [r for r in rows if r['model'] == name]
        folds = [np.mean([p['AUROC'] for p in r['paper_threshold_per_patient']]) for r in selected]
        summary[name] = {'finished_folds': len(selected), 'expected_folds': 25,
                         'mean_AUROC': float(np.mean(folds)) if folds else None,
                         'SD_fold_AUROC': float(np.std(folds, ddof=1)) if len(folds)>1 else None,
                         'paper': PAPER[name], 'results': selected}
    temporary = args.output/f'table1_reproduction.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    temporary.replace(args.output/'table1_reproduction.json')
    return summary


def parallel(args):
    queue = [(seed,fold) for seed in args.seeds for fold in args.folds]
    active = []
    logdir = args.output/'logs'
    logdir.mkdir(exist_ok=True)
    while queue or active:
        while queue and len(active) < args.parallel_workers:
            seed,fold = queue.pop(0)
            command = [sys.executable,'-u','-X','utf8',str(Path(__file__).resolve()),
                       '--upstream',str(args.upstream),'--output',str(args.output),
                       '--seeds',str(seed),'--folds',str(fold),'--epochs',str(args.epochs),
                       '--device',args.device,'--models',*args.models]
            if args.allow_count_mismatch:
                command.append('--allow-count-mismatch')
            log = (logdir/f'train_seed{seed}_fold{fold}.log').open('w',encoding='utf-8')
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                       env=dict(os.environ,PYTHONHASHSEED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
            active.append((process,log,seed,fold))
            print('START FOLD',seed,fold,flush=True)
        for process,log,seed,fold in list(active):
            if process.poll() is None:
                continue
            log.close()
            active.remove((process,log,seed,fold))
            if process.returncode:
                raise RuntimeError(f'Training failed seed {seed} fold {fold}; inspect log')
            print('DONE FOLD',seed,fold,flush=True)
        if queue or active:
            time.sleep(2)
    summary = aggregate(args)
    print(json.dumps({name:{k:v for k,v in d.items() if k!='results'} for name,d in summary.items()}),flush=True)


def score(y, prediction, patient, thresholds=None):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
    rows = []
    for subject in np.unique(patient):
        mask = patient == subject
        truth, values = y[mask], prediction[mask]
        if len(np.unique(truth)) != 2:
            raise ValueError(f'Patient {subject} has only one class')
        precision, recall, _ = precision_recall_curve(truth, values)
        row = {'patient': int(subject), 'n': len(truth), 'positive': int(truth.sum()),
               'AUROC': float(roc_auc_score(truth, values)), 'AUPRC': float(auc(recall, precision))}
        if thresholds is not None:
            threshold = thresholds(values) if callable(thresholds) else thresholds
            tn, fp, fn, tp = confusion_matrix(truth, values > threshold, labels=[0, 1]).ravel()
            row.update(Sensitivity=float(tp/(tp+fn)), Specificity=float(tn/(tn+fp)), threshold=float(threshold))
            row['Youden'] = row['Sensitivity'] + row['Specificity'] - 1
        rows.append(row)
    return rows


def paper_threshold(val_y, val_p, val_patient):
    from sklearn.metrics import roc_curve
    ns = []
    for subject in np.unique(val_patient):
        mask = val_patient == subject
        values = val_p[mask]
        fpr, tpr, thresholds = roc_curve(val_y[mask], values)
        threshold = thresholds[int(np.argmax(tpr-fpr))]
        mad = np.median(np.abs(values-np.median(values)))
        if mad == 0 or not np.isfinite(threshold):
            raise ValueError('Paper threshold undefined for a validation patient; do not silently replace it')
        ns.append((threshold-np.median(values))/mad)
    return float(np.median(ns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-count-mismatch', action='store_true')
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--folds', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--parallel-workers', type=int, default=1)
    args = p.parse_args()
    args.upstream, args.output = args.upstream.resolve(), args.output.resolve()
    cohort = json.loads((args.output / 'cohort.json').read_text())
    if not cohort['paper_counts_match'] and not args.allow_count_mismatch:
        raise RuntimeError('Cohort counts do not match paper. See cohort.json; do not silently claim full reproduction.')
    if args.parallel_workers > 1:
        parallel(args)
        return
    os.chdir(args.upstream / 'src')
    models = load_upstream(args.upstream)
    import torch
    import dataset
    from train import train_model
    from evaluate import get_preds, calculate_youden_threshold
    torch.set_num_threads(4)
    # Original glob/set order is not published. Freeze lexical input order and save actual splits.
    original_glob = dataset.glob.glob
    dataset.glob.glob = lambda pattern: [Path(path).as_posix() for path in sorted(original_glob(pattern))]
    training_dir = args.output / 'training'
    training_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((args.upstream/'src').glob('*.py'))}
    protocol = {'upstream_commit': '90062e8c5fe5593713125eaf5f573faedb739f65', 'source_sha256': source_hashes,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'seeds': [1,2,3,4,5], 'folds': list(range(5)), 'epochs': args.epochs, 'models': MODELS,
                'source_compatibility': 'Isolated original TorchEEG RandomNoise loader; released torch 2.1.0+cu121 replaces unavailable development build. CPU filter jobs bounded. Glob paths normalized to POSIX before original string substitutions (Windows compatibility).',
                'order_and_rng': 'Lexical subject-file order; PYTHONHASHSEED=0; explicit Python/NumPy/Torch RNG seeds. Original notebook only seeds folds.',
                'thresholds': ['paper Appendix C validation-only median/MAD rule', 'released-code pooled validation Youden rule'],
                'cohort_counts_match': cohort['paper_counts_match'],
                'training_rule': 'Early stopping after 10 epochs without validation AUROC improvement. 1000-epoch ceiling follows released train_model default; the paper does not state its final-training ceiling. Ten epochs in Appendix D describe tuning, and the example notebook explicitly says its 10-epoch run differs from the paper.',
                'scope': 'Reconstruction of three primary models and held-out predictions; cohort/implementation discrepancies prevent a claim of exact paper reproduction. Channel sensitivity and outcome analyses require separately recorded stages.'}
    protocol_path = training_dir / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError('Existing training protocol differs; use a new output directory')
    temporary = training_dir/f'protocol.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    temporary.replace(protocol_path)
    for seed in args.seeds:
        for fold in args.folds:
            file_paths = sorted(original_glob('../data/mean/X_stim*.npy'))
            splits = dataset.get_splits(file_paths, seed=seed)[fold]
            split_names = {k: [Path(file_paths[i]).stem.removeprefix('X_stim_') for i in v] for k, v in splits.items()}
            if any(set(split_names[a]) & set(split_names[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
                raise RuntimeError('Patient leakage')
            split_path = training_dir / f'split_seed{seed}_fold{fold}.json'
            split_path.write_text(json.dumps(split_names, indent=2))
            train_loader, val_loader, test_loader, weight = dataset.create_dataset('data/mean', 'data/std', fold, seed, batch_size=8)
            for name in args.models:
                slug = name.replace(' ', '_').replace('(', '').replace(')', '')
                run_dir = training_dir / f'{slug}_seed{seed}_fold{fold}'
                run_dir.mkdir(exist_ok=True)
                done = run_dir / 'metrics.json'
                if done.exists():
                    continue
                run_seed = 100000 * seed + 100 * fold + list(MODELS).index(name)
                random.seed(run_seed); np.random.seed(run_seed); torch.manual_seed(run_seed); torch.cuda.manual_seed_all(run_seed)
                cfg = dict(MODELS[name]); learning_rate = cfg.pop('lr')
                net = models.get_model_instance(name, **cfg).to(args.device)
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(weight), device=args.device))
                optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
                begin = time.perf_counter()
                print('TRAIN', name, seed, fold, flush=True)
                net = train_model(net, train_loader, val_loader, optimizer, criterion,
                                  save_path=str(run_dir / 'checkpoint.pth'), device=args.device, num_epochs=args.epochs)
                val_y, val_p, val_patient = get_preds(net, val_loader, args.device, mode='validation')
                y, prediction, coords, lobes, patient = get_preds(net, test_loader, args.device, mode='test')
                # Keep all test labels inside numerical judge artifacts, never agent payloads.
                np.savez_compressed(run_dir / 'predictions.npz', y=y, prediction=prediction, patient=patient,
                                    coords=coords, lobes=lobes, val_y=val_y, val_prediction=val_p, val_patient=val_patient)
                _, code_threshold = calculate_youden_threshold(val_y, val_p)
                threshold_error = None
                try:
                    n = paper_threshold(val_y, val_p, val_patient)
                    paper_rows = score(y, prediction, patient, lambda v: np.median(v)+n*np.median(np.abs(v-np.median(v))))
                except ValueError as error:
                    n, paper_rows, threshold_error = None, score(y, prediction, patient), str(error)
                result = {'model': name, 'seed': seed, 'fold': fold, 'training_rng_seed': run_seed,
                          'seconds': time.perf_counter()-begin, 'paper_mad_n': n, 'paper_threshold_error': threshold_error,
                          'paper_threshold_per_patient': paper_rows,
                          'released_code_threshold_per_patient': score(y, prediction, patient, code_threshold)}
                result_temporary = run_dir/'metrics.tmp'
                result_temporary.write_text(json.dumps(result, indent=2))
                result_temporary.replace(done)
                aggregate(args)
                print('DONE', name, seed, fold, 'AUROC', np.mean([r['AUROC'] for r in paper_rows]), 'seconds', result['seconds'], flush=True)
                del net, optimizer
                torch.cuda.empty_cache()
            del train_loader, val_loader, test_loader
    summary = aggregate(args)
    print(json.dumps({name: {k:v for k,v in d.items() if k != 'results'} for name,d in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
