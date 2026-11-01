"""Post-fit diagnostic only: no tuning, re-fitting, or changes to agent inputs."""
from pathlib import Path
import argparse
import json
import sys
import joblib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(); root = args.project.resolve(); sys.path.insert(0, str(root))
from experiments.spes_agent_v2.data import SLUGS, RESPONSE_FEATURES, patient_features, validation_patient_ids
from experiments.spes_agent_v2.verify import auc

output = args.output.resolve(); old = root / 'outputs/reproduction/norris2024/training'
cache = {}; scores = {'transformer': [], 'prediction_fusion': [], 'feature_fusion': []}; coefficients = []
for seed in range(1, 6):
    for fold in range(5):
        base = [np.load(old / f'{slug}_seed{seed}_fold{fold}/predictions.npz') for slug in SLUGS]
        first = base[0]; split = json.loads((old / f'split_seed{seed}_fold{fold}.json').read_text())
        patient = validation_patient_ids(first['val_patient'], split)
        x = np.column_stack([b['val_prediction'] for b in base]); response = np.zeros((len(patient), 15))
        for p in set(patient):
            if int(p) not in cache:cache[int(p)] = patient_features(root, int(p))[0]
            response[patient == p] = cache[int(p)]
        paths = output / 'fusion' / f'seed{seed}_fold{fold}'
        small, full = [joblib.load(paths / (name + '.joblib')) for name in ['prediction_fusion', 'feature_fusion']]
        coefficients.append(full.named_steps['classifier'].coef_[0])
        predictions = {'transformer': x[:, 2], 'prediction_fusion': small.predict_proba(x)[:, 1], 'feature_fusion': full.predict_proba(np.c_[x, response])[:, 1]}
        for name, prediction in predictions.items():
            for p in sorted(set(patient)):
                mask = patient == p; scores[name].append({'seed': seed, 'fold': fold, 'patient': int(p), 'AUROC': auc(first['val_y'][mask], prediction[mask])})
summary = json.loads((output / 'fusion_results.json').read_text()); coef = np.array(coefficients)
result = {'purpose': 'Post-fit diagnosis; no new tuning or agent prompt changes',
 'caveat': 'Fusion validation scores are resubstitution on fitting patients, not independent validation. The Transformer also used validation for early stopping.',
 'validation_resubstitution_mean_patient_AUROC': {k: float(np.mean([r['AUROC'] for r in v])) for k,v in scores.items()},
 'held_out_mean_patient_AUROC': {k: summary['mean_patient_AUROC'][k] for k in scores},
 'scaled_coefficient_across_25_fits': {name: {'mean': float(coef[:,i].mean()), 'sd': float(coef[:,i].std()), 'positive_fits': int((coef[:,i]>0).sum())} for i,name in enumerate(SLUGS+RESPONSE_FEATURES)}}
(output / 'fusion_postfit_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='scaled_coefficient_across_25_fits'},indent=2))
