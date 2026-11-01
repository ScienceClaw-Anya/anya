"""Real response features and validation-only fusion; test labels stay in the judge."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SLUGS = ['CNN_divergent', 'CNN_convergent', 'Transformer_base']
SCORERS = ['transformer', 'mean_ensemble', 'prediction_fusion', 'feature_fusion']
RESPONSE_FEATURES = ['incoming_early', 'incoming_middle', 'incoming_late',
 'outgoing_early', 'outgoing_middle', 'outgoing_late', 'incoming_valid_fraction',
 'outgoing_valid_fraction', 'incoming_peak_magnitude', 'incoming_peak_latency',
 'outgoing_peak_magnitude', 'outgoing_peak_latency', 'direction_early', 'direction_middle', 'direction_late']

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    temporary.replace(path)

def rank(values):
    values = np.asarray(values, dtype=float)
    return rankdata(values, method='average') / len(values)

def direction_features(array):
    waveform = np.asarray(array[:, :, :-1], dtype=float)
    distance = array[:, :, -1]
    valid = np.isfinite(distance) & (distance > 13) & np.isfinite(waveform).all(axis=2)
    windows = [(0, 47), (47, 175), (175, waveform.shape[-1])]
    features = []
    for lo, hi in windows:
        rms = np.sqrt(np.mean(waveform[:, :, lo:hi] ** 2, axis=2))
        features.append(rank([np.median(row[mask]) if mask.any() else 0. for row, mask in zip(rms, valid)]))
    features.append(valid.mean(axis=1))
    magnitude, latency = [], []
    for trace, mask in zip(waveform, valid):
        if mask.any():
            summary = np.median(np.abs(trace[mask]), axis=0)
            magnitude.append(float(summary.max())); latency.append(float(summary.argmax()))
        else:
            magnitude.append(0.); latency.append(0.)
    features.extend([rank(magnitude), rank(latency)])
    return np.column_stack(features)

def patient_features(project, patient):
    folder = project / 'vendor/norris2024/data'
    subject = f'ccepAgeUMCU{int(patient):02d}'
    incoming = np.load(folder / 'mean' / f'X_recording_{subject}.npy')
    outgoing = np.load(folder / 'mean' / f'X_stim_{subject}.npy')
    a, b = direction_features(incoming), direction_features(outgoing)
    if len(a) != len(b): raise ValueError('Incoming/outgoing contact order length differs')
    features = np.column_stack([a[:, :3], b[:, :3], a[:, 3], b[:, 3], a[:, 4:6], b[:, 4:6], b[:, :3] - a[:, :3]])
    if features.shape[1] != len(RESPONSE_FEATURES) or not np.isfinite(features).all():
        raise ValueError('Invalid measured response features')
    labels = np.load(folder / 'main' / f'y_{subject}.npy')
    return features, labels

def equal_patient_weights(patient):
    weights = np.array([1 / np.sum(patient == p) for p in patient], dtype=float)
    return weights / weights.mean()

def validation_patient_ids(indices, split):
    # The released validation loader stores local positions 0..6, unlike test IDs.
    mapping = np.array([int(name[-2:]) for name in split['val']], dtype=int)
    indices = np.asarray(indices, dtype=int)
    if set(indices) != set(range(len(mapping))):
        raise ValueError('Unexpected validation-local patient indices')
    return mapping[indices]

def fit_fusion(x, y, patient):
    model = Pipeline([('scale', StandardScaler()), ('classifier', LogisticRegression(
        C=.1, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42))])
    weight = equal_patient_weights(patient)
    model.fit(x, y, scale__sample_weight=weight, classifier__sample_weight=weight)
    return model

def auc_by_patient(y, scores, patient):
    return [{'patient': int(p), 'AUROC': float(roc_auc_score(y[patient == p], scores[patient == p]))}
            for p in sorted(set(patient))]

def source_hashes(project):
    project = Path(project)
    paths = list((project / 'experiments/spes_agent_v2').glob('*.py'))
    paths += [project / name for name in ['skills/neurotarget-spes-tools/scripts/spes_tools.py',
        'skills/neurotarget-spes-tools/SKILL.md', 'neurotarget/scienceclaw_runtime.py', 'config/upstream-lock.json']]
    return {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def prepare(project, output, seeds=(1, 2, 3, 4, 5)):
    project, output = Path(project).resolve(), Path(output).resolve()
    if any((output / 'studies').glob('*/agent_protocol.json')):
        raise ValueError('Agent study already frozen; do not overwrite its cases or trained models')
    old = project / 'outputs/reproduction/norris2024'
    protocol = {'version': 2, 'scope': 'Exploratory follow-up on the already inspected public-code cohort',
       'feature_names': ['CNN_divergent', 'CNN_convergent', 'Transformer_base'] + RESPONSE_FEATURES,
       'models': ['prediction_fusion', 'feature_fusion'], 'C': .1,
       'fit_patients': 'Only seven original validation patients per outer split; original 21 numerical-training patients and seven test patients are separate',
       'sample_weights': 'Equal total weight for each validation patient',
       'tuning': 'No test-driven tuning or hyperparameter search; one fixed L2 logistic configuration',
       'limitations': ['Validation patients also supported original numerical early stopping', 'This cohort was already inspected; results are exploratory',
                      'Response features are summaries and ranks, not validated physiological mechanisms'], 'seeds': list(seeds)}
    marker = output / 'fusion/protocol.json'
    if marker.exists() and json.loads(marker.read_text()) != protocol:
        raise ValueError('Existing fusion protocol differs; choose a new output directory')
    write_json(marker, protocol)
    cache = {}
    def features_for(patient):
        if int(patient) not in cache: cache[int(patient)] = patient_features(project, int(patient))
        return cache[int(patient)]
    records, timings = [], []
    for seed in seeds:
        for fold in range(5):
            data = [np.load(old / 'training' / f'{slug}_seed{seed}_fold{fold}' / 'predictions.npz') for slug in SLUGS]
            first = data[0]
            for value in data[1:]:
                for key in ['val_patient', 'val_y', 'patient', 'y', 'coords']:
                    if not np.array_equal(value[key], first[key]): raise ValueError('Numerical prediction row order differs')
            split = json.loads((old / 'training' / f'split_seed{seed}_fold{fold}.json').read_text())
            val_p, test_p = validation_patient_ids(first['val_patient'], split), first['patient']
            train_ids = {int(p[-2:]) for p in split['train']}
            if train_ids & set(test_p) or set(val_p) & set(test_p) or train_ids & set(val_p):
                raise ValueError('Patient leakage')
            val_response = np.zeros((len(val_p), len(RESPONSE_FEATURES)))
            test_response = np.zeros((len(test_p), len(RESPONSE_FEATURES)))
            for patient in set(val_p):
                features, labels = features_for(patient)
                if not np.array_equal(labels, first['val_y'][val_p == patient]): raise ValueError('Validation contact order differs from released tensors')
                val_response[val_p == patient] = features
            for patient in set(test_p):
                features, labels = features_for(patient)
                if not np.array_equal(labels, first['y'][test_p == patient]): raise ValueError('Test contact order differs from released tensors')
                test_response[test_p == patient] = features
            x_val = np.column_stack([d['val_prediction'] for d in data])
            x_test = np.column_stack([d['prediction'] for d in data])
            started = time.perf_counter()
            small = fit_fusion(x_val, first['val_y'], val_p)
            full = fit_fusion(np.column_stack([x_val, val_response]), first['val_y'], val_p)
            fit_seconds = time.perf_counter() - started
            predictions = np.column_stack([x_test[:, 2], x_test.mean(axis=1), small.predict_proba(x_test)[:, 1],
                 full.predict_proba(np.column_stack([x_test, test_response]))[:, 1]])
            folder = output / 'fusion' / f'seed{seed}_fold{fold}'
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(small, folder / 'prediction_fusion.joblib'); joblib.dump(full, folder / 'feature_fusion.joblib')
            np.savez_compressed(folder / 'predictions.npz', scores=predictions, y=first['y'], patient=test_p)
            reliability = {slug: float(np.mean([r['AUROC'] for r in auc_by_patient(first['val_y'], x_val[:, i], val_p)])) for i, slug in enumerate(SLUGS)}
            receipt = {'seed': seed, 'fold': fold, 'validation_patient_count': len(set(val_p)), 'test_patient_count': len(set(test_p)),
                'fit_seconds': fit_seconds, 'validation_model_AUROC': reliability,
                'input_prediction_sha256': {slug: hashlib.sha256((old / 'training' / f'{slug}_seed{seed}_fold{fold}' / 'predictions.npz').read_bytes()).hexdigest() for slug in SLUGS}}
            write_json(folder / 'receipt.json', receipt); timings.append(receipt)
            for patient in sorted(set(test_p)):
                mask = test_p == patient
                case_id = hashlib.sha256(f'{seed}:{fold}:{int(patient)}'.encode()).hexdigest()[:16]
                coords = first['coords'][mask]; distance = cdist(coords, coords)
                np.fill_diagonal(distance, np.inf)
                neighbors = np.argsort(distance, axis=1)[:, :min(3, len(coords)-1)]
                contacts = [{'id': f'c{i:03d}', 'models': [round(float(v), 5) for v in x_test[mask][i]],
                    'responses': [round(float(v), 5) for v in test_response[mask][i]],
                    'neighbors': [f'c{int(j):03d}' for j in neighbors[i]]} for i in range(mask.sum())]
                public = {'task': 'Rank measured contacts by SOZ likelihood from supplied SPES data. Retrospective classification, not a treatment recommendation.',
                  'model_order': SLUGS, 'response_feature_order': RESPONSE_FEATURES, 'contacts': contacts,
                  'feature_notes': 'RMS/peak ranks summarize valid measured responses beyond 13 mm. Early/middle/late windows follow the existing reconstruction. Peak latency is a within-patient sample-index rank. Geometric neighbors are not causal pathways.'}
                # Tools receive public numerical outputs only; no clinical labels or patient identifiers.
                tool_data = {'signal': {'contact_ids': [c['id'] for c in contacts], 'features': [c['responses'][:6] + c['responses'][8:12] for c in contacts]},
                  'network': {'contact_ids': [c['id'] for c in contacts], 'features': [c['responses'][6:8] + c['responses'][12:] for c in contacts], 'neighbors': [c['neighbors'] for c in contacts]},
                  'reliability': {'validation_patient_count': len(set(val_p)), 'validation_model_AUROC': reliability, 'note': 'Validation evidence only; no held-out accuracy is available.'},
                  'scorers': {name: [round(float(v), 6) for v in predictions[mask, i]] for i, name in enumerate(SCORERS)}}
                case = output / 'cases' / case_id
                write_json(case / 'public.json', public); write_json(case / 'tools.json', tool_data)
                write_json(case / 'judge.json', {'seed': seed, 'fold': fold, 'patient': int(patient), 'y': first['y'][mask].astype(int).tolist()})
                metrics = {name: float(roc_auc_score(first['y'][mask], predictions[mask, i])) for i, name in enumerate(SCORERS)}
                records.append({'case_id': case_id, 'seed': seed, 'fold': fold, 'patient': int(patient), 'metrics': metrics})
    patients = sorted({r['patient'] for r in records})
    averages = {name: np.array([np.mean([r['metrics'][name] for r in records if r['patient'] == p]) for p in patients]) for name in SCORERS}
    rng = np.random.default_rng(42); draws = rng.integers(0, len(patients), (10000, len(patients)))
    differences = {}
    for name in SCORERS[1:]:
        delta = averages[name] - averages['transformer']
        differences[name + ' minus transformer'] = {'mean': float(delta.mean()), 'patient_bootstrap_95CI': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
    result = {'protocol': protocol, 'patients': len(patients), 'held_out_cases': len(records), 'trained_fusion_models': len(timings) * 2,
      'fit_seconds_sum': sum(r['fit_seconds'] for r in timings), 'mean_patient_AUROC': {k: float(v.mean()) for k, v in averages.items()},
      'paired_comparisons': differences, 'cases': records}
    write_json(output / 'fusion_results.json', result)
    return {k: v for k, v in result.items() if k not in {'cases', 'protocol'}}

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(json.dumps(prepare(args.project, args.output), indent=2))
