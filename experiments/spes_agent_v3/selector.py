"""Fixed low-dimensional patient selector with an OOF evidence gate."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from .common import SELECTOR, WEIGHTS

def features(predictions):
    x = np.asarray(predictions, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError('Expected finite divergent/convergent/Transformer probabilities')
    k = min(10, len(x)); t = set(np.argsort(x[:,2], kind='stable')[-k:]); c = set(np.argsort(x[:,1], kind='stable')[-k:])
    return np.array([x[:,2].std(), np.abs(x[:,1]-x[:,2]).mean(), np.abs(x[:,0]-x[:,2]).mean(), len(t&c)/k])

def candidates(predictions):
    return {name: np.asarray(predictions, dtype=float) @ np.asarray(weight) for name, weight in WEIGHTS.items()}

def fit_selector(predictions, y, patient):
    predictions, y, patient = np.asarray(predictions), np.asarray(y), np.asarray(patient)
    patients = sorted(set(patient)); x = []; delta = []; base_auc = []; candidate_auc = []
    alternatives = list(WEIGHTS)[1:]
    for p in patients:
        mask = patient == p; values = candidates(predictions[mask]); x.append(features(predictions[mask]))
        scores = {name: roc_auc_score(y[mask], value) for name, value in values.items()}
        base_auc.append(scores['transformer']); candidate_auc.append([scores[n] for n in alternatives])
        delta.append([scores[n]-scores['transformer'] for n in alternatives])
    x, delta = np.array(x), np.array(delta)
    model = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=SELECTOR['ridge_alpha']))])
    model.fit(x, delta)
    rng = np.random.default_rng(SELECTOR['bootstrap_seed'])
    draw = rng.integers(0,len(patients),(SELECTOR['bootstrap_draws'],len(patients)))
    lower = np.quantile(delta[draw].mean(axis=1), SELECTOR['family_alpha']/len(alternatives), axis=0)
    evidence = {name: {'OOF_mean_gain':float(delta[:,i].mean()), 'bootstrap_lower_bound':float(lower[i]),
        'eligible':bool(lower[i]>SELECTOR['minimum_gain']), 'OOF_mean_AUROC':float(np.mean(candidate_auc,axis=0)[i])}
        for i,name in enumerate(alternatives)}
    return {'model':model, 'alternatives':alternatives, 'evidence':evidence, 'training_patients':list(map(int,patients)),
        'OOF_transformer_AUROC':float(np.mean(base_auc)), 'config':SELECTOR,
        'training_features':x, 'training_targets':delta}

def select(fitted, predictions):
    expected = fitted['model'].predict(features(predictions)[None])[0]
    eligible = [(float(expected[i]), name) for i,name in enumerate(fitted['alternatives'])
        if fitted['evidence'][name]['eligible'] and expected[i]>SELECTOR['minimum_gain']]
    name = max(eligible)[1] if eligible else 'transformer'
    return {'candidate':name, 'scores':candidates(predictions)[name],
        'predicted_gain':{n:float(expected[i]) for i,n in enumerate(fitted['alternatives'])},
        'patient_features':features(predictions).tolist(), 'weights':WEIGHTS[name]}
