"""Pure score transformations. No clinical labels enter runtime acquisition."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

RECIPES = ['raw_platt', 'prior_platt', 'centered_platt']
POLICIES = ['prior_uncertainty', 'learned_boundary', 'zero_boundary']


def logits(score):
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise ValueError('Scores must be finite and in [0,1]')
    return logit(np.clip(score, 1e-12, 1-1e-12))


def location_scale(score):
    z = logits(score)
    center = float(np.median(z))
    scale = max(float(np.median(np.abs(z-center))), 1e-6)
    return z, center, scale


def features(score, recipe, ratio):
    z, center, scale = location_scale(score)
    if ratio <= 0 or not np.isfinite(ratio):
        raise ValueError('Positive training-only class ratio required')
    if recipe == 'raw_platt': return z
    if recipe == 'prior_platt': return z-2*np.log(ratio)
    if recipe == 'centered_platt': return (z-center)/scale
    raise ValueError('Unknown calibration recipe')


def fit_platt(x, y, weights, prevalence):
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000)
    model.fit(np.asarray(x).reshape(-1, 1), y, sample_weight=weights)
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    fallback = a <= 0
    if fallback: a, b = 0., float(logit(np.clip(prevalence, 1e-12, 1-1e-12)))
    return {'slope': a, 'intercept': b, 'nonpositive_slope_constant_fallback': fallback}


def calibrate(score, recipe, ratio, params):
    return expit(params['slope']*features(score, recipe, ratio)+params['intercept'])


def boundary_offset(score, labels):
    """Youden-optimal finite threshold, expressed in this patient's logit MADs."""
    z, center, scale = location_scale(score)
    fpr, tpr, threshold = roc_curve(labels, z, drop_intermediate=False)
    valid = np.isfinite(threshold)
    best = np.max((tpr-fpr)[valid])
    threshold = float(np.median(threshold[valid & np.isclose(tpr-fpr, best, rtol=0, atol=1e-12)]))
    return float((threshold-center)/scale)


def uncertainty_weights(matrix, policy, ratio, boundary_n):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError('Require baseline/MTE/HAT scores for each recording contact')
    if policy == 'prior_uncertainty':
        q = expit(logits(matrix)-2*np.log(ratio))
        return .05+q.std(axis=1)+.1*(1-2*np.abs(q[:, 2]-.5))
    z, center, scale = location_scale(matrix[:, 2])
    if policy == 'learned_boundary': target = center+boundary_n*scale
    elif policy == 'zero_boundary': target = 0.
    else: raise ValueError('Unknown acquisition policy')
    return .05+np.exp(-np.abs((z-target)/scale))


def acquisition_table(kernel, selected, available, matrix, policy, ratio, boundary_n):
    current = kernel[:, selected].max(axis=1) if selected else np.zeros(len(kernel))
    weights = uncertainty_weights(matrix, policy, ratio, boundary_n)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Invalid uncertainty weights')
    return [{'id': f'p{q:03d}', 'gain': float(np.average(np.maximum(kernel[:, q]-current, 0.), weights=weights))}
            for q in available if q not in selected]


def choose(rows):
    if not rows: raise ValueError('No available physical pair')
    return min(rows, key=lambda r: (-r['gain'], r['id']))['id']
