"""Causal envelope and finite phase-search development diagnostics.

These activity proxies are not clinically calibrated seizure detectors.
"""
import numpy as np


def causal_variance(signal, window=50):
    y = np.asarray(signal, dtype=float)
    if y.ndim != 2 or len(y) < window or not np.isfinite(y).all():
        raise ValueError('Invalid recording')
    # Remove a fixed first-sample offset for numerical accuracy. Window variance
    # is translation invariant and each valid output uses only its own history.
    y = y - y[0]
    sums = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y, axis=0)])
    squares = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y*y, axis=0)])
    mean = (sums[window:] - sums[:-window]) / window
    variance = (squares[window:] - squares[:-window]) / window - mean*mean
    out = np.full_like(y, np.nan)
    out[window-1:] = np.maximum(variance, 0.)
    return out


def training_scale(variance, indices):
    scale = np.mean(variance[indices], axis=0)
    positive = scale[scale > 0]
    if not len(positive):
        raise ValueError('No training activity')
    return np.maximum(scale, .01*np.median(positive))


def envelope(signal, scale, window=50):
    return np.log1p(causal_variance(signal, window) / scale)


def phase_search(simulated, observed_prefix, times, offsets):
    """No observations after the supplied prefix enter the profile."""
    if len(observed_prefix) != len(times):
        raise ValueError('Prefix/time mismatch')
    losses = np.array([np.mean((simulated[times+o]-observed_prefix)**2)
                       for o in offsets])
    if not np.isfinite(losses).all():
        raise ValueError('Nonfinite phase loss')
    best = int(np.argmin(losses))
    near = np.asarray(offsets)[losses <= losses[best]*1.05 + 1e-12]
    return int(offsets[best]), losses, {
        'offset': int(offsets[best]), 'mse': float(losses[best]),
        'near_minimum_count': len(near),
        'near_minimum_offset_range': [int(near.min()), int(near.max())],
        'boundary_optimum': best in (0, len(offsets)-1)}


def fit_ar(training, lag, ridge):
    y = np.asarray(training)
    mean = y.mean(axis=0)
    centered = y-mean
    x = np.stack([centered[lag-k-1:len(y)-k-1] for k in range(lag)], axis=-1).reshape(-1,lag)
    target = centered[lag:].ravel()
    coefficients = np.linalg.solve(x.T@x + ridge*len(x)*np.eye(lag), x.T@target)
    companion = np.zeros((lag,lag)); companion[0] = coefficients
    if lag > 1: companion[1:,:-1] = np.eye(lag-1)
    radius = float(np.max(abs(np.linalg.eigvals(companion))))
    return coefficients, mean, radius


def ar_forecast(prefix, coefficients, mean, steps):
    history = [x.copy() for x in np.asarray(prefix)[-len(coefficients):]]
    result = []
    for _ in range(steps):
        pred = mean + coefficients @ (np.stack(history[::-1])-mean)
        pred = np.maximum(pred, 0.)
        result.append(pred); history = history[1:]+[pred]
    return np.asarray(result)


def trend_forecast(prefix, damping, steps):
    recent = np.asarray(prefix)[-10:]
    t = np.arange(len(recent), dtype=float); t -= t.mean()
    slope = t @ recent / (t@t)
    distance = np.cumsum(damping**np.arange(1,steps+1))
    return np.maximum(prefix[-1] + distance[:,None]*slope, 0.)


def episodes(series, times, threshold, minimum=3):
    """Return a high-activity proxy's episodes, preserving boundary censoring."""
    active = np.asarray(series) > threshold
    changes = np.diff(np.r_[False, active, False].astype(int))
    starts = np.flatnonzero(changes == 1); stops = np.flatnonzero(changes == -1)
    return [dict(first_active_sample=int(times[a]), last_active_sample=int(times[b-1]),
                 onset_sample=None if a == 0 else int(times[a]),
                 offset_sample=None if b == len(times) else int(times[b]),
                 left_censored=bool(a == 0), right_censored=bool(b == len(times)))
            for a,b in zip(starts,stops) if b-a >= minimum]


def errors(prediction, observed, times, threshold):
    pred = np.asarray(prediction); obs = np.asarray(observed)
    pg = pred.mean(axis=1); og = obs.mean(axis=1)
    return dict(mse=float(np.mean((pred-obs)**2)), mae=float(np.mean(abs(pred-obs))),
                high_activity_proxy_mismatch=float(np.mean((pg>threshold)!=(og>threshold))),
                predicted_high_activity_fraction=float(np.mean(pg>threshold)),
                observed_high_activity_fraction=float(np.mean(og>threshold)),
                predicted_episodes=episodes(pg,times,threshold),
                observed_episodes=episodes(og,times,threshold))
