"""Observation-only policies; no generator input or evaluator outcomes."""
import numpy as np
from scipy.ndimage import uniform_filter1d


def sensor_onset_scores(observation,window=50):
    """Exploratory simulated-SEEG onset proxy, not a clinical SOZ annotation.

    50-sample local variance, threshold four times baseline variance (first
    100 samples). Require 25 active samples in a 50-sample centered window.
    Project inverse onset latency through row-normalized absolute bipolar gain.
    """
    y=observation.sensors
    variance=np.maximum(0,uniform_filter1d(y*y,window,axis=0)-uniform_filter1d(y,window,axis=0)**2)
    base=np.maximum(np.median(variance[:100],axis=0),1e-12)
    active=uniform_filter1d((variance>4*base).astype(float),window,axis=0)>=0.5
    active[:100]=False
    first=np.where(active.any(axis=0),active.argmax(axis=0),np.inf)
    sensor_score=np.where(np.isfinite(first),1/(1+first/100),0)
    g=observation.gain
    g=g/np.maximum(g.sum(axis=1,keepdims=True),1e-30)
    return g.T@sensor_score


def ranking(score,labels):
    return sorted(range(len(labels)),key=lambda i:(-float(score[i]),labels[i]))


def choose_ranked(score,labels,candidates):
    return min(candidates,key=lambda a:(-sum(score[i] for i in a),tuple(labels[i] for i in a)))


def network_scores(observation,localization):
    # W[target,source], so column sum is outgoing weighted strength.
    strength=observation.weights.sum(axis=0)
    return localization*(1+strength/max(float(strength.max()),1e-30))
