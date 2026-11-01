import numpy as np
from intervention.temporal import causal_variance, phase_search, fit_ar, ar_forecast, episodes


def test_causal_envelope_matches_direct_windows_and_ignores_future():
    rng = np.random.default_rng(81)
    signal = rng.normal(size=(130,4)) + 12
    actual = causal_variance(signal, 20)
    expected = np.stack([signal[t-19:t+1].var(axis=0) for t in range(19,130)])
    np.testing.assert_allclose(actual[19:], expected, atol=1e-12)
    altered = signal.copy(); altered[70:] += 1000*rng.normal(size=(60,4))
    np.testing.assert_array_equal(actual[:70], causal_variance(altered,20)[:70])


def test_phase_alignment_known_shift_and_channel_permutation():
    rng = np.random.default_rng(83)
    trajectory = rng.normal(size=(200,5))
    times = np.arange(20,70); observed = trajectory[times+30]
    offset, losses, profile = phase_search(trajectory, observed, times, np.arange(0,60,5))
    assert offset == 30 and losses.min() == 0 and profile['near_minimum_count'] == 1
    perm = [4,0,3,1,2]
    other = phase_search(trajectory[:,perm], observed[:,perm], times, np.arange(0,60,5))
    np.testing.assert_allclose(losses, other[1], atol=1e-14)
    changed = trajectory.copy(); changed[140:] = 9999
    assert phase_search(changed,observed,times,np.arange(0,60,5))[0] == 30


def test_recursive_prediction_and_censored_activity_are_explicit():
    prefix = np.array([[3.,5.],[2.,3.]])
    actual = ar_forecast(prefix,np.array([.5]),np.array([0.,1.]),3)
    np.testing.assert_allclose(actual,[[1.,2.],[.5,1.5],[.25,1.25]])
    fit = fit_ar(np.tile(np.arange(50)[:,None],(1,3))*.01,1,.001)
    perm = fit_ar(np.tile(np.arange(50)[:,None],(1,3))[:,[2,0,1]]*.01,1,.001)
    np.testing.assert_allclose(fit[0],perm[0])
    events = episodes([2,2,2,0,0,2,2,2],np.arange(8)*10,1)
    assert events[0]['onset_sample'] is None and events[0]['offset_sample'] == 30
    assert events[1]['onset_sample'] == 50 and events[1]['offset_sample'] is None
