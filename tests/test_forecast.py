import dataclasses
import numpy as np
from intervention.data import PlannerObservation
from intervention.forecast import sensor_network_operator,fit_linear,predict


def test_observation_only_schema_and_atlas_permutation():
    assert {f.name for f in dataclasses.fields(PlannerObservation)}=={'subject','labels','weights','gain','sensors'}
    w=np.array([[0.,.2,.4],[.2,0.,.1],[.4,.1,0.]])
    g=np.array([[1.,.2,.1],[.1,1.,.2],[.2,.1,1.],[.4,.2,.3]])
    perm=[2,0,1]
    a=sensor_network_operator(w,g,.01)
    b=sensor_network_operator(w[np.ix_(perm,perm)],g[:,perm],.01)
    np.testing.assert_allclose(a,b,atol=1e-12)
    np.testing.assert_allclose(sensor_network_operator(np.zeros_like(w),g,.01),0,atol=0)


def test_multistep_forecast_uses_only_origin_and_training():
    rng=np.random.default_rng(42)
    w=np.array([[0.,.5],[.5,0.]])
    g=np.eye(2);op=sensor_network_operator(w,g,.01)
    y=np.zeros((500,2))
    a=.8*np.eye(2)+.1*op
    for t in range(len(y)-1): y[t+1]=a@y[t]+rng.normal(0,.1,2)
    m=fit_linear(y[:350],op)
    origin=y[350:351].copy()
    p=predict(m,origin,10)
    y[351:]=1e9
    np.testing.assert_array_equal(p,predict(m,origin,10))
    manual=origin.copy()
    for _ in range(10): manual=(manual-m.mean)@m.transition.T+m.mean
    np.testing.assert_allclose(p,manual,atol=1e-12)
