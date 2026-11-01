import numpy as np
from intervention.covariance_fit import source_variances


def test_covariance_inverse_recovery_and_sensor_permutation():
    g=np.array([[1.,.1,.2],[.2,1.,.1],[.1,.2,1.],[.4,.1,.3]])
    q=np.array([.2,2.,.8]);c=(g*q)@g.T
    fitted,error=source_variances(c,g,ridge=1e-8)
    np.testing.assert_allclose(fitted,q,rtol=1e-6)
    assert error<1e-6 and (fitted>=0).all()
    perm=[2,0,3,1]
    again,_=source_variances(c[np.ix_(perm,perm)],g[perm],ridge=1e-8)
    np.testing.assert_allclose(fitted,again,atol=1e-12)
