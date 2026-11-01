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


def test_recording_loader_does_not_require_hidden_assets(tmp_path,monkeypatch):
    import zipfile
    from intervention.data import load_recording
    root=tmp_path/'VirtualEpilepticCohort'
    anatomy=root/'derivatives/tvb/sub-003/struct'
    anatomy.mkdir(parents=True)
    (root/'sub-003/ses-01/ieeg').mkdir(parents=True)
    (root/'vep_atlas.tsv').write_text('label\tregion_name\n1\tA\n2\tB\n3\tC\n')
    with zipfile.ZipFile(anatomy/'sub-003_connectome.zip','w') as z:
        z.writestr('centres.txt','A 0 0 0\nB 1 0 0\nC 0 1 0\n')
        z.writestr('weights.txt','0 1 0\n1 0 1\n0 1 0\n')
    np.savetxt(anatomy/'sub-003_gain.txt',np.arange(12).reshape(4,3)+1.)
    (root/'sub-003/sub-003_electrodes.tsv').write_text('name\nA1\nA2\nB1\nB2\n')
    prefix=root/'sub-003/ses-01/ieeg/sub-003_ses-01_task-simulatedseizure_acq-clinicalhypothesis_run-01_ieeg'
    prefix.with_suffix('.vhdr').write_text('DataOrientation=MULTIPLEXED\nBinaryFormat=IEEE_FLOAT_32\nCh1=A1-2,,0.1,µV\nCh2=B1-2,,0.1,µV\n')
    np.arange(20,dtype='<f4').tofile(prefix.with_suffix('.eeg'))
    def forbidden(*args,**kwargs): raise AssertionError('NPZ/source access forbidden')
    monkeypatch.setattr(np,'load',forbidden)
    observation=load_recording(tmp_path)
    assert observation.sensors.shape==(10,2)
    assert isinstance(observation,PlannerObservation)
