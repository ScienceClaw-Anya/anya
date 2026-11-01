"""VEC inputs. Planners receive Observation only; generators stay evaluator-side."""
from dataclasses import dataclass
import csv
from pathlib import Path
import re
import zipfile
import numpy as np
from .model import normalized_weights


@dataclass(frozen=True)
class Observation:
    subject: str
    labels: tuple
    weights: np.ndarray
    gain: np.ndarray
    sensors: np.ndarray
    volumes_unverified: np.ndarray
    source: np.ndarray


@dataclass(frozen=True)
class PlannerObservation:
    """Only anatomy and recorded sensors; no simulated sources or parameters."""
    subject: str
    labels: tuple
    weights: np.ndarray
    gain: np.ndarray
    sensors: np.ndarray


def array(s):
    a=np.fromstring(s.strip('[]').replace('\n',' '),sep=' ')
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite parameters')
    return a


def load_recording(root,subject='sub-003',run='01'):
    root=Path(root)/'VirtualEpilepticCohort';p=root/'derivatives/tvb'/subject
    with zipfile.ZipFile(p/'struct'/f'{subject}_connectome.zip') as z:
        labels=tuple(l.split()[0] for l in z.read('centres.txt').decode().splitlines())
        weights,scale=normalized_weights(np.loadtxt(z.open('weights.txt')))
    atlas=list(csv.DictReader((root/'vep_atlas.tsv').open(),delimiter='\t'))
    atlas_names=tuple(row['region_name'] for row in atlas if int(row['label'])>0)
    if labels!=atlas_names:
        raise ValueError('Atlas/connectome label order mismatch')
    mono=np.loadtxt(p/'struct'/f'{subject}_gain.txt')
    contacts=list(csv.DictReader((root/subject/f'{subject}_electrodes.tsv').open(),delimiter='\t'))
    if mono.shape!=(len(contacts),len(labels)):
        raise ValueError('Gain shape mismatch')
    names=[r['name'] for r in contacts];mapped={}
    for i in range(len(names)-1):
        a=re.fullmatch(r"([A-Za-z']+)(\d+)",names[i]);b=re.fullmatch(r"([A-Za-z']+)(\d+)",names[i+1])
        if a and b and a[1]==b[1]:
            mapped[f'{a[1]}{a[2]}-{b[2]}']=abs(mono[i+1]-mono[i])
    prefix=root/subject/'ses-01/ieeg'/f'{subject}_ses-01_task-simulatedseizure_acq-clinicalhypothesis_run-{run}_ieeg'
    header=prefix.with_suffix('.vhdr').read_text()
    channels=[line.split('=',1)[1].split(',')[0] for line in header.splitlines() if re.match(r'^Ch\d+=',line)]
    gain=np.stack([mapped[ch] for ch in channels])
    # IEEE_FLOAT_32, MULTIPLEXED; 0.1 microvolt header resolution verified per channel.
    if 'DataOrientation=MULTIPLEXED' not in header or 'BinaryFormat=IEEE_FLOAT_32' not in header:
        raise ValueError('Unsupported BrainVision format')
    resolutions=[float(line.split(',')[2]) for line in header.splitlines() if re.match(r'^Ch\d+=',line)]
    sensors=np.fromfile(prefix.with_suffix('.eeg'),dtype='<f4').reshape(-1,len(channels))*resolutions
    return PlannerObservation(subject,labels,weights,gain,sensors)


def load_observation(root,subject='sub-003',run='01'):
    """Legacy evaluator diagnostic loader. Never pass this to model fitting."""
    o=load_recording(root,subject,run)
    p=Path(root)/'VirtualEpilepticCohort/derivatives/tvb'/subject
    with zipfile.ZipFile(p/'struct'/f'{subject}_connectome.zip') as z:
        volumes=np.loadtxt(z.open('volumes.txt'))
    source=np.load(p/'ses-01/clinicalhypothesis'/f'{subject}_simulated_source_timeseries_run-{run}.npz')['source_signal']
    return Observation(subject,o.labels,o.weights,o.gain,o.sensors,volumes,source)


def load_generator(root,subject='sub-003',run='01'):
    p=Path(root)/'VirtualEpilepticCohort/derivatives/tvb'/subject/'ses-01/clinicalhypothesis/parameters'
    ep=next(csv.DictReader((p/f'{subject}_epileptor_parameters_run-{run}.tsv').open(),delimiter='\t'))
    sp=next(csv.DictReader((p/f'{subject}_simulator_parameters_run-{run}.tsv').open(),delimiter='\t'))
    n=len(array(ep['x0']))
    params=np.stack([np.broadcast_to(array(ep[k]),(n,)) for k in ['x0','Iext','Iext2','slope','r','Ks','Kf','Kvf']])
    return dict(p=params,initial=array(sp['init_cond']),nsig=array(sp['noise_coeffs']),factor=float(sp['coupling_factor']),dt=float(sp['dt']),period=float(sp['period']))
