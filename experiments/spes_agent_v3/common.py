from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

SLUGS = ['CNN_divergent', 'CNN_convergent', 'Transformer_base']
MODEL_NAMES = ['CNN (divergent)', 'CNN (convergent)', 'Transformer (base)']
# Columns are divergent, convergent, Transformer; every alternative retains >=80% Transformer.
WEIGHTS = {'transformer': [0., 0., 1.], 'convergent_10': [0., .1, .9],
           'divergent_10': [.1, 0., .9], 'both_20': [.1, .1, .8]}
SELECTOR = {'ridge_alpha': 100., 'minimum_gain': .005, 'family_alpha': .05,
            'bootstrap_draws': 10000, 'bootstrap_seed': 4701,
            'features': ['transformer_std', 'convergent_disagreement', 'divergent_disagreement', 'convergent_top10_overlap'],
            'eligibility': 'Bonferroni one-sided patient-bootstrap lower bound exceeds minimum_gain on inner OOF predictions',
            'activation': 'Eligible candidate with highest ridge-predicted patient AUROC gain, only if predicted gain exceeds minimum_gain'}

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    temporary.replace(path)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes(project, paths):
    project = Path(project)
    return {str(p): hashlib.sha256((project / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}

def check_groups(train, stop, predict, outer_test):
    groups = [set(train), set(stop), set(predict)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i+1,3)):
        raise ValueError('A patient occurs in training, early stopping and/or OOF prediction')
    if set.union(*groups) & set(outer_test): raise ValueError('Outer test patient used in inner model development')

@contextmanager
def exclusive(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try: handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError: raise RuntimeError('A coordinator lock exists; do not run concurrent coordinators or remove a live lock') from None
    try:
        os.write(handle, json.dumps({'pid': os.getpid()}).encode()); os.close(handle)
        yield
    finally: path.unlink(missing_ok=True)
