"""Audit public ds004080 metadata; never infer eligibility from model performance."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root: Path) -> dict:
    subjects = []
    for folder in sorted(root.glob('sub-*')):
        electrodes = sorted(folder.glob('ses-*/ieeg/*electrodes.tsv'))
        if not electrodes:
            continue
        ed = pd.read_csv(electrodes[0], sep='\t', index_col=0)
        if 'soz' not in ed or not ed.soz.eq('yes').any():
            continue
        runs = []
        for f in sorted(folder.glob('ses-*/ieeg/*ieeg.eeg')):
            pointer = f.read_text()
            match = re.search(r'SHA256E-s(\d+)--([a-f0-9]{64})\.eeg', pointer)
            if not match:
                raise ValueError(f'Not an annex pointer: {f}')
            stem = f.name[:-9]
            channels = pd.read_csv(f.with_name(stem + '_channels.tsv'), sep='\t', index_col=0)
            included = set(channels.index[channels.status_description.eq('included')])
            events = pd.read_csv(f.with_name(stem + '_events.tsv'), sep='\t')
            stim = events[events.trial_type.eq('electrical_stimulation')].copy()
            stim['pair'] = stim.electrical_stimulation_site.map(lambda x: '-'.join(sorted(x.split('-'))))
            eligible = set()
            pairs = []
            for pair, group in stim.groupby('pair'):
                names = pair.split('-')
                if len(group) < 5 or len(names) != 2 or not set(names).issubset(ed.index):
                    continue
                midpoint = ed.loc[names, ['x', 'y', 'z']].mean().to_numpy()
                record = ed.loc[ed.index.isin(included)]
                distances = np.linalg.norm(record[['x', 'y', 'z']].to_numpy() - midpoint, axis=1)
                receivers = set(record.index[distances > 13]) - set(names)
                if receivers:
                    eligible.update(set(names) & included)
                    pairs.append(pair)
            runs.append({'eeg_path': f.relative_to(root).as_posix(), 'bytes': int(match[1]),
                         'sha256': match[2], 'preflight_stimulated': sorted(eligible), 'pairs': pairs})
        stim_names = set().union(*(set(r['preflight_stimulated']) for r in runs)) if runs else set()
        positives = sorted(set(ed.index[ed.soz.eq('yes')]) & stim_names)
        subjects.append({'subject': folder.name, 'soz_contacts_in_metadata': int(ed.soz.eq('yes').sum()),
                         'preflight_stimulated_soz': positives, 'runs': runs})
    return {'dataset': 'ds004080', 'metadata_only': True,
            'eligibility_warning': 'Final eligibility and electrode counts require original preprocessing, artifact exclusions and both incoming/outgoing responses. Do not discard subjects to match paper numbers.',
            'subjects_with_any_soz': len(subjects), 'recording_bytes': sum(r['bytes'] for s in subjects for r in s['runs']),
            'subjects': subjects}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'subjects': result['subjects_with_any_soz'], 'GiB': result['recording_bytes']/2**30,
                      'no_stimulated_soz_preflight': [s['subject'] for s in result['subjects'] if not s['preflight_stimulated_soz']]}))
