"""Run original preprocessing per subject, with resumable receipts and saved contact IDs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def process(args):
    import numpy as np
    import pandas as pd
    import mne
    os.chdir(args.upstream / 'src')
    sys.path.insert(0, str(args.upstream / 'src'))
    from create_dataset import BIDSDataLoader, StimulationDataProcessor, DatasetCreator, combine_stats
    mne.set_log_level('WARNING')
    original_filter = mne.io.BaseRaw.filter

    def bounded_filter(self, *a, **kw):
        kw['n_jobs'] = args.filter_jobs
        return original_filter(self, *a, **kw)

    mne.io.BaseRaw.filter = bounded_filter
    loader = BIDSDataLoader(str(args.raw_root))
    subject = args.subject.removeprefix('sub-')
    session = loader.load_session_data(subject)
    parts = []
    processor = StimulationDataProcessor(tmin=0.009, tmax=1)
    for run in session['runs']:
        print('Processing', subject, run, flush=True)
        data = loader.load_run_data(subject, run)
        part = processor.process_run_data(data['eeg'], data['events_df'], data['channels_df'], subject)
        if part is not None:
            parts.append(part)
        del data, part
        gc.collect()
    if not parts:
        raise RuntimeError(f'No valid responses for {subject}')
    responses = pd.concat(parts)
    grouped = responses.groupby(['recording', 'stim_1', 'stim_2'])
    responses = pd.concat([pd.concat(combine_stats(group)) for _, group in grouped])
    for subdir in ['mean', 'std', 'main']:
        (args.upstream / 'data' / subdir).mkdir(exist_ok=True, parents=True)
    # Run the author's numerical transformation without changing its formulas.
    DatasetCreator(responses).process_for_analysis(subject, session['electrodes_tsv'])
    output = args.output / 'subjects'
    output.mkdir(parents=True, exist_ok=True)
    responses.to_pickle(output / f'{subject}_responses.pkl')
    y_path = args.upstream / 'data' / 'main' / f'y_{subject}.npy'
    result = {'subject': subject, 'runs': session['runs'], 'response_rows': len(responses),
              'eligible': y_path.exists(), 'filter_jobs': args.filter_jobs}
    if y_path.exists():
        y = np.load(y_path)
        result.update(electrodes=len(y), positives=int(y.sum()))
        # Reconstruct the author's exact set order, using this process's frozen hash seed.
        ed = session['electrodes_tsv']
        df = responses[responses.metric.eq('mean')].copy()
        p1 = ed.loc[df.stim_1, ['x', 'y', 'z']].to_numpy()
        p2 = ed.loc[df.stim_2, ['x', 'y', 'z']].to_numpy()
        pr = ed.loc[df.recording, ['x', 'y', 'z']].to_numpy()
        distances = np.linalg.norm((p1+p2)/2-pr, axis=1)
        df = df.loc[distances > 13].copy()
        df['distances'] = distances[distances > 13]
        df = df.sort_values('distances')
        names = list(set(df.recording.unique()).intersection(set(df.stim_2.unique()).union(set(df.stim_1.unique()))))
        coords = np.load(args.upstream / 'data' / 'main' / f'coords_{subject}.npy')
        if not np.array_equal(y, ed.loc[names].soz.eq('yes').to_numpy()) or not np.allclose(coords, ed.loc[names, ['x', 'y', 'z']].to_numpy()):
            raise RuntimeError('Contact identity reconstruction does not match saved tensors')
        result['contacts'] = names
        result['arrays'] = {}
        for path in sorted((args.upstream / 'data').glob(f'*/*{subject}.npy')):
            result['arrays'][path.relative_to(args.upstream).as_posix()] = {
                'shape': list(np.load(path, mmap_mode='r').shape),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / f'{subject}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


def supervise(args):
    manifest = json.loads(args.manifest.read_text())
    waiting = [s['subject'] for s in manifest['subjects']]
    active = {}
    logs = args.output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while waiting or active:
        for subject in list(waiting):
            if (args.output / 'subjects' / f'{subject.removeprefix("sub-")}.json').exists():
                waiting.remove(subject)
                continue
            if len(active) >= args.workers:
                break
            folder = args.raw_root / subject
            vhdrs = list(folder.rglob('*ieeg.vhdr'))
            # A subject may be copied before all its large recordings are verified.
            expected = next(s for s in manifest['subjects'] if s['subject'] == subject)
            eegs = [args.raw_root / r['eeg_path'] for r in expected['runs']]
            if not eegs or not all(p.with_suffix('.eeg.verified.json').exists() and p.with_suffix('.vhdr.verified.json').exists() and p.with_suffix('.vmrk.verified.json').exists() for p in eegs):
                continue
            log = (logs / f'preprocess_{subject}.log').open('w', encoding='utf-8')
            command = [sys.executable, '-u', '-X', 'utf8', str(Path(__file__).resolve()),
                       '--subject', subject, '--raw-root', str(args.raw_root), '--upstream', str(args.upstream),
                       '--output', str(args.output), '--filter-jobs', str(args.filter_jobs)]
            env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            active[subject] = (proc, log)
            waiting.remove(subject)
            print('Started preprocessing', subject, flush=True)
        for subject, (proc, log) in list(active.items()):
            if proc.poll() is None:
                continue
            log.close()
            del active[subject]
            if proc.returncode:
                raise RuntimeError(f'Preprocessing failed for {subject}; inspect {logs / ("preprocess_" + subject + ".log")}')
            print('Finished preprocessing', subject, flush=True)
        if waiting or active:
            if time.monotonic() - started > args.max_wait_hours * 3600:
                raise TimeoutError('Preprocessing/download wait exceeded explicit limit')
            time.sleep(3)
    results = [json.loads(p.read_text()) for p in sorted((args.output / 'subjects').glob('*.json'))]
    eligible = [r for r in results if r['eligible']]
    summary = {'subjects': len(eligible), 'electrodes': sum(r['electrodes'] for r in eligible),
               'positives': sum(r['positives'] for r in eligible), 'records': results,
               'paper_expected': {'subjects': 35, 'electrodes': 2066, 'positives': 298}}
    summary['paper_counts_match'] = all(summary[k] == summary['paper_expected'][k] for k in ['subjects', 'electrodes', 'positives'])
    (args.output / 'cohort.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--subject')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--filter-jobs', type=int, default=2)
    p.add_argument('--max-wait-hours', type=float, default=4)
    args = p.parse_args()
    process(args) if args.subject else supervise(args)
