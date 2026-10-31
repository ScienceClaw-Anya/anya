"""Fetch public complete BrainVision files with annex SHA-256 verification and resume."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from pathlib import Path

import requests

BASE = 'https://s3.amazonaws.com/openneuro.org/ds004080/'


def fetch(path: Path, root: Path, output: Path) -> dict:
    relative = path.relative_to(root).as_posix()
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    pointer = path.read_text()
    m = re.search(r'SHA256E-s(\d+)--([a-f0-9]{64})\.', pointer)
    if not m:
        raise ValueError(f'Expected public annex pointer: {path}')
    size, checksum = int(m[1]), m[2]
    receipt = target.with_suffix(target.suffix + '.verified.json')
    if target.exists() and target.stat().st_size == size and receipt.exists():
        r = json.loads(receipt.read_text())
        if r['sha256'] == checksum:
            return r
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        start = partial.stat().st_size if partial.exists() else 0
        if start > size:
            raise RuntimeError(f'Partial is larger than expected: {relative}')
        if start < size:
            headers = {'Range': f'bytes={start}-'} if start else {}
            with requests.get(BASE + relative, headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if start and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                    raise RuntimeError(f'Resume not honored: {relative}')
                try:
                    with partial.open('ab' if start else 'wb') as out:
                        for chunk in response.iter_content(4 * 1024**2):
                            out.write(chunk)
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    continue
        if partial.stat().st_size != size:
            if attempt == 3:
                raise RuntimeError(f'Size mismatch: {relative}')
            continue
        digest = hashlib.sha256()
        with partial.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != checksum:
            raise RuntimeError(f'SHA256 mismatch: {relative}')
        partial.replace(target)
        result = {'path': relative, 'bytes': size, 'sha256': checksum, 'source': BASE + relative}
        receipt.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'verified': relative, 'MiB': round(size/2**20, 1)}), flush=True)
        return result
    raise RuntimeError(relative)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata-root', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--subject')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    subjects = [s for s in manifest['subjects'] if not args.subject or s['subject'] == args.subject]
    tasks = []
    # Copy ordinary metadata, never Git directories or unverified pointer files.
    for subject in subjects:
        for source in sorted((args.metadata_root / subject['subject']).rglob('*')):
            if not source.is_file():
                continue
            target = args.raw_root / source.relative_to(args.metadata_root)
            if source.suffix in {'.eeg', '.vhdr', '.vmrk'}:
                tasks.append(source)
            elif source.suffix in {'.json', '.tsv'}:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    for name in ['dataset_description.json', 'participants.tsv', 'participants.json', 'events.json', 'README', 'CHANGES']:
        target = args.raw_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((args.metadata_root / name).read_bytes())
    begin = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, path, args.metadata_root, args.raw_root) for path in tasks]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    receipt = {'files': sorted(results, key=lambda x: x['path']), 'seconds': time.perf_counter()-begin}
    (args.raw_root / ('download_receipts' + ('_' + args.subject if args.subject else '') + '.json')).write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'finished': len(results), 'seconds': receipt['seconds']}), flush=True)


if __name__ == '__main__':
    main()
