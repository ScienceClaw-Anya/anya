"""Acquire only the exact audited development asset list on GCP."""
import json
from pathlib import Path
from acquire_vec import acquire
ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
rows=json.loads((ROOT/'data/metadata/vec-inventory.json').read_text())
names=json.loads((ROOT/'manifests/development-v2-assets.json').read_text())
assert all('/sub-010/' in n and '/img/' not in n for n in names)
assert sum(r['bytes'] for r in rows if r['name'] in names)<64*1024**2
acquire(rows,names)
