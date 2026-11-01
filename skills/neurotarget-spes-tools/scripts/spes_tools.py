"""Read only an approved, label-free scientific tool result."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--project', type=Path, required=True); p.add_argument('--public-case', type=Path, required=True)
p.add_argument('--operation', choices=['signal', 'network', 'reliability', 'fusion'], required=True)
p.add_argument('--format', default='json')
args = p.parse_args(); path = args.public_case.resolve(); allowed = (args.project / 'outputs/spes_agent_v2/cases').resolve()
if allowed not in path.parents or path.name != 'tools.json': raise ValueError('Unapproved public tool-bank path')
raw = path.read_bytes(); data = json.loads(raw)
if set(data) != {'signal', 'network', 'reliability', 'scorers'}: raise ValueError('Unexpected fields in public tool bank')
key = 'scorers' if args.operation == 'fusion' else args.operation
print(json.dumps({'operation': args.operation, 'source_sha256': hashlib.sha256(raw).hexdigest(), 'result': data[key]}))
