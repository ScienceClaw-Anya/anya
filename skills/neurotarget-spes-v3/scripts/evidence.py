import argparse
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--public-case',type=Path,required=True)
p.add_argument('--operation',choices=['baseline','selector','validation','distribution'],required=True);p.add_argument('--format',default='json')
a=p.parse_args();path=a.public_case.resolve();allowed=(a.project/'outputs/spes_agent_v3/cases').resolve()
if allowed not in path.parents or path.name!='tools.json':raise ValueError('Only an approved public evidence bank may be read')
raw=path.read_bytes();data=json.loads(raw)
if set(data)!={'baseline','selector','validation','distribution'}:raise ValueError('Unexpected public evidence schema')
print(json.dumps({'operation':a.operation,'source_sha256':hashlib.sha256(raw).hexdigest(),'result':data[a.operation]}))
