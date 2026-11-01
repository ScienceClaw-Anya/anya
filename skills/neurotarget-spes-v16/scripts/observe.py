import argparse
import json
from pathlib import Path
import sys

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',required=True); p.add_argument('--state-path',required=True); p.add_argument('--format',default='json'); a=p.parse_args()
    sys.path.insert(0,str(Path(a.project).resolve()))
    from experiments.spes_agent_v16.public import read_public
    print(json.dumps(read_public(a.project,a.state_path),allow_nan=False))
