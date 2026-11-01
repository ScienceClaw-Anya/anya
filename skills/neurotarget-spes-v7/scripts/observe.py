import argparse
import json
from pathlib import Path

def read_public(project,path):
    root=Path(project).resolve()/'outputs/spes_agent_v7/trajectories';path=Path(path).resolve();path.relative_to(root)
    if not path.name.endswith('_public.json'):raise ValueError('Public snapshots only')
    data=json.loads(path.read_text(encoding='utf8'))
    if set(data)!={'task','budget','evidence'} or set(data['evidence'])!={'geometry','observed','model_uncertainty','limitations'}:raise ValueError('Unexpected public fields')
    forbidden={'y','labels','truth','SOZ','outcome','patient','subject','unobserved_waveforms','full_response_prediction'}
    def check(x):
        if isinstance(x,dict):
            if set(x)&forbidden:raise ValueError('Hidden fields rejected')
            for v in x.values():check(v)
        elif isinstance(x,list):
            for v in x:check(v)
    check(data);return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--state-path',required=True);p.add_argument('--format',default='json');a=p.parse_args()
    print(json.dumps(read_public(a.project,a.state_path),allow_nan=False))
