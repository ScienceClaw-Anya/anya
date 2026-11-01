"""Native ScienceClaw public-only observation gate; no model or private loader."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def read_public(project,path):
    root=Path(project).resolve()/'outputs/spes_agent_v4/active/trajectories';path=Path(path).resolve()
    path.relative_to(root)
    if not path.name.endswith('_public.json'):raise ValueError('Only public observation snapshots are allowed')
    data=json.loads(path.read_text(encoding='utf-8'))
    if set(data)!={'task','budget','evidence'}:raise ValueError('Unexpected public schema')
    if set(data['evidence'])!={'geometry','observed','model_uncertainty','limitations'}:raise ValueError('Unexpected evidence')
    forbidden={'y','labels','truth','SOZ','outcome','full_response_prediction','unobserved_waveforms','patient','subject'}
    def check(value):
        if isinstance(value,dict):
            if set(value)&forbidden:raise ValueError('Private fields cannot enter a public tool result')
            for child in value.values():check(child)
        elif isinstance(value,list):
            for child in value:check(child)
    check(data);return data

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--state-path',required=True);p.add_argument('--format',default='json');a=p.parse_args()
    print(json.dumps(read_public(a.project,a.state_path),allow_nan=False))
