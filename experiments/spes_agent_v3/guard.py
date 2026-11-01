"""Refuse uninformative or user-stopped studies before any model client exists."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from .common import read_json, write_json

def has_ranking_choice(scores):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=2 or scores.shape[1]!=2 or not np.isfinite(scores).all():raise ValueError('Invalid executable scorer bank')
    return bool(np.any(np.sign(scores[:,0,None]-scores[None,:,0])!=np.sign(scores[:,1,None]-scores[None,:,1])))

def feasibility(output):
    output=Path(output)
    if (output/'interruption.json').exists():
        return {'launch':False,'reason':'user_stopped_study','model_requests_issued':0}
    audit=read_json(output/'preparation_audit.json')
    if not audit.get('verified'):raise ValueError('A successful preparation audit is required')
    prepared=read_json(output/'prepared.json')
    # No labels or accuracy results are used. This is an action-space test, not test-set model selection.
    distinct=sum(has_ranking_choice(np.load(output/'cases'/r['case_id']/'scores.npz')['scores']) for r in prepared['cases'])
    return {'launch':distinct>0,'reason':'distinct_rankings_available' if distinct else 'no_effective_ranking_choice',
        'ranking_choice_patients':distinct,'patients':len(prepared['cases']),'model_requests_issued':0,
        'scope':'Unlabeled mathematical feasibility only; it does not prove a selector or agents will improve accuracy'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);a=p.parse_args();decision=feasibility(a.output)
    write_json(a.output/'launch_feasibility.json',decision);print(decision,flush=True)
    if not decision['launch']:return
    from .engine import main as run
    sys.argv=[sys.argv[0],'--project',str(a.project),'--output',str(a.output),'--workers',str(a.workers)];run()

if __name__=='__main__':main()
