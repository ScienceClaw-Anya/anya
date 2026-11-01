"""Post-hoc interface diagnostic only; NEVER rescore counterfactual trajectories."""
import argparse
from pathlib import Path
from collections import Counter
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v2.providers import parse_json
from .experiment import validate,write_json

def normalize_refs(answer,public):
    roots=[]
    for ref in answer['evidence_ids']:
        path=ref.removeprefix('evidence.');parts=path.split('.');value=public['evidence']
        for part in parts:
            if not isinstance(value,dict) or part not in value:raise ValueError('Reference does not resolve')
            value=value[part]
        if parts[0] not in roots:roots.append(parts[0])
    return {**answer,'evidence_ids':roots}

def run(project):
    root=Path(project)/'outputs/spes_agent_v6';rows=[]
    for path in root.glob('trajectories/*/*/*_decision.json'):
        decision=read_json(path)
        if decision.get('origin')!='spent_default':continue
        usage=read_json(path.with_name(path.name.replace('_decision','_usage')));public=read_json(path.with_name(path.name.replace('_decision','_public')))
        reason='other_schema_or_generation';resolvable=False
        try:
            original=parse_json(usage.get('response_content'));normalized=normalize_refs(original,public);validate(normalized,public)
            resolvable=original['evidence_ids']!=normalized['evidence_ids'];reason='valid_nested_reference_rejected' if resolvable else 'other_provider_or_schema'
        except (ValueError,TypeError,KeyError,AttributeError):pass
        rows.append({'policy':path.parent.name,'step':decision['step'],'case_id':path.parent.parent.name,'kind':reason,'otherwise_executable_with_resolved_refs':resolvable})
    result={'scope':'Post-hoc interface diagnosis; historical executed fallback scores remain unchanged',
       'fallbacks':len(rows),'kinds':dict(Counter(r['kind'] for r in rows)),'rows':rows,
       'not_claimed':'No counterfactual AUROC, no repaired full-trajectory performance, no API request replay',
       'implication':'The pilot is confounded by over-strict evidence-reference validation; its scores do not isolate collaboration ability.'}
    write_json(root/'schema_diagnostic.json',result);print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();run(a.project)
