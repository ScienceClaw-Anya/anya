"""Actual transport-enforced three-query rankings, with spent-attempt accounting."""
from experiments.spes_agent_v2.providers import Provider,ProviderUnavailable,parse_json

EVIDENCE=['geometry','observed','model_uncertainty','limitations']


def response_format(public):
    ids=[r['id'] for r in public['evidence']['geometry']['candidates']]
    if len(ids)<3 or len(set(ids))!=len(ids): raise ValueError('Invalid action catalog')
    schema={'type':'object','properties':{
        'ranked_queries':{'type':'array','items':{'type':'string','enum':ids},'minItems':3,'maxItems':3},
        'evidence_ids':{'type':'array','items':{'type':'string','enum':EVIDENCE},'minItems':1},
        'reason':{'type':'string'},'counterargument':{'type':'string'}},
        'required':['ranked_queries','evidence_ids','reason','counterargument'],'additionalProperties':False}
    return {'type':'json_schema','json_schema':{'name':'physical_independent_rank','strict':True,'schema':schema}}


def validate(answer,public):
    if not isinstance(answer,dict) or set(answer)!={'ranked_queries','evidence_ids','reason','counterargument'}: raise ValueError('Invalid proposal schema')
    ids=answer['ranked_queries']; allowed={r['id'] for r in public['evidence']['geometry']['candidates']}
    if not isinstance(ids,list) or len(ids)!=3 or not all(isinstance(q,str) for q in ids) or len(set(ids))!=3 or not set(ids)<=allowed: raise ValueError('Three distinct available pairs required')
    evidence=answer['evidence_ids']
    if not isinstance(evidence,list) or not evidence or not all(isinstance(s,str) and s in EVIDENCE for s in evidence): raise ValueError('Unknown evidence reference')
    for field in ['reason','counterargument']:
        if not isinstance(answer[field],str) or not 0<len(answer[field].strip())<=6000: raise ValueError('Incomplete/excessive reasoning')
    return {**answer,'accepted_output':True,'origin':'schema_constrained_model'}


class RankedProvider(Provider):
    def __init__(self,project): super().__init__(project,'argon'); self.current_format=None
    def metadata(self): return {**super().metadata(),'response_format':self.current_format,'structured_output':True}
    def ask(self,prompt,destination):
        self.current_format=response_format(prompt); resource=self.native.gemini_client.chat.completions; original=resource.create; fmt=self.current_format
        def constrained(*args,**kwargs):
            kwargs['response_format']=fmt
            return original(*args,**kwargs)
        resource.create=constrained
        try: return super().ask(prompt,destination)
        finally: resource.create=original
