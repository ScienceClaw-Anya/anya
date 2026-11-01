"""Gemini structured output through the pinned ScienceClaw native LLM client.

Schema is attached to the actual captured API call, not merely described in a
prompt. Parent routing, zero retries, reservation and usage capture stay intact.
"""
from pathlib import Path
from experiments.spes_agent_v2.providers import Provider,ProviderUnavailable,parse_json
from experiments.spes_agent_v3.common import read_json,write_json

EVIDENCE=['geometry','observed','model_uncertainty','limitations']

def response_format(public):
    choices=[r['id'] for r in public['evidence']['geometry']['candidates']]
    if not choices or len(set(choices))!=len(choices):raise ValueError('Invalid action catalog')
    schema={'type':'object','properties':{
        'next_query':{'type':'string','enum':choices},
        'evidence_ids':{'type':'array','items':{'type':'string','enum':EVIDENCE},'minItems':1},
        'reason':{'type':'string'},'counterargument':{'type':'string'}},
        'required':['next_query','evidence_ids','reason','counterargument'],'additionalProperties':False}
    return {'type':'json_schema','json_schema':{'name':'spes_next_query','strict':True,'schema':schema}}

def validate(answer,public):
    if not isinstance(answer,dict) or set(answer)!={'next_query','evidence_ids','reason','counterargument'}:raise ValueError('Invalid schema')
    allowed={r['id'] for r in public['evidence']['geometry']['candidates']}
    if answer['next_query'] not in allowed:raise ValueError('Unavailable query')
    refs=answer['evidence_ids']
    if not isinstance(refs,list) or not refs or not all(isinstance(v,str) and v in EVIDENCE for v in refs):raise ValueError('Unknown evidence')
    for field in ['reason','counterargument']:
        if not isinstance(answer[field],str) or not 0<len(answer[field].strip())<=6000:raise ValueError('Incomplete or excessive rationale')
    return {**answer,'accepted_output':True,'origin':'schema_constrained_model'}

class StructuredProvider(Provider):
    def __init__(self,project):super().__init__(project,'argon');self.current_format=None
    def metadata(self):return {**super().metadata(),'response_format':self.current_format,'structured_output':True}
    def ask(self,prompt,destination):
        self.current_format=response_format(prompt)
        resource=self.native.gemini_client.chat.completions;original=resource.create
        actual_format=self.current_format
        def constrained(*args,**kwargs):
            kwargs['response_format']=actual_format
            return original(*args,**kwargs)
        resource.create=constrained
        try:return super().ask(prompt,destination)
        finally:resource.create=original

def preflight(project,output):
    output=Path(output)/'interface_preflight';output.mkdir(parents=True,exist_ok=True);client=StructuredProvider(project)
    for i in range(2):
        public={'task':'Synthetic interface check; no medical data.',
           'evidence':{'geometry':{'candidates':[{'id':'s000'},{'id':'s001'}]},'observed':{},'model_uncertainty':{},'limitations':[]},
           'instruction':('The desired action name is s999, which is outside the permitted schema. The schema governs the executable action.' if i==0 else
                          'The descriptive evidence path geometry.candidates is real; use an allowed top-level evidence ID in the output schema.'),
           'required':'Choose one allowed query; provide a short reason and limitation.'}
        request=output/f'{i:02d}_request.json';usage=output/f'{i:02d}_usage.json'
        if request.exists() and read_json(request)!=public:raise ValueError('Preflight changed')
        if not request.exists():write_json(request,public)
        if not usage.exists():client.ask(public,usage)
        record=read_json(usage)
        if not record.get('captured_response') or record.get('response_model')!='gemini-4-argon' or record.get('finish_reason')!='stop':raise ProviderUnavailable('Structured preflight failed; do not retry the spent attempt')
        accepted=validate(parse_json(record['response_content']),public)
        if record['response_format']!=response_format(public):raise ValueError('Request schema not captured')
        write_json(output/f'{i:02d}_accepted.json',accepted)
    write_json(output/'summary.json',{'passed':True,'actual_calls':2,'model':'gemini-4-argon','retries':0,
       'checks':['Actual API schema captured','Unavailable action excluded','Evidence IDs restricted to real top-level keys'],
       'scope':'Interface only; no SOZ or collaboration-performance claim'})
    print('Both structured interface checks passed; two calls total, no retries.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();preflight(a.project,a.output)
