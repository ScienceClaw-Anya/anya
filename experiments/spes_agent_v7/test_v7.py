"""Actual transport-schema injection and full protocol budgets without HTTP."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from experiments.spes_agent_v6.test_v6 import Environment,Framework,public
from experiments.spes_agent_v6.diagnose_schema import normalize_refs
from . import experiment as e
from .provider import StructuredProvider,ProviderUnavailable,response_format,validate

class FakeProvider:
    calls=0
    def __init__(self,*args):pass
    def ask(self,prompt,path):
        FakeProvider.calls+=1;answer={'next_query':prompt['evidence']['geometry']['candidates'][0]['id'],'evidence_ids':['geometry'],
          'reason':'Complete evidence rationale.','counterargument':'No guarantee of accuracy.'}
        Path(path).write_text(json.dumps({'captured_response':True,'response_model':'gemini-4-argon','finish_reason':'stop',
           'response_format':response_format(prompt),'response_content':json.dumps(answer)}))

class SchemaTests(unittest.TestCase):
    def client(self,fail=False):
        calls=[]
        def send(**kwargs):
            calls.append(kwargs)
            if fail:raise RuntimeError('Synthetic transport failure')
            answer={'next_query':'s004','evidence_ids':['geometry'],'reason':'Supported','counterargument':'Limited'}
            return SimpleNamespace(model='gemini-4-argon',usage=SimpleNamespace(model_dump=lambda:{'prompt_tokens':1,'completion_tokens':1}),
                 choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(answer)))])
        resource=SimpleNamespace(create=send);native=SimpleNamespace(gemini_client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        def call(text,**kwargs):return resource.create(model='gemini-4-argon',messages=[{'role':'user','content':text}],**kwargs).choices[0].message.content
        native.call=call;c=StructuredProvider.__new__(StructuredProvider);c.project=Path('.');c.name='argon';c.model='gemini-4-argon';c.reasoning='medium';c.max_tokens=32768
        c.base_url='https://generativelanguage.googleapis.com/v1beta/openai/';c.native=native;c.current_format=None
        return c,calls,resource,send
    def test_real_parent_capture_sends_schema_and_records_it_once(self):
        c,calls,resource,original=self.client();p=public([0,1,2,3])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'usage.json';answer=c.ask(p,path);saved=json.loads(path.read_text())
            self.assertEqual(len(calls),1);self.assertEqual(calls[0]['response_format'],response_format(p));self.assertEqual(saved['response_format'],response_format(p))
            validate(json.loads(answer),p)
            with self.assertRaises(ValueError):c.ask(p,path)
            self.assertEqual(len(calls),1)
        self.assertIs(resource.create,original)
    def test_transport_failure_stays_spent_without_retry(self):
        c,calls,resource,original=self.client(fail=True)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'usage.json'
            with self.assertRaises(ProviderUnavailable):c.ask(public([]),path)
            self.assertEqual(len(calls),1);self.assertFalse(json.loads(path.read_text())['captured_response'])
        self.assertIs(resource.create,original)
    def test_dynamic_catalog_and_real_nested_reference_diagnostic(self):
        p=public([0,1,2,3]);f=response_format(p)
        self.assertNotIn('s000',f['json_schema']['schema']['properties']['next_query']['enum'])
        a={'next_query':'s999','evidence_ids':['geometry'],'reason':'x','counterargument':'y'}
        with self.assertRaises(ValueError):validate(a,p)
        old={'ranked_queries':['s004'],'evidence_ids':['evidence.geometry.candidates'],'reason':'x','counterargument':'y'}
        self.assertEqual(normalize_refs(old,p)['evidence_ids'],['geometry'])
        with self.assertRaises(ValueError):normalize_refs({**old,'evidence_ids':['geometry.nonexistent']},p)
    def test_full_trajectory_has_eight_attempts_and_matching_feedback(self):
        for policy in ['single','multi','multi_reason_hidden','coverage']:
            with self.subTest(policy=policy),tempfile.TemporaryDirectory() as temporary:
                project=Path(temporary);output=project/'outputs/spes_agent_v7';case={'case_id':'unit','subject':'ccepAgeUMCU07','outer':0,'pilot':True}
                metric=project/'outputs/spes_agent_v4/active/trajectories/unit/random_0/metrics.json';metric.parent.mkdir(parents=True);metric.write_text(json.dumps({'queries':[f's{i:03d}' for i in range(8)]}))
                hashes={}
                for family in ['baseline','mte','hat']:
                    file=project/f'outputs/spes_agent_v4/multitrial/neural/outer0/{family}/checkpoint.pth';file.parent.mkdir(parents=True);file.write_bytes(b'test');hashes[f'0/{family}']=e.sha(file)
                output.mkdir(parents=True);e.write_json(output/'protocol.json',{'source_sha256':{},'model_sha256':hashes,'prefix_metric_sha256':{'unit':e.sha(metric)}})
                FakeProvider.calls=0
                with patch.object(e,'Acquisition',Environment),patch.object(e,'StructuredProvider',FakeProvider),patch('neurotarget.scienceclaw_runtime.Framework',Framework),patch.dict(sys.modules,{'torch':SimpleNamespace(set_num_threads=lambda _:None)}):
                    result=e.run_case(project,output,case,policy)
                self.assertEqual(FakeProvider.calls,0 if policy=='coverage' else 8);self.assertEqual(result['native_artifacts'],24)
                self.assertEqual(result['accepted_model_outputs'],0 if policy=='coverage' else 8)
                folder=output/'trajectories/unit'/policy
                self.assertEqual([json.loads((folder/f'{i:02d}_public.json').read_text())['budget']['used'] for i in range(8)],[4,4,5,5,6,6,7,7])

if __name__=='__main__':unittest.main()
