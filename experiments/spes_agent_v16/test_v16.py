"""No-HTTP evidence gating, independent prompts and real transport-schema tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from .logic import prompts,visible_history,merge
from .public import public_state,project_columns,check_public,read_public,recording_catalog
from experiments.spes_agent_v3.common import sha
from .provider import RankedProvider,ProviderUnavailable,response_format,validate
from .experiment import execute_trajectory


def public(selected):
    remaining=[q for q in range(12) if q not in selected]
    return {'task':'Synthetic public physical state','budget':{'used':len(selected),'maximum_measurements':8},
        'evidence':{'geometry':{'candidates':[{'id':f'p{q:03d}','score_focus_gain':q/1000} for q in remaining]},
                    'observed':{'selected':[f'p{q:03d}' for q in selected],'queries':[]},'model_uncertainty':{},'limitations':[]}}


class FakeEnv:
    def __init__(self): self.selected=[]; self.predictions={'hat':np.array([.2,.4,.6,.8])}; self.pairs=[{'id':f'p{i:03d}'} for i in range(12)]
    def observe(self,q):
        if q in self.selected or len(self.selected)>=8: raise ValueError('Repeated/over-budget physical query')
        self.selected.append(q); self.predictions['hat']=self.predictions['hat']+.001
        return {'measurement':len(self.selected),'anatomical_pair':f'p{q:03d}','real_feedback':True}
    def final_judge(self):
        if len(self.selected)!=8: raise ValueError('Full budget required')
        return {'AUROC':.5,'prediction':self.predictions['hat'].tolist(),'queries':[f'p{q:03d}' for q in self.selected]}


class FakeFramework:
    def __init__(self,project):
        self.project=project; self.calls=0; self.records=[]; self.registry=SimpleNamespace(get_skill=lambda _:True)
        self.executor=SimpleNamespace(execute_skill=self.execute)
    def execute(self,name,metadata,args,timeout):
        self.calls+=1
        return {'status':'success','result':read_public(self.project,args['state_path'])}
    def artifact(self,agent,skill,payload,kind,parents):
        ids={r['id'] for r in self.records}
        if not set(parents)<=ids: raise ValueError('Invalid native lineage')
        new=f'a{len(self.records):03d}'; self.records.append({'id':new,'agent':agent,'payload':payload,'kind':kind,'parents':parents})
        return SimpleNamespace(artifact_id=new)


class FakeProvider:
    calls=[]
    def __init__(self,*args): pass
    def ask(self,prompt,path):
        for slot in [0,1]:
            if not path.with_name(f'{prompt["round"]-1:02d}_{slot}_request.json').exists(): raise ValueError('Peer prompt was not frozen before first response')
        self.calls.append(copy.deepcopy(prompt)); ids=[r['id'] for r in prompt['evidence']['geometry']['candidates']]
        answer={'ranked_queries':ids[:3],'evidence_ids':['geometry'],'reason':f'Secret rationale round{prompt["round"]}slot{prompt["slot"]}',
                'counterargument':'Observed feedback may revise the next proposal.'}
        Path(path).write_text(json.dumps({'captured_response':True,'response_model':'gemini-4-argon','finish_reason':'stop',
            'prompt_sha256':hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest(),
            'response_format':response_format(prompt),'response_content':json.dumps(answer)}))


class IndependentTests(unittest.TestCase):
    def test_smoke_stops_after_first_spent_invalid_response(self):
        class InvalidProvider(FakeProvider):
            def ask(self,prompt,path):
                super().ask(prompt,path); saved=json.loads(path.read_text()); saved['response_model']='wrong-model'; path.write_text(json.dumps(saved))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'outputs/spes_agent_v16'; env=FakeEnv(); framework=FakeFramework(root); FakeProvider.calls=[]
            with self.assertRaises(RuntimeError): execute_trajectory(root,out,{'case_id':'bad','pilot':True},'single',env,framework,[0,1,2,3],None,None,
                strict_smoke=True,client_factory=InvalidProvider,state_builder=lambda env,*args:public(env.selected))
            self.assertEqual(len(FakeProvider.calls),1); self.assertEqual(env.selected,[0,1,2,3]); self.assertTrue((out/'smoke_interruption.json').exists())

    def test_prompt_isolation_delayed_exchange_and_ablation(self):
        base=public([0,1,2,3]); batch=prompts(base,'multi',0,[])
        self.assertEqual(batch[0]['evidence'],batch[1]['evidence']); self.assertEqual(batch[1]['history'],[])
        batch[0]['evidence']['limitations'].append('Mutation')
        self.assertEqual(batch[1]['evidence']['limitations'],[])
        history=[{'round':1,'proposals':[{'slot':0,'ranked_queries':['p004'],'reason':'Own','counterargument':'Own caveat'},
                                       {'slot':1,'ranked_queries':['p005'],'reason':'Peer','counterargument':'Peer caveat'}],
                  'executed':'p004','feedback':{'measured':True}}]
        hidden=visible_history(history,'multi_reason_hidden',0)
        self.assertEqual(hidden[0]['proposals'][0]['reason'],'Own'); self.assertNotIn('reason',hidden[0]['proposals'][1])
        self.assertEqual(hidden[0]['feedback'],history[0]['feedback']); self.assertEqual(hidden[0]['proposals'][1]['ranked_queries'],['p005'])
        self.assertEqual(prompts(base,'multi',1,history)[0]['history'],history)
        with self.assertRaises(ValueError): prompts(base,'multi',0,history)

    def test_merger_is_symmetric_and_has_no_reviewer_veto(self):
        base=public([0,1,2,3]); a={'ranked_queries':['p004','p005','p006']}; b={'ranked_queries':['p005','p006','p004']}
        self.assertEqual(merge([a,b],base),merge([b,a],base)); self.assertEqual(merge([a,b],base)['choice'],'p005')
        with self.assertRaises(ValueError): merge([a],base)

    def test_complete_arms_match_budget_and_actual_feedback(self):
        for policy in ['single','multi','multi_reason_hidden']:
            with self.subTest(policy=policy),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); out=root/'outputs/spes_agent_v16'; env=FakeEnv(); framework=FakeFramework(root); FakeProvider.calls=[]
                result=execute_trajectory(root,out,{'case_id':'test','pilot':True},policy,env,framework,[0,1,2,3],None,None,
                    client_factory=FakeProvider,state_builder=lambda env,*args:public(env.selected))
                self.assertEqual(result['LLM_attempts'],8); self.assertEqual(result['accepted_model_outputs'],8)
                self.assertEqual(framework.calls,8); self.assertEqual(len(env.selected),8); self.assertEqual(len(set(env.selected)),8)
                self.assertEqual(result['native_artifacts'],28)
                for r in range(4):
                    first,second=FakeProvider.calls[r*2:r*2+2]
                    self.assertEqual(len(first['history']),r); self.assertEqual(len(second['history']),r)
                    self.assertNotIn(f'Secret rationale round{r+1}slot0',json.dumps(second))
                    self.assertEqual(first['evidence'],second['evidence'])
                proposals=[row for row in framework.records if row['kind']=='independent_ranking']
                for row in proposals: self.assertEqual(len(row['parents']),1)


class PublicTests(unittest.TestCase):
    def test_future_waveforms_cannot_affect_public_quality(self):
        rng=np.random.default_rng(16); raw=rng.normal(size=(4,2,12,641)); raw[...,0]=20.
        env=SimpleNamespace(index=np.tile(np.arange(12),(4,1)),selected=[0,1,2,3],available=list(range(12)),
            pairs=[{'id':f'p{q:03d}','recorded_conditions':[{'electrical_stimulation_current':'0.004'}]} for q in range(12)],
            predictions={k:np.array([.1,.4,.6,.9]) for k in ['baseline','mte','hat']})
        def base():
            data=public(env.selected); data['evidence']['anatomy']={}; data['evidence']['model_uncertainty']={'contacts':[{'id':f'c{i:03d}'} for i in range(4)]}
            return data
        env.public=base; catalog={'contacts':[],'frame':{}}
        original=public_state(env,catalog,raw); changed=raw.copy(); changed[:,:,4:,-512:]+=10000
        self.assertEqual(public_state(env,catalog,changed),original)
        changed[:,:,0,-512:]*=10
        self.assertNotEqual(public_state(env,catalog,changed),original)
        with self.assertRaises(ValueError): check_public({**original,'labels':[0,1]})

    def test_clinical_columns_do_not_enter_projected_anatomy(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'electrodes.tsv'; header='name\tx\ty\tz\tsoz\tresected\tDestrieux_label_text\n'
            path.write_text(header+'C01\t1\t2\t3\tyes\tno\tregion\n'); first=project_columns(path)
            path.write_text(header+'C01\t1\t2\t3\tno\tyes\tregion\n')
            self.assertEqual(project_columns(path),first); self.assertNotIn('soz',first[0]); self.assertNotIn('resected',first[0])

    def test_duplicate_coordinates_preserve_identity_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); coords=root/'vendor/norris2026_hat/data/coords_unit.npz'; coords.parent.mkdir(parents=True)
            np.savez_compressed(coords,data=np.array([[1.,2.,3.]]))
            electrode=root/'public_electrodes.tsv'; electrode.write_text('name\tx\ty\tz\tDestrieux_label_text\nA\t1\t2\t3\tregion\nB\t1\t2\t3\tregion\n')
            mapping={'metadata_sha256':{'public_electrodes.tsv':sha(electrode)},'coords_sha256':sha(coords),'coordinate_frame':{}}
            row=recording_catalog(root,'unit',mapping)['contacts'][0]
            self.assertIsNone(row['electrode']); self.assertEqual(row['possible_electrodes'],['A','B']); self.assertEqual(row['declared_anatomy'],'region')


class TransportTests(unittest.TestCase):
    def client(self,path,fail=False):
        calls=[]
        def send(**kwargs):
            self.assertEqual(json.loads(path.read_text())['status'],'attempt_reserved'); calls.append(kwargs)
            if fail: raise RuntimeError('Synthetic failure')
            answer={'ranked_queries':['p004','p005','p006'],'evidence_ids':['geometry'],'reason':'Evidence based.','counterargument':'Not certain.'}
            return SimpleNamespace(model='gemini-4-argon',usage=SimpleNamespace(model_dump=lambda:{'prompt_tokens':1,'completion_tokens':1}),
                choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(answer)))])
        resource=SimpleNamespace(create=send); native=SimpleNamespace(gemini_client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        native.call=lambda text,**kwargs:resource.create(model='gemini-4-argon',messages=[{'role':'user','content':text}],**kwargs).choices[0].message.content
        client=RankedProvider.__new__(RankedProvider); client.project=Path('.'); client.name='argon'; client.model='gemini-4-argon'; client.reasoning='medium'; client.max_tokens=32768
        client.base_url='https://generativelanguage.googleapis.com/v1beta/openai/'; client.native=native; client.current_format=None
        return client,calls,resource,send
    def test_actual_schema_and_spent_reservation_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'usage.json'; client,calls,resource,original=self.client(path); base=public([0,1,2,3])
            answer=client.ask(base,path); self.assertEqual(calls[0]['response_format'],response_format(base)); validate(json.loads(answer),base)
            self.assertEqual(json.loads(path.read_text())['response_format'],response_format(base))
            with self.assertRaises(ValueError): client.ask(base,path)
            self.assertEqual(len(calls),1); self.assertIs(resource.create,original)
        invalid={'ranked_queries':['p004']*3,'evidence_ids':['geometry'],'reason':'x','counterargument':'y'}
        with self.assertRaises(ValueError): validate(invalid,base)
    def test_failure_is_spent_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'usage.json'; client,calls,resource,original=self.client(path,True)
            with self.assertRaises(ProviderUnavailable): client.ask(public([0,1,2,3]),path)
            self.assertEqual(len(calls),1); self.assertIs(resource.create,original); self.assertFalse(json.loads(path.read_text())['captured_response'])


if __name__=='__main__': unittest.main()
