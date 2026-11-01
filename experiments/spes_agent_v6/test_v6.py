"""Equal feedback/call budgets and rationale visibility, without key/data/GPU."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import experiment as e

def public(selected):
    return {'task':'test','budget':{'used':len(selected),'maximum_measurements':8},'evidence':{
      'geometry':{'candidates':[{'id':f's{i:03d}','coverage_gain':.1,'score_focus_gain':.1,'uncertainty_focus_gain':.1} for i in range(16) if i not in selected]},
      'observed':{'queries':[{'stimulus':f's{i:03d}'} for i in selected]},'model_uncertainty':{},'limitations':[]}}

class Environment:
    def __init__(self,*args):self.selected=[]
    def observe(self,q):
        if q in self.selected or len(self.selected)>=8:raise ValueError('Budget or duplicate')
        self.selected.append(q);return {'measurement':len(self.selected),'stimulus':f's{q:03d}'}
    def public(self):return public(self.selected)
    def final_judge(self):
        if len(self.selected)!=8:raise ValueError('Premature label access')
        return {'patient':7,'outer':0,'AUROC':.5,'queries':[f's{i:03d}' for i in self.selected],'prediction':[.4,.6]}

class Framework:
    def __init__(self,*args):
        self.n=0;self.registry=SimpleNamespace(get_skill=lambda _: {'type':'tool'})
        self.executor=SimpleNamespace(execute_skill=lambda name,meta,parameters,**kwargs:{'status':'success','result':json.loads(Path(parameters['state_path']).read_text())})
    def artifact(self,*args):self.n+=1;return SimpleNamespace(artifact_id=f'a{self.n}')

class Provider:
    calls=0
    def __init__(self,*args):pass
    def ask(self,prompt,destination):
        Provider.calls+=1;query=prompt['evidence']['geometry']['candidates'][0]['id']
        answer={'ranked_queries':[query],'evidence_ids':['geometry'],'reason':'Complete evidence rationale.','counterargument':'No guarantee of accuracy.'}
        Path(destination).write_text(json.dumps({'captured_response':True,'response_model':'gemini-4-argon','finish_reason':'stop','response_content':json.dumps(answer)}))

class BudgetTests(unittest.TestCase):
    def test_all_arms_have_identical_feedback_and_eight_attempts(self):
        for policy in ['single','multi','multi_reason_hidden','coverage']:
            with self.subTest(policy=policy),tempfile.TemporaryDirectory() as temporary:
                project=Path(temporary);output=project/'outputs/spes_agent_v6';case={'case_id':'unit','subject':'ccepAgeUMCU07','outer':0,'pilot':True}
                metric=project/'outputs/spes_agent_v4/active/trajectories/unit/random_0/metrics.json';metric.parent.mkdir(parents=True)
                metric.write_text(json.dumps({'queries':[f's{i:03d}' for i in range(8)]}))
                hashes={}
                for family in ['baseline','mte','hat']:
                    file=project/f'outputs/spes_agent_v4/multitrial/neural/outer0/{family}/checkpoint.pth';file.parent.mkdir(parents=True);file.write_bytes(b'test')
                    hashes[f'0/{family}']=e.sha(file)
                output.mkdir(parents=True);e.write_json(output/'protocol.json',{'source_sha256':{},'model_sha256':hashes,'prefix_metric_sha256':{'unit':e.sha(metric)}})
                Provider.calls=0
                with patch.object(e,'Acquisition',Environment),patch('neurotarget.scienceclaw_runtime.Framework',Framework),patch('experiments.spes_agent_v2.providers.Provider',Provider),patch.dict(sys.modules,{'torch':SimpleNamespace(set_num_threads=lambda _:None)}):
                    result=e.run_case(project,output,case,policy)
                self.assertEqual(Provider.calls,0 if policy=='coverage' else 8)
                self.assertEqual(result['measurements'],8);self.assertEqual(result['native_artifacts'],24)
                folder=output/'trajectories/unit'/policy
                self.assertEqual([json.loads((folder/f'{i:02d}_public.json').read_text())['budget']['used'] for i in range(8)],[4,4,5,5,6,6,7,7])
    def test_peer_rationales_hidden_but_choices_and_own_reason_preserved(self):
        states=[{'role':'hypothesis','ranked_queries':['s004'],'reason':'peer secret','counterargument':'peer limitation'},
                {'role':'auditor','ranked_queries':['s005'],'reason':'own reason','counterargument':'own limit'}]
        hidden=e.visible_states(states,'multi_reason_hidden','auditor')
        self.assertNotIn('reason',hidden[0]);self.assertEqual(hidden[0]['ranked_queries'],['s004']);self.assertEqual(hidden[1]['reason'],'own reason')
        self.assertEqual(states[0]['reason'],'peer secret')
    def test_complete_longer_rationale_preserved_without_probability_generation(self):
        answer={'ranked_queries':['s004'],'evidence_ids':['geometry'],'reason':'x'*1200,'counterargument':'Valid complete limitation.'}
        self.assertEqual(len(e.validate(answer,public([0,1,2,3]))['reason']),1200)
        with self.assertRaises(ValueError):e.validate({**answer,'probabilities':[.2,.8]},public([]))
    def test_local_atomic_retry_does_not_repeat_any_model_request(self):
        with patch.object(e,'_atomic_write',side_effect=[PermissionError('locked'),None]) as write,patch.object(e.time,'sleep'):
            e.write_json(Path('unused'),{'a':1});self.assertEqual(write.call_count,2)

if __name__=='__main__':unittest.main()
