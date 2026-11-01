"""Meaningful observation and decision isolation checks; no data/key/GPU required."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from .acquisition import gated_input,numeric_choice
from .active import validate_answer,make_prompt

class GatingTests(unittest.TestCase):
    def test_unobserved_responses_cannot_change_model_input(self):
        a=np.arange(2*3*5*4,dtype=float).reshape(2,3,5,4);b=a.copy();b[:,:,[0,2,4],:]+=1e7
        first=gated_input(a,[1,3]);second=gated_input(b,[1,3])
        np.testing.assert_array_equal(first,second)
        self.assertEqual(first.shape,(2,3,2,4));first[:]=0
        self.assertNotEqual(float(a.sum()),0.)
    def test_invalid_and_repeated_queries_rejected(self):
        x=np.zeros((2,3,5,4))
        for query in [[1,1],[-1],[5],[1.5]]:
            with self.assertRaises(ValueError):gated_input(x,query)
    def public(self):
        return {'task':'test','budget':{'used':0,'maximum_measurements':8},'evidence':{
            'geometry':{'candidates':[{'id':'s001','coverage_gain':.2,'score_focus_gain':.3,'uncertainty_focus_gain':.1},
                                      {'id':'s002','coverage_gain':.3,'score_focus_gain':.1,'uncertainty_focus_gain':.2}]},
            'observed':{'queries':[]},'model_uncertainty':{},'limitations':[]}}
    def test_free_probabilities_or_hidden_query_rejected(self):
        answer={'ranked_queries':['s001'],'evidence_ids':['geometry'],'reason':'Known geometry','counterargument':'No measured responses yet'}
        validate_answer(answer,self.public())
        for mutation in [{'probabilities':[.1]}, {'ranked_queries':['s999']},{'ranked_queries':['s001','s001']},{'evidence_ids':['truth']},{'reason':'x'*901}]:
            with self.assertRaises((ValueError,TypeError)):validate_answer({**answer,**mutation},self.public())
    def test_independent_first_round_and_ablation_visibility(self):
        states=[{'reason':'A secret other role rationale'}]
        self.assertEqual(make_prompt(self.public(),'multi',0,states)['shared_decisions'],[])
        self.assertEqual(make_prompt(self.public(),'multi',4,states)['shared_decisions'],states)
        self.assertEqual(make_prompt(self.public(),'multi_no_exchange',4,states)['shared_decisions'],[])
        self.assertEqual(make_prompt(self.public(),'single',1,states)['shared_decisions'],states)
    def test_numeric_controls_use_different_known_objectives(self):
        p=self.public();self.assertEqual(numeric_choice(p,'coverage'),'s002');self.assertEqual(numeric_choice(p,'score_focus'),'s001')
    def test_actual_native_cli_rejects_hidden_judge_and_nested_labels(self):
        project=Path(__file__).resolve().parents[2];script=project/'skills/neurotarget-spes-v4/scripts/observe.py'
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'outputs/spes_agent_v4/active/trajectories/test';folder.mkdir(parents=True)
            public=folder/'00_public.json';public.write_text(json.dumps(self.public()),encoding='utf8')
            def run(path):return subprocess.run([sys.executable,str(script),'--project',str(root),'--state-path',str(path),'--format','json'],capture_output=True,text=True)
            result=run(public);self.assertEqual(result.returncode,0,result.stderr)
            data=self.public();data['evidence']['observed']['labels']=[0,1];public.write_text(json.dumps(data),encoding='utf8')
            self.assertNotEqual(run(public).returncode,0)
            judge=folder/'judge.json';judge.write_text(json.dumps(self.public()),encoding='utf8');self.assertNotEqual(run(judge).returncode,0)

if __name__=='__main__':unittest.main()
