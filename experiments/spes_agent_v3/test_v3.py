"""Offline guards for patient isolation, scorer execution and matched information."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .common import WEIGHTS, check_groups, exclusive, read_json, write_json
from .crossfit import schedule
from .selector import candidates, features, fit_selector, select
from .engine import POLICIES, prompt_for, validate_answer, visible_indices, run_policy, execute_tool
from .guard import feasibility, has_ranking_choice
from experiments.spes_agent_v2.providers import ProviderUnavailable

PROJECT=Path(__file__).resolve().parents[2]
FACTS={'baseline':{'definition':'Transformer default'},'candidate_evidence':{'patients':28},
       'selector_decision':{'candidate':'transformer'},'distribution':{'features':[0,0,0,1]},'limitations':{'score_vectors_differ':True}}
PUBLIC={'task':'Research SOZ ranking','default_action':'transformer','allowed_actions':['transformer','conservative_selector'],
        'evidence':FACTS,'scorer_order':['transformer','conservative_selector'],
        'contacts':[{'id':'c000','scorers':[.1,.2]},{'id':'c001','scorers':[.9,.8]}],'constraints':[]}
TOOL_BANK={'baseline':FACTS['baseline'],'selector':FACTS['selector_decision'],'validation':FACTS['candidate_evidence'],'distribution':FACTS['distribution']}

class V3Tests(unittest.TestCase):
    def test_launch_guard_skips_identical_ordering_without_model_calls(self):
        self.assertFalse(has_ranking_choice([[.1,.2],[.8,.9]]))
        self.assertTrue(has_ranking_choice([[.1,.9],[.8,.2]]))
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);write_json(output/'preparation_audit.json',{'verified':True})
            write_json(output/'prepared.json',{'cases':[{'case_id':'unit'}]})
            case=output/'cases/unit';case.mkdir(parents=True);np.savez(case/'scores.npz',scores=[[.1,.2],[.8,.9]])
            with patch('experiments.spes_agent_v3.engine.Provider') as provider:
                decision=feasibility(output);provider.assert_not_called()
            self.assertFalse(decision['launch']);self.assertEqual(decision['model_requests_issued'],0)

    def test_launch_guard_preserves_user_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);write_json(output/'interruption.json',{'reason':'User stop'})
            self.assertEqual(feasibility(output)['reason'],'user_stopped_study')

    def test_inner_predictions_exclude_outer_test_and_early_stop_patients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);people=[f'patient{i:02d}' for i in range(35)]
            for fold in range(5):
                test=people[fold*7:(fold+1)*7];val=people[((fold+1)%5)*7:((fold+1)%5+1)*7]
                write_json(root/f'outputs/reproduction/norris2024/training/split_seed1_fold{fold}.json',
                    {'train':sorted(set(people)-set(test)-set(val)),'val':val,'test':test})
            rows=schedule(root);self.assertEqual(len(rows),20)
            for outer in range(5):
                selected=[r for r in rows if r['outer']==outer]
                self.assertEqual(len({p for r in selected for p in r['predict']}),28)
                self.assertEqual(sum(len(r['predict']) for r in selected),28)
                for row in selected:
                    self.assertEqual([len(row[k]) for k in ['train','early_stop','predict','outer_test']],[17,4,7,7])
                    check_groups(row['train'],row['early_stop'],row['predict'],row['outer_test'])

    def test_outer_test_patient_cannot_enter_early_stopping(self):
        with self.assertRaises(ValueError):check_groups([1,2],[3],[4],[3,5])
        with self.assertRaises(ValueError):check_groups([1,2],[2],[4],[5])

    def test_candidates_are_convex_and_keep_transformer_weight(self):
        x=np.array([[0,1,.4],[1,0,.8],[.5,.4,.2]])
        for name,w in WEIGHTS.items():
            self.assertAlmostEqual(sum(w),1);self.assertGreaterEqual(w[2],.8)
            self.assertTrue(np.allclose(candidates(x)[name],x@np.array(w)))
            self.assertLessEqual(np.max(np.abs(candidates(x)[name]-x[:,2])),.2+1e-12)

    def test_selector_is_fitted_and_defaults_when_evidence_fails(self):
        x=np.tile([[.9,.8,.1],[.2,.1,.9],[.8,.7,.2],[.1,.2,.8]],(12,1))
        y=np.tile([0,1,0,1],12);patient=np.repeat(np.arange(12),4)
        model=fit_selector(x,y,patient)
        self.assertEqual(model['model'].named_steps['ridge'].coef_.shape,(3,4))
        self.assertFalse(any(v['eligible'] for v in model['evidence'].values()))
        self.assertEqual(select(model,x[:4])['candidate'],'transformer')
        self.assertTrue(np.array_equal(select(model,x[:4])['scores'],x[:4,2]))

    def test_prediction_summary_rejects_unbounded_or_nonfinite_scores(self):
        for x in [[[0,1,2]],[[0,float('nan'),.3]]]:
            with self.assertRaises(ValueError):features(x)

    def test_selector_can_activate_when_independent_predictions_support_gain(self):
        x=np.tile([[.9,.1,.5],[.1,.9,.5],[.8,.2,.5],[.2,.8,.5]],(12,1))
        y=np.tile([0,1,0,1],12);patient=np.repeat(np.arange(12),4)
        fitted=fit_selector(x,y,patient);decision=select(fitted,x[:4])
        self.assertTrue(fitted['evidence']['convergent_10']['eligible'])
        self.assertEqual(decision['candidate'],'convergent_10')
        self.assertTrue(np.allclose(decision['scores'],.9*x[:4,2]+.1*x[:4,1]))

    def test_all_agent_policies_receive_identical_public_information(self):
        for policy in POLICIES[2:]:
            prompt=prompt_for(PUBLIC,policy,0,[],{'result':TOOL_BANK['baseline']})
            self.assertEqual({k:prompt[k] for k in PUBLIC},PUBLIC)
            self.assertEqual(prompt['default_action'],'transformer')
            self.assertIn('candidate_evidence',prompt['evidence'])
            for key in ['y','patient','test_accuracy','prevalence']:
                self.assertNotIn('"'+key+'"',json.dumps(prompt))

    def test_independence_and_full_structured_explanation(self):
        state={'step':1,'role':'ranking','action':'transformer','evidence_ids':['baseline'],
            'reason':'r'*850,'counterargument':'c'*800,'accepted_output':True}
        self.assertEqual(prompt_for(PUBLIC,'multi_independent',1,[state],{})['shared_board'],[])
        visible=prompt_for(PUBLIC,'multi_serial',1,[state],{})['shared_board'][0]
        self.assertEqual(len(visible['reason']),850);self.assertEqual(len(visible['counterargument']),800)
        self.assertEqual(visible_indices('multi_no_exchange',7),[])

    def test_free_probabilities_or_unknown_evidence_are_rejected(self):
        good={'action':'transformer','evidence_ids':['baseline'],'reason':'Evidence','counterargument':'Uncertain'}
        self.assertEqual(validate_answer(good,PUBLIC)['action'],'transformer')
        for extra in [{'scores':[.1,.9]},{'weights':[1,0]},{'action':'invented'},{'evidence_ids':['hidden_test_accuracy']}]:
            with self.assertRaises(ValueError):validate_answer({**good,**extra},PUBLIC)

    def test_coordinator_lock_does_not_allow_concurrent_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'test.lock'
            with exclusive(path):
                with self.assertRaises(RuntimeError):
                    with exclusive(path):pass
            self.assertFalse(path.exists())

    def test_real_native_tool_blocks_hidden_judge(self):
        from neurotarget.scienceclaw_runtime import Framework
        parent=PROJECT/'outputs/spes_agent_v3/cases';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            case=Path(temporary);write_json(case/'tools.json',TOOL_BANK);write_json(case/'judge.json',{'y':[0,1]})
            framework=Framework(PROJECT,case/'native')
            self.assertEqual(execute_tool(framework,case,'baseline')['result'],TOOL_BANK['baseline'])
            meta=framework.registry.get_skill('neurotarget-spes-v3')
            response=framework.executor.execute_skill('neurotarget-spes-v3',meta,
                {'project':str(PROJECT),'public_case':str(case/'judge.json'),'operation':'baseline','format':'json'})
            self.assertNotEqual(response['status'],'success')

    def test_numeric_baseline_has_eight_slots_and_no_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary,patch('experiments.spes_agent_v3.engine.execute_tool',return_value={'result':{}}),patch('experiments.spes_agent_v3.engine.Provider') as provider:
            output=Path(temporary);case=output/'cases/unit';write_json(case/'public.json',PUBLIC);write_json(case/'tools.json',TOOL_BANK)
            write_json(case/'judge.json',{'patient':99,'outer':0,'y':[0,1]});np.savez(case/'scores.npz',scores=[[.1,.2],[.9,.8]])
            result=run_policy(PROJECT,output,'unit','numeric_transformer')
            provider.assert_not_called();self.assertEqual(result['model_attempts'],0);self.assertEqual(result['tool_queries'],8)
            self.assertEqual(result['prediction'],[.1,.9]);self.assertEqual(result['native_artifacts'],17)

    def test_spent_transport_attempt_is_not_replayed_and_uses_transformer(self):
        calls=[]
        class FakeProvider:
            def __init__(self,*args):pass
            def ask(self,prompt,path):
                step=len(calls);calls.append(prompt)
                record={'prompt_sha256':hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()}
                if step==2:
                    write_json(path,{**record,'captured_response':False,'status':'uncaptured_failure'})
                    raise ProviderUnavailable('Synthetic transport failure')
                answer={'action':'conservative_selector','evidence_ids':['candidate_evidence'],'reason':'Evidence','counterargument':'Uncertain'}
                text=json.dumps(answer);write_json(path,{**record,'captured_response':True,'response_model':'gemini-4-argon','finish_reason':'stop','response_content':text})
                return text
        with tempfile.TemporaryDirectory() as temporary,patch('experiments.spes_agent_v3.engine.execute_tool',return_value={'result':{}}),patch('experiments.spes_agent_v3.engine.Provider',FakeProvider):
            output=Path(temporary);case=output/'cases/unit';write_json(case/'public.json',PUBLIC);write_json(case/'tools.json',TOOL_BANK)
            write_json(case/'judge.json',{'patient':99,'outer':0,'y':[0,1]});np.savez(case/'scores.npz',scores=[[.1,.2],[.9,.8]])
            with self.assertRaises(ProviderUnavailable):run_policy(PROJECT,output,'unit','multi_independent')
            result=run_policy(PROJECT,output,'unit','multi_independent')
            self.assertEqual(len(calls),8);self.assertEqual(result['accepted_model_outputs'],7)
            state=read_json(output/'policies/unit/multi_independent/state_02.json');self.assertEqual(state['action'],'transformer')
            run_policy(PROJECT,output,'unit','multi_independent');self.assertEqual(len(calls),8)

if __name__=='__main__':unittest.main()
