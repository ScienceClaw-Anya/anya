"""Meaningful guards for the clinical-label gate, patient splits, and scoring."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from train_reproduction import score, paper_threshold
from agent_comparison import public_case, validate, traced_ask
from final_report import summarize_agents


class ScoringTests(unittest.TestCase):
    def test_patient_average_is_not_electrode_pooled(self):
        truth = np.array([0, 1, 0, 0, 0, 1])
        prediction = np.array([.1, .9, .9, .8, .7, .1])
        patient = np.array([1, 1, 2, 2, 2, 2])
        rows = score(truth, prediction, patient)
        self.assertEqual([r['AUROC'] for r in rows], [1., 0.])
        self.assertEqual(np.mean([r['AUROC'] for r in rows]), .5)

    def test_threshold_uses_validation_only(self):
        n = paper_threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), np.array([1, 1, 1, 1]))
        self.assertAlmostEqual(n, (.8-.5)/.35)
        test = np.array([.2, .25, .4, .5])
        threshold = np.median(test) + n*np.median(np.abs(test-np.median(test)))
        self.assertTrue(np.isfinite(threshold))

    def test_one_class_is_not_fabricated_chance(self):
        with self.assertRaises(ValueError):
            score(np.array([0, 0]), np.array([.1, .2]), np.array([1, 1]))

    def test_hidden_labels_do_not_change_agent_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            up = root/'upstream'
            for folder in ['mean', 'main']:
                (up/'data'/folder).mkdir(parents=True)
            raw = np.zeros((4, 2, 509), dtype=np.float32)
            raw[:, :, -1] = 20
            raw[:, :, :508] = np.arange(4)[:, None, None]+1
            for kind in ['recording', 'stim']:
                np.save(up/'data/mean'/f'X_{kind}_ccepAgeUMCU07.npy', raw)
            slugs = ['CNN_divergent', 'CNN_convergent', 'Transformer_base']
            def save(truth):
                for slug in slugs:
                    folder = root/'training'/f'{slug}_seed1_fold0'
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez(folder/'predictions.npz', y=truth, prediction=[.1,.2,.3,.4],
                             patient=[7,7,7,7], coords=np.arange(12).reshape(4,3))
            save([0,0,1,1])
            first, judge_first = public_case(root, root, up, 1, 0, 7)
            save([1,1,0,0])
            second, judge_second = public_case(root, root, up, 1, 0, 7)
            self.assertEqual(first, second)
            self.assertFalse(np.array_equal(judge_first['y'], judge_second['y']))

    def test_invalid_agent_scores_fail_closed(self):
        for values in [[float('nan'), .1], [.5], [1.1,.3]]:
            with self.assertRaises(ValueError):
                validate({'scores': values, 'reason': 'test', 'reflection': 'test'}, 2)

    def test_usage_trace_keeps_native_request_and_restores_client(self):
        from types import SimpleNamespace
        import json
        calls=[]
        usage=SimpleNamespace(model_dump=lambda:{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15})
        response=SimpleNamespace(usage=usage,model='test-model',choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{}'))])
        def original(**kwargs):calls.append(kwargs);return response
        resource=SimpleNamespace(create=original)
        client=SimpleNamespace(gemini_client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        fake=SimpleNamespace(client=lambda role:client,ask=lambda role,prompt:resource.create(model='test-model',max_tokens=2800))
        with tempfile.TemporaryDirectory() as directory:
            result=traced_ask(fake,'signal',{},Path(directory),0)
            saved=json.loads((Path(directory)/'api_usage_00.json').read_text())
            self.assertIs(result,response)
            self.assertEqual(calls,[{'model':'test-model','max_tokens':2800}])
            self.assertEqual(saved['usage']['total_tokens'],15)
            self.assertIs(resource.create,original)

    def test_repeated_evaluations_are_clustered_by_patient(self):
        records=[]
        for subject,count,value in [(1,5,.8),(2,1,.2)]:
            for repeat in range(count):
                strategies={policy:{'metrics':{'AUROC':value+(.05 if policy=='multi' else 0)},
                                    'calls':8,'seconds':1,'api_tokens':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15},'response_models':['test-model']}
                            for policy in ['single','multi']}
                numerical={name:{'AUROC':value} for name in ['CNN (divergent)','CNN (convergent)','Transformer (base)','fixed_mean_ensemble']}
                records.append({'patient':subject,'strategies':strategies,'numerical_baselines':numerical})
        result=summarize_agents(records)
        self.assertAlmostEqual(result['mean_patient_AUROC']['single'],.5)
        self.assertAlmostEqual(result['mean_patient_AUROC']['multi'],.55)
        self.assertEqual(result['usage']['single']['requests'],48)
        interval=result['paired_comparisons']['multi minus single']['patient_bootstrap_95CI']
        self.assertTrue(np.allclose(interval,[.05,.05]))


if __name__ == '__main__':
    unittest.main()
