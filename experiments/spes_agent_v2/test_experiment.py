"""Offline guards for leakage, communication interventions and spent-call recovery."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .data import equal_patient_weights, validation_patient_ids, fit_fusion, write_json, source_hashes
from .engine import POLICIES, visible_indices, provider_for, make_prompt, fallback, validate_answer, run_policy, execute_tool
from .providers import Provider, ProviderUnavailable
from .verify import read_native, auc

PROJECT = Path(__file__).resolve().parents[2]
PUBLIC = {'task': 'Retrospective SOZ ranking', 'model_order': ['a', 'b', 'c'], 'response_feature_order': [],
          'contacts': [{'id': 'c000', 'models': [.1, .2, .3], 'responses': [], 'neighbors': []},
                       {'id': 'c001', 'models': [.8, .7, .6], 'responses': [], 'neighbors': []}], 'feature_notes': 'test'}
TOOL_BANK = {'signal': {'contact_ids': ['c000', 'c001'], 'features': [[.1], [.9]]},
             'network': {'features': [[.2], [.8]], 'neighbors': [['c001'], ['c000']]},
             'reliability': {'validation_patient_count': 7},
             'scorers': {key: [.3, .6] for key in ['transformer', 'mean_ensemble', 'prediction_fusion', 'feature_fusion']}}


class ExperimentGuards(unittest.TestCase):
    def test_source_freeze_covers_real_runtime_and_skill(self):
        hashes = source_hashes(PROJECT)
        self.assertIn('neurotarget/scienceclaw_runtime.py', hashes)
        self.assertIn('skills/neurotarget-spes-tools/scripts/spes_tools.py', hashes)

    def test_local_validation_indices_are_not_patient_ids(self):
        split = {'val': [f'ccepAgeUMCU{v:02d}' for v in [2, 4, 8, 10, 12, 14, 16]]}
        actual = validation_patient_ids([0, 0, 1, 2, 3, 4, 5, 6], split)
        self.assertEqual(actual.tolist(), [2, 2, 4, 8, 10, 12, 14, 16])
        with self.assertRaises(ValueError): validation_patient_ids([0, 1, 2], split)

    def test_patients_receive_equal_total_fit_weight(self):
        patient = np.array([2, 2, 2, 2, 3, 3]); weight = equal_patient_weights(patient)
        self.assertAlmostEqual(weight[patient == 2].sum(), weight[patient == 3].sum())
        x = np.arange(18).reshape(6, 3) / 20
        model = fit_fusion(x, [0, 0, 1, 1, 0, 1], patient)
        self.assertTrue(np.allclose(model.named_steps['scale'].mean_, np.average(x, axis=0, weights=weight)))

    def test_independence_and_no_exchange_are_enforced(self):
        board = [{'step': i+1, 'role': 'signal', 'scores': [.1, .9], 'reason': 'reason', 'reflection': 'review', 'accepted_agent_output': True} for i in range(8)]
        for step in range(8):
            self.assertEqual(visible_indices('multi_serial', step), list(range(step)))
            self.assertEqual(visible_indices('multi_no_exchange', step), [])
            independent = make_prompt(PUBLIC, 'multi_independent_first', step, board[:step])
            self.assertEqual(len(independent['shared_board']), 0 if step < 4 else step)
        self.assertEqual(fallback(PUBLIC, 'multi_independent_first', 3, board[:3]), [.3, .6])
        self.assertEqual(fallback(PUBLIC, 'multi_no_exchange', 7, board[:7]), [.3, .6])

    def test_matched_provider_and_call_schedule(self):
        for policy in POLICIES:
            schedule = [provider_for(i) for i in range(8)]
            self.assertEqual(schedule.count('argon'), 4); self.assertEqual(schedule.count('nemotron'), 4)
            self.assertEqual([provider_for(i, 'argon_control') for i in range(8)], ['argon'] * 8)

    def test_held_out_truth_is_not_in_prompt_or_tool_selection(self):
        prompt = make_prompt(PUBLIC, 'single_tools', 0, [], TOOL_BANK, {'result': TOOL_BANK['signal']})
        for key in ['y', 'patient', 'SOZ_labels', 'test_accuracy', 'prevalence']:
            self.assertNotIn('"' + key + '"', json.dumps(prompt))
        self.assertEqual(validate_answer({'scorer': 'transformer', 'reason': 'Measured model', 'reflection': 'Uncertain'}, 2, TOOL_BANK)['scores'], [.3, .6])

    def test_invalid_prediction_consumes_no_schema_retry(self):
        for scores in [[float('nan'), .2], [.1], [-.1, .9], [float('inf'), .1]]:
            with self.assertRaises(ValueError): validate_answer({'scores': scores, 'reason': 'r', 'reflection': 'r'}, 2)
        self.assertEqual(auc([0, 1, 0, 1], [.1, .8, .8, .9]), .875)

    def test_provider_never_replays_reserved_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'usage.json'; write_json(path, {'status': 'attempt_reserved'})
            provider = object.__new__(Provider)
            with self.assertRaises(ValueError): provider.ask({}, path)

    def test_native_tools_reject_judge_and_execute_public_bank(self):
        from neurotarget.scienceclaw_runtime import Framework
        folder = PROJECT / 'outputs/spes_agent_v2/cases'; folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as directory:
            case = Path(directory); write_json(case / 'tools.json', TOOL_BANK); write_json(case / 'judge.json', {'y': [0, 1]})
            framework = Framework(PROJECT, case / 'native')
            result = execute_tool(framework, case, 'signal')
            self.assertEqual(result['result'], TOOL_BANK['signal'])
            metadata = framework.registry.get_skill('neurotarget-spes-tools')
            rejected = framework.executor.execute_skill('neurotarget-spes-tools', metadata,
                {'project': str(PROJECT), 'public_case': str(case / 'judge.json'), 'operation': 'signal', 'format': 'json'})
            self.assertNotEqual(rejected['status'], 'success')

    def test_resume_counts_spent_attempt_and_keeps_independent_fallback(self):
        calls = []
        class FakeProvider:
            def __init__(self, project, name): self.name = name
            def ask(self, prompt, destination):
                step = len(calls); calls.append(prompt)
                record = {'provider': self.name, 'prompt_sha256': hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()}
                if step == 2:
                    write_json(destination, {**record, 'captured_response': False, 'status': 'uncaptured_failure'})
                    raise ProviderUnavailable('synthetic transport failure')
                text = json.dumps({'scores': [.2, .8], 'reason': 'Measured evidence', 'reflection': 'Uncertain'})
                write_json(destination, {**record, 'captured_response': True, 'response_content': text, 'response_model': 'gemini-4-argon', 'finish_reason': 'stop'})
                return text
        with tempfile.TemporaryDirectory() as directory, patch('experiments.spes_agent_v2.engine.Provider', FakeProvider):
            output = Path(directory); case = output / 'cases/test'; write_json(case / 'public.json', PUBLIC)
            write_json(case / 'tools.json', TOOL_BANK); write_json(case / 'judge.json', {'y': [0, 1], 'patient': 99, 'seed': 1, 'fold': 0})
            with self.assertRaises(ProviderUnavailable): run_policy(PROJECT, output, 'test', 'multi_independent_first', 'argon_control')
            result = run_policy(PROJECT, output, 'test', 'multi_independent_first', 'argon_control')
            self.assertEqual(len(calls), 8); self.assertEqual(result['accepted'], 7)
            root = output / 'studies/argon_control/agents/test/multi_independent_first'
            third = json.loads((root / 'response_02.json').read_text())
            self.assertEqual(third['scores'], [.3, .6])
            self.assertEqual(len(read_native(root / 'scienceclaw')), 9)
            self.assertEqual(run_policy(PROJECT, output, 'test', 'multi_independent_first', 'argon_control'), result)
            self.assertEqual(len(calls), 8)


if __name__ == '__main__': unittest.main()
