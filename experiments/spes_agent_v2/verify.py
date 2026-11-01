"""Offline replay of patient isolation, predictions, budgets and native lineage."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from .data import SLUGS, SCORERS, patient_features, source_hashes, write_json


def auc(y, prediction):
    """Independent pairwise AUROC with half credit for ties."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    a, b = prediction[y == 1], prediction[y == 0]
    if not len(a) or not len(b): raise ValueError('AUROC requires both classes')
    difference = a[:, None] - b[None, :]
    return float(((difference > 0).sum() + .5 * (difference == 0).sum()) / difference.size)


def read_native(state):
    """Read all producer stores, including those created in earlier invocations."""
    state = Path(state); records = {}
    for path in sorted((state / 'artifacts').glob('*/store.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['artifact_id'] in records: raise ValueError('Duplicate native artifact ID')
            # Native Artifact._hash_payload is canonical sorted-key JSON.
            digest = hashlib.sha256(json.dumps(row['payload'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if digest != row['content_hash']: raise ValueError('Native content hash differs')
            records[row['artifact_id']] = row
    seen = set()
    index = state / 'artifacts/global_index.jsonl'
    if not index.exists(): return records
    for line in index.read_text(encoding='utf-8').splitlines():
        entry = json.loads(line); row = records[entry['artifact_id']]
        if row['artifact_id'] in seen or not set(row['parent_artifact_ids']).issubset(seen):
            raise ValueError('Duplicate or forward native lineage')
        if entry['content_hash'] != row['content_hash']: raise ValueError('Native index hash differs')
        seen.add(row['artifact_id'])
    if seen != set(records): raise ValueError('Native store/index mismatch')
    return records


def verify_fusion(project, output):
    project, output = Path(project), Path(output)
    summary = json.loads((output / 'fusion_results.json').read_text())
    collected = {name: {} for name in SCORERS}; model_count = 0; case_count = 0
    feature_cache = {}
    manifest = {}
    def features(patient):
        if patient not in feature_cache:
            feature_cache[patient] = patient_features(project, patient)[0]
            subject = f'ccepAgeUMCU{patient:02d}'
            for prefix in ['X_recording', 'X_stim']:
                path = project / 'vendor/norris2024/data/mean' / f'{prefix}_{subject}.npy'
                manifest[path.relative_to(project).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return feature_cache[patient]
    for seed in range(1, 6):
        for fold in range(5):
            old = project / 'outputs/reproduction/norris2024/training'
            split = json.loads((old / f'split_seed{seed}_fold{fold}.json').read_text())
            groups = [{int(p[-2:]) for p in split[key]} for key in ['train', 'val', 'test']]
            assert [len(g) for g in groups] == [21, 7, 7]
            assert all(not a & b for i, a in enumerate(groups) for b in groups[i+1:])
            base = [np.load(old / f'{slug}_seed{seed}_fold{fold}/predictions.npz') for slug in SLUGS]
            first = base[0]
            mapping = np.array([int(p[-2:]) for p in split['val']])
            val_p = mapping[first['val_patient'].astype(int)]; test_p = first['patient']
            assert set(val_p) == groups[1] and set(test_p) == groups[2]
            xv = np.column_stack([v['val_prediction'] for v in base]); xt = np.column_stack([v['prediction'] for v in base])
            vr = np.zeros((len(val_p), 15)); tr = np.zeros((len(test_p), 15))
            for p in set(val_p): vr[val_p == p] = features(int(p))
            for p in set(test_p): tr[test_p == p] = features(int(p))
            folder = output / 'fusion' / f'seed{seed}_fold{fold}'
            receipt = json.loads((folder / 'receipt.json').read_text())
            for slug in SLUGS:
                path = old / f'{slug}_seed{seed}_fold{fold}/predictions.npz'
                assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['input_prediction_sha256'][slug]
            result = [xt[:, 2], xt.mean(axis=1)]
            for name, train, test in [('prediction_fusion', xv, xt), ('feature_fusion', np.c_[xv, vr], np.c_[xt, tr])]:
                path = folder / (name + '.joblib'); model = joblib.load(path)
                scale, classifier = model.named_steps['scale'], model.named_steps['classifier']
                w = np.array([1 / np.sum(val_p == p) for p in val_p])
                assert np.allclose(scale.mean_, np.average(train, axis=0, weights=w), atol=1e-10)
                assert classifier.C == .1 and classifier.n_features_in_ == test.shape[1]
                pred = expit(((test - scale.mean_) / scale.scale_) @ classifier.coef_[0] + classifier.intercept_[0])
                result.append(pred); model_count += 1
                manifest[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            predictions = np.column_stack(result); saved = np.load(folder / 'predictions.npz')
            assert np.allclose(predictions, saved['scores'], atol=1e-12)
            assert np.array_equal(saved['patient'], test_p) and np.array_equal(saved['y'], first['y'])
            for p in sorted(set(test_p)):
                mask = test_p == p
                record = next(r for r in summary['cases'] if r['seed'] == seed and r['fold'] == fold and r['patient'] == p)
                case = output / 'cases' / record['case_id']; judge = json.loads((case / 'judge.json').read_text())
                public = json.loads((case / 'public.json').read_text()); tools = json.loads((case / 'tools.json').read_text())
                assert set(public) == {'task', 'model_order', 'response_feature_order', 'contacts', 'feature_notes'}
                assert set(tools) == {'signal', 'network', 'reliability', 'scorers'}
                assert set(judge) == {'seed', 'fold', 'patient', 'y'} and judge['y'] == first['y'][mask].astype(int).tolist()
                assert np.allclose([c['models'] for c in public['contacts']], xt[mask], atol=5.01e-6)
                assert np.allclose([c['responses'] for c in public['contacts']], tr[mask], atol=5.01e-6)
                for i, name in enumerate(SCORERS):
                    assert np.allclose(tools['scorers'][name], predictions[mask, i], atol=5.01e-7)
                    value = auc(judge['y'], predictions[mask, i])
                    assert abs(value - record['metrics'][name]) < 1e-12
                    collected[name].setdefault(int(p), []).append(value)
                case_count += 1
    means = {k: float(np.mean([np.mean(v) for v in values.values()])) for k, values in collected.items()}
    assert all(abs(means[k] - summary['mean_patient_AUROC'][k]) < 1e-12 for k in means)
    result = {'verified': True, 'patients': 35, 'cases': case_count, 'fusion_models': model_count,
              'mean_patient_AUROC': means, 'checks': ['Patient-disjoint train/validation/test groups',
                'Validation local indices mapped to actual patient IDs', 'Validation-only weighted scaler means',
                'Manual logistic prediction replay from saved coefficients', 'Independent pairwise AUROC',
                'Label-free public/tool schemas and exact contact alignment'], 'sha256': manifest}
    write_json(output / 'fusion_audit.json', result)
    return {k: v for k, v in result.items() if k != 'sha256'}


def verify_study(project, output, routing, allow_incomplete=False):
    from .engine import POLICIES, provider_for, visible_indices
    root = Path(output) / 'studies' / routing
    protocol = json.loads((root / 'agent_protocol.json').read_text())
    assert source_hashes(project) == protocol['source_sha256']
    for name, digest in protocol['data_sha256'].items():
        assert hashlib.sha256((Path(output) / name).read_bytes()).hexdigest() == digest
    completed = list((root / 'agents').glob('*/*/metrics.json'))
    expected = 35 * len(protocol['policies'])
    if not allow_incomplete and len(completed) != expected: raise ValueError('Study incomplete; no complete-cohort inference')
    attempts = accepted = artifacts = 0
    for path in completed:
        folder = path.parent; case_id, policy = folder.parent.name, folder.name
        public = json.loads((Path(output) / 'cases' / case_id / 'public.json').read_text())
        judge = json.loads((Path(output) / 'cases' / case_id / 'judge.json').read_text())
        tools = json.loads((Path(output) / 'cases' / case_id / 'tools.json').read_text())
        native = read_native(folder / 'scienceclaw'); ids = json.loads((folder / 'native_ids.json').read_text())
        rows = [json.loads((folder / f'response_{i:02d}.json').read_text()) for i in range(8)]
        assert len(list(folder.glob('api_usage_*.json'))) == 8 and len(ids['states']) == 8
        assert len(native) == 9 + (8 if POLICIES[policy][2] else 0)
        for i, row in enumerate(rows):
            prompt = json.loads((folder / f'request_{i:02d}.json').read_text())
            usage = json.loads((folder / f'api_usage_{i:02d}.json').read_text())
            assert [v['step'] for v in prompt['shared_board']] == [j + 1 for j in visible_indices(policy, i)]
            assert usage['provider'] == provider_for(i, routing)
            digest = hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()
            assert usage['prompt_sha256'] == digest
            assert native[ids['states'][i]]['payload'] == row
            expected_parents = [ids['observation']] + [ids['states'][j] for j in visible_indices(policy, i)]
            if POLICIES[policy][2]: expected_parents.append(ids['tools'][i])
            assert native[ids['states'][i]]['parent_artifact_ids'] == expected_parents
            if not row['accepted_agent_output']:
                visible = visible_indices(policy, i)
                prior = rows[visible[-1]]['scores'] if visible else [c['models'][2] for c in public['contacts']]
                assert row['scores'] == prior
            elif POLICIES[policy][2]: assert row['scores'] == tools['scorers'][row['selected_scorer']]
            accepted += row['accepted_agent_output']; attempts += 1
        pred = np.array([r['scores'] for r in rows[-4:]]).mean(axis=0)
        metrics = json.loads(path.read_text())
        assert np.allclose(pred, metrics['prediction']) and abs(auc(judge['y'], pred) - metrics['AUROC']) < 1e-12
        artifacts += len(native)
    result = {'verified': True, 'routing': routing, 'complete_policy_cases': len(completed), 'expected_policy_cases': expected,
              'full_cohort_complete': len(completed) == expected, 'attempts_completed_cases': attempts,
              'accepted_completed_cases': accepted, 'native_artifacts': artifacts}
    write_json(root / 'audit.json', result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--routing', choices=['mixed', 'argon_control']); p.add_argument('--allow-incomplete', action='store_true')
    args = p.parse_args()
    result = verify_study(args.project, args.output, args.routing, args.allow_incomplete) if args.routing else verify_fusion(args.project, args.output)
    print(json.dumps(result, indent=2))
