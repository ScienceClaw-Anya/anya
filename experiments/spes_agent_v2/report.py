"""English repository report from audited evidence; incomplete arms stay pending."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .data import write_json

LABELS = {'single_serial': 'Single generalist, serial memory', 'multi_serial': 'Four specialists, serial discussion',
 'multi_independent_first': 'Four specialists, independent first round', 'multi_no_exchange': 'Four specialists, no exchange',
 'single_tools': 'Single generalist with numerical tools', 'multi_independent_tools': 'Independent-first specialists with numerical tools'}


def summarize(output):
    output = Path(output)
    fusion = json.loads((output / 'fusion_results.json').read_text())
    audit = json.loads((output / 'fusion_audit.json').read_text())
    assert audit['verified'] and audit['cases'] == 175
    studies = {}
    for routing in ['argon_control', 'mixed']:
        root = output / 'studies' / routing
        if not (root / 'agent_protocol.json').exists():
            studies[routing] = {'status': 'not_run'}; continue
        protocol = json.loads((root / 'agent_protocol.json').read_text())
        rows = [json.loads(p.read_text()) for p in (root / 'agents').glob('*/*/metrics.json')]
        usages = [json.loads(p.read_text()) for p in (root / 'agents').glob('*/*/api_usage_*.json')]
        counts = {policy: sum(r['policy'] == policy for r in rows) for policy in protocol['policies']}
        data = {'status': 'incomplete', 'completed_cases': counts, 'total_request_attempts': len(usages),
                'captured_responses': sum(v.get('captured_response', False) for v in usages),
                'usage_tokens': {key: sum((v.get('usage') or {}).get(key, 0) or 0 for v in usages) for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']},
                'protocol': protocol}
        if all(v == 35 for v in counts.values()):
            checked = json.loads((root / 'audit.json').read_text())
            assert checked['full_cohort_complete'] and checked['verified']
            data['status'] = 'complete_audited'
            data['mean_patient_AUROC'] = {k: float(np.mean([r['AUROC'] for r in rows if r['policy'] == k])) for k in counts}
            data['accepted_agent_outputs'] = sum(r['accepted'] for r in rows)
            data['native_artifacts'] = checked['native_artifacts']
            contrasts = [('multi_independent_first', 'multi_serial'), ('multi_independent_first', 'single_serial'),
                         ('multi_independent_first', 'multi_no_exchange'), ('multi_independent_tools', 'single_tools')]
            data['contrasts'] = {}
            draws = np.random.default_rng(42).integers(0, 35, (10000, 35))
            patients = sorted(set(r['patient'] for r in rows))
            assert len(patients) == 35
            for a, b in contrasts:
                if a not in counts or b not in counts: continue
                left = {r['patient']: r['AUROC'] for r in rows if r['policy'] == a}
                right = {r['patient']: r['AUROC'] for r in rows if r['policy'] == b}
                delta = np.array([left[p] - right[p] for p in patients])
                data['contrasts'][a + ' minus ' + b] = {'mean': float(delta.mean()), 'patient_bootstrap_95CI': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                    'patients_improved': int((delta > 0).sum()), 'patients_worse': int((delta < 0).sum()), 'patients_tied': int((delta == 0).sum())}
            data['feedback_diagnostics'] = {}
            for policy in counts:
                paired = []
                same_final = 0
                for r in rows:
                    if r['policy'] != policy: continue
                    states = [json.loads((root / 'agents' / r['case_id'] / policy / f'response_{i:02d}.json').read_text()) for i in range(8)]
                    judge = json.loads((output / 'cases' / r['case_id'] / 'judge.json').read_text())
                    from .verify import auc
                    first = np.array([s['scores'] for s in states[:4]]).mean(axis=0)
                    paired.append(r['AUROC'] - auc(judge['y'], first))
                    same_final += all(s['scores'] == states[4]['scores'] for s in states[5:])
                data['feedback_diagnostics'][policy] = {'second_minus_first_round_AUROC': float(np.mean(paired)),
                    'identical_final_four_vectors_cases': same_final, 'cases': 35}
        studies[routing] = data
    return {'fusion': {k: v for k, v in fusion.items() if k != 'cases'}, 'fusion_audit': {k: v for k, v in audit.items() if k != 'sha256'},
            'studies': studies, 'first_repeat_baselines': {name: float(np.mean([r['metrics'][name] for r in fusion['cases'] if r['seed'] == 1])) for name in fusion['mean_patient_AUROC']}}


def write_report(project, output):
    project, output = Path(project), Path(output)
    result = summarize(output); fusion = result['fusion']
    lines = ['# SPES follow-up: independent deliberation and trained fusion', '',
        'The fixed validation-trained fusion models did not beat the reconstructed Transformer. Communication experiments use a separate first-repeat, 35-patient cohort and equal call budgets. This is retrospective SOZ contact ranking, not evidence of therapeutic benefit.', '',
        '## Numerical fusion: completed and independently replayed', '',
        '| Scorer | Mean patient AUROC, all five repeats |', '|---|---:|']
    for name, value in fusion['mean_patient_AUROC'].items(): lines.append(f'| {name} | {value:.4f} |')
    lines += ['', 'These results average the five held-out predictions for each of 35 distinct patients; 175 patient-repeat cases are not 175 independent patients. Fifty fixed L2 logistic models were fitted: two per split, five folds, five repeats. Total fitting time was '
        + f"{fusion['fit_seconds_sum']:.3f} seconds on the local CPU. No cloud GPU was started.", '',
        'The prediction-only fusion uses three numerical model probabilities. The response-feature fusion adds 15 measured SPES response summaries. Both use C=0.1, weighted scaling and equal total weight per validation patient. Only the original seven validation patients per outer split fit the fusion; the 21 numerical-training and seven test patients are disjoint. Validation also supported original model early stopping. Hyperparameters were fixed and not selected against test scores.', '']
    for key, value in fusion['paired_comparisons'].items():
        lo, hi = value['patient_bootstrap_95CI']; lines.append(f"- {key}: {value['mean']:+.4f}; 95% patient-bootstrap interval [{lo:+.4f}, {hi:+.4f}].")
    lines += ['', 'The negative response-feature result is evidence against this particular small, fixed fusion model on this cohort. Limited validation-patient diversity, redundant or unstable summaries, and model misspecification are plausible explanations; this experiment does not distinguish them. It does not establish that all response features or all trained fusion methods fail.', '',
        '## Communication experiment', '',
        'All arms use eight model request attempts and aggregate the final four score states. Four roles are signal, network, uncertainty and critic. Serial discussion exposes every previous proposal; independent-first hides all proposals for the first four calls and then permits discussion. No-exchange always hides other proposals. The single generalist has the same serial memory and model-call budget.', '',
        'Tool arms execute eight fixed queries through native ScienceClaw SkillExecutor. They retrieve measured signal/network summaries, validation reliability and saved trained-model predictions. Both arms can select the same four executable numerical scorers from the first step. These are cached scientific outputs, not fresh physiological measurements. Tool arms also change the output interface to scorer selection; comparison against manual score vectors cannot isolate tool access alone.', '',
        'The public packet includes three predictions, 15 response summaries and geometric neighbors. It excludes patient identity, clinical labels, outcome, SOZ prevalence and held-out accuracy. Later rounds receive critique, not answer-key feedback. Invalid or truncated answers consume budget and retain the last visible state, or the shared Transformer baseline when the policy has no visible state. Transport failures halt; resuming never repeats a spent attempt.', '']
    for routing, study in result['studies'].items():
        lines += ['### ' + ('Gemini 4 Argon control' if routing == 'argon_control' else 'Argon + Nemotron mixture'), '']
        if study['status'] == 'not_run':
            lines += ['Not run. The Scientific AI credential failed the official Token Factory read-only model-permission test (HTTP 401). A dedicated Token Factory or NVIDIA model API credential is required. No Nemotron results are claimed.', '']
            continue
        lines += [f"Status: **{study['status']}**. Recorded request attempts: {study['total_request_attempts']}.", '']
        if study['status'] != 'complete_audited':
            lines += ['Incomplete arms are not used for full-cohort performance claims.', '', '| Policy | Completed patient cases |', '|---|---:|']
            lines += [f'| {LABELS[k]} | {v}/35 |' for k, v in study['completed_cases'].items()]
            lines += ['']; continue
        lines += ['| Policy | Mean patient AUROC, first repeat only |', '|---|---:|']
        lines += [f'| {LABELS[k]} | {v:.4f} |' for k, v in study['mean_patient_AUROC'].items()]
        lines += [f'| Same-repeat Transformer baseline | {result["first_repeat_baselines"]["transformer"]:.4f} |', '']
        for key, value in study['contrasts'].items():
            lo, hi = value['patient_bootstrap_95CI']; lines.append(f"- {key}: {value['mean']:+.4f}; 95% paired patient-bootstrap interval [{lo:+.4f}, {hi:+.4f}].")
        lines += ['', 'The independent-first minus serial contrast is primary. An interval spanning zero does not demonstrate a reliable benefit. Improvements against another arm alone do not establish that independence or communication is the cause. Additional contrasts are exploratory and unadjusted for multiplicity.', '',
            '| Policy | Round 2 minus round 1 AUROC | Cases with identical final four vectors |', '|---|---:|---:|']
        for key, value in study['feedback_diagnostics'].items():
            lines.append(f"| {LABELS[key]} | {value['second_minus_first_round_AUROC']:+.4f} | {value['identical_final_four_vectors_cases']}/35 |")
        lines += ['', 'Identical final vectors quantify agreement, not correctness or a proven anchoring mechanism. The controlled visibility contrast is the more direct test of the anchoring hypothesis.', '']
    lines += ['## Scope and reproducibility', '',
        'The public-code cohort has 35 patients, 2,145 contacts and 304 SOZ contacts, differing from the paper\'s 35 / 2,066 / 298. The already inspected cohort makes this an exploratory follow-up. The frozen earlier reconstruction is preserved. Numerical CNN/Transformer checkpoints are reused; no LLM fine-tuning or new EEG foundation model was trained.', '',
        'Gemini is requested as gemini-4-argon with medium reasoning and a 32,768 combined completion-token cap. Every response records actual model identity, tokens, finish status and prompt hash. Argon uses the pinned native ScienceClaw LLMClient; Nemotron uses an explicit OpenAI-compatible adapter. No upstream autonomous heartbeat or Infinite publication is run.', '',
        'See [the experiment README](../experiments/spes_agent_v2/README.md) for exact commands. Full local evidence is under outputs/spes_agent_v2; raw recordings, labels, credentials and generated model files are excluded from Git. The portable [summary](../examples/spes_agent_v2_summary.json) contains aggregates and protocols only.', '',
        'Source evidence: frozen reconstruction outputs, fusion/protocol.json, all 25 fusion receipts and saved models, fusion_audit.json, study protocols and per-request native artifacts. Official API references: [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai/) and [Nemotron 3 Super](https://docs.nvidia.com/nemotron/latest/usage-cookbook/Nemotron-3-Super/OpenScaffoldingResources/README.html).', '']
    destination = project / 'docs/spes-agent-v2-report.md'; destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(('\n'.join(lines)).encode())
    write_json(project / 'examples/spes_agent_v2_summary.json', result)
    return destination


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(write_report(args.project, args.output))
