"""Add a source-backed interpretation after the frozen experiment's offline audit.

Reporting only: this file never trains a model, calls an API or changes prompts.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(); project, output = args.project.resolve(), args.output.resolve()
sys.path.insert(0, str(project))
from experiments.spes_agent_v2.report import write_report

destination = write_report(project, output)
summary_path = project / 'examples/spes_agent_v2_summary.json'
summary = json.loads(summary_path.read_text())
study = summary['studies']['argon_control']
extra = ['## Interpretation of the completed comparison', '']
if study['status'] == 'complete_audited':
    key = 'multi_independent_first minus multi_serial'
    primary = study['contrasts'][key]; lo, hi = primary['patient_bootstrap_95CI']
    if lo > 0:
        conclusion = 'Independent first-round deliberation improved SOZ ranking relative to serial specialist discussion in this exploratory comparison.'
    elif hi < 0:
        conclusion = 'Independent first-round deliberation reduced SOZ ranking performance relative to serial specialist discussion in this exploratory comparison.'
    else:
        conclusion = 'This comparison did not demonstrate a consistent benefit from independent first-round deliberation.'
    extra += [conclusion + f" The paired mean AUROC difference was {primary['mean']:+.4f}, with a 95% patient-bootstrap interval of [{lo:+.4f}, {hi:+.4f}]. "
        'The cohort was already inspected and is small; this is not external clinical validation.', '',
        'Intervals resample the 35 patients conditional on the recorded model generations and numerical checkpoints. There is one agent run per patient and policy in this first-repeat study; the intervals do not quantify all variability from rerunning stochastic model generation or retraining the numerical models.', '',
        'All roles in this control use the same base model and observation packet. Independence changes which proposals are visible; it does not create independent training data, separately trained experts, or new physiological measurements. The result concerns this prompted-role architecture and cannot establish that every multi-agent architecture succeeds or fails.', '',
        'A change in agreement or second-round accuracy alone does not identify anchoring as the mechanism. The matched visibility intervention is the primary evidence; agreement and reflection diagnostics help describe what the policies did.', '',
        'Every arm used the same number of request attempts and the same maximum completion-token cap. Realized context length, token use and monetary cost can differ. The comparison establishes a call-budget match, not exact monetary-cost equality. No price estimate is invented.', '']
    scores = study['mean_patient_AUROC']; baseline = summary['first_repeat_baselines']['transformer']
    tool_delta = study['contrasts']['multi_independent_tools minus single_tools']
    tool_lo, tool_hi = tool_delta['patient_bootstrap_95CI']
    extra += [f"The single generalist with numerical tools reached {scores['single_tools']:.4f}, and tool-enabled specialists reached {scores['multi_independent_tools']:.4f}; "
        f"the matching Transformer baseline was {baseline:.4f}. Multi-agent minus single-agent in the tool comparison was {tool_delta['mean']:+.4f} "
        f'[{tool_lo:+.4f}, {tool_hi:+.4f}] (95% paired patient-bootstrap interval). This comparison also does not demonstrate a consistent collaboration benefit.', '',
        f"The corresponding manual-score generalist reached {scores['single_serial']:.4f}. Tool arms are descriptively closer to the numerical baseline, "
        'but they jointly change scorer access, validation evidence and the output interface. The data do not identify which component caused that difference. The practical recommendation is to preserve the held-out numerical baseline for contact ranking and require evidence before accepting LLM score revisions.', '',
        f"Serial specialists produced identical final four vectors for {study['feedback_diagnostics']['multi_serial']['identical_final_four_vectors_cases']}/35 patients; "
        f"independent-first specialists did so for {study['feedback_diagnostics']['multi_independent_first']['identical_final_four_vectors_cases']}/35. "
        'Thus greater output diversity did not translate into a demonstrated ranking benefit. The controlled comparison does not establish first-round copying as the main performance limit.', '',
        f"Independent-first second-round AUROC changed by {study['feedback_diagnostics']['multi_independent_first']['second_minus_first_round_AUROC']:+.4f}, "
        f"while no-exchange changed by {study['feedback_diagnostics']['multi_no_exchange']['second_minus_first_round_AUROC']:+.4f}. "
        'A small second-round gain is therefore not, by itself, evidence that shared reflection added value. These round-change diagnostics are descriptive and exploratory.', '']
    extra += ['Public model probabilities and response summaries are serialized to five decimal places; executable scorer vectors use six. Numerical baseline tables use the original full-precision predictions. Tiny rank-tie differences are therefore possible against a raw baseline; all policies within each communication comparison share the same serialization.', '']
    extra += ['Token totals sum captured usage records. Any uncaptured request has unknown token consumption; these totals do not establish complete billed usage.', '']
    study['token_accounting_note'] = 'Sum of captured usage records only; uncaptured requests have unknown consumption, not verified zero cost.'
    root = output / 'studies/argon_control'
    failure_accounting = {}
    selections = {}
    extra += ['### Recorded failures and executable choices', '', '| Policy | Attempts | Schema carries | Transport carries |', '|---|---:|---:|---:|']
    for policy in study['completed_cases']:
        state_paths = list((root / 'agents').glob('*/' + policy + '/response_*.json'))
        states = [json.loads(p.read_text()) for p in state_paths]
        counts = Counter(s.get('failure_kind') for s in states if not s['accepted_agent_output'])
        failure_accounting[policy] = {'attempts': len(states), 'schema_carries': counts['schema'], 'transport_carries': counts['transport']}
        extra.append(f"| {policy} | {len(states)} | {counts['schema']} | {counts['transport']} |")
        if policy.endswith('tools'):
            selections[policy] = dict(Counter(s.get('selected_scorer', 'policy_carry') for s in states if s['step'] > 4))
    summary['failure_accounting'] = failure_accounting
    summary['final_round_scorer_selections'] = selections
    extra += ['', 'All failed attempts remain in the primary comparison through the predeclared carry rule; no failed case or patient is dropped.', '',
              '| Tool policy | Final-round scorer selections, 140 opportunities |', '|---|---|']
    for policy, counts in selections.items():
        extra.append('| ' + policy + ' | ' + '; '.join(f'{name}: {count}' for name, count in sorted(counts.items())) + ' |')
    extra += ['', 'Scorer-selection counts describe executable decisions, not clinical effects. Averaging the final four selections can produce a numerical mixture even when no single scorer is used unanimously.', '']
    latest = max(p.stat().st_mtime for p in (root / 'agents').glob('*/*/metrics.json'))
    usage_paths = list((root / 'agents').glob('*/*/api_usage_*.json'))
    first_request = min(p.stat().st_mtime - json.loads(p.read_text()).get('seconds', 0) for p in usage_paths)
    elapsed = max(0., latest - first_request)
    extra += [f"All six policies completed 35 patients each: {study['total_request_attempts']:,} formal attempts, "
        f"{study['accepted_agent_outputs']:,} accepted outputs and {study['native_artifacts']:,} verified native artifacts. "
        f'Elapsed API-run time, estimated from local request receipts, was about {elapsed/60:.1f} minutes with 16 local coordinators waiting on the remote API. '
        'The synthetic preflight is excluded from these counts.', '']
    summary['interpretation'] = {'primary_contrast': key, 'conclusion': conclusion, 'elapsed_seconds': elapsed,
        'note': 'Post-run reporting; no changes to frozen agent or training inputs'}
else:
    extra += ['The communication study is incomplete. No full-cohort conclusion is drawn.', '']

diagnostic_path = output / 'fusion_postfit_diagnostics.json'
if diagnostic_path.exists():
    diagnostic = json.loads(diagnostic_path.read_text())
    extra += ['## Why the fixed response-feature fusion may generalize poorly', '',
        '| Model | Score on patients used for fitting / checkpoint selection | Held-out patient AUROC |', '|---|---:|---:|']
    for name, value in diagnostic['validation_resubstitution_mean_patient_AUROC'].items():
        extra.append(f"| {name} | {value:.4f} | {diagnostic['held_out_mean_patient_AUROC'][name]:.4f} |")
    extra += ['', 'The response-feature fusion has a larger fit-to-test gap than the Transformer. That is consistent with overfitting or weak transportability of these summaries across patients. It is not proof of one causal explanation, nor independent evidence that the added features contain no signal. Fusion scores on fitting patients are resubstitution, and the Transformer also used validation for early stopping.', '',
        'This diagnostic was computed after fitting, without refitting, hyperparameter search, or modifications to the ongoing communication experiment. The next defensible model-development step would use patient-disjoint development data and a separate untouched evaluation cohort; the current test set should not become a tuning set.', '',
        'Reproduce the diagnostic with `python analysis/spes_v2_fusion_diagnostics.py --project . --output outputs/spes_agent_v2`. '
        'Rebuild this interpreted report, after the offline audit, with `python analysis/finalize_spes_v2_report.py --project . --output outputs/spes_agent_v2`.', '']
    summary['fusion_postfit_diagnostics'] = diagnostic

summary['reporting_source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [project / 'analysis/spes_v2_fusion_diagnostics.py', Path(__file__).resolve()]}
text = destination.read_text(encoding='utf-8')
text = text.replace('| -0.0000 |', '| +0.0000 |')
heading, body = text.split('\n', 1)
destination.write_bytes((heading + '\n\n' + '\n'.join(extra) + '\n' + body.lstrip()).encode())
summary_path.write_bytes((json.dumps(summary, indent=2, allow_nan=False) + '\n').encode())
print(destination)
