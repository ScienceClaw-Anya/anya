"""Matched-model communication interventions with a blinded numerical judge."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import SCORERS, source_hashes, write_json
from .providers import Provider, ProviderUnavailable, parse_json

ROLES = ['signal', 'network', 'uncertainty', 'critic']
POLICIES = {'single_serial': ('single', 'serial', False),
            'multi_serial': ('multi', 'serial', False),
            'multi_independent_first': ('multi', 'independent_first', False),
            'multi_no_exchange': ('multi', 'no_exchange', False),
            'single_tools': ('single', 'serial', True),
            'multi_independent_tools': ('multi', 'independent_first', True)}
EXPERTISE = {'signal': 'Assess measured temporal response summaries and measurement limitations.',
 'network': 'Assess directional response patterns and measured neighborhood evidence; do not invent pathways.',
 'uncertainty': 'Assess model disagreement, validation reliability when supplied, and unjustified score revisions.',
 'critic': 'Check specific earlier claims, identify counterevidence and allow retaining the numerical baseline.',
 'single': 'Assess signal, network evidence, uncertainty and earlier reasoning as a generalist.'}
TOOLS = ['signal', 'network', 'reliability', 'fusion']

def visible_indices(policy, step):
    schedule = POLICIES[policy][1]
    if schedule == 'no_exchange' or (schedule == 'independent_first' and step < 4): return []
    return list(range(step))

def provider_for(step, routing='mixed'):
    if routing == 'argon_control': return 'argon'
    if routing != 'mixed': raise ValueError('Unknown model routing')
    return ['argon', 'nemotron', 'argon', 'nemotron'][step % 4]

def make_prompt(public, policy, step, board, tool_data=None, current_tool=None):
    mode, _, tools = POLICIES[policy]
    role = 'single' if mode == 'single' else ROLES[step % 4]
    visible = [{k: row[k] for k in ['step', 'role', 'scores', 'reason', 'reflection', 'accepted_agent_output']}
               for i, row in enumerate(board) if i in visible_indices(policy, step)]
    for row in visible:
        row['scores'] = [round(float(v), 4) for v in row['scores']]
        row['reason'] = row['reason'][:400]; row['reflection'] = row['reflection'][:400]
    result = {**public, 'role': role, 'expertise': EXPERTISE[role], 'round': step // 4 + 1,
      'shared_board': visible, 'constraints': ['No patient identity, SOZ/resection/outcome labels, reference prevalence or test accuracy is supplied.',
       'The observations are fixed across rounds; critique has no answer-key feedback.',
       'Use only supplied evidence; retain a baseline if no measured evidence justifies a revision.',
       'Write English explanations, at most 100 words each. Return only one JSON object.']}
    if tools:
        result['available_numerical_scorers'] = tool_data['scorers']
        result['validation_reliability'] = tool_data['reliability']
        result['executed_tool'] = current_tool
        result['required_json_shape'] = {'scorer': '|'.join(SCORERS), 'reason': 'English justification', 'reflection': 'English uncertainty or critique'}
        result['instruction'] = 'Choose one supplied, executable numerical scorer. The program will produce the contact scores; do not generate a score vector.'
    else:
        result['required_json_shape'] = {'scores': ['one finite number in [0,1] per contact, in order'], 'reason': 'English justification', 'reflection': 'English uncertainty or critique'}
        result['instruction'] = 'Return exactly one score per contact. Review only visible proposals. Revise scores only with a supported explanation.'
    return result

def validate_answer(answer, n, tool_data=None):
    if not isinstance(answer, dict): raise ValueError('Expected object')
    result = {}
    for key in ['reason', 'reflection']:
        if not isinstance(answer.get(key), str) or not answer[key].strip() or len(answer[key]) > 1800:
            raise ValueError('Missing or oversized explanation')
        result[key] = answer[key]
    if tool_data is not None:
        scorer = answer.get('scorer')
        if scorer not in SCORERS: raise ValueError('Unknown executable scorer')
        values = tool_data['scorers'][scorer]; result['selected_scorer'] = scorer
    else:
        values = answer.get('scores')
    if not isinstance(values, list) or len(values) != n: raise ValueError('Wrong score-vector length')
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any(): raise ValueError('Invalid scores')
    return {**result, 'scores': values.tolist(), 'accepted_agent_output': True, 'state_origin': 'agent' if tool_data is None else 'agent_selected_numerical_tool'}

def fallback(public, policy, step, board):
    visible = visible_indices(policy, step)
    return board[visible[-1]]['scores'] if visible else [c['models'][2] for c in public['contacts']]

def execute_tool(framework, case_dir, operation):
    name = 'neurotarget-spes-tools'
    metadata = framework.registry.get_skill(name)
    if not metadata: raise ValueError('SPES skill was not discovered')
    response = framework.executor.execute_skill(name, metadata,
        {'project': str(framework.project), 'public_case': str(case_dir / 'tools.json'), 'operation': operation, 'format': 'json'}, timeout=90)
    if response.get('status') != 'success' or not isinstance(response.get('result'), dict):
        raise RuntimeError('Native scientific tool execution failed')
    return response['result']

def run_policy(project, output, case_id, policy, routing='mixed'):
    project, output = Path(project), Path(output)
    sys.path.insert(0, str(project))
    from neurotarget.scienceclaw_runtime import Framework
    case = output / 'cases' / case_id; folder = output / 'studies' / routing / 'agents' / case_id / policy
    complete = folder / 'metrics.json'
    if complete.exists(): return json.loads(complete.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    public = json.loads((case / 'public.json').read_text()); tool_data = json.loads((case / 'tools.json').read_text())
    framework = Framework(project, folder / 'scienceclaw')
    session_file = folder / 'native_ids.json'
    ids = json.loads(session_file.read_text()) if session_file.exists() else {'observation': None, 'states': [], 'tools': []}
    if ids['observation'] is None:
        observation = framework.artifact('recordings', 'spes-v2-observation', public, 'spes_observation')
        ids['observation'] = observation.artifact_id; write_json(session_file, ids)
    board, clients = [], {}
    began = time.perf_counter()
    for step in range(8):
        request_file = folder / f'request_{step:02d}.json'; usage_file = folder / f'api_usage_{step:02d}.json'; state_file = folder / f'response_{step:02d}.json'
        if state_file.exists():
            if len(ids['states']) <= step: raise ValueError('Decision is missing its native artifact')
            board.append(json.loads(state_file.read_text())); continue
        if len(ids['states']) > step:
            # Recover the narrow artifact-before-state-file crash window without another API call.
            from .verify import read_native
            recovered = read_native(folder / 'scienceclaw')[ids['states'][step]]['payload']
            write_json(state_file, recovered); board.append(recovered); continue
        current_tool = None
        parents = [ids['observation']] + [ids['states'][i] for i in visible_indices(policy, step)]
        if POLICIES[policy][2]:
            tool_file = folder / f'tool_{step:02d}.json'
            if tool_file.exists(): current_tool = json.loads(tool_file.read_text())
            else:
                current_tool = execute_tool(framework, case, TOOLS[step % 4]); write_json(tool_file, current_tool)
            if len(ids['tools']) <= step:
                artifact = framework.artifact('scientific-tools', 'neurotarget-spes-tools', current_tool, 'scientific_tool_result', [ids['observation']])
                ids['tools'].append(artifact.artifact_id); write_json(session_file, ids)
            parents.append(ids['tools'][step])
        prompt = make_prompt(public, policy, step, board, tool_data if POLICIES[policy][2] else None, current_tool)
        if request_file.exists():
            if json.loads(request_file.read_text()) != prompt: raise ValueError('Frozen prompt changed on resume')
        else: write_json(request_file, prompt)
        previous = fallback(public, policy, step, board)
        error_kind = None
        if usage_file.exists():
            usage = json.loads(usage_file.read_text())
            text = usage.get('response_content') or ''
            if not usage.get('captured_response'): error_kind = 'transport'
        else:
            provider_name = provider_for(step, routing)
            if provider_name not in clients: clients[provider_name] = Provider(project, provider_name)
            try: text = clients[provider_name].ask(prompt, usage_file)
            except ProviderUnavailable:
                # The already-reserved attempt remains spent; a later run continues the unspent budget.
                write_json(folder / 'failure.json', {'step': step, 'attempt_counted': True, 'provider': provider_name})
                raise
        try:
            if error_kind: raise ValueError('Uncaptured response')
            usage = json.loads(usage_file.read_text())
            expected_model = 'gemini-4-argon' if provider_for(step, routing) == 'argon' else 'nvidia/nemotron-3-super-120b-a12b'
            if usage.get('response_model') != expected_model or usage.get('finish_reason') != 'stop':
                raise ValueError('Unexpected model identity or incomplete response')
            value = validate_answer(parse_json(text), len(previous), tool_data if POLICIES[policy][2] else None)
        except (ValueError, TypeError, KeyError):
            value = {'scores': previous, 'reason': 'The request consumed its budget. The policy retained its visible prior state or the shared Transformer baseline.',
                     'reflection': 'No accepted model prediction was obtained for this attempt.', 'accepted_agent_output': False,
                     'state_origin': 'policy_carry', 'failure_kind': error_kind or 'schema'}
        value.update(step=step + 1, role='single' if POLICIES[policy][0] == 'single' else ROLES[step % 4], provider=provider_for(step, routing))
        artifact = framework.artifact('paper-' + value['role'], 'spes-v2-policy', value, 'soz_policy_state', parents)
        ids['states'].append(artifact.artifact_id); write_json(session_file, ids)
        write_json(state_file, value); board.append(value)
    prediction = np.mean([r['scores'] for r in board[-4:]], axis=0)
    # No hidden labels are read until all policy decisions have been recorded.
    judge = json.loads((case / 'judge.json').read_text())
    result = {'case_id': case_id, 'policy': policy, 'routing': routing, 'patient': judge['patient'], 'seed': judge['seed'], 'fold': judge['fold'],
      'AUROC': float(roc_auc_score(judge['y'], prediction)), 'prediction': prediction.tolist(), 'attempts': 8,
      'accepted': sum(r['accepted_agent_output'] for r in board), 'tool_calls': len(ids['tools']),
      'seconds_this_invocation': time.perf_counter() - began, 'native_artifacts': 1 + len(ids['states']) + len(ids['tools'])}
    write_json(complete, result)
    print('COMPLETE', case_id, policy, flush=True)
    return result

def freeze_protocol(project, output, policies, routing='mixed'):
    protocol = {'version': 2, 'scope': 'Exploratory first-repeat study: 35 patients, seven held-out per fold; old cohort already inspected',
      'policies': policies, 'routing': routing, 'calls_per_policy_case': 8, 'model_schedule': [provider_for(i, routing) for i in range(8)],
      'models': {'argon': {'model': 'gemini-4-argon', 'reasoning_effort': 'medium', 'max_tokens': 32768},
                 'nemotron': {'model': 'nvidia/nemotron-3-super-120b-a12b', 'thinking': 'enabled', 'max_tokens': 32768}},
      'aggregation': 'Mean of the final four policy score states for every arm',
      'primary_contrast': 'multi_independent_first minus multi_serial, with identical model schedule and observations',
      'additional_contrasts': ['multi_independent_first minus single_serial', 'multi_independent_first minus multi_no_exchange', 'multi_independent_tools minus single_tools'],
      'isolation': 'First four independent-first prompts have empty boards; later prompts see all earlier states. No-exchange never sees another proposal.',
      'tool_arm_caveat': 'Tool arms bundle equal fixed tool calls with executable-scorer selection; comparison with manual vectors does not isolate tool access alone',
      'failures': 'Invalid responses consume attempts and carry a visible prior state or shared baseline. Transport failures halt; resume never repeats spent attempts.',
      'single_definition': 'One logical generalist with the same model routing and serial memory as the specialists; mixed routing uses four calls to each provider',
      'model_control': 'Argon-only is a separate control study and must not be presented as the requested Argon/Nemotron mixture',
      'tool_budget': 'Tool arms each execute eight cached-measurement/scorer CLI queries via native SkillExecutor; identical scorer/reliability access from the first step',
      'data_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((output / 'cases').glob('*/*.json'))},
      'source_sha256': source_hashes(project)}
    target = output / 'studies' / routing / 'agent_protocol.json'
    if target.exists() and json.loads(target.read_text()) != protocol: raise ValueError('Frozen agent protocol or source changed')
    write_json(target, protocol)
    return protocol

def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4); p.add_argument('--policies', nargs='+', choices=list(POLICIES), default=list(POLICIES))
    p.add_argument('--routing', choices=['mixed', 'argon_control'], default='mixed')
    p.add_argument('--max-patients', type=int, default=35)
    args = p.parse_args(); args.project = args.project.resolve(); args.output = args.output.resolve()
    if not 1 <= args.workers <= 24 or not 1 <= args.max_patients <= 35: raise ValueError('Invalid bounded run size')
    for name in sorted({provider_for(i, args.routing) for i in range(8)}): Provider(args.project, name)
    freeze_protocol(args.project, args.output, args.policies, args.routing)
    rows = json.loads((args.output / 'fusion_results.json').read_text())['cases']
    cases = sorted((r for r in rows if r['seed'] == 1), key=lambda r: (r['fold'], r['patient']))[:args.max_patients]
    tasks = [(r['case_id'], policy) for r in cases for policy in args.policies]
    # Only one task per worker is in flight. A failure cannot silently queue the entire cohort.
    pending = iter(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        def submit():
            item = next(pending, None)
            if item is not None:
                future = pool.submit(run_policy, args.project, args.output, *item, args.routing)
                futures[future] = item
        for _ in range(args.workers): submit()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            # Resolve the whole completed batch before submitting any additional work.
            for future in done: future.result()
            for future in done: del futures[future]
            for _ in done: submit()
    print('ALL REQUESTED POLICIES COMPLETE', flush=True)

if __name__ == '__main__': main()
