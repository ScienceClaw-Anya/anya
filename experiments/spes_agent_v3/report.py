"""English report from audited v3 evidence, with incomplete stages kept explicit."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from .common import SELECTOR, WEIGHTS, read_json, sha, write_json
from .engine import POLICIES

LABELS={'numeric_transformer':'Numerical Transformer','numeric_selector':'Numerical conservative selector',
    'single':'Single generalist','multi_serial':'Serial specialists','multi_independent':'Independent-first specialists','multi_no_exchange':'Specialists without exchange'}

def build(project,output):
    project,output=Path(project),Path(output)
    train=[read_json(p) for p in (output/'crossfit').glob('*/*/receipt.json')]
    summary={'version':3,'status':'inner_training','completed_inner_fits':len(train),'expected_inner_fits':60,
        'training_fit_seconds_sum':sum(r['seconds'] for r in train),'selector_config':SELECTOR,'candidate_weights':WEIGHTS,
        'source_evidence':{'training_protocol_sha256':sha(output/'training_protocol.json')},
        'scope':'Exploratory first-repeat study on 35 previously inspected public-code patients; not external clinical validation'}
    lines=['# SPES v3: conservative selection with matched information','',
        'This experiment makes Transformer the explicit shared default, replaces free probability generation with executable decisions, and trains a small conservative selector from patient-isolated cross-predictions. The previous experiments remain frozen.','',
        '## Completion status','',f'Inner numerical model fits: **{len(train)}/60**. No incomplete stage is reported as a completed comparison.','']
    interruption=read_json(output/'interruption.json') if (output/'interruption.json').exists() else None
    usage_all=[read_json(p) for p in (output/'policies').glob('*/*/usage_*.json')]
    summary['actual_model_attempts']=len(usage_all)
    summary['captured_model_responses']=sum(v.get('captured_response',False) for v in usage_all)
    summary['usage_status_counts']=dict(Counter(v['status'] for v in usage_all))
    summary['captured_usage_tokens']={k:sum((v.get('usage') or {}).get(k,0) or 0 for v in usage_all) for k in ['prompt_tokens','completion_tokens','total_tokens']}
    if interruption:
        summary['interruption']=interruption
        summary['source_evidence']['interruption_sha256']=sha(output/'interruption.json')
        lines+=['## The uninformative API run was stopped','',
            f"The user requested stopping unnecessary execution after the audit found no distinct rankings. **{interruption['formal_attempts_recorded']} attempts** were recorded; "
            f"**{interruption['remaining_attempts_not_scheduled']} remaining attempts were not scheduled**. In-flight reserved attempts remain spent; their token consumption is unknown.",'',
            'All 60 neural fits and five selector fits completed. The agent study did not complete all 35 patients per policy. Partial traces and completed cases are retained; they are not presented as a full observed agent comparison. Ranking equivalence for all 35 patients follows from the identical audited score vectors, not from a completed model-call study.','',
            'A new label-free launch guard now rejects identical-ordering score banks and user-stopped studies before any model client is created. The original frozen policy controller is preserved for replay.','']
    prepared_path=output/'prepared.json';audit_path=output/'preparation_audit.json';policy_audit=output/'policy_audit.json'
    if prepared_path.exists() and audit_path.exists() and read_json(audit_path)['verified']:
        prepared=read_json(prepared_path);audit=read_json(audit_path);cases=prepared['cases']
        summary.update(status='numerical_preparation_audited',patients=35,distinct_action_patients=audit['distinct_action_patients'],
            selector_fits=5,gates=audit['gates'],distinct_ranking_patients=audit['distinct_ranking_patients'],
            numeric_AUROC={key:float(np.mean([r[key+'_AUROC'] for r in cases])) for key in ['transformer','selector']})
        summary['selector_training_evidence']=[{'outer':r['outer'],'OOF_transformer_AUROC':r['OOF_transformer_AUROC'],'candidate_evidence':r['evidence']} for r in prepared['training_receipts']]
        gate_values=[v for r in prepared['training_receipts'] for v in r['evidence'].values()]
        summary['gate_diagnostics']={'candidate_fold_combinations':len(gate_values),
            'positive_mean_gains':sum(v['OOF_mean_gain']>0 for v in gate_values),
            'positive_lower_bounds':sum(v['bootstrap_lower_bound']>0 for v in gate_values),
            'largest_mean_gain':max(v['OOF_mean_gain'] for v in gate_values),
            'largest_lower_bound':max(v['bootstrap_lower_bound'] for v in gate_values)}
        summary['source_evidence'].update(selector_protocol_sha256=sha(output/'selector_protocol.json'),preparation_audit_sha256=sha(audit_path))
        lines+=['## Numerical results and actual decision space','',
            '| Numerical policy | Mean patient AUROC, first repeat |','|---|---:|']
        for key,value in summary['numeric_AUROC'].items():lines.append(f'| {key} | {value:.4f} |')
        lines+=['','These two numerical scores cover all 35 prepared and independently replayed scorer vectors. The execution table below counts completed eight-step policy traces, a separate quantity; no full-cohort empirical agent score is inferred from those partial traces.']
        lines+=['',f"**{audit['distinct_action_patients']}/35 patients** have different executable vectors, and **{audit['distinct_ranking_patients']}/35** have different contact orderings for Transformer and the conservative selector.", '']
        if audit['distinct_ranking_patients']==0:
            lines+=['The allowed scorers have the same contact ordering throughout this held-out cohort. Equal ranking performance is forced by the action space; it cannot establish that collaboration generally lacks value.','']
            if audit['distinct_action_patients']==0:lines+=['Every selector output equals the exact Transformer vector.','']
        gd=summary['gate_diagnostics']
        lines += [f"Of {gd['candidate_fold_combinations']} development fold/candidate combinations, {gd['positive_mean_gains']} had positive mean gains; "
            f"the largest was {gd['largest_mean_gain']:+.4f} AUROC. {gd['positive_lower_bounds']} had positive bootstrap lower bounds. "
            'The tested bounded CNN blends did not establish a robust development gain under this protocol.','',
            'The global eligibility veto can also suppress a candidate that helps a predictable subgroup. This experiment does not establish that such subgroup selection is impossible; it has not independently validated such a policy. Improving candidate complementarity or subgroup evidence requires a new development protocol, rather than interpreting a constrained tie as a general result about agents.','']
        lines+=['| Outer fold | Eligible alternatives |','|---|---|']
        for outer,gates in audit['gates'].items():lines.append(f"| {outer} | {', '.join(k for k,v in gates.items() if v) or 'None; retain Transformer'} |")
        lines+=['','| Outer fold | Candidate | Inner-OOF mean gain | Gate lower bound | Eligible |','|---|---|---:|---:|---|']
        for receipt in prepared['training_receipts']:
            for name,value in receipt['evidence'].items():
                lines.append(f"| {receipt['outer']} | {name} | {value['OOF_mean_gain']:+.4f} | {value['bootstrap_lower_bound']:+.4f} | {value['eligible']} |")
        lines+=['','These are development-stage gate statistics, not independent held-out scores of the trained selector. The 0.005 margin is a fixed research safeguard, not a validated clinical effect threshold.','']
    metrics=[read_json(p) for p in (output/'policies').glob('*/*/metrics.json')]
    if policy_audit.exists():summary['completed_case_audit']=read_json(policy_audit)
    if metrics:
        count={policy:sum(r['policy']==policy for r in metrics) for policy in POLICIES};summary['completed_policy_cases']=count
        if all(n==35 for n in count.values()) and policy_audit.exists() and read_json(policy_audit)['full_cohort_complete']:
            checked=read_json(policy_audit);summary['status']='complete_audited';summary['audit']=checked
            summary['source_evidence']['policy_audit_sha256']=sha(policy_audit)
            patients=sorted(set(r['patient'] for r in metrics));assert len(patients)==35
            mean_auc={p:float(np.mean([r['AUROC'] for r in metrics if r['policy']==p])) for p in POLICIES}
            summary['mean_patient_AUROC']=mean_auc
            summary['mean_patient_AP']={}
            for policy in POLICIES:
                values=[]
                for row in metrics:
                    if row['policy']!=policy:continue
                    truth=read_json(output/'cases'/row['case_id']/'judge.json')['y'];values.append(average_precision_score(truth,row['prediction']))
                summary['mean_patient_AP'][policy]=float(np.mean(values))
            contrasts=[('multi_independent','single'),('multi_serial','single'),('multi_independent','multi_serial'),
                ('multi_independent','multi_no_exchange'),('numeric_selector','numeric_transformer')]
            contrasts += [(p,'numeric_selector') for p in POLICIES[2:]]
            rng=np.random.default_rng(42);draw=rng.integers(0,35,(10000,35));summary['contrasts']={}
            for a,b in contrasts:
                left={r['patient']:r['AUROC'] for r in metrics if r['policy']==a};right={r['patient']:r['AUROC'] for r in metrics if r['policy']==b}
                delta=np.array([left[p]-right[p] for p in patients]);lo,hi=np.quantile(delta[draw].mean(axis=1),[.025,.975])
                summary['contrasts'][a+' minus '+b]={'mean':float(delta.mean()),'patient_bootstrap_95CI':[float(lo),float(hi)]}
            primary=summary['contrasts']['multi_independent minus single'];lo,hi=primary['patient_bootstrap_95CI']
            if summary['distinct_ranking_patients']==0:verdict='There was no effective ranking choice between the allowed scorers; all policies necessarily tie on ranking.'
            elif lo>0:verdict='Independent-first specialists improved the mean ranking relative to the generalist in this exploratory comparison.'
            elif hi<0:verdict='Independent-first specialists performed worse than the generalist in this exploratory comparison.'
            else:verdict='This comparison does not demonstrate a consistent ranking benefit from independent-first specialists over the generalist.'
            summary['interpretation']=verdict
            lines+=['## Completed matched-budget comparison','',verdict,'',
                '| Policy | Mean patient AUROC | Mean patient AP | Actual Gemini attempts |','|---|---:|---:|---:|']
            for policy in POLICIES:
                attempts=sum(r['model_attempts'] for r in metrics if r['policy']==policy)
                lines.append(f'| {LABELS[policy]} | {mean_auc[policy]:.4f} | {summary["mean_patient_AP"][policy]:.4f} | {attempts} |')
            lines+=['',f"Primary contrast, independent-first specialists minus generalist: **{primary['mean']:+.4f} AUROC**, 95% patient-bootstrap interval **[{lo:+.4f}, {hi:+.4f}]**.",'']
            for key,value in summary['contrasts'].items():
                if key=='multi_independent minus single':continue
                lower,upper=value['patient_bootstrap_95CI'];lines.append(f"- {key}: {value['mean']:+.4f}, interval [{lower:+.4f}, {upper:+.4f}].")
            usage=[read_json(p) for p in (output/'policies').glob('*/*/usage_*.json')]
            summary['actual_model_attempts']=len(usage);summary['captured_model_responses']=sum(v.get('captured_response',False) for v in usage)
            summary['captured_usage_tokens']={k:sum((v.get('usage') or {}).get(k,0) or 0 for v in usage) for k in ['prompt_tokens','completion_tokens','total_tokens']}
            summary['policy_details']={}
            lines+=['','## Defaults, disagreements and resource use','',
                '| Policy | Patients below / above / tied with Transformer | Schema defaults | Transport defaults |','|---|---|---:|---:|']
            baseline={r['patient']:r['AUROC'] for r in metrics if r['policy']=='numeric_transformer'}
            for policy in POLICIES:
                states=[read_json(p) for p in (output/'policies').glob('*/'+policy+'/state_*.json')]
                failures=Counter(s.get('failure_kind') for s in states if not s['accepted_output'])
                delta=[r['AUROC']-baseline[r['patient']] for r in metrics if r['policy']==policy]
                detail={'patients_below_transformer':sum(d < -1e-12 for d in delta),'patients_above_transformer':sum(d > 1e-12 for d in delta),
                    'patients_tied_transformer':sum(abs(d)<=1e-12 for d in delta),'failures':dict(failures),
                    'final_round_actions':dict(Counter(s['action'] for s in states if s['step']>4))}
                summary['policy_details'][policy]=detail
                counts=f"{detail['patients_below_transformer']} / {detail['patients_above_transformer']} / {detail['patients_tied_transformer']}"
                lines.append(f"| {LABELS[policy]} | {counts} | {failures['schema']} | {failures['transport']} |")
            lines+=['',f"There were **{len(usage):,} formal Gemini attempts**, **{checked['accepted_model_outputs_completed_cases']:,} accepted model outputs**, "
                f"**{checked['native_tool_queries']:,} native tool queries** and **{checked['native_artifacts']:,} verified native artifacts**. No failed patient or attempt was discarded.",'',
                'All policies have eight decision/query slots and an eight-model-call upper limit. The numerical policies make zero language-model calls. Common ceilings are not exact cost equality. Captured token totals do not establish the billed usage of uncaptured attempts. Synthetic preflight calls are excluded.','']
        else:
            lines+=['## Policy execution was stopped' if interruption else '## Policy execution remains incomplete','','| Policy | Completed patients |','|---|---:|']
            lines += [f'| {LABELS[p]} | {count[p]}/35 |' for p in POLICIES];lines+=['','No incomplete arm is used for full-cohort performance claims.','']
    if (output/'development_headroom.json').exists() and (output/'conditional_selector_diagnostic.json').exists():
        headroom=read_json(output/'development_headroom.json');conditional=read_json(output/'conditional_selector_diagnostic.json')
        summary['development_headroom']=headroom;summary['conditional_selector_diagnostic']=conditional
        lines+=['## What could justify a useful next experiment','',
            f"A cheap development-only hindsight diagnostic found an average best-of-four gain of {headroom['mean_of_fold_hindsight_gains']:+.4f} AUROC across the five development pools. "
            'This chooses after seeing development labels: it is optimistic, not a deployable score and not proof that a selector can identify the benefiting patients. The pools overlap and are not independent cohorts.','',
            f"Ignoring the global eligibility veto while keeping the existing ridge predictions and 0.005 margin would propose alternatives for {conditional['ungated_unlabeled_outer_activations_total']}/35 unlabeled outer cases. "
            'This counts possible actions only; no claim is made that their held-out accuracy improves. No new training, neural inference or API calls were performed for these diagnostics.','',
            'The current global rule tests whether each candidate is reliably beneficial on average. A personalized selector instead needs independent evaluation of its complete conditional policy. A candidate can have a negative overall average yet help a predictable subgroup; the present global veto prevents testing that possibility. Any new policy must be assessed with patient isolation through every fitted component, rather than treating selector-training scores as validation.','',
            'The next useful work is numerical feasibility before agents: define a single conditional-policy hypothesis, validate it on development patients, and stop if it cannot beat Transformer reliably. This cohort has already been inspected repeatedly, so any further replay is exploratory; an independent cohort is needed for final validation. Agent comparison should resume only with a meaningful decision space and a reason to expect value beyond the numerical policy.','']
    lines+=['## How the numerical gate was trained','',
        'Five outer folds each hold out seven patients. Four new inner folds within the 28-patient development pool use 17 fitting patients, four early-stopping patients and seven OOF prediction patients. All three base architectures are fitted in each inner job: 60 new fits. The outer test patients never enter fitting, normalization or early stopping.','',
        'Five fixed Ridge(alpha=100) selectors learn patient-level AUROC gains from four prediction-summary features. All candidate alternatives retain at least 80% Transformer weight. Eligibility requires a Bonferroni one-sided patient-bootstrap lower bound above 0.005 gain; the personalized predicted gain must also exceed 0.005. Otherwise the exact Transformer vector is returned. No test-driven parameter search was performed.','',
        'The gate is a conservative development heuristic, not a clinical guarantee. Overlapping inner models affect uncertainty, candidate OOF performance trains the selector, and its individual gain predictions are not independently calibrated. Inner base fits use 17 plus four early-stop patients; reused outer models use 21 plus seven. This training-size shift is a limitation.','',
        '## What is held constant','',
        'Every numerical and agent policy receives the same explicit baseline, OOF reliability, permitted scorer outputs and distribution evidence. Agents output only a scorer action and structured evidence references. Probability vectors and manual weights are rejected. Complete reasons are retained within the declared length bound.','',
        'Four Gemini policies share Gemini 4 Argon, medium reasoning, a 32,768 combined completion-token cap and an eight-request trajectory cap. Completed trajectories use eight attempts; interrupted trajectories retain only their spent prefix. The primary contrast is independent-first specialists versus one generalist. Serial specialists and no-exchange specialists are controls. Completed final scores are programmatic means of the last four executable vectors in full precision. Invalid or uncaptured attempts default to Transformer and are never replaced by an extra request.','',
        'These are static recorded-data decisions: later rounds receive discussion, not newly measured SPES data or hidden accuracy feedback. Roles share the same base model; they are not separately trained clinical experts.','',
        '## Evidence and limits','',
        f"Observed sum of individual neural fit durations: **{summary['training_fit_seconds_sum']:.1f} seconds**. Fits overlap on the local RTX 4090; this sum is not elapsed wall time. No additional Nebius GPU job was created. Gemini inference uses the official Google API.",'',
        'The public-code cohort contains 35 patients, 2,145 contacts and 304 SOZ contacts; the paper reports 35 / 2,066 / 298. The cohort and earlier results were already inspected. This follow-up is exploratory and requires independent patients for external validation. AUROC and AP measure retrospective SOZ ranking, not treatment benefit. Bootstrap intervals resample patients conditional on recorded fits and generations, and do not capture all retraining or sampling variability. Secondary comparisons are exploratory.','',
        'The preparation audit verifies every inner split including early stopping, checkpoint/prediction hashes, label alignment, ridge normal-equation coefficients, bootstrap gate and exact scorer execution for all 35 cases. The policy audit verifies packets, budgets and native lineage for completed trajectories; interrupted prefixes are preserved, not certified as complete. Neural outputs are frozen captured predictions; the audit does not claim bit-exact re-inference of stochastic CNN channel sampling.','',
        'See the [v3 README](../experiments/spes_agent_v3/README.md) for exact commands and the [aggregate summary](../examples/spes_agent_v3_summary.json). Full private evidence stays in outputs/spes_agent_v3; raw data, labels, model files and credentials are excluded from Git.','']
    summary['reporting_source_sha256']=sha(Path(__file__))
    if interruption:summary['status']='stopped_uninformative_action_space'
    if summary.get('distinct_ranking_patients')==0:
        lines[2]='The conservative selector retained the exact Transformer ranking for all 35 held-out patients. No tested alternative cleared the fixed development gate. The resulting policy tie is forced by the permitted action space and cannot identify a general collaboration benefit or failure. Execution and audit status are listed below.'
    path=project/'docs/spes-agent-v3-report.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('\n'.join(lines)).encode())
    write_json(project/'examples/spes_agent_v3_summary.json',summary)
    print(path);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
