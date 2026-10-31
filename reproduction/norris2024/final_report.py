"""Finalize the English report and compact, portable evidence after every stage completes."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from report import document
from train_reproduction import MODELS

PAPER_TABLE={
    'CNN (divergent)':{'AUROC':(.574,.065),'Specificity':(.632,.118),'Sensitivity':(.478,.169),'Youden':(.109,.106)},
    'CNN (convergent)':{'AUROC':(.666,.067),'Specificity':(.692,.147),'Sensitivity':(.531,.199),'Youden':(.223,.095)},
    'Transformer (base)':{'AUROC':(.730,.082),'Specificity':(.731,.130),'Sensitivity':(.589,.218),'Youden':(.319,.134)},
}


def summarize_agents(records):
    subjects=sorted({row['patient'] for row in records})
    methods=list(MODELS)+['fixed_mean_ensemble','single','multi']
    means={}
    for method in methods:
        values=[]
        for subject in subjects:
            subset=[r for r in records if r['patient']==subject]
            aucs=[r['strategies'][method]['metrics']['AUROC'] if method in ['single','multi']
                  else r['numerical_baselines'][method]['AUROC'] for r in subset]
            values.append(np.mean(aucs))
        means[method]=np.array(values)
    rng=np.random.default_rng(42)
    bootstrap=rng.integers(0,len(subjects),(10000,len(subjects)))
    comparisons={}
    for left,right in [('multi','single'),('multi','Transformer (base)'),('single','Transformer (base)'),('multi','fixed_mean_ensemble')]:
        delta=means[left]-means[right]
        comparisons[f'{left} minus {right}']={'mean_difference':float(delta.mean()),
            'patient_bootstrap_95CI':np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist()}
    costs={}
    for policy in ['single','multi']:
        selected=[r['strategies'][policy] for r in records]
        tokens={}
        for key in ['prompt_tokens','completion_tokens','total_tokens']:
            counts=[r.get('api_tokens',{}).get(key) for r in selected]
            tokens[key]=sum(counts) if all(v is not None for v in counts) else None
        costs[policy]={'requests':sum(r['calls'] for r in selected),'api_tokens':tokens,
                       'accepted_agent_outputs':sum(r.get('accepted_agent_outputs',8) for r in selected),
                       'schema_failures':sum(r.get('schema_failures',0) for r in selected),
                       'transport_failures':sum(r.get('transport_failures',0) for r in selected),
                       'unknown_usage_attempts':sum(r.get('unknown_usage_attempts',0) for r in selected),
                       'captured_tokens_lower_bound':{key:sum((r.get('captured_tokens_lower_bound',r['api_tokens']).get(key) or 0) for r in selected) for key in ['prompt_tokens','completion_tokens','total_tokens']},
                       'cumulative_case_seconds':sum(r['seconds'] for r in selected),
                       'response_models':sorted({name for r in selected for name in r['response_models']})}
    return {'distinct_patients':len(subjects),'held_out_cases':len(records),
            'mean_patient_AUROC':{name:float(value.mean()) for name,value in means.items()},
            'paired_comparisons':comparisons,'usage':costs,
            'uncertainty_unit':'35 patients; average their five repeated held-out evaluations first',
            'budget_note':'Request counts and configured output caps are matched; measured token totals may differ.'}


def build(project):
    output=project/'outputs/reproduction/norris2024'
    table=json.loads((output/'table1_reproduction.json').read_text())
    if any(value['finished_folds']!=25 for value in table.values()):raise ValueError('All 75 primary runs required')
    records=json.loads((output/'agent_results.json').read_text())
    if len(records)!=175:raise ValueError('All 175 held-out agent cases required')
    verification=json.loads((output/'verification.json').read_text())
    if verification['verified_primary_runs']!=75 or verification['verified_agent_cases']!=175 or verification['status'].startswith('partial'):
        raise ValueError('Complete independent verification required')
    sensitivity=json.loads((output/'channel_sensitivity.json').read_text())
    agent=summarize_agents(records)
    metrics={}
    for name,value in table.items():
        result={'runs':value['finished_folds'],'AUROC_mean':value['mean_AUROC'],'AUROC_SD_across_folds':value['SD_fold_AUROC'],'paper':value['paper']}
        for rule in ['paper_threshold_per_patient','released_code_threshold_per_patient']:
            result[rule]={}
            for metric in ['Sensitivity','Specificity','Youden','AUPRC']:
                folds=[np.mean([row[metric] for row in run[rule]]) for run in value['results'] if all(metric in row for row in run[rule])]
                result[rule][metric]={'mean':float(np.mean(folds)) if folds else None,
                                     'SD_across_folds':float(np.std(folds,ddof=1)) if len(folds)>1 else None,'defined_folds':len(folds)}
        metrics[name]=result
    summary={'title':'Public-code SPES reconstruction and matched-task agents',
             'exact_paper_reproduction':False,'cohort':{'patients':35,'electrodes':2145,'SOZ':304},
             'paper_cohort':{'patients':35,'electrodes':2066,'SOZ':298},
             'upstream_commit':'90062e8c5fe5593713125eaf5f573faedb739f65',
             'primary_models':metrics,'paper_table1_reported_metrics':PAPER_TABLE,'channel_sensitivity':sensitivity['summary'],'agents':agent,
             'verification':{k:v for k,v in verification.items() if k!='checkpoint_sha256'},
             'remaining_paper_gaps':['different cohort counts','unreleased original RNG and split manifests','unverified Engel outcomes and modified outcome-model details','original spatial example patient not specified']}
    calibration_path=output/'token_calibration_accounting.json'
    if calibration_path.exists():summary['separate_token_calibration']=json.loads(calibration_path.read_text())
    extras='<article><h2>Published and reconstructed model metrics</h2><p>Published values are from Table 1. Standard deviations describe the 25 fold/repeat summaries, not confidence intervals. Cohort and implementation differences remain disclosed above.</p><table><tr><th>Model</th><th>Metric</th><th>Published mean ± SD</th><th>Rerun: Appendix C threshold</th><th>Rerun: released-code threshold</th></tr>'
    for name,result in metrics.items():
        for metric in ['AUROC','Specificity','Sensitivity','Youden','AUPRC']:
            published=PAPER_TABLE[name].get(metric)
            paper_text=f'{published[0]:.3f} ± {published[1]:.3f}' if published else 'Not in Table 1'
            observed=[]
            for rule in ['paper_threshold_per_patient','released_code_threshold_per_patient']:
                if metric=='AUROC':text=f'{result["AUROC_mean"]:.4f} ± {result["AUROC_SD_across_folds"]:.4f}'
                else:
                    item=result[rule][metric]
                    text=(f'{item["mean"]:.4f} ± {item["SD_across_folds"]:.4f} ({item["defined_folds"]}/25)' if item['SD_across_folds'] is not None else 'Undefined')
                observed.append(text)
            extras+=f'<tr><td>{html.escape(name)}</td><td>{metric}</td><td>{paper_text}</td><td>{observed[0]}</td><td>{observed[1]}</td></tr>'
    extras+='</table></article><article><h2>Completed comparison and measured API use</h2><table><tr><th>Method</th><th>Mean patient AUROC</th></tr>'
    for method,value in agent['mean_patient_AUROC'].items():extras+=f'<tr><td>{html.escape(method)}</td><td>{value:.4f}</td></tr>'
    extras+='</table><table><tr><th>Paired comparison</th><th>Mean AUROC difference</th><th>95% patient bootstrap CI</th></tr>'
    for name,value in agent['paired_comparisons'].items():
        lo,hi=value['patient_bootstrap_95CI'];extras+=f'<tr><td>{html.escape(name)}</td><td>{value["mean_difference"]:+.4f}</td><td>[{lo:+.4f}, {hi:+.4f}]</td></tr>'
    extras+='</table><p>The primary collaboration comparison is multi-agent versus single agent. Comparisons with numerical baselines are additional controls. Intervals resample patients after averaging repeated evaluations; they do not treat 175 evaluations as 175 independent patients.</p><table><tr><th>Policy</th><th>Requests</th><th>Input tokens</th><th>Output tokens</th><th>Provider model</th></tr>'
    for policy,cost in agent['usage'].items():
        extras+=f'<tr><td>{policy}</td><td>{cost["requests"]}</td><td>{cost["api_tokens"]["prompt_tokens"]}</td><td>{cost["api_tokens"]["completion_tokens"]}</td><td>{html.escape(", ".join(cost["response_models"]))}</td></tr>'
    extras+='</table><p>'+html.escape(agent['budget_note'])+' Provider SDK response metadata is saved for every successful request. Malformed returned score vectors consume a request; the policy retains its previous scores (initially the shared Transformer). Such states are explicitly labelled, with no extra request or invented agent explanation.</p><p>'+'; '.join(f'{policy}: {value["accepted_agent_outputs"]} accepted outputs, {value["schema_failures"]} malformed outputs, {value["transport_failures"]} counted uncaptured failures' for policy,value in agent['usage'].items())+'.</p><p>A halted episode may continue only its unspent attempts. An uncaptured failure is retained as a labelled carry state, with unknown provider token consumption. Previously completed requests are never replayed; a partial token total is a lower bound. Case elapsed times can include the recovery pause.</p></article>'
    if 'separate_token_calibration' in summary:
        calibration=summary['separate_token_calibration']
        extras+=f'<article><h2>Separate API token calibration</h2><p>The initial 3,824-token combined ceiling truncated {calibration["truncated_captured_responses"]} of {calibration["captured_api_responses"]} captured responses. The stopped trial contains {calibration["attempted_requests"]} attempted requests and {calibration["unresolved_inflight_attempts"]} unresolved in-flight attempts. These records are preserved separately and excluded from performance comparisons; settings were changed based on truncation, without selecting by localization accuracy.</p><p>The fresh formal comparison uses an identical 8,192-token combined ceiling and low reasoning effort for both strategies. Captured calibration token usage is a lower bound because in-flight responses were not captured. Formal and calibration request/token consumption are reported separately.</p></article>'
    extras+='<article><h2>Channel sensitivity</h2><table><tr><th>Requested channels</th><th>Mean patient AUROC</th><th>95% patient bootstrap CI</th></tr>'
    for row in sensitivity['summary']:
        lo,hi=row['patient_cluster_bootstrap_95CI'];extras+=f'<tr><td>{row["channels"]}</td><td>{row["mean_patient_AUROC"]:.4f}</td><td>[{lo:.4f}, {hi:.4f}]</td></tr>'
    extras+='</table><p>The saved Transformer is reused without retraining. Cases with fewer valid channels provide their available responses, following the released implementation. The original inference sampling and bootstrap code were not released.</p></article>'
    example=min((r for r in records if r['seed']==1),key=lambda r:r['patient'])
    case_id=hashlib.sha256(f'{example["seed"]}:{example["fold"]}:{example["patient"]}'.encode()).hexdigest()[:16]
    extras+='<article><h2>Predetermined example: two rounds of specialist critique</h2><p>The lowest-ID eligible held-out patient in the first repeat was chosen by protocol, not by performance. Contact identifiers shown to agents are opaque. The generated explanations below are interpretations, not independently validated mechanisms or clinical conclusions. Every round sees the same measured data; there is no test-label feedback.</p><table><tr><th>Round</th><th>Role</th><th>Generated reason</th><th>Generated reflection</th></tr>'
    roles=['signal','network','uncertainty','critic']
    for step in range(8):
        response=json.loads((output/'agents'/case_id/'multi'/f'response_{step:02d}.json').read_text())
        label=roles[step%4]+(' (policy carry; unaccepted output)' if not response['accepted_agent_output'] else '')
        extras+=f'<tr><td>{step//4+1}</td><td>{label}</td><td>{html.escape(response["reason"])}</td><td>{html.escape(response["reflection"])}</td></tr>'
    extras+='</table></article><article><h2>Independent verification</h2><p>All 75 primary numerical runs and all 175 held-out agent cases were replayed from saved artifacts. Checks cover source hashes, disjoint patient splits, label order, numerical and agent scoring, request budgets, role schedule, provider usage records and native ScienceClaw artifact hashes/lineage. Raw recordings and clinical annotation tables are kept outside Git.</p><p>These are public-code reconstruction results on a disclosed different cohort. They do not establish exact reproduction of every paper result or therapeutic benefit.</p></article>'
    body=document(output).replace('</html>',extras+'</html>')
    (output/'final_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (project/'outputs/reports/paper_reproduction.html').write_text(body,encoding='utf-8')
    examples=project/'examples';examples.mkdir(exist_ok=True)
    (examples/'spes_reconstruction_report.html').write_text(body,encoding='utf-8')
    (examples/'spes_reconstruction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args()
    result=build(args.project.resolve());print(json.dumps(result['agents'],indent=2))
