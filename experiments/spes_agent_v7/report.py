"""English result report for the fully schema-valid collaboration pilot."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from .experiment import write_json

def build(project):
    project=Path(project);root=project/'outputs/spes_agent_v7';decision=read_json(root/'pilot_decision.json');audit=read_json(root/'audit.json');behavior=read_json(root/'behavior.json')
    lines=['# Schema-valid collaboration reproduces geometric coverage','',
      'The repaired interface accepted **120/120 formal model decisions**. Single-agent, multi-agent and rationale-hidden policies all scored **0.790440 mean patient SOZ AUROC**, equal to the geometric-coverage policy on the same five patients. The fixed pilot rule does **not** permit expansion. No multi-agent superiority is established.','',
      '## Matched results','', '| Policy | Mean patient AUROC |','|---|---:|']
    for name,value in decision['mean_AUROC'].items():lines.append(f'| {name} | {value:.6f} |')
    lines+=['', '| Multi minus control | Mean gain | Exploratory paired 95% interval |','|---|---:|---:|']
    for name,v in decision['comparisons_multi_minus'].items():lines.append(f"| {name} | {v['mean_gain']:+.6f} | {v['paired_95CI']} |")
    lines+=['','Five patients provide a small exploratory screen. Repeatedly inspected cohort identities and adaptive project iterations prevent an untouched-test interpretation. The zero interval versus single describes these identical recorded outcomes, not a population guarantee of equivalence.','',
      '## What the agents actually did','',
      f"Reviewers selected the maximum-coverage query in **{behavior['choice_counts']['review_coverage_True']}/60** final choices. They replaced **{behavior['review_counts']['changed_proposal']}** proposals; every replacement moved to maximum geometric coverage. **{behavior['same_query_set_as_coverage']}/15** LLM trajectories selected the same final query set and **{behavior['same_ranking_as_coverage']}/15** produced the same ordered ranking as the coverage control. All 15 had its same AUROC.",'',
      'This is a behavioral observation. It supports a narrower explanation than a general claim about LLM medical knowledge: the workflow repeatedly returned to a common geometric proxy instead of producing an independently useful acquisition rule. The reviewer had final authority, and some distinct proposals were discarded. Neither geometric coverage nor model disagreement had been trained to predict actual localization gain. This pilot does not causally separate role authority, prompt emphasis, model priors and the available tool set.','',
      'The next fixed experiment learns measurement value from development-patient replay, using inner models that excluded each target patient from fitting and early stopping. It first tests numerical utility policies. More LLM calls are not justified merely by changing role names.','',
      '## Interface repair and fairness','',
      'The previous v6 validator rejected real nested citations. Historical v6 fallback outcomes remain unchanged. The new interface attaches [JSON Schema to the actual Gemini API request](https://ai.google.dev/gemini-api/docs/openai#structured-output): one currently available query ID, allowed evidence-root IDs, complete rationale and limitation, and no probability fields. Two synthetic calls checked the interface. A complete eight-call smoke trajectory passed before the rest of the pilot; those eight calls count inside the 120 formal attempts.','',
      'Each arm starts with the same four real measurements. Four adaptive cycles each contain a proposal call, review call, execution and real recorded-response feedback. Single and multi share exactly the same feedback schedule, measurements, numerical tools and final original HAT. The no-peer-rationale control retains measured feedback and query choices. Gemini 4 Argon uses medium reasoning and zero retries. No sparse v5 weights are adopted.','',
      f"The audit verified **{audit['native_artifacts']} native artifacts**, raw model/executed-decision agreement, actual API schemas, request budgets, peer-rationale isolation, independent AUROC and exact reproduction of the old random_0 control. The two synthetic calls are reported separately from the formal study.",'',
      '## Scope','',
      'This is a retrospective subset-acquisition replay. Stimulus IDs are array columns, with no independently reconstructed mapping to named clinical electrode pairs. Geometry and released preprocessing availability are assumed known. Physiological state changes, clinical safety and treatment effects are not simulated. The endpoint is SOZ ranking, not seizure control.','',
      'The new processed cohort has 35 patients / 2,117 contacts / 296 SOZ contacts. A source audit found 34 patients in common with the older 35-patient reconstruction: patient 39 replaces patient 47. Across 2,036 unambiguous coordinate matches in shared patients, scored labels agreed. Different patient/contact populations prevent direct cross-version architecture claims. All v7 policy comparisons use the same current cohort/pilot.','',
      'Waveform computation ran locally on the RTX 4090; language inference used the official Google API. No new cloud resources were created.','']
    (project/'docs/spes-agent-v7-report.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v7_summary.json',{'status':'completed_valid_no_gain_not_expanded','pilot_decision':decision,'audit':audit,'behavior':behavior,
        'separate_synthetic_preflight_calls':2,'formal_calls':120,'formal_accepted':120,'multi_agent_superiority_established':False,'clinical_benefit_established':False})
    print('English v7 report and compact evidence saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();build(a.project)
