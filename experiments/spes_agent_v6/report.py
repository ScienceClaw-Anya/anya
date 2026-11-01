"""English report for the frozen common-prefix proposal/review study."""
from analysis.action_mapping_correction import ACTION_NOTICE
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json
from .experiment import write_json,NUMERIC,LLM

def build(project):
    project=Path(project);root=project/'outputs/spes_agent_v6';protocol=read_json(root/'protocol.json')
    rows=[read_json(p) for p in root.glob('trajectories/*/*/metrics.json')];decision=read_json(root/'pilot_decision.json') if (root/'pilot_decision.json').exists() else None
    lines=[ACTION_NOTICE,'','# Feedback-matched SPES collaboration','',
      'This exploratory experiment gives every policy the same four initial measurements, then four adaptive measurements. Each adaptive query receives a proposal and review call before execution. Actual measured feedback arrives at the same points for every LLM arm. The original v4 HAT remains the final scorer; sparse v5 models are not adopted.','',
      'The single agent proposes and self-reviews; the team alternates hypothesis/uncertainty proposers and audit/coverage reviewers. A team ablation hides peer rationales while retaining query choices and physiological feedback. Every LLM arm has eight attempts and eight total measurements. Numerical policies use no LLM calls. All share the same numerical tools. No agent-generated probabilities are executed.','',
      'Five fixed pilot patients exclude the five v4 LLM pilot patients. This is still a previously inspected cohort; it is not independent external validation. The new numerical random_0 control reproduces the old exact scores, verifying the same responses and scorer.','',
      '| Policy | Completed pilot patients | Mean SOZ AUROC | Accepted model outputs |','|---|---:|---:|---:|']
    for name in NUMERIC+LLM:
        selected=[r for r in rows if r['policy']==name and r['pilot']]
        if selected:lines.append(f"| {name} | {len(selected)} | {np.mean([r['AUROC'] for r in selected]):.6f} | {sum(r['accepted_model_outputs'] for r in selected)} |")
    lines+=['','Partial, unequal completed counts are progress only; do not compare unpaired partial means.','']
    if decision:
        d=decision['comparisons_multi_minus']['single'];lines+=['## Frozen pilot decision','',
          f"Multi minus single: **{d['mean_gain']:+.6f}**, exploratory paired 95% patient-bootstrap interval **{d['paired_95CI']}**. Frozen-rule expansion: **{decision['expand']}**. Accepted fraction: **{decision['accepted_fraction']:.1%}**.",'',
          'The expansion rule requires >0.005 gain over single, no more than 0.02 loss versus the best fixed numerical pilot policy, and at least 90% accepted decisions. Passing is a screening signal, not a significance or clinical-benefit claim.','']
        lines+=['| Multi minus same-patient control | Gain | Exploratory paired 95% interval |','|---|---:|---:|']
        for name,v in decision['comparisons_multi_minus'].items():lines.append(f"| {name} | {v['mean_gain']:+.6f} | {v['paired_95CI']} |")
    diagnostic_path=root/'schema_diagnostic.json';diagnostic=read_json(diagnostic_path) if diagnostic_path.exists() else None
    if diagnostic:
        lines+=['','## Interface confound found after the pilot','',
          f"The executed study contained {diagnostic['fallbacks']} fallbacks. Of these, **{diagnostic['kinds'].get('valid_nested_reference_rejected',0)}** used real nested evidence references (for example `geometry.candidates`) that the top-level-only validator rejected. This was an over-strict interface design. These scores therefore do not isolate collaboration ability.",'',
          'All historical outcomes remain the actually executed fallback-policy outcomes. We do not retroactively invent accuracy for the unexecuted choices: accepting a different proposal could change later observations and later model answers. The follow-up uses actual API-enforced JSON Schema, permits only available query IDs and real evidence-root IDs, and adds a full smoke trajectory before another pilot. This interface problem does not explain all older negative experiments.','']
    lines+=['','## Evidence boundaries','',
      'The [MDAgents architecture](https://doi.org/10.52202/079017-2522) motivates relevant specialists and structured review; its QA/diagnosis findings are not evidence for this epilepsy task. Full Methods sections 3.1-3.5 in the official proceedings PDF were inspected, alongside the canonical Chrome record. The literature search was focused and non-exhaustive.','',
      'Public observations contain only known geometry/availability, already measured responses, and model scores recomputed from those observed columns. Hidden labels, future responses and full-response test predictions are excluded. Stimulus IDs are array columns; their mapping to clinical electrode-pair names has not been independently reconstructed. Known released preprocessing availability may overstate prospective knowledge.','',
      'All waveform computation runs locally on the RTX 4090. Gemini inference uses the official Google API, medium reasoning, no retries. Every malformed or uncaptured attempt stays spent. Full reasons are preserved; the 6,000-character cap applies equally across arms. Short local filesystem retries do not resend API calls.','',
      'SOZ concordance is a diagnostic ranking endpoint, not therapeutic benefit. No physical stimulation is delivered. Any pilot improvement needs a larger matched comparison, repeat-generation sensitivity and a fresh cohort before a strong generalization claim.','']
    (project/'docs/spes-agent-v6-report.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v6_summary.json',{'status':'completed_interface_confounded_not_expanded' if diagnostic else ('pilot_complete' if decision else 'pilot_in_progress'),'pilot_decision':decision,'schema_diagnostic':diagnostic,
        'completed':[{k:v for k,v in r.items() if k not in ['patient','prediction','SOZ']} for r in rows],
        'multi_agent_superiority_established':False,'clinical_benefit_established':False})
    print('English v6 report and portable summary saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();build(a.project)
