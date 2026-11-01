"""English hypothetical-tool feasibility report; report all fixed outcomes."""
from analysis.action_mapping_correction import ACTION_NOTICE
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output);r=read_json(output/'numeric_results.json');a=read_json(output/'audit.json');d=read_json(output/'hypothesis_diagnostic.json')
    if a['status']!='verified':raise ValueError('Independent audit required')
    primary=r['policies']['bag_mean'];passed=r['further_LLM_eligible']
    lines=[ACTION_NOTICE,'','# Hypothetical-response specialists: numerical acquisition test','',
      f"The fixed primary scored **{primary['mean_patient_AUROC']:.6f} mean patient SOZ AUROC**, a gain of **{primary['gain_over_random_0']:+.6f}** over random_0, paired 95% interval {primary['paired_95CI']}. The future-LLM spending gate {'passed' if passed else 'failed'}. This is a numerical-tool feasibility study; multi-agent superiority and clinical benefit remain unestablished.",'',
      '## Complete matched results','', '| Policy | Mean patient AUROC | Gain over random_0 | Paired 95% interval |','|---|---:|---:|---:|']
    for name,value in r['policies'].items():lines.append(f"| {name} | {value['mean_patient_AUROC']:.6f} | {value['gain_over_random_0']:+.6f} | {value['paired_95CI']} |")
    lines+=['','Each policy receives the same four initial real measurements, then four adaptive measurements. The final scorer is unchanged original v4 HAT. Candidates are not prefiltered. All five fixed numerical controls reproduce v8; they must be compared with the same-prefix policies here rather than earlier all-eight-query rules.','',
      f"The fixed primary is bag_mean. Its one-sided 95% patient-bootstrap lower bound versus random_0 is {r['primary_one_sided_95_lower_vs_random0']:+.6f}. Before outer evaluation, the spending rule required its mean to exceed the strongest numerical control by more than 0.005, the lower bound to be positive, and at least ten changed final query sets. All secondary policies remain visible; none replaces the primary after evaluation.",'',
      '## Substantive tool change','',
      'The v8 candidate-ranking diagnostic showed weak transfer to unseen development patients. This version adds two different hypotheses about an unmeasured response: interpolation from observed stimulation sites with similar geometry profiles, and interpolation with clipped radial amplitude correction. Both use known distances/availability and observed waveforms only. They never read candidate waveforms or clinical labels. Trial-index alignment across sites is computational; it does not imply simultaneous physiological states.','',
      'Source-held-out HATs evaluate the hypothetical additions. Twenty observed-only features are supplemented by predicted score/rank changes, response amplitude summaries and disagreement between the two hypotheses, for 46 features total. The imagined waveform is always a prediction and does not count as acquired evidence. The final judge uses only the eight actual released measurements.','',
      'The source models and supervised candidate targets are exactly the twenty inner HATs and phase-zero banks from v8. Each target patient was excluded from source fitting and early stopping; outer patients never enter the new banks or regressors. All settings were fixed before new outer-policy outcomes. Five outer learners contain 55 small regressors in total: ridge, five absolute-gain bags and five relative-gain bags per fold. No new neural fit, language-model call or hyperparameter search was used.','',
      '## Retrospective prediction diagnostic','', '| Response hypothesis | Within-state gain rank correlation | Score-change cosine | Label-assisted pick gain over average |','|---|---:|---:|---:|']
    for name,v in d['summary'].items():lines.append(f"| {name} | {v['within_state_AUROC_change_spearman']} | {v['score_change_vector_cosine']} | {v['hindsight_label_assisted_pick_minus_average']} |")
    lines+=['','This diagnostic uses clinical labels retrospectively to assess hypothetical predictions on development replay. Label-assisted picks are not executable policies and are never exposed to agents. The 140 patient-fold cases and 560 states per hypothesis contain correlated repetitions, not new independent patients.','',
      '## Audit, compute and interpretation','',
      f"The independent audit reconstructed all {a['feature_rows_reconstructed']:,} hypothetical feature rows from observed waveforms and recorded hypothetical scores, checked source/model/target lineage and patient isolation, recomputed all {a['numeric_trajectories']} final AUROCs by pairwise comparison, and checked actual measurements, recorded argmax choices and hypothetical compute counts. Recorded hypothetical HAT outputs were not all re-inferred by the audit; source generation separately reproduced observed scores at every development state.",'',
      f"Development feature generation used {a['source_hypothetical_forward_calls']:,} candidate-hypothesis scoring passes; outer evaluation used {r['hypothetical_forward_calls']:,}. These counts represent full-contact hypothetical scoring passes, each internally minibatched, not actual stimulations or LLM calls. Waveform/CPU/GPU work ran locally on the RTX 4090 and workstation memory; no new cloud resource was created.",'',
      ('The tool clears the spending screen. The next step is a separately frozen, matched-tool/budget comparison of single-agent, multi-agent and direct numerical selection. Passing this screen alone does not establish that language-model collaboration adds value.' if passed else 'The tool does not clear the spending screen, so this version launches no additional Gemini comparison. Increasing debate or model-call budgets on unsupported tools is not justified by these outcomes.'),'',
      'All outcomes remain exploratory after repeated inspection of this 35-patient / 2,117-contact / 296-SOZ processed cohort. Thirty-four patients overlap the older reconstruction; patient 39 replaces patient 47. Training isolation prevents direct leakage but does not correct adaptive test reuse. Patient-bootstrap intervals condition on recorded runs and do not adjust for the complete research search. A fresh cohort is needed for stronger confirmation.','',
      'This is retrospective diagnostic acquisition using anonymous stimulation columns and assumed known preprocessing availability. It does not reconstruct clinical electrode-pair labels, deliver stimulation, model physiological state carry-over or establish seizure-control benefit.','']
    (project/'docs/spes-agent-v9-report.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v9_summary.json',{'status':'numeric_gate_passed' if passed else 'numeric_gate_failed_no_LLM_launched','results':r,'audit':a,'hypothesis_diagnostic':d,'multi_agent_superiority_established':False,'clinical_benefit_established':False})
    print('English v9 report and complete compact outcomes saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
