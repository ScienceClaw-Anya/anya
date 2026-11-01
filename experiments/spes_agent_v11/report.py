"""English report for the first physically aligned acquisition controls."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output);r=read_json(output/'numeric_results.json');a=read_json(output/'audit.json')
    if a['status']!='verified':raise ValueError('Verified physical audit required')
    lines=['# Verified physical-pair acquisition: numerical controls','',
      'This is the first completed acquisition comparison using one real anatomical stimulation pair per query. Earlier v4-v10 column-ablation means are a different task and are not comparator results here. No agent or treatment benefit is established.','',
      '| Policy | Mean patient SOZ AUROC | Gain over random_0 | Paired 95% interval |','|---|---:|---:|---:|']
    for name,v in r['policies'].items():lines.append(f"| {name} | {v['mean_patient_AUROC']:.6f} | {v['gain_over_random_0']:+.6f} | {v['paired_95CI']} |")
    lines+=['','The fixed primary is score_focus minus random_0. Its observed gain is about 0.0215, but the interval includes zero. This does not demonstrate a stable improvement. The highest-scoring secondary policy is not substituted for the primary.','',
      f"All 35 patients have different final rankings under the fixed query rules. Mean within-patient AUROC range across five policies is {r['mean_withinpatient_AUROC_range']:.6f}. This passes a predefined query-sensitivity check: actions matter. It is not evidence that an LLM or collaboration will select better actions.",'',
      '## Same task and budgets','',
      'Five policies share the same four initial random_0 measurements and choose four further anatomical pairs. Each trajectory uses exactly eight distinct verified pairs, with the same original patient-held-out HAT as final scorer. Baseline and MTE are also recomputed from observed measurements to provide descriptive model disagreement. Every fit, normalizer and early-stop set excludes the tested patient. Numerical policies use no language-model requests.','',
      'Responses are gathered from different compact columns at different recording contacts when required by the physical-pair map. Geometry-ambiguous cells remain absent. Only the first twenty trials are used. Every candidate has one recorded parameter tuple. Provider anatomy is fsaverage with millimetre coordinates. Released preprocessing availability is assumed known and may exceed prospective knowledge.','',
      f"The independent audit checked **{a['physical_trajectories']} trajectories / {a['actual_physical_queries']} physical-pair query slots**, model/source/map lineage, patient exclusion, common prefixes, exact fixed-policy decisions and final AUROCs by pairwise comparison. **{a['queries_requiring_different_compact_columns_across_contacts']}** query slots required different compact columns across recording contacts. This reinforces why a shared-column action was invalid.",'',
      '## Next decision','',
      'Rebuild source-held-out development states with these physical queries before training contact-posterior or value models. The old v8/v9 utility models and v10 column-state classifiers are not adopted. The 20 existing inner HATs remain reusable because their original full-response contact-classification task did not require global action alignment. Fixed feature/learner recipes may be reused, but the training states must be replaced.','',
      'A subsequent single/multi-agent comparison must share tools, observations, measurements, request budget and feedback timing. Real anatomical identities can now support neuroanatomy and waveform/uncertainty roles. Independent proposals and an explicit communication ablation should test collaboration without a unilateral reviewer defaulting to one proxy. Direct numerical selection with those same tools remains a control.','',
      'All patient identities have been repeatedly inspected. These 35 patients / 2,117 contacts / 296 SOZ contacts are one exploratory cohort, not 175 independent patients. Thirty-four patients overlap the older reconstruction; patient 39 replaces patient 47. Paired bootstrap intervals condition on the recorded run and do not adjust for all research iterations. Independent confirmation is required. SOZ ranking does not measure treatment efficacy.','',
      'No physical stimulation is delivered, physiological state carry-over is not modeled, and clinical safety is not established. Computation ran locally on the RTX 4090 and CPU. No new cloud resource or model API request was used.','']
    (project/'docs/spes-agent-v11-report.md').write_text('\n'.join(lines),encoding='utf8');write_json(project/'examples/spes_agent_v11_summary.json',{'status':'physically_valid_numeric_controls_complete','results':r,'audit':a,'multi_agent_superiority_established':False,'clinical_benefit_established':False});print('English v11 report saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
