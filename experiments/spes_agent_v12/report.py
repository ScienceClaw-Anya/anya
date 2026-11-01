"""English contact-specialist results and calibration diagnostic."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output);r=read_json(output/'results.json');a=read_json(output/'audit.json')
    if a['status']!='verified':raise ValueError('Audit required')
    probabilities=[];prevalence=[]
    for path in (output/'evaluation').glob('*/predictions.npz'):
        with np.load(path) as d:probabilities.append(float(d['8/hat'].mean()));prevalence.append(float(d['label'].mean()))
    diagnostic={'mean_patient_raw_HAT_score':float(np.mean(probabilities)),'mean_patient_SOZ_prevalence':float(np.mean(prevalence)),
      'observed_gap':'Raw weighted-classifier scores are not directly calibrated SOZ probabilities.',
      'verified_training_code':'Inverse-class and inverse-patient weighted sampling plus BCE positive weight n_negative/n_positive.',
      'causal_caveat':'Weighting motivates a prior-correction test; finite training and partial-input distribution shift may also contribute.'}
    lines=['# Physical-state contact specialists: ranking gain not established','',
      'The fixed logistic specialist scored **0.733662 mean patient SOZ AUROC** at eight verified physical measurements, versus HAT **0.740463**. Gain **-0.006801**, paired 95% interval **[-0.063261, 0.048103]**. The predefined next-tool gate fails. These classifiers are not adopted as an improved localization scorer, and no new Gemini comparison is launched for this version.','',
      '| Measurements | Scorer | Mean patient AUROC | Gain over HAT | Paired 95% interval | Mean patient Brier |','|---:|---|---:|---:|---:|---:|']
    for key,v in r['results'].items():
        k,name=key.split('/');lines.append(f"| {k} | {name} | {v['mean_patient_AUROC']:.6f} | {v['gain_over_hat']:+.6f} | {v['paired_95CI']} | {v['mean_patient_Brier']} |")
    lines+=['','Four-measurement logistic ranking is numerically higher than HAT, but its interval also includes zero. It remains secondary and does not replace the fixed primary. The rank-mean score is not interpreted as a probability and receives no Brier score.','',
      '## Meaningful training and independent checks','',
      'All input states use the verified physical-pair map. Ten small classifiers were fitted using 28 development patients per outer fold, equally weighted. Twenty existing inner HATs generate source-held-out scores, with the target patient absent from neural fitting and early stopping. Outer patients never enter fitted contact models. Twenty-six observed-only features describe morphology, sample-index energy, FFT-bin energy, geometry and HAT scores; unknown 128-feature semantics are not assumed. No new neural fit or LLM request was used.','',
      f"The audit reconstructed **{a['training_feature_rows_reconstructed']:,} contact/state feature rows**, checked label and physical-action identity, patient/source/model provenance, replayed both fitted classifiers on every test state, and independently recomputed all **{a['scorer_state_results']}** AUROC/Brier results. Original HAT predictions exactly match the v11 same-query control. Recorded HAT outputs were not all re-inferred by this audit.",'',
      'The 42,340 rows are correlated repetitions across states/folds, not new patients. This remains a repeatedly inspected 35-patient cohort. The outer-label exclusion prevents direct leakage but does not remove adaptive research bias. No general medical or collaboration claim follows from these results.','',
      '## A separate problem: absolute score interpretation','',
      f"Mean raw HAT score across held-out patients is **{diagnostic['mean_patient_raw_HAT_score']:.6f}**, while mean patient SOZ prevalence is **{diagnostic['mean_patient_SOZ_prevalence']:.6f}**. Its eight-query Brier score is about **0.6003**, versus about **0.1134** for logistic. This gap concerns absolute output quality; it does not make the logistic ranking better. A development-prevalence constant is still needed as a proper calibration comparator.",'',
      'The local training code applies inverse-class/inverse-patient sampling and a positive-class weighted BCE loss. The author [supporting methods, S6, PDF pp. 6-7](https://www.medrxiv.org/content/medrxiv/early/2026/07/31/2026.07.27.26358819/DC1/embed/media-1.pdf?download=true) describe the same combination. This is an intentional ranking/training strategy, not evidence that raw sigmoid scores equal unweighted clinical probabilities. The preprint supplement was read directly; the associated article is a preprint version, not a peer-reviewed clinical validation.','',
      'This motivates a separately frozen calibration study before agents use absolute confidence. It should compare training-only analytic prior adjustment, patient-separated sigmoid calibration, a development-prevalence constant and an adaptive decision boundary; then test whether corrected uncertainty changes physical query selection. Weighting is a mechanistic hypothesis for part of the gap, not the demonstrated sole cause. Sparse-input shift and incomplete fitting may also matter.','',
      'The supplementary protocol uses repeated pulses and variable recorded currents. The benchmark budget is eight anatomical-pair response groups, each containing up to twenty retained trials; pulse count, charge, timing and safety are not equated across query sets. Each action uses its recorded condition. Anatomical-position effects cannot be separated causally from dose with this replay.','',
      'The unused 128-feature block remains unresolved after checking the primary Methods, supplement and available model/dataset code. It is not called a prestimulus baseline or spectral block and is excluded from the specialist features.','',
      'All computation ran locally on the workstation. Reports and code are in English, raw data/labels/weights stay outside Git, and no new cloud resource was created. The endpoint is clinical SOZ ranking, not seizure-control efficacy.','']
    (project/'docs/spes-agent-v12-report.md').write_text('\n'.join(lines),encoding='utf8');write_json(project/'examples/spes_agent_v12_summary.json',{'status':'ranking_gate_failed_not_adopted','results':r,'audit':a,'posthoc_calibration_diagnostic':diagnostic,'multi_agent_superiority_established':False,'clinical_benefit_established':False});print('English v12 negative ranking result and calibration diagnostic saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
