"""English report of all frozen calibration/acquisition outcomes."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output)
    calibration=read_json(output/'calibration_results.json'); acquisition=read_json(output/'acquisition_results.json'); audit=read_json(output/'audit.json')
    if audit['status']!='verified': raise ValueError('Verified audit required')
    lines=['# Calibration improved modestly; physical query selection did not','',
      'The fixed learned-boundary acquisition policy scored **0.744035** mean patient SOZ AUROC, versus **0.761978** for the frozen score-focus numerical control. Gain **-0.017943**, paired 95% interval **[-0.043149, +0.005348]**. The acquisition gate fails. No Gemini comparison is launched for this hypothesis, and no multi-agent advantage is claimed.','',
      '## Calibration is a separate outcome','',
      '| Eight-group score | Mean patient Brier (lower is better) | Gain over development-prevalence constant | Paired 95% interval | AUROC |',
      '|---|---:|---:|---|---:|']
    for name in ['raw','analytic_prior','constant','raw_platt','prior_platt','centered_platt']:
        v=calibration['summary']['8/'+name]
        lines.append(f"| {name} | {v['mean_patient_Brier']:.6f} | {v['Brier_gain_over_constant']:+.6f} | {v['paired_95CI']} | {v['mean_patient_AUROC']:.6f} |")
    lines+=['',
      'Patient-centered sigmoid calibration beats the constant by 0.003551, with a positive paired interval, but misses the fixed practical-gain requirement of 0.005. Its eight-group AUROC is unchanged from HAT. This is a modest score-quality finding on this cohort, not better localization or clinical probabilities. Four-group outcomes remain secondary in the machine-readable summary. Some source-fitted sigmoid slopes were nonpositive and followed the predeclared constant fallback; this explains AUROC reductions for those variants rather than a silently reversed ranking.','',
      'The analytic prior hypothesis reduces the large raw-score offset, but does not beat the development-prevalence constant. Correcting average prevalence is insufficient to establish individual calibration. The ideal odds derivation is not a proof that finite, sparsely queried neural outputs have that distribution.','',
      '## All physical acquisition variants','',
      '| Policy | Mean patient AUROC | Gain over score-focus | Paired 95% interval |','|---|---:|---:|---|']
    for name,v in acquisition['summary'].items():
        d=v['comparisons']['score_focus']; lines.append(f"| {name} | {v['mean_patient_AUROC']:.6f} | {d['gain']:+.6f} | {d['paired_95CI']} |")
    lines+=['',
      'All policies receive the same four random0 initial physical groups and four adaptive choices. The final scorer is the unchanged outer HAT applied only to revealed responses. The prior-corrected disagreement policy and matched zero-logit boundary control also fail to beat score-focus. No secondary variant replaces the primary.','',
      '## Training, evidence and checks','',
      'Seventy-five small one-dimensional calibrators were fitted on 28 development patients per outer fold, using twenty existing source-held-out HATs and audited physical states. Source patients were absent from neural fitting and early stopping. Calibration weights give each patient equal influence at natural class prevalence. Median Youden boundary offsets are learned only from those development patients.','',
      'The [official HAT preprint supplement](https://www.medrxiv.org/content/medrxiv/early/2026/07/31/2026.07.27.26358819/DC1/embed/media-1.pdf?download=true), S3/S6, supports the threshold and weighted-training motivation. The acquisition rule in logit-MAD units is a new project adaptation. The source is a preprint with an inspected official supplement, not a peer-reviewed clinical validation. Atomic evidence locations are recorded in `research/spes-v13/claims.json`.','',
      f"Independent audit refitted **{audit['calibrators_refit']}** calibrators, enumerated **{audit['independent_patient_boundary_checks']}** patient/state boundaries independently, recomputed **{audit['calibration_scorer_state_results']}** calibration outcomes, and reconstructed **{audit['adaptive_choices_reconstructed']}** adaptive choices across **{audit['physical_trajectories']}** physical trajectories. All physical identities, patient splits and frozen hashes passed. Stored neural scores were not all re-inferred in the audit.",'',
      'The study used the local workstation, zero new neural fits and zero API calls. Eight response groups do not equal eight pulses or equal current, charge or time. The repeatedly inspected 35-patient cohort is exploratory; SOZ ranking does not measure seizure-control efficacy. All failures are retained. The next distinct hypothesis is supervised physical query value, with genuinely separate specialist feature views, before any new paid collaboration comparison.','']
    (project/'docs/spes-agent-v13-report.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v13_summary.json',{'status':'acquisition_gate_failed','calibration':calibration,'acquisition':acquisition,'audit':audit,'multi_agent_superiority_established':False})
    print('English v13 report and complete results saved.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
