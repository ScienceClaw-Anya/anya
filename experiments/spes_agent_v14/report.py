"""English, audit-gated physical query-value specialist report."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output); result=read_json(output/'results.json'); audit=read_json(output/'audit_full.json')
    if audit['status']!='verified' or audit['training_rows_reconstructed']!=3360 or audit['physical_trajectories']!=105:
        raise ValueError('Complete independent audit required')
    primary=result['summary']['expert_consensus']; d=primary['comparisons']['score_focus']; mono=primary['comparisons']['monolithic']
    status='numerical_feasibility_gate_passed' if result['next_agent_gate'] else 'numerical_feasibility_gate_failed'
    lines=['# Physical query-value specialists','',
      f"The fixed conservative expert policy scored **{primary['mean_patient_AUROC']:.6f}** mean patient SOZ AUROC. Against the frozen v11 score-focus controller, gain was **{d['gain']:+.6f}**, paired 95% interval **{d['paired_95CI']}**. Against the all-feature boosted controller, gain was **{mono['gain']:+.6f}**, interval **{mono['paired_95CI']}**.",'',
      f"The fixed next-agent feasibility gate **{'passes' if result['next_agent_gate'] else 'fails'}**. These are trained numerical specialists, not a completed Gemini collaboration comparison. No multi-agent improvement or clinical benefit is established.",'',
      '| Controller | Mean patient AUROC | Gain over score-focus | Paired 95% interval |','|---|---:|---:|---|']
    for name,v in result['summary'].items():
        delta=v['comparisons']['score_focus']; lines.append(f"| {name} | {v['mean_patient_AUROC']:.6f} | {delta['gain']:+.6f} | {delta['paired_95CI']} |")
    lines+=['','## What was trained','',
      'Three fixed feature views describe geometry/coverage, neural-state scores and already observed waveform morphology. Each specialist predicts the one-query SOZ-AUROC change using a Huber boosted regressor. The fixed consensus uses their mean minus half their spread. Spread is a disagreement heuristic, not a calibrated confidence interval. An all-feature boosted regressor and an all-feature ridge model receive the same source targets and patients.','',
      'The 25 fitted regressors use 3,360 correlated candidate/state targets, derived from twenty existing source-held-out HATs and 28 development patients per outer fold. Each candidate slate is determined before its future response and clinical label are used to compute the supervised target. The slate includes the next seeded random query, coverage and raw-rank focus leaders, plus seeded alternatives. All current features use selected responses only. Source target patients are absent from source neural fitting and early stopping; outer patients never enter model fitting.','',
      'Only identical, audited physical random0-prefix scores are reused from v12. Responses for alternative physical pairs are newly inferred with the correct per-recording-contact index. No old array-column utility bank or trained value learner is adopted. Relative scores use an unsupervised patient logit median/MAD transform, not learned v13 parameters or clinical probabilities.','',
      '## What the comparison can resolve','',
      f"Specialists proposed different top choices in **{result['diverse_expert_steps']}/{result['consensus_steps']}** consensus-policy steps. Diversity alone cannot demonstrate useful complementarity. The primary must beat score-focus and the same-information monolithic model by more than 0.01, each with a positive paired interval, before an agent trial for this tool version.",'',
      'Every controller receives the same four initial physical response groups, then four adaptive groups. The unchanged outer HAT supplies all final rankings. Candidate choices, current scores, observed-only feature tables and every specialist output are recorded at each round. All three controllers and all five frozen v11 comparisons remain in the machine-readable summary; a secondary winner cannot replace the primary.','',
      '## Verification and limitations','',
      f"Audit reconstructed all **{audit['training_rows_reconstructed']}** source candidate feature/target rows and **{audit['source_states']}** source states, checked frozen source/model/map/raw hashes and patient exclusion, then replayed **{audit['adaptive_choices_replayed']}** adaptive decisions across **{audit['physical_trajectories']}** trajectories. Every final AUROC was independently recomputed with scikit-learn. Source next-query scores match audited physical prefix scores. Alternative-response and runtime neural outputs were checked from stored predictions, not all re-inferred in the audit.",'',
      'Random-prefix training may differ from learned-policy runtime states. Source HATs fit 17 patients and outer HATs fit 21, creating possible score/input shift. Model fitting uses patient-balanced rows, but these are not independent new participants. Repeated same-cohort research creates adaptive-selection bias; nominal bootstrap intervals are not adjusted for the full research search. A favorable result would still require a separate matched-budget Gemini comparison and external confirmation.','',
      'The local workstation performed this experiment with zero new neural fits and zero API calls. The anatomical-pair response-group budget does not equate current, pulse count, charge, time or safety. SOZ contact ranking measures retrospective diagnostic discrimination, not seizure-control efficacy.','']
    (project/'docs/spes-agent-v14-report.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v14_summary.json',{'status':status,'results':result,'audit':audit,'multi_agent_superiority_established':False})
    print('English v14 report and complete comparison saved.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
