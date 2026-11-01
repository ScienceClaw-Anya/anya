"""Publish aggregate-only results of the failed relative-ranking feasibility test."""
import argparse
import json
from pathlib import Path


def build(project,output):
    project,output=Path(project),Path(output)
    r=json.loads((output/'results.json').read_text()); a=json.loads((output/'audit.json').read_text())
    if a['status']!='verified': raise ValueError('Independent audit required')
    rows=['# Relative query-ranking feasibility test','','The specialist panel did not improve physical diagnostic-query selection. Its mean patient SOZ AUROC is **0.742124**, versus frozen score-focus **0.761978**. The fixed next-tool gate fails. No Gemini call was added and no new weights are adopted.','',
          '| Controller | Mean patient AUROC | Gain versus score-focus | Nominal paired 95% interval |','|---|---:|---:|---|']
    for name,v in r['summary'].items():
        d=v['comparisons']['score_focus']; rows.append(f"| {name} | {v['mean_patient_AUROC']:.6f} | {d['gain']:+.6f} | {d['paired_95CI']} |")
    rows+=['','## What this tests','',
           'For each outer fold, 28 development patients provide 112 observed states and 672 candidate rows from audited physical counterfactual banks. Six candidates yield 15 unordered comparisons per state. The target is the difference in after-query AUROC, so the common before-query AUROC cancels. Pair weights preserve the original total source weight; algebraic pairs are not new participants. No v16 test-patient oracle choices enter fitting.','',
           'Twenty fixed-alpha pairwise Ridge models and five pointwise controls are fitted. Geometry, neural-state and waveform specialists combine percentile ranks with a fixed equal average. The all-feature pairwise and pointwise models see the same features and training patients. All controllers compute the same observed-only descriptors, receive four common initial groups plus four adaptive groups, and retain the same original outer HAT scorer.','',
           'All 35 patients complete all three policies, for 105 physical trajectories. The pairwise all-feature model scores 0.757274 versus pointwise 0.752502, gain +0.004772 with interval [-0.019706, +0.032186]. The expert panel also trails the all-feature pairwise model by -0.015150, interval [-0.052355, +0.020759]. Neither changing to relative ranking nor splitting these descriptors into experts establishes an improvement.','',
           '## Interpretation','',
           'The training target removes one nuisance term but still has noisy patient-specific responses, random-prefix versus adaptive-prefix shift, and only 28 development patients per outer fold. These are plausible limitations, not experimentally isolated causes. Experts disagree, yet their unweighted combination can discard useful interactions present in the all-feature model. This result weakens the case for promoting this particular expert tool to an expensive LLM experiment. It does not show that every multi-agent approach must fail.','',
           '## Verification','',
           f"Independent auditing verifies {a['pair_models_verified_by_centered_loss']} pairwise models through an equivalent centered-loss calculation, refits all five pointwise controls, replays {a['physical_trajectories']} trajectories and {a['adaptive_choices_replayed']} adaptive choices, and reconstructs {a['candidate_feature_rows_reconstructed']:,} observed-only candidate feature rows. The original uncached feature recipe agrees for one common state per patient. Prior v16 API attempts remain 120 and new API calls remain zero.",'',
           'Stored runtime neural scores were not all re-inferred by this audit. The final-scorer weights and physical mappings stay pinned. Parameter counts differ between expert views and the all-feature control. Repeated investigation of this single-centre cohort limits generalization, and nominal intervals do not correct the entire research search. Groups do not imply equal pulses, current, charge or time. SOZ ranking is not evidence of treatment efficacy.','',
           'The legacy 0.7201 reconstruction used 35 patients and 2,145 contacts with different preprocessing and all measured responses. This physical-query test uses 2,117 contacts and eight selected groups. Its scores cannot establish an improvement over the legacy reconstruction. A separate same-input complete-cohort benchmark now addresses that comparison.','']
    (project/'docs/spes-agent-v17-report.md').write_text('\n'.join(rows),encoding='utf8')
    public={**r,'audit':a}; (project/'examples/spes_agent_v17_summary.json').write_text(json.dumps(public,indent=2)+'\n',encoding='utf8')
    print('Saved English report and aggregate-only summary.')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); build(a.project,a.output)
