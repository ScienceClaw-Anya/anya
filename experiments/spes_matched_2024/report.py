"""English, aggregate-only report for the exact legacy-cohort comparison."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from .experiment import read,write,sha


def build(project,output):
    project,output=Path(project),Path(output); r=read(output/'results.json'); a=read(output/'audit.json'); p=read(output/'protocol.json')
    if a['status']!='verified': raise ValueError('Audit must finish before reporting')
    gain=r['paired_gain']; low,high=r['paired_nominal_95CI']
    if low>0: verdict='The mean-input HAT adaptation improves AUROC in this exploratory matched comparison.'
    elif high<0: verdict='The mean-input HAT adaptation performs worse in this exploratory matched comparison.'
    else: verdict='The matched comparison does not establish a reliable improvement by the mean-input HAT adaptation.'
    receipts=[read(output/f"fits/seed{j['seed']}_fold{j['fold']}/receipt.json") for j in p['jobs']]
    lines=['# Same-input complete-cohort benchmark','',verdict,'',
           f"The original Transformer scores **{r['baseline_mean_patient_AUROC']:.6f}** and the new mean-input HAT scores **{r['candidate_mean_patient_AUROC']:.6f}**. Paired mean gain is **{gain:+.6f}**, nominal patient-bootstrap 95% interval **[{low:+.6f}, {high:+.6f}]**. This numerical architecture result does not establish multi-agent value or treatment benefit.",'',
           '| Measure | Original Transformer | Mean-input HAT |','|---|---:|---:|',
           f"| Mean patient AUROC across five repeats | {r['baseline_mean_patient_AUROC']:.6f} | {r['candidate_mean_patient_AUROC']:.6f} |",'| Patients | 35 | 35 |','| Contacts / SOZ | 2,145 / 304 | 2,145 / 304 |','| Patient-disjoint fits | 25 reused and verified | 25 newly trained |','',
           f"After averaging five repeats per patient, {r['patients_improved']} patients improve and {r['patients_worsened']} worsen. The bootstrap resamples 35 patient means, not 175 repeated cases or 2,145 contacts. The baseline recomputed from pinned predictions reproduces the earlier 0.7201 result.",'',
           '## What is held fixed','',
           'Every original seed/fold split is reused, with 21 fitting, seven validation and seven test patients. Both classifiers consume the old convergent mean representation from the identical raw processed arrays and training-patient normalization. Batch size, class-positive loss weight, shuffle sampler, AdamW learning rate and default weight decay, dropout, old noise/channel-drop augmentation, 1,000-epoch ceiling and patience-ten stopping are unchanged. The baseline comprises all 25 previously audited fits; no favorable folds are chosen or replaced. All candidate fits complete before any new test prediction is generated.','',
           'The architecture changes from the original convolutional response encoder plus channel Transformer to the HAT embedding and interleaved attention structure, with published depth and head counts. Parameter counts and compute differ. The learning rate, dropout and augmentation are held to the original baseline rather than tuned on these test patients. A negative result would not prove that all HAT hyperparameter configurations fail.','',
           '## What this adaptation can test','',
           'Legacy arrays store mean and SD summaries, not individual trials. Both evaluated classifiers use only the convergent mean. The new architecture therefore has one mean token per channel. SD is never relabelled as a second trial. This experiment tests transfer of the architecture under identical available information; it cannot replicate the full cross-trial HAT method or demonstrate active physical-query selection.','',
           'Independent reconstruction confirms an aliasing quirk in the unchanged historical normalizer: shifting an overlapping response view copies the distance into feature one after feature zero has been overwritten. Both arms share this historical 509-feature representation. The comparison preserves it for compatibility. A preprocessing repair would require another frozen protocol with both arms retrained.','',
           '## Why previous numbers differ','',
           'The old reconstruction uses 35 patients, 2,145 contacts and 304 SOZ. The newer processed dataset uses 35 patients, 2,117 contacts and 296 SOZ, with only 34 overlapping patients. The newer 0.7620 score-focus result also selects eight physical response groups, while the old reconstruction uses full responses. The five-patient 0.8119 score-focus result comes from a small selected subset and is not a full-cohort result. These cross-protocol numbers do not establish an improvement over 0.7201.','',
           '## Repeat sensitivity','',
           '| Original split seed | Transformer AUROC | HAT AUROC | Paired gain |','|---|---:|---:|---:|']
    repeats=[]
    for seed in range(1,6):
        rows=[c for c in r['cases'] if c['seed']==seed]
        b=float(np.mean([c['baseline_AUROC'] for c in rows])); h=float(np.mean([c['candidate_AUROC'] for c in rows]))
        repeats.append({'seed':seed,'baseline_AUROC':b,'candidate_AUROC':h,'gain':h-b}); lines.append(f'| {seed} | {b:.6f} | {h:.6f} | {h-b:+.6f} |')
    lines+=['','## Verification and resources','',
            f"Independent audit reconstructs **{a['normalized_array_comparisons']}** normalized mean/SD input arrays, re-infers **{a['checkpoint_prediction_reinferences']}** saved validation/test prediction sets, verifies all 25 source splits and checkpoint receipts, and recomputes 175 patient-repeat AUROCs and the 35-patient bootstrap. Maximum saved-versus-reinferred prediction difference is {a['maximum_prediction_difference']:.3g}. The original 75-run reproduction audit remains intact.",'',
            f"Training used the local RTX 4090 with two workers, **zero paid API calls** and no cloud resources. Summed worker training time was {sum(v['seconds'] for v in receipts)/60:.1f} minutes; this is not elapsed wall time. Candidate parameter count is {receipts[0]['parameters']:,}. Best validation checkpoint epochs range from {min(v['best_epoch'] for v in receipts)} to {max(v['best_epoch'] for v in receipts)}. No model is selected by test-patient outcomes.",'',
            f"Frozen protocol SHA256: `{sha(output/'protocol.json')}`.",'',
            'This is an exploratory comparison on an already inspected, single-centre public cohort. Confidence intervals are nominal and do not correct the whole research search. A supported collaboration claim still requires a stronger patient-isolated numerical/tool policy, a matched-budget single-agent control, and independent evaluation. No new Gemini requests were made here. Patient-level data, labels, requests and weights stay outside Git.','']
    report='\n'.join(lines); (project/'docs/spes-matched-2024-report.md').write_text(report,encoding='utf8')
    public={k:v for k,v in r.items() if k not in ['cases','patient_means']}; public.update({'verdict':verdict,'repeat_summary':repeats,'audit':a})
    write(project/'examples/spes_matched_2024_summary.json',public)
    paragraphs='\n'.join('<p>'+html.escape(line)+'</p>' for line in lines if line)
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>Complete-cohort comparison</title><style>body{max-width:1000px;margin:48px auto;padding:0 28px;font:18px/1.6 Arial;color:#20364b}p{white-space:pre-wrap}</style><body>'+paragraphs+'</body></html>'
    (project/'examples/spes_matched_2024_report.html').write_text(page,encoding='utf8')
    print('Saved English matched-cohort report and aggregate summary.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); build(a.project,a.output)
