"""Document every fixed sparse-training result and its audit in English."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,write_json,sha
from experiments.spes_agent_v2.verify import auc
from experiments.spes_agent_v4.neural import load_array

def build(project):
    project=Path(project);root=project/'outputs/spes_agent_v5';protocol=read_json(root/'sparse_protocol.json');result=read_json(root/'fixed_sequence_results.json')
    if any(sha(project/p)!=h for p,h in protocol['sources'].items()):raise ValueError('Frozen training source changed')
    fits=[]
    for job in protocol['jobs']:
        groups=[set(job[k]) for k in ['train','early_stop','predict']]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):raise ValueError('Patient overlap')
        for family in ['baseline','hat']:
            folder=root/f'neural/outer{job["outer"]}/{family}';r=read_json(folder/'receipt.json')
            if sha(folder/'checkpoint.pth')!=r['checkpoint_sha256'] or r['protocol_sha256']!=sha(root/'sparse_protocol.json'):raise ValueError('Artifact changed')
            if r['train']!=job['train'] or r['early_stop']!=job['early_stop'] or r['outer_test']!=job['predict']:raise ValueError('Fit provenance')
            if r['best_epoch']!=max(r['trace'],key=lambda x:x['validation_two_subset_mean_AUROC'])['epoch']:raise ValueError('Best epoch differs')
            fits.append({k:r[k] for k in ['outer','family','best_epoch','best_validation_AUROC','seconds']})
    for row in result['rows']:
        data=np.load(root/'predictions'/row['case_id']/f'{row["policy"]}.npz');y=load_array(project/f'vendor/norris2026_hat/data/y_ccepAgeUMCU{row["patient"]:02d}.npz')
        if not np.array_equal(y,data['y']):raise ValueError('Label alignment')
        for method in ['old_hat','sparse_baseline','sparse_hat','sparse_rank_mean']:
            if abs(auc(y,data[method])-row[method])>1e-12:raise ValueError('Independent pairwise AUROC differs')
    audit={'verified':True,'new_neural_fits':len(fits),'heldout_patients':35,'fixed_query_sets':175,'new_API_calls':0,
      'checks':['Disjoint train/early-stop/test patients','Checkpoint/protocol hashes','First maximum validation epoch','Exact label alignment','Independent pairwise AUROC for all methods/sequences'],
      'not_claimed':'Bitwise training reproduction or external clinical validation'}
    write_json(root/'audit.json',audit)
    lines=['# Sparse-observation training: negative result','',
      'Ten new local fits tested whether matching training inputs to an eight-measurement acquisition task improves the numerical scorer. The fixed experiment did **not** improve performance and these weights are **not adopted**. No Gemini requests were made.','',
      '## Controlled comparison','',
      'The same 35 patients and five outer splits were used. Each model fitted 21 patients and used seven different patients for early stopping. Every outer test patient remained excluded from fitting, normalization and early stopping. Training sampled 1-8 whole stimulation columns per patient per batch. Early stopping used two fixed eight-column validation subsets. Architecture/optimizer presets, the 25-epoch ceiling and patience 5 were fixed.','',
      'The test replays exactly the five v4 numerical policies\' existing eight-stimulation sets. Only the scorer changes. No new acquisition controller is tested by this replay. The primary sequence is random_0; the other sequences are reported as declared secondary checks.','',
      '| Fixed measurement sequence | Original HAT | Sparse baseline | Sparse HAT | Sparse rank mean |','|---|---:|---:|---:|---:|']
    for policy,methods in result['methods'].items():lines.append('| '+policy+' | '+' | '.join(f"{methods[n]['mean_patient_AUROC']:.6f}" for n in ['old_hat','sparse_baseline','sparse_hat','sparse_rank_mean'])+' |')
    primary=result['methods']['random_0']['sparse_hat'];lines+=['',
      f"Primary sparse-HAT minus original-HAT gain: **{primary['gain_over_old_hat']:+.6f}**, paired patient-bootstrap 95% interval **{primary['paired_95CI']}**. Sparse HAT was worse in {primary['patients_worse']} patients and better in {primary['patients_better']}; ties account for the remainder.",'',
      '## What this resolves','',
      'Changing the mask alone did not produce a better scorer. Training on fewer observations also reduces the information available to learn from. Published full-response hyperparameters may not suit sparse training from scratch; extra trial masking and dropout may further weaken that signal. These are hypotheses, not causes established by this experiment. Warm-start adaptation or a learned acquisition-value model would require separate preregistered tests.','',
      'The negative result does not prove that missing-column mismatch is irrelevant, nor that all sparse models fail. It does rule out adopting this particular fixed recipe as an improvement. The next collaboration test retains the stronger original HAT and matches single/multi feedback timing.','',
      '## Audit and scope','',f"Verified {audit['new_neural_fits']} new fits and {audit['fixed_query_sets']} held-out query sets, with independent AUROC replay and source/checkpoint provenance. The local RTX 4090 was sufficient; no cloud job or API call was needed.",'',
      'This is adaptive exploration on a repeatedly inspected single-centre cohort. It is not untouched external validation and it establishes no clinical or multi-agent benefit. All negative results remain recorded.','']
    (project/'docs/spes-agent-v5-report.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v5_summary.json',{'status':'completed_negative_not_adopted','patients':35,'new_fits':10,'new_API_calls':0,'methods':result['methods'],'fits':fits,'audit':audit})
    print(audit);print('English v5 report and compact results saved.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();build(a.project)
