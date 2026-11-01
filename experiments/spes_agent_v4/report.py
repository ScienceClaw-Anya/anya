"""English progress/results report; preserves failure history and task boundaries."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,write_json
from .screen import bootstrap_delta

def build(project):
    project=Path(project);root=project/'outputs/spes_agent_v4';screen=read_json(root/'screen/screen_results.json')
    lines=['# SPES research v4: evidence before additional agent calls','',
      'This is an adaptive exploratory study. The previously stopped v3 study remains stopped; its unused API budget has not been resumed.', '',
      '## Why the method changes','',
      'Previous agents mostly recombined the same fixed observations, and v3\'s safety gate made every executable option identical. More debate could not improve those rankings. The new programme first tests cheap learned ranking controls, then adds complementary physiological representations, then tests whether agents can acquire useful new measurements.','',
      '## Fixed cheap screen on the old cohort','',
      'All methods use the same old first-repeat 35-patient outer partitions and v3\'s patient-isolated OOF base predictions. No new neural fit or LLM request was used. These are exploratory outcomes on a repeatedly inspected cohort.','',
      '| Method | Mean patient AUROC | Gain over Transformer | Paired patient-bootstrap 95% interval |','|---|---:|---:|---:|']
    for name,v in screen['methods'].items():
        lo,hi=v['patient_bootstrap_95CI'];lines.append(f"| {name} | {v['mean_patient_AUROC']:.6f} | {v['mean_gain']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','Removing the global veto did not help. The 10% pairwise-ranker blend had only a tiny, uncertain gain. These routes did not justify another large static-agent run.','',
      '## Complementary multi-trial models','',
      'A [2026 preprint](https://doi.org/10.64898/2026.07.27.26358819) and its [public author code](https://github.com/norrisjamie23/SOZ_localisation_SPES_HAT) motivate testing trial-averaged, mean-trial-embedding (MTE), and hierarchical-attention (HAT) representations. The preprint models cross-trial and cross-channel dependencies; it does not study LLM collaboration. Its reported SOZ improvement did not establish postsurgical benefit. Full HTML Methods/Results/Discussion were verified in Chrome. The literature search was focused and non-exhaustive.','',
      'The newly released processed arrays contain **35 patients, 2,117 contacts and 296 SOZ contacts**. The old project cohort contains 2,145 contacts and 304 SOZ contacts. These are not interchangeable denominators.','',
      'The new local runner fixes five patient-held-out folds and published architecture presets before evaluation. All fitting, normalization and early stopping exclude each outer test patient. The best validation weights are cloned. We do not load upstream checkpoint pickles. There is no new hyperparameter search. The runner uses the paper\'s 25-epoch maximum and patience 5, not the CLI defaults. Streamed float64 statistics, batch padding, weight snapshot handling and deterministic seed assignment differ from upstream. This is an adaptation with one repeat, not an exact reproduction of the full paper.','']
    numerical_path=root/'multitrial/numerical_results.json';numerical=None
    if numerical_path.exists():
        numerical=read_json(numerical_path)
        lines+=['| Same-cohort method | Mean patient AUROC | Gain over HAT | Paired 95% interval |','|---|---:|---:|---:|']
        for name,v in numerical['methods'].items():
            lo,hi=v['paired_95CI_over_hat'];lines.append(f"| {name} | {v['mean_patient_AUROC']:.6f} | {v['gain_over_hat']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
        d=numerical['HAT_vs_baseline'];lines+=['',f"HAT minus the retrained same-data baseline: {d['mean_gain']:+.6f}, paired 95% interval {d['paired_95CI']}. These numerical results alone establish no multi-agent advantage.",'']
    else:lines+=['Training/evaluation is still in progress; no neural superiority claim is made.','']
    lines+=['## New task: acquire eight real SPES measurements','',
      'The active replay reveals one real stimulation-column response at a time. The three trained models recompute contact scores using only revealed columns. Hidden waveforms, SOZ labels, outcomes and full-response predictions never enter agent prompts. The final scorer is always the same trained HAT; agents choose measurements and cannot write probabilities.','',
      'Known distances and the author-released preprocessing availability map are assumed available before acquisition. That map may include knowledge unavailable prospectively. No real stimulation is delivered; clinical safety, voltage choices and prospective acquisition cost are outside this benchmark.','',
      'Every policy receives eight measurement slots and eight native ScienceClaw public-tool queries. Each LLM policy has at most eight Gemini 4 Argon attempts, medium reasoning, no retries or replacement calls. Numerical controls use no LLM calls. The single agent acts sequentially and sees all earlier feedback. Four specialist roles (coverage, hypothesis testing, uncertainty, audit) act independently in batch one, then share real feedback and rationales in batch two. The no-exchange control shares the same type of measured feedback but hides other roles\' rationales. Duplicate proposals use ordered alternatives and a declared geometric fallback. This gives the single agent a strong sequential-feedback control.','',
      'The initial five patients are fixed in advance (one from each outer fold). A numerical feasibility gate must show that changing queries meaningfully changes final rankings before any paid agent pilot. A successful pilot is not a final efficacy result; later patients remain separately reported.','']
    active=root/'active';metrics=[]
    if active.exists():
        metrics=[read_json(p) for p in active.glob('trajectories/*/*/metrics.json')]
        if (active/'active_feasibility.json').exists():
            gate=read_json(active/'active_feasibility.json');lines+=[f"Numerical pilot feasibility: launch={gate['launch_agent_pilot']}, mean within-patient AUROC range={gate['mean_AUROC_range']:.6f}.",'']
        pilot=[m for m in metrics if m['pilot']]
        lines+=['### Matched five-patient pilot','', '| Active policy | Completed pilot patients | Mean AUROC at eight measurements | Accepted model decisions |','|---|---:|---:|---:|']
        for name in sorted({m['policy'] for m in pilot}):
            selected=[m for m in pilot if m['policy']==name];lines.append(f"| {name} | {len(selected)} | {np.mean([m['AUROC'] for m in selected]):.6f} | {sum(m['accepted_model_outputs'] for m in selected)} |")
        multi={m['case_id']:m for m in pilot if m['policy']=='multi'};single={m['case_id']:m for m in pilot if m['policy']=='single'}
        if len(multi)==len(single)==5:
            delta=[multi[c]['AUROC']-single[c]['AUROC'] for c in sorted(multi)]
            lines+=['',f"Multi minus single on the same five patients: **{np.mean(delta):+.6f}**, exploratory paired 95% interval {bootstrap_delta(delta)}. The multi-agent pilot is not expanded to the remaining 30 patients. This is a small negative feasibility result, not proof that all collaboration methods fail.",'',
               'The 120 model attempts produced 109 accepted decisions and 11 declared fallbacks. Rejected outputs include oversized rationales and unavailable alternatives; no spent request was replayed. A transient Windows lock interrupted one numerical artifact-index write; a verified completed temporary transaction was recovered and the unfinished numerical trajectory resumed without an API request.','']
        lines+=['### Complete numerical acquisition controls','', '| Policy | Completed patients | Mean AUROC at eight measurements |','|---|---:|---:|']
        for name in ['random_0','random_1','coverage','score_focus','uncertainty']:
            selected=[m for m in metrics if m['policy']==name]
            if selected:lines.append(f"| {name} | {len(selected)} | {np.mean([m['AUROC'] for m in selected]):.6f} |")
        lines+=['','The 35-patient numerical means must not be directly compared with the five-patient LLM means. The next follow-up tests sparse-observation training: masking individual trials during full-response training does not necessarily train a scorer for whole missing stimulation positions. A controlled, fixed-sequence scorer test precedes further LLM calls.','']
    lines+=['## Limits and next decision','',
      'All 35 patient identities have been examined in earlier work. Patient-isolated training prevents direct train/test leakage, but repeated research on this cohort still makes improvements exploratory. Preserve all trials; do not keep tuning to a reported test winner. A meaningful claim needs stable paired gains versus both single-agent and the strongest numerical control, repeat-generation sensitivity, and ideally a fresh external cohort. SOZ agreement is not treatment effectiveness.','',
      'All current neural training and waveform inference run on the local workstation\'s RTX 4090. Gemini inference uses Google\'s API. No new Nebius resources are created.','']
    text='\n'.join(lines)+'\n';(project/'docs/spes-agent-v4-report.md').write_text(text,encoding='utf8')
    public={'status':'v4_agent_pilot_complete_not_expanded' if len([m for m in metrics if m['policy'] in ['single','multi','multi_no_exchange']])==15 else 'exploratory_research_in_progress','cheap_screen':screen['methods'],'multi_trial_numerical':None if numerical is None else numerical['methods'],
       'active_completed':[{k:v for k,v in m.items() if k not in ['prediction','patient','SOZ','trace_sha256']} for m in metrics],
       'clinical_benefit_established':False,'multi_agent_superiority_established':False}
    write_json(project/'examples/spes_agent_v4_summary.json',public);print('English v4 report and portable summary updated.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();build(a.project)
