"""English negative sparse-readout result, requiring independent audit."""
import argparse
from pathlib import Path
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json


def build(project,output):
    project,output=Path(project),Path(output); result=read_json(output/'results.json'); audit=read_json(output/'audit.json')
    if audit['status']!='verified' or audit['readouts_refitted']!=10 or audit['heldout_scorer_state_results']!=280: raise ValueError('Complete readout audit required')
    primary=result['summary']['8/context']
    lines=['# Sparse pretrained readouts: ranking gain not established','',
      f"The fixed context readout scored **{primary['mean_patient_AUROC']:.6f}** mean patient SOZ AUROC on the same eight random0 physical groups, versus original HAT **0.740463**. Gain **{primary['gain_over_hat']:+.6f}**, paired 95% interval **{primary['paired_95CI']}**. The next-tool gate fails. These readouts are not adopted as improved localization scorers.",'',
      '| Groups | Scorer | Mean patient AUROC | Gain over original HAT | Paired 95% interval | Mean patient Brier |','|---:|---|---:|---:|---|---:|']
    for key,v in result['summary'].items():
        k,name=key.split('/'); lines.append(f"| {k} | {name} | {v['mean_patient_AUROC']:.6f} | {v['gain_over_hat']:+.6f} | {v['paired_95CI']} | {v['mean_patient_Brier']:.6f} |")
    lines+=['',
      'The simpler linear readout has a small positive eight-group mean difference, but its interval crosses zero and it remains secondary. The four-group linear result is also uncertain. Neither replaces the primary. Better absolute output scale is distinct from better within-patient ranking; the development-prevalence constant is included as a proper Brier comparator.','',
      '## Adaptation and patient separation','',
      'Each of five pretrained outer HAT encoders remains fixed. Its own two separately fitted readouts use only its 21 original fitting patients, with three seeded physical paths at four, six and eight response groups. The seven original early-stop patients and seven outer test patients never enter readout fitting. All ten fixed C=0.1 logistic fits are frozen before new outer outcomes. Patient weights balance total contribution while preserving natural class prevalence.','',
      'The 16-dimensional features are extracted immediately before the original classifier; their linear reconstruction is checked against original HAT outputs. The context readout adds count and missingness information and representation/count interactions. Each encoder has its own readout coordinate system. Latent dimensions from independently trained neural models are never pooled into one classifier. These fitting features are in-sample for their frozen encoder, explicitly not out-of-fold neural features.','',
      f"There are **{audit['training_banks']}** fitting banks and **{audit['training_contact_state_rows']:,}** correlated contact/state rows. The audit independently reconstructs physical availability and context features, checks all lineage hashes and patient exclusion, refits all ten readouts, and replays **{audit['heldout_scorer_state_results']}** held-out scorer/state outcomes. Five representative source encoder states were re-inferred to verify feature capture. Remaining neural features were checked from stored outputs, not all re-inferred. Original HAT outputs reproduce the frozen v12 physical-input reference.",'',
      '## Implication for further collaboration research','',
      'Replacing the last readout alone does not establish better localization. This does not rule out adapting the encoder, obtaining additional independent data, or a different acquisition strategy. No new Gemini call was used. A new physically valid language-agent pilot must use a separately frozen protocol and common tools, evidence, response groups, API attempts, feedback cadence and final scorer; it must not adopt failed v12-v15 predictors as established improvements.','',
      'All computation ran locally. No new neural fit or cloud resource was required. Repeated inspection of this 35-patient cohort makes these exploratory outcomes; nominal bootstrap intervals are not adjusted for the entire research search. Eight response groups do not equate pulse/charge/time budgets. SOZ ranking does not establish seizure-control benefit.','']
    (project/'docs/spes-agent-v15-report.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v15_summary.json',{'status':'readout_ranking_gate_failed','results':result,'audit':audit,'multi_agent_superiority_established':False})
    print('English v15 result and complete audit saved.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.project,a.output)
