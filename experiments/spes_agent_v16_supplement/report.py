"""Audited, English presentation figures and diagnostic conclusions."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json
from experiments.spes_agent_v8.crossfit import write_json

LABEL={'random_0':'Random 0','random_1':'Random 1','coverage':'Coverage','score_focus':'Score focus','uncertainty':'Uncertainty','single':'Single agent','multi':'Multi agent','multi_reason_hidden':'Multi, peer reasons hidden'}


def plot_interval(ax,labels,values,colors):
    y=np.arange(len(labels)); mean=np.array([v['mean'] for v in values]); low=np.array([v['paired_95CI'][0] for v in values]); high=np.array([v['paired_95CI'][1] for v in values])
    for i in y:
        ax.errorbar(mean[i],i,xerr=[[max(0.,mean[i]-low[i])],[max(0.,high[i]-mean[i])]],fmt='o',color=colors[i],capsize=3,elinewidth=1.6,markersize=6)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.axvline(0,color='#768398',linestyle='--',linewidth=1)
    ax.grid(axis='x',alpha=.18); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)


def build(project,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    project,output=Path(project),Path(output); budget=read_json(output/'budget_results.json'); oracle=read_json(output/'oracle_results.json'); audit=read_json(output/'audit.json')
    if audit['status']!='verified': raise ValueError('Audit required before figures/report')
    images=project/'docs/images'; images.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13.4,5.5),layout='constrained')
    for ax,population,title in zip(axes,['full35','pilot5'],['Numerical cohort (35 patients)','Agent pilot (5 patients)']):
        summary=budget['summary'][population]; names=list(summary)
        plot_interval(ax,[LABEL[n] for n in names],[summary[n] for n in names],['#d26927' if n in ['single','multi','multi_reason_hidden'] else '#216a99' for n in names])
        ax.set_title(title+f"\nCommon four-group AUROC: {next(iter(summary.values()))['mean_AUROC4']:.4f}")
        ax.set_xlabel('AUROC gain: eight minus four groups')
    fig.suptitle('Additional measurements help unevenly; the pilot begins stronger',fontsize=14)
    for suffix in ['png','svg']: fig.savefig(images/f'spes_v16_budget_gain.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.4,3.5),layout='constrained'); names=list(oracle['summary'])
    for ax,key,title in zip(axes,['oracle_minus_actual','oracle_minus_matched_random_oracle'],['Room within proposed candidates','Proposed-pool quality versus matched random pools']):
        plot_interval(ax,[LABEL[n] for n in names],[oracle['summary'][n][key] for n in names],['#7b4e9b']*3)
        ax.set_xlabel('Conditional final-query AUROC gap'); ax.set_title(title)
    fig.suptitle('Posthoc last-step oracle: true labels used only for judging (5 patients)',fontsize=13)
    for suffix in ['png','svg']: fig.savefig(images/f'spes_v16_candidate_diagnostic.{suffix}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    for name in ['spes_v16_budget_gain.svg','spes_v16_candidate_diagnostic.svg']:
        path=images/name; path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    full=budget['summary']['full35']['score_focus']; pilot=budget['summary']['pilot5']; multi=oracle['summary']['multi']
    lines=['# Supplemental diagnostics for the physical multi-agent pilot','',
      'Additional physical response groups can improve the numerical benchmark, but the five-patient pilot started from a stronger four-group result. The candidate analysis finds label-privileged selection room within agent proposals; it does not establish that the proposals beat equally sized random pools or that a usable selector can find the better choice. No model API call or training fit was added.','',
      '## Four to eight measured response groups','',
      f"On the 35-patient numerical cohort, score-focus rises from **{full['mean_AUROC4']:.6f}** to **{full['mean_AUROC8']:.6f}**, gain **{full['mean']:+.6f}**, paired nominal95%interval **{full['paired_95CI']}**. It improves25patients and worsens9, with one tie. This compares measurement budgets, not score-focus against random at the same budget.",'',
      f"The five-patient pilot already scores **{pilot['multi']['mean_AUROC4']:.6f}** after the common four groups. Eight-group score-focus is **{pilot['score_focus']['mean_AUROC8']:.6f}** (gain **{pilot['score_focus']['mean']:+.6f}**), while multi is **{pilot['multi']['mean_AUROC8']:.6f}** (gain **{pilot['multi']['mean']:+.6f}**). All pilot four-to-eight intervals include zero. The pilot is not representative of the 35-patient mean starting accuracy; its five IDs were fixed before v11 outcomes, so this is a selection/generalization limitation rather than evidence of outcome-based cherry-picking.",'',
      '![Paired four-to-eight group gains](images/spes_v16_budget_gain.png)','',
      '| Population | Policy | AUROC4 | AUROC8 | Paired gain | Nominal95%interval |','|---|---|---:|---:|---:|---|']
    for population,summary in budget['summary'].items():
        for name,v in summary.items(): lines.append(f"| {population} | {LABEL[name]} | {v['mean_AUROC4']:.6f} | {v['mean_AUROC8']:.6f} | {v['mean']:+.6f} | {v['paired_95CI']} |")
    lines+=['','## Is the final candidate pool useful?','',
      'Each actual seven-group prefix stays fixed. The judge tests the union of both three-position proposals and five seeded random pools of the same size. For each random pool, it takes its best true-label AUROC, then averages those five maxima. True labels are unavailable to the agents and are used only in this offline evaluator.','',
      '| Arm | Executed AUROC8 | Best proposed candidate | Mean matched-random oracle | Oracle minus executed | Proposal oracle minus random oracle |','|---|---:|---:|---:|---:|---:|']
    for name,v in oracle['summary'].items():
        lines.append(f"| {LABEL[name]} | {v['mean_actual_AUROC8']:.6f} | {v['mean_proposal_oracle_AUROC8']:.6f} | {v['mean_matched_random_oracle_AUROC8']:.6f} | {v['oracle_minus_actual']['mean']:+.6f} | {v['oracle_minus_matched_random_oracle']['mean']:+.6f} |")
    lines+=['',
      f"For multi, the privileged best final candidate yields **{multi['mean_proposal_oracle_AUROC8']:.6f}**, versus executed **{multi['mean_actual_AUROC8']:.6f}**: room **{multi['oracle_minus_actual']['mean']:+.6f}** within the proposals. This gap is nonnegative by construction and is not a demonstrated improvement by a usable agent. Proposed-pool quality versus a matched random oracle is **{multi['oracle_minus_matched_random_oracle']['mean']:+.6f}**, interval **{multi['oracle_minus_matched_random_oracle']['paired_95CI']}**, which includes zero.",'',
      '![Conditional candidate-pool diagnostics](images/spes_v16_candidate_diagnostic.png)','',
      'A perfect judge of the last query does not define an achievable eight-step oracle trajectory: changing earlier decisions would change every later state and proposal. The result suggests candidate ranking is a worthwhile training target, while evidence that proposal generation itself is superior remains uncertain. Replacing just the last decision with the score-focus leader barely changes the multi mean (+0.000275 on the same actual seven-group prefixes), so a simple final-step geometric override is not a demonstrated fix.','',
      '## Verification and reporting limits','',
      f"The supplement used **{oracle['new_local_scorer_calls']} local HAT counterfactual evaluations**, zero API calls and zero new training fits. Independent audit reconstructed **{audit['budget_patient_policy_rows']}** patient/policy budget rows, **{audit['conditional_oracle_rows']}** conditional-pool rows, every seeded pool and all paired bootstrap intervals. Original API attempts remain exactly120. Stored new neural predictions were not all re-inferred by this supplement audit; actual executed predictions match the separately re-inferred v16 results.",'',
      'The bootstrap unit is the patient. Arms, electrodes and random-pool repeats are correlated observations, not new participants. All results are posthoc on a repeatedly inspected cohort; nominal intervals are not adjusted for the whole research search. Response groups are not single pulses or equal current/charge/time. SOZ discrimination does not demonstrate treatment efficacy.','',
      'For presentation, report the validated benchmark and negative collaboration result, then these two bottleneck findings. A next local experiment may learn relative candidate ranking using source-held-out development patients and compare separate specialists with an all-feature single model. It must not use these test-label oracle choices as deployed predictions or unreported tuning targets.','']
    (project/'docs/spes-agent-v16-supplement.md').write_text('\n'.join(lines),encoding='utf8')
    write_json(project/'examples/spes_agent_v16_supplement_summary.json',{'scope':'Posthoc diagnostics, not executable agent performance','budget':budget['summary'],'candidate_oracle':oracle['summary'],'audit':audit,'new_API_calls':0,'multi_agent_superiority_established':False})
    print('English diagnostic report and four standalone figure files saved.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); a=parser.parse_args(); build(a.project,a.output)
