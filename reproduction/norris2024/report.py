"""Build an English report from observed artifacts; missing stages remain visibly pending."""
from __future__ import annotations
import argparse
import base64
import html
import io
import json
from pathlib import Path

import numpy as np

EXPECTED={'CNN (divergent)':.574,'CNN (convergent)':.666,'Transformer (base)':.730}


def document(output:Path)->str:
    cohort_path=output/'cohort.json'
    cohort=json.loads(cohort_path.read_text()) if cohort_path.exists() else None
    table_path=output/'table1_reproduction.json'
    table=json.loads(table_path.read_text()) if table_path.exists() else {}
    finished=sum(row['finished_folds'] for row in table.values())
    all_done=finished==75
    rows=[]
    for name,value in EXPECTED.items():
        actual=table.get(name,{})
        observed=actual.get('mean_AUROC')
        text=f'{observed:.4f}' if observed is not None else 'Pending'
        delta=f'{observed-value:+.4f}' if observed is not None else 'Pending'
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{value:.3f}</td><td>{text}</td><td>{delta}</td><td>{actual.get("finished_folds",0)}/25</td></tr>')
    cohort_text=(f'{cohort["subjects"]} patients, {cohort["electrodes"]} electrodes, {cohort["positives"]} SOZ-positive electrodes. Published counts match: {cohort["paper_counts_match"]}.' if cohort else 'Preprocessing is still running; final cohort counts are pending.')
    metric_table=''
    for rule in ['paper_threshold_per_patient','released_code_threshold_per_patient']:
        subrows=[]
        for name in EXPECTED:
            runs=table.get(name,{}).get('results',[])
            values={key:[np.mean([r[key] for r in run[rule]]) for run in runs if all(key in r for r in run[rule])] for key in ['Sensitivity','Specificity','Youden','AUPRC']}
            subrows.append('<tr><td>'+html.escape(name)+'</td>'+''.join(f'<td>{np.mean(v):.4f} ({len(v)} folds)</td>' if v else '<td>Pending / undefined</td>' for v in values.values())+'</tr>')
        metric_table+=f'<h3>{html.escape(rule.replace("_"," "))}</h3><table><thead><tr><th>Model</th><th>Sensitivity</th><th>Specificity</th><th>Youden</th><th>AUPRC</th></tr></thead><tbody>'+''.join(subrows)+'</tbody></table>'
    agent_path=output/'agent_results.json'
    completed_cases=len(list((output/'agents').glob('*/judge_metrics.json')))
    agents=f'Pending: {completed_cases}/175 complete held-out patient evaluations. No multi-agent advantage is claimed.'
    if agent_path.exists():
        data=json.loads(agent_path.read_text())
        subjects=sorted(set(r['patient'] for r in data))
        agent_rows=[]
        for policy in ['single','multi']:
            per_subject=np.array([np.mean([r['strategies'][policy]['metrics']['AUROC'] for r in data if r['patient']==subject]) for subject in subjects])
            delta=np.array([np.mean([r['strategies'][policy]['metrics']['AUROC']-r['numerical_baselines']['Transformer (base)']['AUROC'] for r in data if r['patient']==subject]) for subject in subjects])
            rng=np.random.default_rng(42)
            draws=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            agent_rows.append(f'<tr><td>{policy}</td><td>{per_subject.mean():.4f}</td><td>{delta.mean():+.4f}</td><td>[{lo:+.4f}, {hi:+.4f}]</td></tr>')
        agents=f'<p>{len(data)} held-out patient evaluations; {len(subjects)} distinct patients. Repeats are clustered by patient for uncertainty.</p><table><tr><th>Policy</th><th>Mean AUROC</th><th>Paired difference vs Transformer</th><th>95% patient bootstrap CI</th></tr>'+''.join(agent_rows)+'</table>'
    figure=''
    if table:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,3.6))
        for index,(name,paper) in enumerate(EXPECTED.items()):
            actual=table.get(name,{}).get('mean_AUROC')
            ax.scatter(index-.08,paper,color='#64748b',marker='x',s=70,label='Published' if index==0 else None)
            if actual is not None:ax.scatter(index+.08,actual,color='#0f766e',s=60,label='Observed rerun' if index==0 else None)
        ax.set(xticks=range(3),xticklabels=['CNN divergent','CNN convergent','CNN Transformer'],ylabel='Patient-averaged AUROC',ylim=(0,1))
        ax.axhline(.5,color='#94a3b8',linestyle='--',linewidth=1)
        ax.legend(frameon=False);fig.tight_layout()
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140);plt.close(fig)
        figure='<img alt="Published and observed AUROC values" src="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">'
    status=('All 75 reconstruction runs complete; cohort/implementation discrepancies prevent a claim of exact paper reproduction.' if all_done else f'Work in progress: {finished}/75 primary runs complete. Values, if shown, are interim.')
    pipeline_path=output/'pipeline_status.json'
    if pipeline_path.exists():status+=' Current stage: '+html.escape(json.loads(pipeline_path.read_text())['stage'])+'.'
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Norris 2024 reproduction audit</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#172536;background:#f5f7fa;max-width:1100px;margin:32px auto;padding:0 24px}}article{{background:white;border:1px solid #dbe3ed;border-radius:16px;padding:28px;margin:20px 0}}h1,h2,h3{{line-height:1.2}}table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #dbe3ed}}th{{background:#e8eff7}}.status{{background:#fff7db;border-left:5px solid #d58d00;padding:16px}}img{{max-width:100%}}a{{color:#036778}}</style><h1>SPES-based SOZ localization: reproduction before agents</h1><p class="status">{status}</p><article><h2>Source and task</h2><p><a href="https://proceedings.mlr.press/v252/norris24a.html">Norris et al., MLHC / PMLR 252 (2024)</a> classifies the same eligible electrodes from measured single-pulse electrical stimulation responses. This is retrospective localization, not treatment efficacy or next-stimulation selection. <a href="https://github.com/norrisjamie23/Localising_SOZ_from_SPES">Official source code</a>; <a href="https://openneuro.org/datasets/ds004080/versions/1.2.4">public dataset</a>.</p><p>{cohort_text}</p><p>Five repeats of five patient-level folds, with three training folds, one validation fold and one test fold. Metrics average patients within each fold, then average the 25 folds. Clinical labels remain in the scoring process.</p></article><article><h2>Primary reproduction</h2><table><thead><tr><th>Model</th><th>Published AUROC</th><th>Observed AUROC</th><th>Difference</th><th>Finished folds</th></tr></thead><tbody>{''.join(rows)}</tbody></table>{figure}{metric_table}</article><article><h2>Same-task agent comparison</h2><p>Frozen out-of-fold numerical models and measured response features are shared by single-agent and four-specialist policies. Each policy receives eight Gemini requests per case. Both average their last four score vectors. The fixed three-model ensemble is an additional control. Critique receives no held-out labels or accuracy feedback.</p>{agents}</article><article><h2>Reproducibility limitations and remaining stages</h2><ul><li>The released notebook does not provide original RNG states, fold manifests or checkpoints. The rerun freezes explicit seeds and lexical file order; exact decimal agreement cannot be guaranteed.</li><li>The original development PyTorch build is replaced by released 2.1.0+cu121. Windows path separators are normalized. The original RandomNoise transform is loaded without unrelated optional spectral dependencies; numerical model source files remain unchanged.</li><li>The published Appendix C median/MAD threshold differs from the released code's pooled validation threshold. Both are reported. If the appendix rule is undefined for a validation patient, it is marked as undefined rather than silently repaired.</li><li>Published selected hyperparameters are used. This does not rerun the original ten-trial Optuna search.</li><li>Channel sensitivity is a separately recorded stage using saved checkpoints. Its original inference RNG and bootstrap implementation are not published.</li><li>Independent Engel outcome labels and the exact outcome-model specification have not been verified in the public release. Figure 4B cannot be claimed reproduced without them. Resection or SOZ labels must not be substituted for outcomes.</li><li>The spatial illustration uses a predetermined held-out patient; the original example patient's identity was not reported.</li></ul><p>Evidence files: <code>cohort.json</code>, <code>training/protocol.json</code>, <code>training/split_*.json</code>, per-run predictions and metrics, <code>table1_reproduction.json</code>, and native ScienceClaw agent artifacts. Missing stages remain pending; published values never stand in for executed results.</p></article></html>'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--html',type=Path,required=True)
    args=p.parse_args();args.html.parent.mkdir(parents=True,exist_ok=True);args.html.write_text(document(args.output),encoding='utf-8');print(args.html)
