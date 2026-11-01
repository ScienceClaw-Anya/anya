"""Independent paired metric and seeded candidate-pool reconstruction."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json
from .analysis import NUMERIC,AGENTS


def mean_ci(values):
    values=np.asarray(values,dtype=float); rng=np.random.default_rng(17671)
    draws=np.array([values[rng.integers(0,len(values),len(values))].mean() for _ in range(10000)])
    return float(values.mean()),np.quantile(draws,[.025,.975])


def audit(project,output):
    project,output=Path(project),Path(output); p=read_json(output/'protocol.json'); budgets=read_json(output/'budget_results.json'); oracle=read_json(output/'oracle_results.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()) or any(sha(project/f)!=h for f,h in p['input_sha256'].items()): raise ValueError('Frozen source/input differs')
    budget_rows=0
    for c in p['cases']:
        with np.load(project/f'outputs/spes_agent_v12/evaluation/{c["case_id"]}/predictions.npz') as d: before=d['4/hat']; labels=d['label']
        for row in [r for r in budgets['rows'] if r['case_id']==c['case_id']]:
            origin=(project/'outputs/spes_agent_v11/numeric'/c['case_id']/row['policy'] if row['policy'] in NUMERIC else project/'outputs/spes_agent_v16/trajectories'/c['case_id']/row['policy'])
            after=read_json(origin/'metrics.json')['prediction']; a=independent_auc(labels,before); b=independent_auc(labels,after)
            np.testing.assert_allclose([row['AUROC4'],row['AUROC8'],row['gain']],[a,b,b-a],rtol=0,atol=1e-12); budget_rows+=1
    for population,summary in budgets['summary'].items():
        for name,metrics in summary.items():
            rr=[r for r in budgets['rows'] if r['policy']==name and (population=='full35' or r['pilot'])]
            mean,interval=mean_ci([r['gain'] for r in rr]); np.testing.assert_allclose(mean,metrics['mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(interval,metrics['paired_95CI'],rtol=0,atol=1e-12)
    oracle_rows=0; counterfactuals=0
    for c in p['pilot_cases']:
        folder=output/'oracle'/c['case_id']; metric=read_json(folder/'metrics.json'); slate_path=folder/'slates.json'; slates=read_json(slate_path)
        if metric['protocol_sha256']!=sha(output/'protocol.json') or metric['slates_sha256']!=sha(slate_path) or metric['predictions_sha256']!=sha(folder/'predictions.npz'): raise ValueError('Oracle lineage/hash differs')
        labels=load_array(project/f'vendor/norris2026_hat/data/y_{c["subject"]}.npz')
        with np.load(folder/'predictions.npz') as d: sequences=d['selected']; predictions=d['prediction']
        lookup={tuple(map(int,q)):independent_auc(labels,score) for q,score in zip(sequences,predictions)}
        if len(lookup)!=len(sequences) or len(lookup)!=metric['counterfactual_scorer_calls']: raise ValueError('Counterfactual caching/count differs')
        counterfactuals+=len(lookup)
        for pi,name in enumerate(AGENTS):
            origin=project/'outputs/spes_agent_v16/trajectories'/c['case_id']/name; public=read_json(origin/'03_public.json'); entry=slates['entries'][name]
            prefix=[int(q[1:]) for q in public['evidence']['observed']['selected']]; available=[int(q['id'][1:]) for q in public['evidence']['geometry']['candidates']]
            pool=list(dict.fromkeys(int(q[1:]) for slot in [0,1] for q in read_json(origin/f'03_{slot}_decision.json')['ranked_queries']))
            if entry['actual_prior_seven']!=prefix or entry['proposal_pool']!=pool or len(prefix)!=7: raise ValueError('Conditional candidate pool differs')
            expected=[]
            for rep in range(5):
                rng=np.random.default_rng(173000+c['outer']*10000+int(c['subject'][-2:])*100+pi*10+rep)
                expected.append(list(map(int,rng.choice(available,len(pool),replace=False))))
            if entry['matched_random_pools']!=expected: raise ValueError('Random oracle pool size/seed differs')
            actual=entry['actual_final_query']; scores={q:lookup[tuple(prefix+[q])] for q in set(pool+[q for rp in expected for q in rp]+[entry['score_focus_query']])}
            chosen=scores[actual]; best=max(scores[q] for q in pool); random_mean=float(np.mean([max(scores[q] for q in rp) for rp in expected])); focus=scores[entry['score_focus_query']]
            row=next(r for r in metric['rows'] if r['policy']==name)
            expected_values={'actual_AUROC8':chosen,'proposal_oracle_AUROC8':best,'matched_random_oracle_mean_AUROC8':random_mean,
                             'score_focus_same_prefix_AUROC8':focus,'oracle_minus_actual':best-chosen,'oracle_minus_matched_random_oracle':best-random_mean,'score_focus_minus_actual':focus-chosen}
            for key,value in expected_values.items(): np.testing.assert_allclose(row[key],value,rtol=0,atol=1e-12)
            if actual not in pool or best<chosen: raise ValueError('Oracle/executed membership differs')
            actual_metric=read_json(origin/'metrics.json'); np.testing.assert_allclose(chosen,actual_metric['AUROC'],rtol=0,atol=1e-12)
            if row!=next(r for r in oracle['rows'] if r['case_id']==c['case_id'] and r['policy']==name): raise ValueError('Oracle summary rows differ')
            oracle_rows+=1
    for name,summary in oracle['summary'].items():
        rr=[r for r in oracle['rows'] if r['policy']==name]
        for key in ['oracle_minus_actual','oracle_minus_matched_random_oracle','score_focus_minus_actual']:
            mean,interval=mean_ci([r[key] for r in rr]); np.testing.assert_allclose(mean,summary[key]['mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(interval,summary[key]['paired_95CI'],rtol=0,atol=1e-12)
    attempts=list((project/'outputs/spes_agent_v16/trajectories').glob('*/*/*_usage.json'))
    if len(attempts)!=120: raise ValueError('Original authorized API-attempt count changed')
    value={'status':'verified','budget_patient_policy_rows':budget_rows,'conditional_oracle_rows':oracle_rows,'local_counterfactual_predictions':counterfactuals,
           'unchanged_original_API_attempts':120,'new_API_calls':0,
           'checks':['Frozen sources and all completed input hashes; paired patient-level four-to-eight metrics independently recomputed.',
                     'Every conditional candidate/random pool, cache key and true-label AUROC independently reconstructed.',
                     'All reported gap bootstrap intervals reproduced with an independent loop; original API count remains120.'],
           'limits':['New counterfactual neural outputs were checked from stored predictions; actual executed choices were matched to the separately re-inferred v16 audit.',
                     'The oracle uses true labels and is not executable agent accuracy or a full-trajectory upper bound.']}
    write_json(output/'audit.json',value); print(value)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); a=parser.parse_args(); audit(a.project,a.output)
