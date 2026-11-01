"""Same-measurement-budget hypothetical-tool policies; no LLM calls."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v4.acquisition import Acquisition,numeric_choice
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v8.crossfit import write_json
from .features import make_features
from .learner import load_model,predict

BASELINES=['random_0','random_1','coverage','score_focus','uncertainty']
POLICIES=['bag_mean','relative_mean','ridge','robust']


def freeze(project,output):
    project,output=Path(project),Path(output);parent=read_json(project/'outputs/spes_agent_v4/active/active_protocol.json')
    r={'study':'v9 fixed outer numeric acquisition','protocol_sha256':sha(output/'protocol.json'),
      'cases':parent['cases'],'policies':BASELINES+POLICIES,'primary':'bag_mean minus random_0 mean patient AUROC',
      'models_sha256':{},'final_neural_sha256':parent['model_sha256'],
      'initial_common_measurements':4,'total_measurements':8,'new_API_calls':0,
      'hypothetical_inference':'Two observed-only predictions per available candidate, batch32, no candidate prefilter. These are compute, not physical observations.',
      'prefix_sha256':{c['case_id']:sha(project/'outputs/spes_agent_v4/active/trajectories'/c['case_id']/'random_0/metrics.json') for c in parent['cases']}}
    for outer in range(5):
        folder=output/f'learners/outer{outer}';receipt=read_json(folder/'receipt.json')
        if receipt['protocol_sha256']!=r['protocol_sha256'] or sha(folder/'models.joblib')!=receipt['models_sha256']:raise ValueError('Learner provenance differs')
        r['models_sha256'][str(outer)]=receipt['models_sha256']
    path=output/'numeric_protocol.json'
    if path.exists() and read_json(path)!=r:raise ValueError('Frozen numeric protocol changed')
    write_json(path,r)


def run_case(project,output,case,policy):
    import torch
    torch.set_num_threads(2);project,output=Path(project),Path(output);protocol=read_json(output/'protocol.json');evaluation=read_json(output/'numeric_protocol.json')
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()) or sha(output/'protocol.json')!=evaluation['protocol_sha256']:raise ValueError('Frozen source changed')
    folder=output/'numeric'/case['case_id']/policy;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return
    outer=case['outer'];training=project/'outputs/spes_agent_v4/multitrial'
    for family in ['baseline','mte','hat']:
        if sha(training/f'neural/outer{outer}/{family}/checkpoint.pth')!=evaluation['final_neural_sha256'][f'{outer}/{family}']:raise ValueError('Final scorer changed')
    env=Acquisition(project,training,outer,case['subject']);prefix_path=project/'outputs/spes_agent_v4/active/trajectories'/case['case_id']/'random_0/metrics.json'
    if sha(prefix_path)!=evaluation['prefix_sha256'][case['case_id']]:raise ValueError('Prefix changed')
    old=read_json(prefix_path)
    for q in old['queries'][:4]:env.observe(int(q[1:]))
    learned=policy in POLICIES;model=raw=None;hypotheses=0
    if learned:
        if sha(output/f'learners/outer{outer}/models.joblib')!=evaluation['models_sha256'][str(outer)]:raise ValueError('Value learner changed')
        model=load_model(output,outer)
        if int(case['subject'][-2:]) in model['fit_patients']:raise ValueError('Outer patient entered value learner')
        raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20]
    seed=6190+outer*100+int(case['subject'][-2:])+(10000 if policy=='random_1' else 0)
    for step in range(4):
        public=env.public()
        if learned:
            candidates=[int(r['id'][1:]) for r in public['evidence']['geometry']['candidates']]
            x,_=make_features(raw,env.selected,env.predictions['hat'],candidates,env.models['hat'],env.stats);values=predict(model,x)
            choice=f's{candidates[int(np.argmax(values[policy]))]:03d}';hypotheses+=2*len(candidates)
            trace={'choice':choice,'candidates':candidates,'features':x.tolist(),'predicted_values':{k:v.tolist() for k,v in values.items()},'hypothetical_not_observed':True}
        else:
            choice=numeric_choice(public,policy,np.random.default_rng(seed+(4+step)*100003));trace={'choice':choice,'rule':policy}
        write_json(folder/f'{step:02d}_choice.json',trace);write_json(folder/f'{step:02d}_observation.json',env.observe(int(choice[1:])))
    result={**env.final_judge(),'case_id':case['case_id'],'policy':policy,'actual_measurements':8,'new_API_calls':0,'hypothetical_forward_calls':hypotheses}
    if policy=='random_0':np.testing.assert_allclose(result['prediction'],old['prediction'],rtol=0,atol=1e-7)
    write_json(folder/'metrics.json',result);print('HYPOTHESIS_POLICY',case['case_id'],policy,round(result['AUROC'],6),flush=True)


def summarize(output):
    output=Path(output);p=read_json(output/'numeric_protocol.json');lookup={(c['case_id'],policy):read_json(output/'numeric'/c['case_id']/policy/'metrics.json') for c in p['cases'] for policy in p['policies']};ids=[c['case_id'] for c in p['cases']]
    base=np.array([lookup[c,'random_0']['AUROC'] for c in ids]);summary={}
    for policy in p['policies']:
        value=np.array([lookup[c,policy]['AUROC'] for c in ids]);delta=value-base
        summary[policy]={'mean_patient_AUROC':float(value.mean()),'gain_over_random_0':float(delta.mean()),'paired_95CI':bootstrap_delta(delta),
          'different_query_sets':sum(set(lookup[c,policy]['queries'])!=set(lookup[c,'random_0']['queries']) for c in ids)}
    delta=np.array([lookup[c,'bag_mean']['AUROC'] for c in ids])-base;rng=np.random.default_rng(49103);lower=float(np.quantile(delta[rng.integers(0,len(ids),(20000,len(ids)))].mean(axis=1),.05))
    eligible=summary['bag_mean']['mean_patient_AUROC']>max(summary[b]['mean_patient_AUROC'] for b in BASELINES)+.005 and lower>0 and summary['bag_mean']['different_query_sets']>=10
    r={'patients':35,'policies':summary,'primary':p['primary'],'primary_one_sided_95_lower_vs_random0':lower,'further_LLM_eligible':bool(eligible),
      'new_API_calls':0,'clinical_or_multi_agent_benefit_established':False,'hypothetical_forward_calls':sum(m['hypothetical_forward_calls'] for m in lookup.values()),
      'rows':[{k:v for k,v in lookup[c,policy].items() if k not in ['prediction','SOZ']} for c in ids for policy in p['policies']]}
    write_json(output/'numeric_results.json',r);print(summary);print('LLM eligible:',eligible)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.freeze_only:freeze(a.project,a.output)
    else:
        if not 1<=a.workers<=2:raise ValueError('One or two GPU workers')
        evaluation=read_json(a.output/'numeric_protocol.json')
        with exclusive(a.output/'numeric.lock'):
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(run_case,a.project,a.output,c,policy) for c in evaluation['cases'] for policy in evaluation['policies']]
                for f in as_completed(futures):f.result()
            summarize(a.output)
