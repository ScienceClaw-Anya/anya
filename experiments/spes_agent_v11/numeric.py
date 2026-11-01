"""Freeze and evaluate five true anatomical-pair acquisition controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v4.acquisition import numeric_choice
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.environment import PhysicalAcquisition

POLICIES=['random_0','random_1','coverage','score_focus','uncertainty']


def freeze(project,output):
    project,output=Path(project),Path(output);parent=read_json(project/'outputs/spes_agent_v4/active/active_protocol.json');audit=read_json(project/'outputs/spes_physical/alignment_audit.json')
    if audit['mapped_patients']!=35:raise ValueError('All physical mappings required')
    sources=['experiments/spes_agent_v11/numeric.py','experiments/spes_physical/environment.py','experiments/spes_physical/mapping.py',
      'experiments/spes_agent_v4/acquisition.py','experiments/spes_agent_v5/neural.py']
    cases=[{k:c[k] for k in ['subject','outer','case_id']} for c in parent['cases']]
    jobs=read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json')['jobs']
    p={'study':'v11 verified physical-pair numerical acquisition controls','task':'Retrospective diagnostic selection of eight real anatomical pairs. No physical delivery, treatment or prospective-safety claim.',
      'cases':cases,'policies':POLICIES,'common_initial_queries':4,'adaptive_queries':4,'measurement_budget':8,'new_API_calls':0,
      'seed_rule':'Random0 seed6190+outer*100+patient, perstep+step*100003. Random1sharesfirstfourthenusesseed+10000. Otherpoliciesusefixed public geometric/score rules.',
      'primary':'score_focus minus random_0, mean patient SOZ AUROC at eight verified physical pairs',
      'secondary':'All five same-prefix policies; patient-paired differences vsrandom0; preserve all results, no posthoc primary swap.',
      'final_scorer':'Unchanged own v4 outer HAT, fit/normalisation/earlystop exclude test patient. Baseline and MTE provide same observed-only disagreement tools.',
      'data_scope':'First20trials only, geometry-uniquecells only, single-recorded-condition pairs only. Availabilityknownfromreleasedpreprocessingisanexplicitretrospectiveassumption.',
      'future_feasibility':'Numeric query sensitivity: mean withinpatient AUROC range>0.03 and atleast20/35patients have different final rankings between fixed controls. This shows actions matter, not that agents can improve.',
      'future_agent_pilot':[j['predict'][3] for j in jobs],'future_agent_scope':'A separate fixed matched-tool/budget pilot is required; no API calls are authorized by a numerical outcome alone outside the standing user research authorization.',
      'limits':'Repeatedlyinspectedcohort, exploratory. ClinicalSOZlabelsnotcompleteEZortreatmentendpoint. No oldcolumnutility/posteriormodeladoption.',
      'source_sha256':{s:sha(project/s) for s in sources},'model_sha256':parent['model_sha256'],
      'mapping_receipt_sha256':{c['subject']:sha(project/'outputs/spes_physical/maps'/c['subject']/'receipt.json') for c in cases},
      'alignment_audit_sha256':sha(project/'outputs/spes_physical/alignment_audit.json')}
    path=output/'numeric_protocol.json'
    if path.exists() and read_json(path)!=p:raise ValueError('Frozen physical benchmark differs')
    write_json(path,p);return p


def run_case(project,output,case,policy):
    import torch
    project,output=Path(project),Path(output);p=read_json(output/'numeric_protocol.json');torch.set_num_threads(2)
    if any(sha(project/s)!=h for s,h in p['source_sha256'].items()) or sha(project/'outputs/spes_physical/maps'/case['subject']/'receipt.json')!=p['mapping_receipt_sha256'][case['subject']]:raise ValueError('Frozen source/map differs')
    folder=output/'numeric'/case['case_id']/policy;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'metrics.json').exists():return
    for family in ['baseline','mte','hat']:
        if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{case["outer"]}/{family}/checkpoint.pth')!=p['model_sha256'][f'{case["outer"]}/{family}']:raise ValueError('Scorer changed')
    env=PhysicalAcquisition(project,case['outer'],case['subject']);patient=int(case['subject'][-2:]);prefix=fixed_queries(env.available,case['outer'],patient,0)[:4]
    for q in prefix:write_json(folder/f'{len(env.selected):02d}_observation.json',env.observe(q))
    seed=6190+case['outer']*100+patient+(10000 if policy=='random_1' else 0)
    for step in range(4,8):
        public=env.public();choice=numeric_choice(public,policy,np.random.default_rng(seed+step*100003))
        write_json(folder/f'{step:02d}_choice.json',{'choice':choice,'policy':policy,'used_physical_pairs':public['evidence']['observed']['selected'],
          'candidate_geometry':public['evidence']['geometry'],'no_labels_or_future_waveforms':True})
        write_json(folder/f'{step:02d}_observation.json',env.observe(int(choice[1:])))
    result={**env.final_judge(),'case_id':case['case_id'],'policy':policy,'actual_physical_measurements':8,'new_API_calls':0,
      'mapping_receipt_sha256':p['mapping_receipt_sha256'][case['subject']],'protocol_sha256':sha(output/'numeric_protocol.json')}
    write_json(folder/'metrics.json',result);print('PHYSICAL_POLICY_DONE',case['case_id'],policy,round(result['AUROC'],6),flush=True)


def summarize(output):
    output=Path(output);p=read_json(output/'numeric_protocol.json');lookup={(c['case_id'],name):read_json(output/'numeric'/c['case_id']/name/'metrics.json') for c in p['cases'] for name in p['policies']};ids=[c['case_id'] for c in p['cases']];base=np.array([lookup[c,'random_0']['AUROC'] for c in ids]);summary={}
    for policy in POLICIES:
        score=np.array([lookup[c,policy]['AUROC'] for c in ids]);delta=score-base;summary[policy]={'mean_patient_AUROC':float(score.mean()),'gain_over_random_0':float(delta.mean()),'paired_95CI':bootstrap_delta(delta)}
    ranges=[max(lookup[c,p]['AUROC'] for p in POLICIES)-min(lookup[c,p]['AUROC'] for p in POLICIES) for c in ids]
    changed=sum(len({tuple(np.argsort(lookup[c,p]['prediction'])) for p in POLICIES})>1 for c in ids)
    value={'study':p['study'],'primary':p['primary'],'policies':summary,'mean_withinpatient_AUROC_range':float(np.mean(ranges)),
      'patients_with_different_final_rankings':changed,'numerical_query_sensitivity':float(np.mean(ranges))>.03 and changed>=20,
      'new_API_calls':0,'patients':35,'physical_trajectories':175,'multi_agent_superiority_established':False,
      'scope':'First physically valid selection controls; no comparison with old column-ablation means.',
      'rows':[{k:v for k,v in lookup[c,policy].items() if k not in ['prediction','SOZ']} for c in ids for policy in POLICIES]}
    write_json(output/'numeric_results.json',value);print({k:v for k,v in value.items() if k!='rows'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.freeze_only:freeze(a.project,a.output)
    else:
        if not 1<=a.workers<=4:raise ValueError('One to four local workers')
        protocol=read_json(a.output/'numeric_protocol.json')
        with exclusive(a.output/'numeric.lock'):
            with ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(run_case,a.project,a.output,c,name) for c in protocol['cases'] for name in protocol['policies']]
                for f in as_completed(futures):f.result()
            summarize(a.output)
