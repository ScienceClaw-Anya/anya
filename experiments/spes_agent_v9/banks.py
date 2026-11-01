"""Freeze and generate hypothetical-response tools on source-held-out patients."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha,exclusive,check_groups
from experiments.spes_agent_v4.neural import load_array,upstream_models
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v8.targets import score_subset
from .features import make_features,FEATURES,METHODS


def freeze(project,output):
    project,output=Path(project),Path(output);source=project/'outputs/spes_agent_v8';cross=read_json(source/'crossfit_protocol.json')
    files=['experiments/spes_agent_v9/'+f for f in ['banks.py','features.py','learner.py','numeric.py']]
    files+=['experiments/spes_agent_v8/'+f for f in ['crossfit.py','targets.py','features.py','numeric.py','learner.py']]
    files+=['experiments/spes_agent_v4/acquisition.py','experiments/spes_agent_v4/neural.py','vendor/norris2026_hat/src/models.py']
    parent={}
    for j in cross['jobs']:
        for path in [source/f'crossfit/outer{j["outer"]}_inner{j["inner"]}/receipt.json',source/f'targets/phase0/outer{j["outer"]}_inner{j["inner"]}/receipt.json']:
            parent[path.relative_to(project).as_posix()]=sha(path)
    v={'study':'v9 observed-only hypothetical-response specialists','scope':'Exploratory tools; no multi-agent or clinical benefit claim',
      'motivation':'v8 primary failed; development diagnostics show poor unseen-patient candidate ranking. Test substantive candidate-specific response hypotheses before more LLM spending.',
      'hypotheses':METHODS,'features':FEATURES,'target':'Reuse exactly v8 phase0 source-heldout candidate gains and prefixes, never outer patients.',
      'waveform_assumptions':'Geometry-profile weighted response interpolation; second specialist additionally applies clipped radial amplitude scaling and distance weights. Trial indices across sites are computationally aligned, not simultaneous measurements.',
      'hypothetical_signal':'Explicit prediction, never an observed measurement or clinical feedback. Candidate raw waveforms are inaccessible to this tool.',
      'source_models':'Reuse twenty own tensor-only v8 inner HATs; no new neural fit.',
      'learner':'Same fixed ridge100 and five patient-exclusion depth2/150tree/lr.03/minleaf30/huber.9 bags for both absolute and relative gain as v8, patient-balanced fitting, no HPO.',
      'policies':['bag_mean','relative_mean','ridge','robust'],'primary':'bag_mean minus same-prefix random_0, mean patient AUROC',
      'outer_plan':'35 patients, common random_0 four-measurement prefix, four adaptive queries, unchanged original v4 HAT. All fixed policies remain reported.',
      'future_LLM_gate':'Primary > strongest fixed numerical baseline by 0.005; positive one-sided95 patient-bootstrap lower versus random_0; at least ten differing final query sets. Passing only permits a separate collaboration test.',
      'new_API_calls':0,'source_sha256':{p:sha(project/p) for p in files},'parent_receipt_sha256':parent,
      'crossfit_protocol_sha256':sha(source/'crossfit_protocol.json'),'v8_query_protocol_sha256':sha(source/'query_protocol.json')}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=v:raise ValueError('Frozen v9 protocol changed')
    write_json(path,v);return v


def run_job(project,output,job):
    import torch
    project,output=Path(project),Path(output);protocol=read_json(output/'protocol.json');source=project/'outputs/spes_agent_v8';outer,inner=job['outer'],job['inner']
    if any(sha(project/p)!=h for p,h in protocol['source_sha256'].items()) or any(sha(project/p)!=h for p,h in protocol['parent_receipt_sha256'].items()):raise ValueError('Frozen source lineage differs')
    check_groups(job['train'],job['early_stop'],job['predict'],job['outer_test'])
    folder=output/f'banks/outer{outer}_inner{inner}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():
        r=read_json(folder/'receipt.json')
        if r['protocol_sha256']!=sha(output/'protocol.json') or any(sha(folder/f'{s}.npz')!=h for s,h in r['bank_sha256'].items()):raise ValueError('Completed v9 bank changed')
        return
    old_folder=source/f'targets/phase0/outer{outer}_inner{inner}';source_folder=source/f'crossfit/outer{outer}_inner{inner}'
    old_receipt=read_json(old_folder/'receipt.json');r=read_json(source_folder/'receipt.json')
    if sha(source_folder/'checkpoint.pth')!=r['checkpoint_sha256']:raise ValueError('Source HAT changed')
    torch.set_num_threads(2);mods=upstream_models(project);model,_,_=mods.create_model('hat');model.load_state_dict(torch.load(source_folder/'checkpoint.pth',map_location='cpu',weights_only=True)['model_state_dict']);model=model.cuda().eval()
    hashes={};inputs={};hyp_calls=0
    for subject in job['predict']:
        if subject in job['train']+job['early_stop']+job['outer_test']:raise ValueError('Source target patient not held out')
        bank_path=old_folder/f'{subject}.npz'
        if sha(bank_path)!=old_receipt['patient_file_sha256'][subject]:raise ValueError('Target rows changed')
        with np.load(bank_path) as d:old={k:d[k] for k in d.files}
        raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
        if sha(raw_path)!=old_receipt['raw_input_sha256'][subject]:raise ValueError('Raw input differs')
        raw=load_array(raw_path)[:,:20];features=np.zeros((len(old['gain']),len(FEATURES)));hyp_scores=np.zeros((len(old['gain']),len(METHODS),len(old['y'])))
        for prefix in np.unique(old['prefix'],axis=0):
            ix=np.flatnonzero((old['prefix']==prefix).all(axis=1));selected=list(map(int,prefix[prefix>=0]));before=old['before_scores'][ix[0]]
            np.testing.assert_allclose(score_subset(model,raw,r['normalisation'],selected),before,rtol=0,atol=1e-7)
            candidates=list(map(int,old['candidate'][ix]));x,details=make_features(raw,selected,before,candidates,model,r['normalisation']);features[ix]=x
            for i,detail in zip(ix,details):hyp_scores[i]=np.vstack([detail[m] for m in METHODS])
            hyp_calls+=len(candidates)*len(METHODS)
        np.savez_compressed(folder/f'{subject}.npz',features=features,gain=old['gain'],prefix=old['prefix'],candidate=old['candidate'],hypothetical_scores=hyp_scores)
        hashes[subject]=sha(folder/f'{subject}.npz');inputs[subject]=sha(bank_path);print('HYPOTHESIS_BANK',outer,inner,subject,len(features),flush=True)
    write_json(folder/'receipt.json',{'outer':outer,'inner':inner,'patients':job['predict'],'outer_test':job['outer_test'],
      'bank_sha256':hashes,'v8_target_sha256':inputs,'source_checkpoint_sha256':r['checkpoint_sha256'],'protocol_sha256':sha(output/'protocol.json'),
      'hypothetical_forward_calls':hyp_calls,'actual_new_measurements':0,'API_calls':0})
    print('HYPOTHESIS_JOB_DONE',outer,inner,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.freeze_only:freeze(a.project,a.output);print('v9 protocol frozen before response hypotheses');return
    if not 1<=a.workers<=2:raise ValueError('One or two GPU workers')
    cross=read_json(a.project/'outputs/spes_agent_v8/crossfit_protocol.json')
    with exclusive(a.output/'banks.lock'):
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(run_job,a.project,a.output,j) for j in cross['jobs']]
            for f in as_completed(futures):f.result()

if __name__=='__main__':main()
