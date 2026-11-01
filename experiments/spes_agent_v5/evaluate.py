"""Frozen fixed-sequence test of sparse experts, with no LLM request."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from experiments.spes_agent_v3.common import read_json,write_json,sha
from experiments.spes_agent_v4.neural import upstream_models,load_array
from experiments.spes_agent_v4.acquisition import gated_input
from experiments.spes_agent_v4.screen import bootstrap_delta
from .neural import FAMILIES

POLICIES=['random_0','random_1','coverage','score_focus','uncertainty']

def freeze(project,output):
    project,output=Path(project),Path(output);active=project/'outputs/spes_agent_v4/active';ap=read_json(active/'active_protocol.json')
    value={'scope':'Exploratory fixed-measurement-sequence evaluation after negative v4 agent pilot',
      'primary':'sparse_hat minus original full-response-trained HAT on random_0 sequence, mean patient AUROC',
      'models':['old_hat','sparse_baseline','sparse_hat','sparse_rank_mean'],'policies':POLICIES,'no_new_model_selection':True,
      'caveat':'This changes the scorer, not the acquisition controller. Fixed adaptive query sets were originally chosen with the old scorer. No multi-agent efficacy claim.',
      'source_sha256':sha(Path(__file__)),'training_protocol_sha256':sha(output/'sparse_protocol.json'),
      'query_inputs':{f'{c["case_id"]}/{p}':sha(active/'trajectories'/c['case_id']/p/'metrics.json') for c in ap['cases'] for p in POLICIES}}
    path=output/'evaluation_protocol.json'
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen sparse evaluation changed')
    write_json(path,value);return value

def run(project,output):
    import torch
    project,output=Path(project),Path(output);protocol=freeze(project,output);training=read_json(output/'sparse_protocol.json')
    active=project/'outputs/spes_agent_v4/active';ap=read_json(active/'active_protocol.json');mods=upstream_models(project);rows=[];torch.set_num_threads(4)
    for row in training['jobs']:
        outer=row['outer'];models={};norm=None
        for family in FAMILIES:
            folder=output/f'neural/outer{outer}/{family}';receipt=read_json(folder/'receipt.json')
            if sha(folder/'checkpoint.pth')!=receipt['checkpoint_sha256'] or receipt['protocol_sha256']!=sha(output/'sparse_protocol.json'):raise ValueError('Checkpoint provenance differs')
            if receipt['train']!=row['train'] or receipt['early_stop']!=row['early_stop'] or receipt['outer_test']!=row['predict']:raise ValueError('Patient isolation differs')
            if norm is None:norm=receipt['normalisation']
            elif norm!=receipt['normalisation']:raise ValueError('Scorer normalization differs')
            net,_,_=mods.create_model(family);saved=torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=True)
            net.load_state_dict(saved['model_state_dict']);models['sparse_'+family]=net.cuda().eval()
        old=read_json(project/f'outputs/spes_agent_v4/multitrial/neural/outer{outer}/hat/receipt.json')
        if norm!=old['normalisation']:raise ValueError('New and old training statistics differ')
        for case in [c for c in ap['cases'] if c['outer']==outer]:
            subject=case['subject'];root=project/'vendor/norris2026_hat/data';raw=load_array(root/f'X_{subject}.npz')[:,:20];y=load_array(root/f'y_{subject}.npz')
            for policy in POLICIES:
                metric_path=active/'trajectories'/case['case_id']/policy/'metrics.json'
                if sha(metric_path)!=protocol['query_inputs'][f'{case["case_id"]}/{policy}']:raise ValueError('Old query inputs changed')
                old=read_json(metric_path);queries=[int(q[1:]) for q in old['queries']];x=gated_input(raw,queries)
                for segment,stat in zip([slice(0,1),slice(1,129),slice(129,None)],norm):
                    view=x[...,segment];mask=np.abs(view)>1e-12;view[mask]=(view[mask]-stat['mean'])/stat['std']
                scores={'old_hat':np.array(old['prediction'])}
                with torch.no_grad():
                    for name,model in models.items():scores[name]=np.concatenate([torch.sigmoid(model(torch.from_numpy(x[b:b+8]).cuda())[:,0]).cpu().numpy() for b in range(0,len(x),8)])
                scores['sparse_rank_mean']=np.mean([rankdata(scores[name])/(len(y)+1) for name in ['sparse_baseline','sparse_hat']],axis=0)
                for value in scores.values():
                    if not np.isfinite(value).all():raise ValueError('Nonfinite prediction')
                folder=output/'predictions'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/f'{policy}.npz',y=y,**scores)
                rows.append({'patient':int(subject[-2:]),'outer':outer,'case_id':case['case_id'],'policy':policy,
                  **{name:float(roc_auc_score(y,value)) for name,value in scores.items()}})
        del models;torch.cuda.empty_cache();print('SPARSE_EVALUATED_OUTER',outer,flush=True)
    summary={}
    for policy in POLICIES:
        selected=[r for r in rows if r['policy']==policy];summary[policy]={};base=np.array([r['old_hat'] for r in selected])
        for name in ['old_hat','sparse_baseline','sparse_hat','sparse_rank_mean']:
            values=np.array([r[name] for r in selected]);delta=values-base
            summary[policy][name]={'mean_patient_AUROC':float(values.mean()),'gain_over_old_hat':float(delta.mean()),'paired_95CI':bootstrap_delta(delta),
                'patients_better':int((delta>1e-12).sum()),'patients_worse':int((delta < -1e-12).sum())}
    write_json(output/'fixed_sequence_results.json',{'protocol':protocol,'patients':35,'methods':summary,'rows':rows})
    print(summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    if a.freeze_only:freeze(a.project,a.output);print('Sparse evaluation frozen; no new test predictions opened.')
    else:run(a.project,a.output)
