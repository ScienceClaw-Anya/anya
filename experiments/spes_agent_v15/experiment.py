"""Fit sparse-input readouts in each encoder's own feature coordinate system."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments.spes_agent_v3.common import read_json,sha,exclusive
from experiments.spes_agent_v4.neural import load_array,upstream_models
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.mapping import physical_input


def design(embedding,k,available_fraction,context):
    embedding=np.asarray(embedding,dtype=float)
    if embedding.ndim!=2 or embedding.shape[1]!=16 or not np.isfinite(embedding).all(): raise ValueError('Expected finite sixteen-dimensional encoder features')
    available_fraction=np.asarray(available_fraction,dtype=float)
    if not isinstance(k,(int,np.integer)) or not 1<=k<=8 or available_fraction.shape!=(len(embedding),) or not np.isfinite(available_fraction).all() or ((available_fraction<0)|(available_fraction>1)).any(): raise ValueError('Invalid physical query count/availability')
    if not context: return embedding.copy()
    return np.column_stack([embedding,embedding*(k/8),np.full(len(embedding),k/8),available_fraction])


def freeze(project,output):
    project,output=Path(project),Path(output); neural=read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json'); physical=read_json(project/'outputs/spes_agent_v11/numeric_protocol.json')
    sources=['experiments/spes_agent_v15/experiment.py','experiments/spes_physical/mapping.py',
             'experiments/spes_agent_v4/neural.py','experiments/spes_agent_v5/neural.py','vendor/norris2026_hat/src/models.py']
    p={'study':'v15 sparse physical readouts on frozen pretrained HAT encoders','scope':'Numerical representation adaptation; no language-model fine-tuning or collaboration result.',
       'jobs':neural['jobs'],'cases':physical['cases'],'training_states':[4,6,8],'training_replicates':[0,1,2],
       'training':'Fit two fixed C=0.1 scaled logistic readouts per outer encoder using only its21 fitting patients. Frozen encoder and original normalization; natural class prevalence, equal patient total weight.',
       'readouts':{'linear':'16 encoder coordinates','context':'16 coordinates, their interactions with query count/8, query count/8, observed trial/contact availability fraction (34features)'},
       'coordinate_rule':'Each encoder has its own separately fitted readouts. Never concatenate unaligned latent coordinates from differently trained neural models.',
       'test':'Same v11 random0 physical path, four/eight groups. All ten readout fits frozen before any new outer evaluation. Original HAT must reproduce audited v12 predictions.',
       'primary':'context readout minus original HAT at eight same physical random0 groups, mean patient SOZ AUROC',
       'secondary':'Both readouts at four/eight groups; Brier versus training21 patient-prevalence constant. No secondary primary swap.',
       'next_gate':'Primary gain>0.01 and positive two-sided95 patient-bootstrap lower bound before rebuilding source tools or running an agent comparison with these readouts.',
       'limits':['Encoder fitting may overfit its own21 patients; readout inputs are in-sample with respect to that encoder, not falsely described as OOF.',
                 'Original encoder early stopping used seven other development patients; no outer patient entered either encoder or readout training.',
                 'Nine input states per patient are correlated, not independent participants.','Repeated cohort exploration; no clinical efficacy or external validation.'],
       'new_neural_fits':0,'new_small_fits':10,'new_API_calls':0,'source_sha256':{f:sha(project/f) for f in sources},
       'mapping_receipt_sha256':physical['mapping_receipt_sha256'],'encoder_sha256':{str(i):physical['model_sha256'][f'{i}/hat'] for i in range(5)},
       'v12_reference_sha256':{c['case_id']:sha(project/'outputs/spes_agent_v12/evaluation'/c['case_id']/'predictions.npz') for c in physical['cases']}}
    path=output/'protocol.json'
    if path.exists() and read_json(path)!=p: raise ValueError('Frozen sparse-readout protocol differs')
    write_json(path,p); print('V15_PROTOCOL_FROZEN',sha(path),flush=True)


def verified(project,output):
    p=read_json(Path(output)/'protocol.json')
    if any(sha(Path(project)/f)!=h for f,h in p['source_sha256'].items()): raise ValueError('Frozen source differs')
    return p


def encoder(project,outer,p):
    import torch
    project=Path(project); source=project/f'outputs/spes_agent_v4/multitrial/neural/outer{outer}/hat'; receipt=read_json(source/'receipt.json')
    if sha(source/'checkpoint.pth')!=p['encoder_sha256'][str(outer)] or receipt['checkpoint_sha256']!=p['encoder_sha256'][str(outer)]: raise ValueError('Frozen encoder changed')
    job=p['jobs'][outer]
    if any(receipt[k]!=job[k] for k in ['train','early_stop','predict']): raise ValueError('Encoder split differs')
    model,_,_=upstream_models(project).create_model('hat'); model.load_state_dict(torch.load(source/'checkpoint.pth',map_location='cpu',weights_only=True)['model_state_dict'])
    return model.cuda().eval(),receipt


def extract(model,observed,stats):
    import torch
    x=observed.copy()
    for segment,stat in zip([slice(0,1),slice(1,129),slice(129,None)],stats):
        view=x[...,segment]; mask=np.abs(view)>1e-12; view[mask]=(view[mask]-stat['mean'])/stat['std']
    saved=[]
    hook=model.fc.register_forward_pre_hook(lambda module,args:saved.append(args[0].detach().cpu().numpy().copy()))
    try:
        with torch.no_grad(): score=np.concatenate([torch.sigmoid(model(torch.from_numpy(x[b:b+8]).cuda())[:,0]).cpu().numpy() for b in range(0,len(x),8)])
    finally: hook.remove()
    embedding=np.vstack(saved); w=model.fc[1].weight.detach().cpu().numpy()[0]; b=float(model.fc[1].bias.detach().cpu().numpy()[0])
    np.testing.assert_allclose(expit(embedding.astype(float)@w+b),score,rtol=0,atol=5e-7)
    fraction=(observed[...,0]>1e-12).mean(axis=(1,2))
    return embedding,score,fraction,w,b


def map_data(project,p,subject):
    project=Path(project); folder=project/'outputs/spes_physical/maps'/subject; receipt=read_json(folder/'receipt.json'); path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
    if sha(folder/'receipt.json')!=p['mapping_receipt_sha256'][subject] or sha(folder/'index.npz')!=receipt['index_sha256'] or sha(path)!=receipt['raw_sha256']: raise ValueError('Physical map/raw changed')
    index=load_array(folder/'index.npz'); raw=load_array(path)[:,:20]; available=[q for q,r in enumerate(receipt['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
    return raw,index,available,receipt


def subject_banks(project,output,subject):
    import torch
    project,output=Path(project),Path(output); p=verified(project,output); torch.set_num_threads(2); jobs=[j for j in p['jobs'] if subject in j['train']]
    if all((output/f'banks/outer{j["outer"]}/{subject}.json').exists() for j in jobs): return
    raw,index,available,mapping=map_data(project,p,subject); y=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
    for j in jobs:
        path=output/f'banks/outer{j["outer"]}/{subject}.npz'
        if path.with_suffix('.json').exists(): continue
        if subject in j['early_stop']+j['predict']: raise ValueError('Readout training patient leak')
        model,receipt=encoder(project,j['outer'],p); rows=[]; queries={}
        for rep in p['training_replicates']:
            sequence=fixed_queries(available,j['outer'],int(subject[-2:]),rep); queries[str(rep)]=sequence
            for k in p['training_states']:
                observed=physical_input(raw,index,sequence[:k]); embedding,score,fraction,w,b=extract(model,observed,receipt['normalisation'])
                rows.append({'embedding':embedding,'original_score':score,'availability':fraction,'state':np.full(len(y),k),'replicate':np.full(len(y),rep),'label':y})
        path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**{key:np.concatenate([r[key] for r in rows]) for key in rows[0]})
        write_json(path.with_suffix('.json'),{'outer':j['outer'],'subject':subject,'rows':9*len(y),'queries':queries,'source_action_kind':'verified_anatomical_pairs',
                   'file_sha256':sha(path),'protocol_sha256':sha(output/'protocol.json'),'encoder_sha256':p['encoder_sha256'][str(j['outer'])],
                   'mapping_receipt_sha256':p['mapping_receipt_sha256'][subject],'original_head_weight':w.tolist(),'original_head_bias':b})
        del model
    print('V15_SPARSE_EMBEDDINGS',subject,len(jobs),flush=True)


def banks(project,output,workers):
    p=verified(project,output); subjects=sorted({s for j in p['jobs'] for s in j['train']})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(subject_banks,project,output,s) for s in subjects]
        for f in as_completed(futures): f.result()


def fit(project,output):
    project,output=Path(project),Path(output); p=verified(project,output)
    for j in p['jobs']:
        path=output/f'learners/outer{j["outer"]}/models.joblib'
        if path.with_suffix('.json').exists(): continue
        chunks=[]; labels=[]; patients=[]; context=[]; hashes={}; prevalence=[]
        for subject in j['train']:
            bank=output/f'banks/outer{j["outer"]}/{subject}.npz'; receipt=read_json(bank.with_suffix('.json'))
            if sha(bank)!=receipt['file_sha256'] or receipt['protocol_sha256']!=sha(output/'protocol.json') or receipt['encoder_sha256']!=p['encoder_sha256'][str(j['outer'])]: raise ValueError('Readout bank changed')
            with np.load(bank) as d:
                chunks.append(d['embedding']); labels.append(d['label']); patients.append(np.full(len(d['label']),int(subject[-2:])))
                context.append(np.vstack([design(d['embedding'][i:i+1],int(d['state'][i]),d['availability'][i:i+1],True) for i in range(len(d['label']))])); prevalence.append(float(d['label'].mean()))
            hashes[subject]=sha(bank)
        x,y,patient=np.vstack(chunks),np.concatenate(labels),np.concatenate(patients); xc=np.vstack(context)
        if len(set(patient))!=21 or set(j['train'])&set(j['early_stop']+j['predict']): raise ValueError('Readout patient isolation differs')
        weight=np.array([len(y)/(21*np.sum(patient==s)) for s in patient]); models={}
        for name,features in [('linear',x),('context',xc)]:
            learner=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=1000))])
            models[name]=learner.fit(features,y,scale__sample_weight=weight,model__sample_weight=weight)
        path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(models,path)
        write_json(path.with_suffix('.json'),{'outer':j['outer'],'fit_patients':j['train'],'early_stop':j['early_stop'],'outer_test':j['predict'],
                   'rows':len(y),'development_prevalence':float(np.mean(prevalence)),'models_sha256':sha(path),'bank_sha256':hashes,'protocol_sha256':sha(output/'protocol.json')})
        print('V15_READOUTS_FIT',j['outer'],len(y),flush=True)
    frozen={'protocol_sha256':sha(output/'protocol.json'),'models_sha256':{str(i):sha(output/f'learners/outer{i}/models.joblib') for i in range(5)}}; path=output/'evaluation_protocol.json'
    if path.exists() and read_json(path)!=frozen: raise ValueError('Frozen readout models differ')
    write_json(path,frozen)


def evaluate(project,output):
    import torch
    project,output=Path(project),Path(output); p=verified(project,output); frozen=read_json(output/'evaluation_protocol.json'); torch.set_num_threads(2)
    if frozen['protocol_sha256']!=sha(output/'protocol.json'): raise ValueError('Evaluation protocol changed')
    rows=[]
    for j in p['jobs']:
        model,receipt=encoder(project,j['outer'],p); path=output/f'learners/outer{j["outer"]}/models.joblib'; fitted=read_json(path.with_suffix('.json'))
        if sha(path)!=frozen['models_sha256'][str(j['outer'])] or fitted['fit_patients']!=j['train']: raise ValueError('Readout changed')
        learners=joblib.load(path)
        for subject in j['predict']:
            c=next(c for c in p['cases'] if c['subject']==subject); folder=output/'evaluation'/c['case_id']
            if (folder/'metrics.json').exists(): rows+=read_json(folder/'metrics.json')['rows']; continue
            raw,index,available,mapping=map_data(project,p,subject); sequence=fixed_queries(available,j['outer'],int(subject[-2:]),0); y=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
            predictions={'label':y}; saved={}; case_rows=[]
            reference=project/'outputs/spes_agent_v12/evaluation'/c['case_id']/'predictions.npz'
            if sha(reference)!=p['v12_reference_sha256'][c['case_id']]: raise ValueError('Frozen physical reference changed')
            with np.load(reference) as d: old={k:d[f'{k}/hat'] for k in [4,8]}
            for k in [4,8]:
                embedding,score,fraction,w,b=extract(model,physical_input(raw,index,sequence[:k]),receipt['normalisation']); np.testing.assert_allclose(score,old[k],rtol=0,atol=1e-7)
                values={'hat':score,'constant':np.full(len(y),fitted['development_prevalence'])}
                for name in ['linear','context']: values[name]=learners[name].predict_proba(design(embedding,k,fraction,name=='context'))[:,1]
                saved[f'{k}/embedding']=embedding; saved[f'{k}/availability']=fraction
                for name,v in values.items():
                    predictions[f'{k}/{name}']=v; case_rows.append({'case_id':c['case_id'],'measurements':k,'scorer':name,'AUROC':independent_auc(y,v),'Brier':float(np.mean((y-v)**2))})
            folder.mkdir(parents=True,exist_ok=True); np.savez_compressed(folder/'predictions.npz',**predictions,**saved)
            write_json(folder/'metrics.json',{'rows':case_rows,'queries':[mapping['pair_rows'][q]['id'] for q in sequence],
                       'predictions_sha256':sha(folder/'predictions.npz'),'evaluation_protocol_sha256':sha(output/'evaluation_protocol.json')}); rows+=case_rows
            print('V15_READOUT_TEST',c['case_id'],flush=True)
        del model
    summary={}
    for k in [4,8]:
        base={r['case_id']:r['AUROC'] for r in rows if r['measurements']==k and r['scorer']=='hat'}
        for name in ['hat','constant','linear','context']:
            rr=[r for r in rows if r['measurements']==k and r['scorer']==name]; delta=np.array([r['AUROC']-base[r['case_id']] for r in rr])
            summary[f'{k}/{name}']={'mean_patient_AUROC':float(np.mean([r['AUROC'] for r in rr])),'gain_over_hat':float(delta.mean()),'paired_95CI':bootstrap_delta(delta),'mean_patient_Brier':float(np.mean([r['Brier'] for r in rr]))}
    primary=summary['8/context']; result={'primary':p['primary'],'summary':summary,'rows':rows,'next_gate':primary['gain_over_hat']>.01 and primary['paired_95CI'][0]>0,
              'new_API_calls':0,'new_neural_fits':0,'multi_agent_superiority_established':False}
    write_json(output/'results.json',result); print('V15_RESULT',{k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['freeze','banks','fit','evaluate'],required=True); parser.add_argument('--workers',type=int,default=2); a=parser.parse_args()
    if not 1<=a.workers<=2: raise ValueError('One or two local GPU workers')
    if a.stage=='freeze': freeze(a.project,a.output)
    else:
        with exclusive(a.output/f'{a.stage}.lock'):
            if a.stage=='banks': banks(a.project,a.output,a.workers)
            else: globals()[a.stage](a.project,a.output)
