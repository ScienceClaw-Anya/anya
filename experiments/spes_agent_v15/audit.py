"""Readout provenance, independent context reconstruction and exact refits."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from .experiment import encoder,extract


def independent_design(embedding,k,fraction,context):
    z=np.asarray(embedding,dtype=float)
    if not context: return z.copy()
    count=np.broadcast_to(np.asarray(k,dtype=float)/8,(len(z),))
    return np.c_[z,z*count[:,None],count,np.asarray(fraction)]


def independent_fraction(raw_distance,index,sequence):
    valid=np.zeros((len(index),raw_distance.shape[1],len(sequence)),dtype=bool)
    for j,q in enumerate(sequence):
        rows=np.flatnonzero(index[:,q]>=0); valid[rows,:,j]=raw_distance[rows,:,index[rows,q]]>1e-12
    return valid.mean(axis=(1,2))


def audit(project,output):
    project,output=Path(project),Path(output); p=read_json(output/'protocol.json'); frozen=read_json(output/'evaluation_protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()) or frozen['protocol_sha256']!=sha(output/'protocol.json'): raise ValueError('Frozen protocol/source differs')
    models={}; training_rows=0; bank_count=0; test_rows=0; source_reinferences=0
    for j in p['jobs']:
        path=output/f'learners/outer{j["outer"]}/models.joblib'; receipt=read_json(path.with_suffix('.json'))
        if receipt['fit_patients']!=j['train'] or receipt['early_stop']!=j['early_stop'] or receipt['outer_test']!=j['predict'] or len(set(j['train']))!=21 or set(j['train'])&set(j['early_stop']+j['predict']):
            raise ValueError('Readout patient isolation differs')
        if sha(path)!=frozen['models_sha256'][str(j['outer'])] or receipt['models_sha256']!=sha(path) or receipt['protocol_sha256']!=sha(output/'protocol.json'): raise ValueError('Readout weights changed')
        models[j['outer']]=joblib.load(path); chunks=[]; contexts=[]; labels=[]; patients=[]; prevalence=[]
        for subject in j['train']:
            bank=output/f'banks/outer{j["outer"]}/{subject}.npz'
            if sha(bank)!=receipt['bank_sha256'][subject]: raise ValueError('Readout fitting input differs')
            with np.load(bank) as d:
                chunks.append(d['embedding']); contexts.append(independent_design(d['embedding'],d['state'],d['availability'],True))
                labels.append(d['label']); patients.append(np.full(len(d['label']),int(subject[-2:]))); prevalence.append(float(d['label'].mean()))
        y=np.concatenate(labels); patient=np.concatenate(patients); weight=np.array([len(y)/(21*np.sum(patient==s)) for s in patient])
        for name,x in [('linear',np.vstack(chunks)),('context',np.vstack(contexts))]:
            refit=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.1,max_iter=1000))])
            refit.fit(x,y,scale__sample_weight=weight,model__sample_weight=weight)
            original=models[j['outer']][name]
            for attribute in ['mean_','scale_']: np.testing.assert_allclose(getattr(refit['scale'],attribute),getattr(original['scale'],attribute),rtol=0,atol=1e-10)
            for attribute in ['coef_','intercept_']: np.testing.assert_allclose(getattr(refit['model'],attribute),getattr(original['model'],attribute),rtol=0,atol=1e-9)
        np.testing.assert_allclose(np.mean(prevalence),receipt['development_prevalence'],rtol=0,atol=1e-15)
    for subject in sorted({s for j in p['jobs'] for s in j['train']}):
        source=project/'outputs/spes_physical/maps'/subject; mapping=read_json(source/'receipt.json'); index=load_array(source/'index.npz')
        raw_path=project/f'vendor/norris2026_hat/data/X_{subject}.npz'
        if sha(source/'receipt.json')!=p['mapping_receipt_sha256'][subject] or sha(source/'index.npz')!=mapping['index_sha256'] or sha(raw_path)!=mapping['raw_sha256']: raise ValueError('Physical source/map differs')
        raw=load_array(raw_path)[:,:20]; labels=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
        available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
        for j in [j for j in p['jobs'] if subject in j['train']]:
            bank=output/f'banks/outer{j["outer"]}/{subject}.npz'; receipt=read_json(bank.with_suffix('.json'))
            if receipt['protocol_sha256']!=sha(output/'protocol.json') or sha(bank)!=receipt['file_sha256'] or receipt['encoder_sha256']!=p['encoder_sha256'][str(j['outer'])] or receipt['source_action_kind']!='verified_anatomical_pairs': raise ValueError('Physical representation bank differs')
            with np.load(bank) as d: values={k:d[k] for k in d.files}
            w=np.array(receipt['original_head_weight']); b=receipt['original_head_bias']
            np.testing.assert_allclose(expit(values['embedding'].astype(float)@w+b),values['original_score'],rtol=0,atol=5e-7)
            for rep in p['training_replicates']:
                sequence=fixed_queries(available,j['outer'],int(subject[-2:]),rep)
                if receipt['queries'][str(rep)]!=sequence: raise ValueError('Physical training subset differs')
                for k in p['training_states']:
                    mask=(values['replicate']==rep)&(values['state']==k)
                    np.testing.assert_array_equal(values['label'][mask],labels)
                    expected=independent_fraction(raw[...,0],index,sequence[:k])
                    np.testing.assert_allclose(values['availability'][mask],expected,rtol=0,atol=1e-15); training_rows+=int(mask.sum())
            # A representative real encoder inference per fold checks the hook.
            if subject==j['train'][0]:
                from experiments.spes_physical.mapping import physical_input
                model,neural=encoder(project,j['outer'],p); sequence=fixed_queries(available,j['outer'],int(subject[-2:]),0)
                z,score,_,_,_=extract(model,physical_input(raw,index,sequence[:4]),neural['normalisation']); mask=(values['replicate']==0)&(values['state']==4)
                np.testing.assert_allclose(z,values['embedding'][mask],rtol=0,atol=1e-6)
                np.testing.assert_allclose(score,values['original_score'][mask],rtol=0,atol=1e-7); source_reinferences+=1; del model
            bank_count+=1
        case=next(c for c in p['cases'] if c['subject']==subject); j=p['jobs'][case['outer']]
        if subject not in j['predict'] or subject in j['train']+j['early_stop']: raise ValueError('Held-out encoder patient differs')
        folder=output/'evaluation'/case['case_id']; metrics=read_json(folder/'metrics.json')
        if sha(folder/'predictions.npz')!=metrics['predictions_sha256'] or metrics['evaluation_protocol_sha256']!=sha(output/'evaluation_protocol.json'): raise ValueError('Test prediction provenance differs')
        sequence=fixed_queries(available,case['outer'],int(subject[-2:]),0)
        if metrics['queries']!=[mapping['pair_rows'][q]['id'] for q in sequence]: raise ValueError('Test physical queries differ')
        reference=project/'outputs/spes_agent_v12/evaluation'/case['case_id']/'predictions.npz'
        if sha(reference)!=p['v12_reference_sha256'][case['case_id']]: raise ValueError('Physical scorer reference changed')
        with np.load(folder/'predictions.npz') as d, np.load(reference) as old:
            np.testing.assert_array_equal(d['label'],labels)
            for k in [4,8]:
                fraction=independent_fraction(raw[...,0],index,sequence[:k]); np.testing.assert_allclose(fraction,d[f'{k}/availability'],rtol=0,atol=1e-15)
                np.testing.assert_allclose(d[f'{k}/hat'],old[f'{k}/hat'],rtol=0,atol=1e-7)
                for name in ['linear','context']:
                    x=independent_design(d[f'{k}/embedding'],k,fraction,name=='context')
                    np.testing.assert_allclose(models[case['outer']][name].predict_proba(x)[:,1],d[f'{k}/{name}'],rtol=0,atol=1e-12)
            for row in metrics['rows']:
                score=d[f'{row["measurements"]}/{row["scorer"]}']
                np.testing.assert_allclose(roc_auc_score(labels,score),row['AUROC'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(np.mean((labels-score)**2),row['Brier'],rtol=0,atol=1e-12); test_rows+=1
        print('V15_AUDIT_PATIENT',subject,flush=True)
    value={'status':'verified','training_banks':bank_count,'training_contact_state_rows':training_rows,'readouts_refitted':10,
           'representative_source_encoder_reinferences':source_reinferences,'heldout_scorer_state_results':test_rows,'new_API_calls':0,
           'checks':['All patient/encoder/map/raw/bank/protocol provenance; per-encoder fitting coordinate systems.',
                     'Physical subsets and trial availability independently reconstructed; original linear head reconstruction for every stored fitting embedding.',
                     'All ten readouts refitted, independent context features and held-out readout predictions replayed.',
                     'Original HAT matches frozen v12; every held-out AUROC and Brier independently recomputed.'],
           'limits':['Five representative source encoder states re-inferred; remaining stored neural features were not all re-inferred.',
                     'Readout features are intentionally in-sample for their frozen encoder; outer testing remains isolated and same-cohort exploratory.']}
    write_json(output/'audit.json',value); print('V15_AUDIT',value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.project,args.output)
