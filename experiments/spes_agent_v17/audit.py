"""Centered-loss equivalence and observed-only physical ranking replay."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_physical.mapping import physical_input
from experiments.spes_agent_v14.audit_features import audit_table
from experiments.spes_agent_v14.tools import table as original_table
from .experiment import fitting_data,verified
from .tools import VIEWS


def independent_expand(z,columns):
    context=[j for j in columns if j<10]; candidate=[j for j in columns if j>=10]
    return np.column_stack([z[:,j] for j in columns]+[z[:,i]*z[:,j] for i in context for j in candidate])


def audit(project,output):
    project,output=Path(project),Path(output); p=verified(project,output); evaluation=read_json(output/'evaluation_protocol.json')
    if evaluation['protocol_sha256']!=sha(output/'protocol.json'): raise ValueError('Evaluation protocol changed')
    models={}; pair_equivalence_checks=0
    for outer in range(5):
        path=output/f'learners/outer{outer}/models.joblib'; receipt=read_json(path.with_suffix('.json'))
        if sha(path)!=receipt['models_sha256'] or sha(path)!=evaluation['models_sha256'][str(outer)] or receipt['protocol_sha256']!=sha(output/'protocol.json'): raise ValueError('Model lineage changed')
        x,y,groups,subjects,hashes,test=fitting_data(project,output,outer)
        if receipt['fit_patients']!=subjects or receipt['source_bank_sha256']!=hashes or receipt['outer_test']!=test: raise ValueError('Fitting provenance changed')
        model=joblib.load(path); scaler=StandardScaler().fit(x)
        np.testing.assert_allclose(model['scaler'].mean_,scaler.mean_,rtol=0,atol=1e-12); np.testing.assert_allclose(model['scaler'].scale_,scaler.scale_,rtol=0,atol=1e-12)
        z=scaler.transform(x); keys=list(dict.fromkeys(groups))
        for name,columns in VIEWS.items():
            design=independent_expand(z,columns); xc=design.copy(); yc=y.copy()
            for key in keys:
                mask=np.array([k==key for k in groups])
                if mask.sum()!=6: raise ValueError('Wrong comparison group size')
                xc[mask]-=xc[mask].mean(axis=0); yc[mask]-=yc[mask].mean()
            # 0.4*sum_{i<j}(residual_i-residual_j)^2 = 2.4*sum_i(centered residual_i)^2.
            independent=Ridge(alpha=100.,fit_intercept=False).fit(xc,yc,sample_weight=np.full(len(y),2.4))
            np.testing.assert_allclose(independent.coef_,model['models'][name].coef_,rtol=0,atol=1e-9); pair_equivalence_checks+=1
        point=Ridge(alpha=100.,fit_intercept=True).fit(independent_expand(z,VIEWS['monolithic']),y)
        np.testing.assert_allclose(point.coef_,model['models']['point'].coef_,rtol=0,atol=1e-10)
        np.testing.assert_allclose(point.intercept_,model['models']['point'].intercept_,rtol=0,atol=1e-10); models[outer]=model
    trajectories=choices=original_comparisons=0; source_rows=0
    for c in p['cases']:
        map_path=project/'outputs/spes_physical/maps'/c['subject']/'receipt.json'; mapping=read_json(map_path); index=load_array(map_path.with_name('index.npz'))
        raw_path=project/f'vendor/norris2026_hat/data/X_{c["subject"]}.npz'
        if sha(map_path)!=p['mapping_receipt_sha256'][c['subject']] or sha(map_path.with_name('index.npz'))!=mapping['index_sha256'] or sha(raw_path)!=mapping['raw_sha256']: raise ValueError('Physical source changed')
        raw=load_array(raw_path)[:,:20]; labels=load_array(project/f'vendor/norris2026_hat/data/y_{c["subject"]}.npz')
        coords=load_array(project/f'vendor/norris2026_hat/data/coords_{c["subject"]}.npz'); centers=np.array([r['center_mm'] for r in mapping['pair_rows']]); distance=np.where(index>=0,cdist(coords,centers),0.)
        available=[q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]; prefix=fixed_queries(available,c['outer'],int(c['subject'][-2:]),0)[:4]
        neural=read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json')['jobs'][c['outer']]
        if c['subject'] not in neural['predict'] or c['subject'] in neural['train']+neural['early_stop']: raise ValueError('Final scorer patient leak')
        for family in ['baseline','mte','hat']:
            if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{c["outer"]}/{family}/checkpoint.pth')!=p['model_sha256'][f'{c["outer"]}/{family}']: raise ValueError('Final scorer changed')
        for policy in p['policies']:
            folder=output/'evaluation'/c['case_id']/policy; result=read_json(folder/'metrics.json'); queries=[int(q[1:]) for q in result['queries']]
            if result['evaluation_protocol_sha256']!=sha(output/'evaluation_protocol.json') or queries[:4]!=prefix or len(set(queries))!=8 or any(q not in available for q in queries) or result['new_API_calls']!=0: raise ValueError('Physical budget/protocol differs')
            np.testing.assert_allclose(roc_auc_score(labels,result['prediction']),result['AUROC'],rtol=0,atol=1e-12)
            for k,q in enumerate(queries):
                pair=mapping['pair_rows'][q]; obs=read_json(folder/f'{k:02d}_observation.json')
                if result['query_contacts'][k]!=pair['contacts'] or obs['contacts']!=pair['contacts'] or obs['anatomical_pair']!=pair['id'] or obs['revealed_recording_contacts']!=int((index[:,q]>=0).sum()): raise ValueError('Physical identity differs')
                if k<4: continue
                trace=read_json(folder/f'{k:02d}_choice.json'); remaining=[v for v in available if v not in queries[:k]]; ids=[mapping['pair_rows'][v]['id'] for v in remaining]
                if trace['ids']!=ids or trace['selected']!=result['queries'][:k]: raise ValueError('Observed history differs')
                observed=physical_input(raw,index,queries[:k]); score=np.array(trace['current_HAT_score']); x=audit_table(distance,observed,score,queries[:k],remaining)
                np.testing.assert_allclose(x,trace['features'],rtol=0,atol=1e-12); source_rows+=len(x)
                if policy==p['policies'][0] and k==4:
                    independent=original_table(distance,observed,score,queries[:k],remaining); np.testing.assert_allclose(x,independent,rtol=0,atol=1e-12); original_comparisons+=1
                model=models[c['outer']]; z=model['scaler'].transform(x)
                values={name:model['models'][name].predict(independent_expand(z,columns)) for name,columns in VIEWS.items()}
                values['pair_monolithic']=values['monolithic']; values['point_monolithic']=model['models']['point'].predict(independent_expand(z,VIEWS['monolithic']))
                values['pair_experts']=sum(rankdata(values[name]) for name in ['geometry','neural','waveform'])/(3*(len(ids)+1))
                for name,value in values.items(): np.testing.assert_allclose(value,trace['values'][name],rtol=0,atol=1e-12)
                # Use the recorded equal-rank aggregate for exact finite-precision ties after numerical equality is verified.
                expected=min(zip(ids,trace['values'][policy]),key=lambda row:(-row[1],row[0]))[0]
                if trace['choice']!=expected or pair['id']!=expected: raise ValueError('Fixed numerical decision differs')
                choices+=1
            trajectories+=1
        print('V17_AUDIT_PATIENT',c['case_id'],flush=True)
    if len(list((project/'outputs/spes_agent_v16/trajectories').glob('*/*/*_usage.json')))!=120: raise ValueError('Prior API budget changed')
    r={'status':'verified','pair_models_verified_by_centered_loss':pair_equivalence_checks,'point_models_refitted':5,'physical_trajectories':trajectories,
       'adaptive_choices_replayed':choices,'candidate_feature_rows_reconstructed':source_rows,'states_checked_against_original_uncached_formula':original_comparisons,
       'new_API_calls':0,'prior_v16_API_attempts_unchanged':120,
       'checks':['All source-model/bank/patient exclusions and frozen model/map/raw/protocol hashes.',
                 'All pair-ranker coefficients independently verified using algebraically equivalent centered loss; point control refitted.',
                 'Every observed-only feature table, transformed specialist score, choice, physical identity and AUROC replayed.',
                 'Original uncached feature formula matched one common state for every patient; no test-oracle choices entered fitting.'],
       'limits':['Stored runtime neural scores were not all re-inferred in this audit.','Correlated pair rows, differing parameter counts and repeated-cohort search prevent a general collaboration claim.']}
    write_json(output/'audit.json',r); print(r,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True); a=parser.parse_args(); audit(a.project,a.output)
