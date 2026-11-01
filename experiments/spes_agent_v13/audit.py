"""Independent arithmetic, patient lineage and physical-action replay checks."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from scipy.spatial.distance import cdist
from experiments.spes_agent_v3.common import read_json, sha, check_groups
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_agent_v8.crossfit import write_json
from .experiment import class_ratio, fit, calibration


def independent_boundary(score, y):
    z = logit(np.clip(np.asarray(score, dtype=float),1e-12,1-1e-12))
    center = np.median(z); mad = max(float(np.median(np.abs(z-center))),1e-6)
    thresholds = np.unique(z)
    youden = np.array([float((z[y==1]>=t).mean()-(z[y==0]>=t).mean()) for t in thresholds])
    best = float(np.median(thresholds[np.isclose(youden,youden.max(),rtol=0,atol=1e-12)]))
    return (best-center)/mad


def audit(project, output):
    project, output = Path(project), Path(output); p = read_json(output/'protocol.json'); evaluation = read_json(output/'evaluation_protocol.json')
    if any(sha(project/f)!=h for f,h in p['source_sha256'].items()) or evaluation['protocol_sha256']!=sha(output/'protocol.json'):
        raise ValueError('Frozen protocol/source differs')
    models = {}; boundary_checks = 0
    for outer in range(5):
        path = output/f'learners/outer{outer}/parameters.json'; model = read_json(path)
        if sha(path)!=evaluation['parameters_sha256'][str(outer)] or model['protocol_sha256']!=sha(output/'protocol.json'):
            raise ValueError('Frozen model provenance differs')
        jobs = [j for j in p['jobs'] if j['outer']==outer]; expected = {s for j in jobs for s in j['predict']}
        if set(model['fit_patients'])!=expected or len(expected)!=28 or expected&set(jobs[0]['outer_test']):
            raise ValueError('Calibration patient isolation differs')
        outer_job = read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json')['jobs'][outer]
        if model['outer_training_ratio']!=class_ratio(project,outer_job['train']): raise ValueError('Outer training ratio differs')
        prevalence = np.mean([load_array(project/f'vendor/norris2026_hat/data/y_{s}.npz').mean() for s in expected])
        np.testing.assert_allclose(model['development_prevalence'],prevalence,rtol=0,atol=1e-15)
        for j in jobs:
            check_groups(j['train'],j['early_stop'],j['predict'],j['outer_test'])
            source = project/f'outputs/spes_agent_v8/crossfit/outer{outer}_inner{j["inner"]}'
            receipt = read_json(source/'receipt.json')
            if any(receipt[k]!=j[k] for k in ['train','early_stop','predict','outer_test']) or sha(source/'checkpoint.pth')!=p['source_HAT_sha256'][f'outer{outer}_inner{j["inner"]}']:
                raise ValueError('Source model lineage differs')
            for subject in j['predict']:
                bank = project/f'outputs/spes_agent_v12/banks/outer{outer}/{subject}.npz'
                if sha(bank)!=p['bank_sha256'][f'outer{outer}/{subject}']: raise ValueError('Source bank differs')
                with np.load(bank) as d:
                    for k in p['states']:
                        mask = d['state']==k
                        value = independent_boundary(d['HAT_score'][mask],d['label'][mask])
                        np.testing.assert_allclose(model['states'][str(k)]['source_patient_offsets'][subject],value,rtol=0,atol=1e-10)
                        boundary_checks += 1
        for k in p['states']:
            np.testing.assert_allclose(model['states'][str(k)]['boundary_n'],np.median(list(model['states'][str(k)]['source_patient_offsets'].values())),rtol=0,atol=1e-12)
        models[outer] = model
    # Refit into a separate audit folder; never overwrite the original frozen models.
    replay = output/'audit_replay'; write_json(replay/'protocol.json',p); fit(project,replay)
    for outer in range(5):
        if read_json(replay/f'learners/outer{outer}/parameters.json')!=models[outer]: raise ValueError('Exact calibrator refit differs')
    calibration(project,replay)
    if read_json(replay/'calibration_results.json')!=read_json(output/'calibration_results.json'): raise ValueError('Calibration replay differs')
    brier_rows = 0; trajectories = 0; decisions = 0
    calibration_rows = read_json(output/'calibration_results.json')['rows']
    for c in p['cases']:
        subject, outer = c['subject'], c['outer']; model = models[outer]
        if subject in model['fit_patients'] or subject not in model['outer_test']: raise ValueError('Held-out subject differs')
        labels = load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
        folder = output/'calibration'/c['case_id']; receipt = read_json(folder/'receipt.json')
        if sha(folder/'predictions.npz')!=receipt['predictions_sha256'] or receipt['evaluation_protocol_sha256']!=sha(output/'evaluation_protocol.json'):
            raise ValueError('Calibration prediction provenance differs')
        with np.load(folder/'predictions.npz') as d:
            np.testing.assert_array_equal(d['label'],labels)
            for row in [r for r in calibration_rows if r['case_id']==c['case_id']]:
                scores = d[f'{row["measurements"]}/{row["scorer"]}']
                np.testing.assert_allclose(np.mean((labels-scores)**2),row['Brier'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(independent_auc(labels,scores),row['AUROC'],rtol=0,atol=1e-12); brier_rows += 1
        source = project/'outputs/spes_physical/maps'/subject; mapping = read_json(source/'receipt.json'); index = load_array(source/'index.npz')
        if sha(source/'receipt.json')!=p['mapping_receipt_sha256'][subject] or sha(source/'index.npz')!=mapping['index_sha256']: raise ValueError('Physical map changed')
        centers = np.array([r['center_mm'] for r in mapping['pair_rows']]); coords = load_array(project/f'vendor/norris2026_hat/data/coords_{subject}.npz')
        kernel = np.where(index>=0,np.exp(-cdist(coords,centers)/40),0.)
        available = [q for q,r in enumerate(mapping['pair_rows']) if (index[:,q]>=0).any() and len(r['recorded_conditions'])==1]
        prefix = fixed_queries(available,outer,int(subject[-2:]),0)[:4]
        for policy in p['policies']:
            folder = output/'acquisition'/c['case_id']/policy; result = read_json(folder/'metrics.json'); queries = [int(q[1:]) for q in result['queries']]
            if result['evaluation_protocol_sha256']!=sha(output/'evaluation_protocol.json') or queries[:4]!=prefix or len(set(queries))!=8 or any(q not in available for q in queries):
                raise ValueError('Query lineage/budget differs')
            np.testing.assert_allclose(independent_auc(labels,result['prediction']),result['AUROC'],rtol=0,atol=1e-12)
            for k,q in enumerate(queries):
                pair = mapping['pair_rows'][q]; obs = read_json(folder/f'{k:02d}_observation.json')
                if obs['anatomical_pair']!=pair['id'] or obs['contacts']!=pair['contacts'] or result['query_contacts'][k]!=pair['contacts']:
                    raise ValueError('Physical identity differs')
                if obs['revealed_recording_contacts']!=int((index[:,q]>=0).sum()): raise ValueError('Recording availability differs')
                if k<4: continue
                trace = read_json(folder/f'{k:02d}_choice.json'); matrix = np.array(trace['scores']); z = logit(np.clip(matrix,1e-12,1-1e-12))
                if policy=='prior_uncertainty':
                    corrected = expit(z-2*np.log(model['outer_training_ratio']))
                    weights = .05+corrected.std(axis=1)+.1*(1-2*np.abs(corrected[:,2]-.5))
                else:
                    hat = z[:,2]; center = np.median(hat); mad = max(float(np.median(np.abs(hat-center))),1e-6)
                    target = 0. if policy=='zero_boundary' else center+model['states'][str(k)]['boundary_n']*mad
                    weights = .05+np.exp(-np.abs((hat-target)/mad))
                current = kernel[:,queries[:k]].max(axis=1)
                expected = [{'id':f'p{v:03d}','gain':float(np.average(np.maximum(kernel[:,v]-current,0),weights=weights))} for v in available if v not in queries[:k]]
                if [r['id'] for r in expected]!=[r['id'] for r in trace['candidates']]: raise ValueError('Candidate identity differs')
                np.testing.assert_allclose([r['gain'] for r in expected],[r['gain'] for r in trace['candidates']],rtol=0,atol=1e-12)
                choice = min(expected,key=lambda r:(-r['gain'],r['id']))['id']
                if trace['choice']!=choice or pair['id']!=choice or trace['selected']!=result['queries'][:k]: raise ValueError('Acquisition decision differs')
                decisions += 1
            trajectories += 1
    value = {'status':'verified', 'calibrators_refit':75, 'independent_patient_boundary_checks':boundary_checks,
             'calibration_scorer_state_results':brier_rows, 'physical_trajectories':trajectories, 'physical_queries':8*trajectories,
             'adaptive_choices_reconstructed':decisions, 'new_API_calls':0,
             'checks':['Frozen source, model, bank, mapping and evaluation hashes; patient/source-model isolation.',
                       'All calibration parameters exactly refitted; source boundaries independently enumerated.',
                       'Every calibration AUROC/Brier and physical trajectory AUROC independently recomputed.',
                       'Every physical query identity, common prefix, availability and numerical choice replayed.'],
             'limits':['Source and runtime neural scores were checked from stored predictions, not all re-inferred.',
                       'Same-cohort adaptation and retrospective availability remain; no clinical or collaboration validation.']}
    write_json(output/'audit.json',value); print('V13_AUDIT',value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); audit(args.project,args.output)
