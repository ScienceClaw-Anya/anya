"""Fresh, frozen calibration and physical-query study; no old study is resumed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.special import expit
from experiments.spes_agent_v3.common import read_json, sha, check_groups, exclusive
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v4.screen import bootstrap_delta
from experiments.spes_agent_v5.neural import fixed_queries
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v8.audit import independent_auc
from experiments.spes_physical.environment import PhysicalAcquisition
from .tools import RECIPES, POLICIES, features, fit_platt, calibrate, logits, boundary_offset, acquisition_table, choose


def class_ratio(project, subjects):
    y = np.concatenate([load_array(Path(project)/f'vendor/norris2026_hat/data/y_{s}.npz') for s in subjects])
    if not 0 < y.sum() < len(y): raise ValueError('Both training classes required')
    return float((len(y)-y.sum())/y.sum())


def freeze(project, output):
    project, output = Path(project), Path(output)
    parent = read_json(project/'outputs/spes_agent_v12/protocol.json')
    if read_json(project/'outputs/spes_agent_v12/audit.json')['status'] != 'verified':
        raise ValueError('Audited physical banks required')
    files = ['experiments/spes_agent_v13/experiment.py', 'experiments/spes_agent_v13/tools.py',
             'experiments/spes_physical/environment.py', 'experiments/spes_physical/mapping.py',
             'experiments/spes_agent_v4/neural.py', 'experiments/spes_agent_v5/neural.py']
    bank_hashes = {f'outer{j["outer"]}/{s}': sha(project/f'outputs/spes_agent_v12/banks/outer{j["outer"]}/{s}.npz')
                   for j in parent['jobs'] for s in j['predict']}
    p = {'study': 'v13 training-only calibration and patient-adaptive physical acquisition',
         'scope': 'Exploratory numerical feasibility on a repeatedly inspected cohort, not a collaboration or clinical trial.',
         'jobs': parent['jobs'], 'cases': parent['cases'], 'states': [4,5,6,7,8], 'policies': POLICIES,
         'calibration_recipes': RECIPES, 'calibration_primary': 'centered_platt minus development-prevalence constant, mean patient Brier at eight random0 physical pair groups; lower is better',
         'calibration_gate': 'Mean Brier gain >0.005 with positive two-sided95 bootstrap lower bound against constant. Better Brier alone cannot pass acquisition gate.',
         'acquisition_primary': 'learned_boundary minus frozen v11 score_focus, mean patient SOZ AUROC at eight physical pair groups',
         'acquisition_gate': 'Mean acquisition gain >0.01 and positive two-sided95 patient-bootstrap lower bound; required before a new paid agent comparison for this hypothesis.',
         'secondary': 'All raw/analytic/three Platt/constant scores at 4 and 8 groups; three new query policies versus all fixed v11 controls. No primary swap.',
         'training': 'Only twenty existing source-held-out HATs and audited physical v12 banks. Per state and fold, train three 1D C=1 sigmoids, natural class prevalence, equal patient weights; nonpositive slope falls back to development-prevalence constant.',
         'boundary_training': 'Median development-patient Youden logit threshold expressed in patient median/MAD units; ties use median optimal finite threshold. New acquisition adaptation of preprint S3, not a paper-reported active policy.',
         'prior_hypothesis': 'Under ideal class/patient sampling and weighted BCE, raw odds scale by training class ratio squared. Subtract 2log(r) as a hypothesis; not guaranteed probability calibration.',
         'runtime': 'Same common random0 first four groups, four adaptive choices, unchanged own outer HAT final ranking. Threshold and class ratios fixed without test labels. Boundary weights .05+exp(-abs(logit distance)/MAD); matched zero-logit control has same form.',
         'measurement_budget': 8, 'new_API_calls': 0, 'new_neural_fits': 0, 'new_small_fits': 75,
         'query_unit': 'Anatomical pair response group, up to20 retained trials; not a single pulse or equal charge/current/time.',
         'source_sha256': {f: sha(project/f) for f in files}, 'bank_sha256': bank_hashes,
         'mapping_receipt_sha256': parent['mapping_receipt_sha256'], 'model_sha256': parent['model_sha256'],
         'source_HAT_sha256': parent['source_HAT_sha256'],
         'v11_results_sha256': sha(project/'outputs/spes_agent_v11/numeric_results.json'),
         'v12_protocol_sha256': sha(project/'outputs/spes_agent_v12/protocol.json'),
         'sources': 'research/spes-v13/claims.json; official HAT preprint supplement S3/S6 inspected, not clinical validation.'}
    path = output/'protocol.json'
    if path.exists() and read_json(path) != p: raise ValueError('Frozen v13 protocol differs')
    write_json(path, p)
    print('V13_PROTOCOL_FROZEN', sha(path), flush=True)


def verified(project, output):
    project, output = Path(project), Path(output)
    p = read_json(output/'protocol.json')
    if any(sha(project/f) != h for f,h in p['source_sha256'].items()): raise ValueError('Frozen source differs')
    return p


def fit(project, output):
    project, output = Path(project), Path(output); p = verified(project, output)
    outer_jobs = read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json')['jobs']
    for outer in range(5):
        path = output/f'learners/outer{outer}/parameters.json'
        if path.exists(): continue
        jobs = [j for j in p['jobs'] if j['outer'] == outer]
        development = sorted({s for j in jobs for s in j['predict']})
        if len(development) != 28 or set(development)&set(jobs[0]['outer_test']): raise ValueError('Development patients differ')
        prevalence = float(np.mean([load_array(project/f'vendor/norris2026_hat/data/y_{s}.npz').mean() for s in development]))
        banks = []
        for j in jobs:
            check_groups(j['train'], j['early_stop'], j['predict'], j['outer_test'])
            ratio = class_ratio(project, j['train'])
            for subject in j['predict']:
                bank = project/f'outputs/spes_agent_v12/banks/outer{outer}/{subject}.npz'
                if sha(bank) != p['bank_sha256'][f'outer{outer}/{subject}']: raise ValueError('Source physical bank changed')
                receipt = read_json(bank.with_suffix('.json'))
                if receipt['source_action_kind'] != 'verified_anatomical_pairs' or receipt['source_checkpoint_sha256'] != p['source_HAT_sha256'][f'outer{outer}_inner{j["inner"]}']:
                    raise ValueError('Source physical bank lineage differs')
                with np.load(bank) as d: banks.append((subject, ratio, d['state'], d['HAT_score'], d['label']))
        states = {}
        for k in p['states']:
            source = [(s, r, score[state==k], y[state==k]) for s,r,state,score,y in banks]
            offsets = [boundary_offset(score, y) for _,_,score,y in source]
            nrows = sum(len(y) for _,_,_,y in source)
            weights = np.concatenate([np.full(len(y), nrows/(28*len(y))) for _,_,_,y in source])
            labels = np.concatenate([y for _,_,_,y in source])
            models = {recipe: fit_platt(np.concatenate([features(score, recipe, r) for _,r,score,_ in source]), labels, weights, prevalence) for recipe in RECIPES}
            states[str(k)] = {'platt': models, 'boundary_n': float(np.median(offsets)),
                              'source_patient_offsets': dict(zip([s for s,_,_,_ in source], offsets)), 'source_rows': nrows}
        value = {'outer': outer, 'fit_patients': development, 'outer_test': jobs[0]['outer_test'],
                 'outer_training_ratio': class_ratio(project, outer_jobs[outer]['train']), 'development_prevalence': prevalence,
                 'states': states, 'protocol_sha256': sha(output/'protocol.json'),
                 'bank_sha256': {k:h for k,h in p['bank_sha256'].items() if k.startswith(f'outer{outer}/')}}
        write_json(path, value); print('V13_CALIBRATORS_FIT', outer, flush=True)
    frozen = {'protocol_sha256': sha(output/'protocol.json'), 'parameters_sha256': {str(i):sha(output/f'learners/outer{i}/parameters.json') for i in range(5)}}
    path = output/'evaluation_protocol.json'
    if path.exists() and read_json(path) != frozen: raise ValueError('Frozen calibration parameters differ')
    write_json(path, frozen)


def parameters(output, outer):
    output = Path(output); frozen = read_json(output/'evaluation_protocol.json'); path = output/f'learners/outer{outer}/parameters.json'
    if frozen['protocol_sha256'] != sha(output/'protocol.json') or sha(path) != frozen['parameters_sha256'][str(outer)]: raise ValueError('Frozen calibration models differ')
    return read_json(path)


def calibration(project, output):
    project, output = Path(project), Path(output); p = verified(project, output); rows = []
    for c in p['cases']:
        model = parameters(output, c['outer'])
        if c['subject'] in model['fit_patients'] or c['subject'] not in model['outer_test']: raise ValueError('Calibration patient leak')
        folder = output/'calibration'/c['case_id']; folder.mkdir(parents=True, exist_ok=True)
        with np.load(project/'outputs/spes_agent_v12/evaluation'/c['case_id']/'predictions.npz') as d:
            y = d['label']; predictions = {'label': y}
            for k in [4,8]:
                raw = d[f'{k}/hat']; ratio = model['outer_training_ratio']
                values = {'raw': raw, 'analytic_prior': expit(logits(raw)-2*np.log(ratio)),
                          'constant': np.full(len(raw), model['development_prevalence']),
                          **{r:calibrate(raw, r, ratio, model['states'][str(k)]['platt'][r]) for r in RECIPES}}
                for name, score in values.items():
                    predictions[f'{k}/{name}'] = score
                    rows.append({'case_id':c['case_id'], 'measurements':k, 'scorer':name, 'AUROC':independent_auc(y, score),
                                 'Brier':float(np.mean((y-score)**2)), 'mean_score':float(score.mean()), 'prevalence':float(y.mean())})
        np.savez_compressed(folder/'predictions.npz', **predictions)
        write_json(folder/'receipt.json', {'evaluation_protocol_sha256':sha(output/'evaluation_protocol.json'), 'predictions_sha256':sha(folder/'predictions.npz')})
    result = {'rows':rows, 'summary':{}}
    for k in [4,8]:
        base = {r['case_id']:r['Brier'] for r in rows if r['measurements']==k and r['scorer']=='constant'}
        for name in ['raw','analytic_prior','constant']+RECIPES:
            rr = [r for r in rows if r['measurements']==k and r['scorer']==name]
            delta = np.array([base[r['case_id']]-r['Brier'] for r in rr])
            result['summary'][f'{k}/{name}'] = {'mean_patient_Brier':float(np.mean([r['Brier'] for r in rr])), 'Brier_gain_over_constant':float(delta.mean()),
                'paired_95CI':bootstrap_delta(delta), 'mean_patient_AUROC':float(np.mean([r['AUROC'] for r in rr])), 'mean_patient_score':float(np.mean([r['mean_score'] for r in rr]))}
    primary = result['summary']['8/centered_platt']
    result['calibration_gate'] = primary['Brier_gain_over_constant']>.005 and primary['paired_95CI'][0]>0
    result['new_API_calls'] = 0
    write_json(output/'calibration_results.json', result)
    print('V13_CALIBRATION', {k:v for k,v in result.items() if k!='rows'}, flush=True)


def run_case(project, output, c, policy):
    import torch
    project, output = Path(project), Path(output); p = verified(project, output); torch.set_num_threads(2)
    model = parameters(output, c['outer']); folder = output/'acquisition'/c['case_id']/policy
    if (folder/'metrics.json').exists(): return
    if c['subject'] in model['fit_patients'] or c['subject'] not in model['outer_test']: raise ValueError('Acquisition patient leak')
    if sha(project/'outputs/spes_physical/maps'/c['subject']/'receipt.json') != p['mapping_receipt_sha256'][c['subject']]: raise ValueError('Physical map changed')
    for f in ['baseline','mte','hat']:
        if sha(project/f'outputs/spes_agent_v4/multitrial/neural/outer{c["outer"]}/{f}/checkpoint.pth') != p['model_sha256'][f'{c["outer"]}/{f}']: raise ValueError('Outer model changed')
    env = PhysicalAcquisition(project, c['outer'], c['subject'])
    prefix = fixed_queries(env.available, c['outer'], int(c['subject'][-2:]), 0)[:4]
    for q in prefix: write_json(folder/f'{len(env.selected):02d}_observation.json', env.observe(q))
    for k in range(4,8):
        matrix = np.column_stack([env.predictions[f] for f in ['baseline','mte','hat']])
        rows = acquisition_table(env.kernel, env.selected, env.available, matrix, policy, model['outer_training_ratio'], model['states'][str(k)]['boundary_n'])
        choice = choose(rows)
        write_json(folder/f'{k:02d}_choice.json', {'selected':[env.pairs[q]['id'] for q in env.selected], 'policy':policy, 'candidates':rows,
            'choice':choice, 'scores':matrix.tolist(), 'no_labels_or_future_waveforms':True})
        write_json(folder/f'{k:02d}_observation.json', env.observe(int(choice[1:])))
    result = {**env.final_judge(), 'case_id':c['case_id'], 'policy':policy, 'actual_physical_measurements':8, 'new_API_calls':0,
              'evaluation_protocol_sha256':sha(output/'evaluation_protocol.json')}
    write_json(folder/'metrics.json', result)
    print('V13_PHYSICAL_DONE', c['case_id'], policy, round(result['AUROC'],6), flush=True)


def acquire(project, output, workers):
    project, output = Path(project), Path(output); p = verified(project, output)
    if not 1 <= workers <= 4: raise ValueError('One to four local workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, project, output, c, policy) for c in p['cases'] for policy in p['policies']]
        for f in as_completed(futures): f.result()
    ids = [c['case_id'] for c in p['cases']]
    parents = read_json(project/'outputs/spes_agent_v11/numeric_results.json')
    if sha(project/'outputs/spes_agent_v11/numeric_results.json') != p['v11_results_sha256']: raise ValueError('Frozen numerical controls differ')
    controls = {name:np.array([next(r['AUROC'] for r in parents['rows'] if r['case_id']==c and r['policy']==name) for c in ids]) for name in parents['policies']}
    rows = [read_json(output/'acquisition'/c/policy/'metrics.json') for c in ids for policy in p['policies']]
    summary = {}
    for policy in p['policies']:
        scores = np.array([next(r['AUROC'] for r in rows if r['case_id']==c and r['policy']==policy) for c in ids])
        summary[policy] = {'mean_patient_AUROC':float(scores.mean()), 'comparisons':{}}
        for name,base in controls.items():
            delta = scores-base
            summary[policy]['comparisons'][name] = {'gain':float(delta.mean()), 'paired_95CI':bootstrap_delta(delta)}
    primary = summary['learned_boundary']['comparisons']['score_focus']
    result = {'primary':p['acquisition_primary'], 'summary':summary, 'acquisition_gate':primary['gain']>.01 and primary['paired_95CI'][0]>0,
              'rows':[{k:v for k,v in r.items() if k!='prediction'} for r in rows], 'trajectories':len(rows), 'new_API_calls':0,
              'multi_agent_superiority_established':False}
    write_json(output/'acquisition_results.json', result)
    print('V13_ACQUISITION', {k:v for k,v in result.items() if k!='rows'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['freeze','fit','calibration','acquire'], required=True); parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    if args.stage == 'freeze': freeze(args.project, args.output)
    else:
        with exclusive(args.output/f'{args.stage}.lock'):
            if args.stage == 'acquire': acquire(args.project, args.output, args.workers)
            else: globals()[args.stage](args.project, args.output)
