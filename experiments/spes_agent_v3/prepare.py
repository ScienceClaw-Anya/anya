"""Fit from inner OOF patients, then build identical label-free decision packets."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from .common import SLUGS, SELECTOR, WEIGHTS, check_groups, read_json, sha, source_hashes, write_json
from .crossfit import freeze
from .selector import candidates, fit_selector, select

def prepare(project, output):
    project, output=Path(project).resolve(),Path(output).resolve()
    if (output/'agent_protocol.json').exists(): raise ValueError('Do not overwrite frozen agent inputs')
    protocol=freeze(project,output)
    selector_protocol={'config':SELECTOR,'weights':WEIGHTS,'training_protocol_sha256':sha(output/'training_protocol.json'),
        'sources':source_hashes(project,['experiments/spes_agent_v3/common.py','experiments/spes_agent_v3/selector.py','experiments/spes_agent_v3/prepare.py'])}
    if (output/'selector_protocol.json').exists() and read_json(output/'selector_protocol.json')!=selector_protocol:
        raise ValueError('Selector protocol changed')
    write_json(output/'selector_protocol.json',selector_protocol)
    records=[];training_receipts=[]
    for outer in range(5):
        chunks=[];ys=[];ps=[];seen=set();receipts=[]
        jobs=[r for r in protocol['jobs'] if r['outer']==outer]
        for row in jobs:
            check_groups(row['train'],row['early_stop'],row['predict'],row['outer_test'])
            folder=output/f"crossfit/outer{outer}_inner{row['inner']}"
            values=[np.load(folder/slug/'predictions.npz') for slug in SLUGS]
            for slug,value in zip(SLUGS,values):
                receipt=read_json(folder/slug/'receipt.json')
                for key in ['train','early_stop','predict','outer_test']:
                    if receipt[key]!=row[key]:raise ValueError('Prediction provenance differs from isolated schedule')
                if sha(folder/slug/'predictions.npz')!=receipt['prediction_sha256']:raise ValueError('OOF predictions changed')
                if not np.array_equal(value['patient'],values[0]['patient']) or not np.array_equal(value['y'],values[0]['y']):raise ValueError('OOF row order differs')
                receipts.append({'path':str((folder/slug/'receipt.json').relative_to(output)),'sha256':sha(folder/slug/'receipt.json')})
            people=set(map(int,values[0]['patient']))
            if seen&people:raise ValueError('Duplicate OOF patient')
            seen|=people; chunks.append(np.column_stack([v['prediction'] for v in values]));ys.append(values[0]['y']);ps.append(values[0]['patient'])
        if len(seen)!=28 or seen & {int(p[-2:]) for p in jobs[0]['outer_test']}:raise ValueError('Incorrect selector fit patients')
        x,y,patient=np.vstack(chunks),np.concatenate(ys),np.concatenate(ps)
        fitted=fit_selector(x,y,patient)
        target=output/f'selectors/outer{outer}';target.mkdir(parents=True,exist_ok=True)
        joblib.dump(fitted,target/'selector.joblib')
        np.savez_compressed(target/'OOF.npz',predictions=x,y=y,patient=patient)
        evidence={'outer':outer,'fit_patient_count':28,'training_patients':fitted['training_patients'],
            'outer_test':jobs[0]['outer_test'],'evidence':fitted['evidence'],'OOF_transformer_AUROC':fitted['OOF_transformer_AUROC'],
            'model_sha256':sha(target/'selector.joblib'),'OOF_sha256':sha(target/'OOF.npz'),'inner_receipts':receipts,
            'limitations':['OOF candidate evidence is training evidence for this selector, not independent validation of the fitted selector',
                          'Inner base models fit 17 patients plus four early-stop patients; outer frozen models fit 21 plus seven early-stop patients',
                          'Bootstrap eligibility is a conservative development heuristic; overlapping inner fits do not provide a clinical guarantee']}
        write_json(target/'receipt.json',evidence);training_receipts.append(evidence)
        # Fit is complete before outer test labels are accessed.
        old=project/'outputs/reproduction/norris2024/training'
        split=read_json(old/f'split_seed1_fold{outer}.json')
        if set(split['train']+split['val'])!=set(jobs[0]['development']):raise ValueError('Outer checkpoint development set differs')
        outer_values=[np.load(old/f'{slug}_seed1_fold{outer}/predictions.npz') for slug in SLUGS]
        original=outer_values[0]
        evidence['outer_split_sha256']=sha(old/f'split_seed1_fold{outer}.json')
        evidence['outer_prediction_sha256']={slug:sha(old/f'{slug}_seed1_fold{outer}/predictions.npz') for slug in SLUGS}
        evidence['outer_checkpoint_sha256']={slug:sha(old/f'{slug}_seed1_fold{outer}/checkpoint.pth') for slug in SLUGS}
        write_json(target/'receipt.json',evidence)
        for value in outer_values[1:]:
            if not np.array_equal(value['patient'],original['patient']):raise ValueError('Outer contact order differs')
        outer_x=np.column_stack([v['prediction'] for v in outer_values])
        oof_reliability={slug:float(np.mean([roc_auc_score(y[patient==p],x[patient==p,i]) for p in seen])) for i,slug in enumerate(SLUGS)}
        for p in sorted(set(original['patient'])):
            mask=original['patient']==p;sample=outer_x[mask];decision=select(fitted,sample)
            case_id=hashlib.sha256(f'v3:1:{outer}:{int(p)}'.encode()).hexdigest()[:16];case=output/'cases'/case_id
            case.mkdir(parents=True,exist_ok=True)
            score_matrix=np.column_stack([sample[:,2],decision['scores']])
            np.savez_compressed(case/'scores.npz',scores=score_matrix)
            facts={'baseline':{'definition':'Transformer is the default and the exact numerical fallback. Never generate probabilities.',
                'OOF_model_AUROC':oof_reliability},
                'candidate_evidence':{'patients':28,'candidates':fitted['evidence'],'minimum_gain':SELECTOR['minimum_gain'],
                    'note':'Inner held-out predictions train this selector. Individual predicted gain is not independently validated.'},
                'selector_decision':{'candidate':decision['candidate'],'weights':decision['weights'],'predicted_gain':decision['predicted_gain']},
                'distribution':{'feature_order':SELECTOR['features'],'features':decision['patient_features'],
                    'OOF_feature_mean':fitted['model'].named_steps['scale'].mean_.tolist(),
                    'OOF_feature_sd':fitted['model'].named_steps['scale'].scale_.tolist(),
                    'note':'Descriptive predictor-distribution summaries; these are not waveform quality or physiological biomarkers.'},
                'limitations':{'static_observations':True,'new_physiological_feedback':False,'clinical_benefit_validated':False,
                    'score_vectors_differ':bool(np.max(np.abs(score_matrix[:,0]-score_matrix[:,1]))>1e-12)}}
            public={'task':'Retrospective SOZ contact ranking. Choose an executable scorer, not a stimulation treatment.',
                'default_action':'transformer','allowed_actions':['transformer','conservative_selector'],
                'evidence':facts,'scorer_order':['transformer','conservative_selector'],
                'contacts':[{'id':f'c{i:03d}','scorers':np.round(score_matrix[i],6).tolist()} for i in range(len(sample))],
                'constraints':['Every arm receives this same evidence and default baseline.',
                    'No score generation, reweighting, hidden labels, prevalence, patient identity or test accuracy.',
                    'Program execution uses full precision; displayed vectors use six decimals.',
                    'A permitted alternative retains at least 80 percent Transformer weight. This is not a guarantee of improved AUROC.']}
            write_json(case/'public.json',public)
            write_json(case/'tools.json',{'baseline':facts['baseline'],'selector':facts['selector_decision'],
                'validation':facts['candidate_evidence'],'distribution':facts['distribution']})
            # Hidden labels enter scoring only, after all selector fitting is complete.
            truth=original['y'][mask].astype(int)
            write_json(case/'judge.json',{'patient':int(p),'outer':outer,'y':truth.tolist()})
            records.append({'case_id':case_id,'outer':outer,'patient':int(p),'selector_candidate':decision['candidate'],
                'distinct_actions':facts['limitations']['score_vectors_differ'],
                'transformer_AUROC':float(roc_auc_score(truth,score_matrix[:,0])),
                'selector_AUROC':float(roc_auc_score(truth,score_matrix[:,1]))})
    result={'patients':35,'outer_repeat':1,'inner_fits':60,'selector_fits':5,'cases':records,'training_receipts':training_receipts}
    write_json(output/'prepared.json',result)
    print({'prepared_patients':len(records),'patients_with_distinct_actions':sum(r['distinct_actions'] for r in records)})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.project,a.output)
