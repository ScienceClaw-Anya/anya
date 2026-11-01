"""Audit every processed patient and save verified geometry-only action maps."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v8.crossfit import write_json
from .mapping import match_pairs

ELECTRODE_FIELDS={'name','x','y','z','hemisphere','Destrieux_label_text'}
EVENT_FIELDS={'electrical_stimulation_site','electrical_stimulation_type','electrical_stimulation_current','electrical_stimulation_frequency','electrical_stimulation_pulsewidth'}


def patient(project,output,subject):
    project,output=Path(project),Path(output);data=project/'vendor/norris2026_hat/data';folder=project/f'data/ds004080/sub-{subject}';coords=load_array(data/f'coords_{subject}.npz');candidates=[]
    for path in folder.rglob('*_electrodes.tsv'):
        d=pd.read_csv(path,sep='\t',usecols=lambda c:c in ELECTRODE_FIELDS).dropna(subset=['x','y','z']).set_index('name')
        error=cdist(coords,d[['x','y','z']].to_numpy(float)).min(axis=1)
        if error.max()<=1e-4:candidates.append((path,d,float(error.max())))
    if len(candidates)!=1:return {'subject':subject,'status':'unresolved_electrode_session','matching_coordinate_sessions':len(candidates)}
    path,d,coord_error=candidates[0];pairs={};hashes={path.relative_to(project).as_posix():sha(path)}
    for event in sorted(path.parent.glob('*events.tsv')):
        e=pd.read_csv(event,sep='\t',usecols=lambda c:c in EVENT_FIELDS);hashes[event.relative_to(project).as_posix()]=sha(event)
        for _,row in e.dropna(subset=['electrical_stimulation_site']).iterrows():
            names=str(row['electrical_stimulation_site']).split('-')
            if len(names)!=2 or any(n not in d.index for n in names):continue
            key=tuple(sorted(names));condition=tuple((c,str(row[c])) for c in sorted(EVENT_FIELDS-{'electrical_stimulation_site'}) if c in row and pd.notna(row[c]))
            pairs.setdefault(key,set()).add(condition)
    names=sorted(pairs);centers=np.array([d.loc[list(p),['x','y','z']].to_numpy(float).mean(axis=0) for p in names]);raw_path=data/f'X_{subject}.npz';full=load_array(raw_path);source_shape=list(full.shape);raw=full[:,:20];distance=raw[...,0];valid=distance>1e-12
    mean=np.divide(distance.sum(axis=1,dtype=float),valid.sum(axis=1),out=np.zeros(distance.shape[::2],dtype=float),where=valid.sum(axis=1)>0)
    # A unique geometry match must persist across every retained trial.
    maximum=np.max(np.where(valid,distance,-np.inf),axis=1);minimum=np.min(np.where(valid,distance,np.inf),axis=1);used=valid.any(axis=1)
    variation=float(np.max((maximum-minimum)[used]))
    index,statistics=match_pairs(mean,coords,centers)
    if variation>1e-4:return {'subject':subject,'status':'unresolved_trial_geometry','max_trial_distance_change_mm':variation}
    canonical=output/'maps'/subject;canonical.mkdir(parents=True,exist_ok=True);np.savez_compressed(canonical/'index.npz',index=index)
    pair_rows=[]
    for i,pair in enumerate(names):
        pair_rows.append({'id':f'p{i:03d}','contacts':list(pair),'center_mm':centers[i].tolist(),
          'declared_anatomy':[str(d.loc[n,'Destrieux_label_text']) if 'Destrieux_label_text' in d else None for n in pair],
          'recorded_conditions':[dict(c) for c in sorted(pairs[pair])]})
    coordinate_files=list(path.parent.glob('*coordsystem.json'))
    if len(coordinate_files)==1:
        frame=read_json(coordinate_files[0]);hashes[coordinate_files[0].relative_to(project).as_posix()]=sha(coordinate_files[0]);frame={k:frame.get(k) for k in ['iEEGCoordinateSystem','iEEGCoordinateUnits','iEEGCoordinateSystemDescription']}
    else:frame={'status':'not_uniquely_verified'}
    result={'subject':subject,'status':'geometry_mapped_with_explicit_exclusions','coordinate_max_match_error_mm':coord_error,
      'max_trial_distance_change_mm':variation,'source_shape':source_shape,'retained_shape':list(raw.shape),'retained_trial_scope':'First twenty trials, matching all current fitted/scored neural protocols. Full forty-trial indexing is not asserted.','anatomical_pairs_in_events':len(names),
      'pairs_with_multiple_recorded_condition_tuples':sum(len(v)>1 for v in pairs.values()),'statistics':statistics,'coordinate_frame':frame,
      'raw_sha256':sha(raw_path),'coords_sha256':sha(data/f'coords_{subject}.npz'),'metadata_sha256':hashes,'index_sha256':sha(canonical/'index.npz'),
      'pair_rows':pair_rows,'excluded_fields':['soz','resected','edge','notes','clinical outcomes'],
      'limits':['Distance-equidistant cells and duplicate recording/pair assignments are excluded, not guessed or pooled.',
        'Maps identify anatomical pairs; multiple stimulation-parameter conditions cannot be separated by distance alone.',
        'Released preprocessing availability remains retrospective, and clinical safety is not modeled.']}
    write_json(canonical/'receipt.json',result);print('PHYSICAL_MAPPING',subject,statistics['valid_cells'],statistics['unambiguous_cells'],sum(c['conflicting_verified_pairs'] for c in statistics['columns']),flush=True);return result


def audit(project,output):
    project,output=Path(project),Path(output);p=read_json(project/'outputs/spes_agent_v4/multitrial/neural_protocol.json');rows=[patient(project,output,r['subject']) for r in p['records']];valid=[r for r in rows if r['status']=='geometry_mapped_with_explicit_exclusions']
    result={'scope':'Action-grain audit of the first twenty trials actually used by current models; no labels, outcomes, model fitting or API calls.','patients':len(rows),'mapped_patients':len(valid),
      'patients_with_verified_mixed_columns':sum(any(c['conflicting_verified_pairs'] for c in r['statistics']['columns']) for r in valid),
      'valid_cells':sum(r['statistics']['valid_cells'] for r in valid),'unambiguous_cells':sum(r['statistics']['unambiguous_cells'] for r in valid),
      'ambiguous_cells':sum(r['statistics']['ambiguous_cells'] for r in valid),'duplicate_recording_pair_assignments':sum(r['statistics']['duplicate_recording_pair_assignments'] for r in valid),
      'conclusion':'Processed column indices are not globally aligned physical stimulation pairs. Historical v4-v10 column-selection budgets are array-ablation operations, not physical-stimulation budgets.',
      'full_response_classification':'The original full-response contact-classification task does not require global action-column alignment and is not invalidated by this finding.',
      'source_sha256':{f:sha(project/f) for f in ['experiments/spes_physical/mapping.py','experiments/spes_physical/audit.py']},
      'rows':[{k:v for k,v in r.items() if k!='pair_rows'} for r in rows]}
    write_json(output/'alignment_audit.json',result);write_json(project/'examples/spes_physical_alignment_summary.json',{k:v for k,v in result.items() if k!='rows'});print({k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.project,a.output)
