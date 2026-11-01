"""Whitelisted anatomy and observed-only quality evidence for every arm."""
import csv
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v10.features import contact_features
from experiments.spes_physical.mapping import physical_input

ROOTS={'geometry','observed','model_uncertainty','limitations'}
FIELDS={'name','x','y','z','hemisphere','Destrieux_label_text'}
FORBIDDEN={'y','labels','truth','SOZ','outcome','patient','subject','unobserved_waveforms','full_response_prediction'}
QUALITY={'observed_trial_fraction':3,'log_rms_mean':6,'log_rms_sd':7,'absolute_trial_coherence':10,
         'relative_difference_energy':11,'first_sample_bin_energy_fraction':12,'last_sample_bin_energy_fraction':19}


def project_columns(path):
    with Path(path).open(encoding='utf8',newline='') as stream:
        reader=csv.reader(stream,delimiter='\t'); header=next(reader)
        selected={k:header.index(k) for k in sorted(FIELDS) if k in header}
        if not {'name','x','y','z'}<=set(selected): raise ValueError('Public electrode fields missing')
        return [{k:row[i] for k,i in selected.items()} for row in reader if row]


def recording_catalog(project,subject,mapping):
    project=Path(project); paths=[f for f in mapping['metadata_sha256'] if f.endswith('_electrodes.tsv')]
    if len(paths)!=1: raise ValueError('Unique verified electrode metadata required')
    path=project/paths[0]
    if sha(path)!=mapping['metadata_sha256'][paths[0]]: raise ValueError('Electrode metadata changed')
    rows=project_columns(path); valid=[]
    for row in rows:
        try: xyz=np.array([float(row[k]) for k in ['x','y','z']])
        except ValueError: continue
        if np.isfinite(xyz).all(): valid.append((row,xyz))
    coords_path=project/f'vendor/norris2026_hat/data/coords_{subject}.npz'
    if sha(coords_path)!=mapping['coords_sha256']: raise ValueError('Recording coordinates changed')
    coords=load_array(coords_path); distances=cdist(coords,np.array([xyz for _,xyz in valid])); catalog=[]
    for i,(xyz,values) in enumerate(zip(coords,distances)):
        matches=np.flatnonzero(values<=1e-4)
        if not len(matches): raise ValueError('Recording coordinates not matched')
        options=[valid[int(j)][0] for j in matches]; row=options[0]
        def shared(field):
            values={option.get(field) for option in options}
            return next(iter(values)) if len(values)==1 else None
        item={'id':f'c{i:03d}','electrode':row['name'] if len(matches)==1 else None,'hemisphere':shared('hemisphere'),
              'declared_anatomy':shared('Destrieux_label_text'),'coordinates_mm':xyz.tolist()}
        if len(matches)>1: item.update(electrode_identity_status='coordinate_ambiguous',possible_electrodes=[option['name'] for option in options])
        catalog.append(item)
    return {'contacts':catalog,'frame':mapping['coordinate_frame'],'source_sha256':{paths[0]:sha(path)},'coords_sha256':sha(coords_path)}


def compact(obj):
    if isinstance(obj,(float,np.floating)): return round(float(obj),6)
    if isinstance(obj,dict): return {k:(float(v) if k.endswith('_gain') and isinstance(v,(float,np.floating)) else compact(v)) for k,v in obj.items()}
    if isinstance(obj,list): return [compact(v) for v in obj]
    return obj


def augment(base,catalog,observed,selected,pairs,predictions,previous=None):
    """No full raw signal or clinical label is an argument to this function."""
    import copy
    public=copy.deepcopy(base); evidence=public['evidence']; evidence['geometry']['recording_contacts']=catalog['contacts']
    evidence['geometry']['coordinate_frame']=catalog['frame']; condition_values=[]
    for q in sorted(pairs):
        condition=pairs[q]['recorded_conditions'][0]
        if condition not in condition_values: condition_values.append(condition)
    conditions={f'r{i:02d}':row for i,row in enumerate(condition_values)}
    for row in evidence['geometry']['candidates']:
        pair=pairs[int(row['id'][1:])]; row['recorded_condition_id']=f'r{condition_values.index(pair["recorded_conditions"][0]):02d}'
    evidence['geometry']['recorded_conditions']=conditions
    evidence['geometry']['condition_units']={'electrical_stimulation_current':'Ampere','electrical_stimulation_frequency':'Hertz','electrical_stimulation_pulsewidth':'seconds','source':'Dataset inherited events.json descriptions, frozen in protocol'}
    evidence['geometry']['gain_definition']='exp(-distance_mm/40) incremental geometric coverage; not an estimated accuracy gain.'
    score=predictions['hat']; quality=contact_features(observed[...,-512:],observed[...,0],score)
    order=['baseline','mte','hat']; ranks=np.column_stack([rankdata(predictions[f])/(len(score)+1) for f in order])
    for i,row in enumerate(evidence['model_uncertainty']['contacts']):
        row['relative_model_ranks']=ranks[i].tolist(); row['observed_quality']={name:float(quality[i,col]) for name,col in QUALITY.items()}
        row['last_HAT_score_change']=None if previous is None else float(score[i]-previous[i])
    evidence['model_uncertainty']['quality_note']='Descriptive summaries of selected post-stimulus traces only. RMS is in released-array units; sample bins are descriptive, not N1/N2 clinical labels. Model scores are not calibrated SOZ probabilities.'
    evidence['observed']['selected_conditions']=[{'pair':pairs[q]['id'],'condition_id':f'r{condition_values.index(pairs[q]["recorded_conditions"][0]):02d}'} for q in selected]
    evidence['limitations']+=['No clinical notes, contact SOZ labels, resection status or outcomes are provided.',
        'Recording coordinates with several matching electrode names retain explicit identity ambiguity; no name is guessed.',
        'Stimulus conditions can vary between anatomical pairs; eight groups do not equal eight pulses or equal current/charge/time.',
        'Score change and disagreement do not measure localization improvement. Anatomical labels alone do not establish pathology.',
        'Future waveforms are inaccessible. Unknown128-feature semantics are excluded.']
    # The base environment exposes anatomy separately; all schema citations use four fixed roots.
    evidence['geometry']['anatomy_provenance']=evidence.pop('anatomy')
    public=compact(public); check_public(public); return public


def public_state(env,catalog,raw,previous=None):
    observed=physical_input(raw,env.index,env.selected)
    pairs={q:env.pairs[q] for q in env.available}
    return augment(env.public(),catalog,observed,env.selected,pairs,env.predictions,previous)


def check_public(data):
    if set(data)!={'task','budget','evidence'} or set(data['evidence'])!=ROOTS: raise ValueError('Unexpected public roots')
    def check(x):
        if isinstance(x,dict):
            if set(x)&FORBIDDEN: raise ValueError('Clinical or hidden field rejected')
            for v in x.values(): check(v)
        elif isinstance(x,list):
            for v in x: check(v)
    check(data)
    rows=data['evidence']['geometry']['candidates']; ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)) or len(ids)<3 or any(not q.startswith('p') for q in ids): raise ValueError('Invalid physical action catalog')
    return data


def read_public(project,path):
    root=Path(project).resolve()/'outputs/spes_agent_v16/trajectories'; path=Path(path).resolve(); path.relative_to(root)
    if not path.name.endswith('_public.json'): raise ValueError('Public snapshots only')
    return check_public(read_json(path))
