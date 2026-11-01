"""Scoped data-quality check: actual scored cohorts, identity and contact grain.

No cohorts are merged. Coordinate matching is a diagnostic within the common
patients, using unique reciprocal matches within 0.001 coordinate units. A
missing match does not establish that a contact is biologically different.
"""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,write_json,sha
from experiments.spes_agent_v4.neural import load_array,header

def run(project):
    project=Path(project);old={};new={};sources={}
    for fold in range(5):
        path=project/f'outputs/reproduction/norris2024/training/Transformer_base_seed1_fold{fold}/predictions.npz';d=np.load(path)
        sources[path.relative_to(project).as_posix()]=sha(path)
        for p in np.unique(d['patient']):
            if int(p) in old:raise ValueError('Duplicate old outer patient')
            mask=d['patient']==p;old[int(p)]={'y':d['y'][mask],'coords':d['coords'][mask] if 'coords' in d.files else None}
        path=project/f'outputs/spes_agent_v4/multitrial/neural/outer{fold}/hat/predictions.npz';d=np.load(path)
        sources[path.relative_to(project).as_posix()]=sha(path)
        for p in np.unique(d['patient']):
            if int(p) in new:raise ValueError('Duplicate new outer patient')
            mask=d['patient']==p;subject=f'ccepAgeUMCU{int(p):02d}';coords_path=project/f'vendor/norris2026_hat/data/coords_{subject}.npz'
            coords=load_array(coords_path);y=load_array(project/f'vendor/norris2026_hat/data/y_{subject}.npz')
            if not np.array_equal(y,d['y'][mask]) or len(coords)!=len(y):raise ValueError('New scored-label-coordinate grain mismatch')
            new[int(p)]={'y':y,'coords':coords};sources[coords_path.relative_to(project).as_posix()]=sha(coords_path)
    shared=sorted(set(old)&set(new));rows=[]
    for p in shared:
        a,b=old[p],new[p];row={'patient':p,'old_contacts':len(a['y']),'new_contacts':len(b['y']),
            'old_SOZ':int(a['y'].sum()),'new_SOZ':int(b['y'].sum())}
        if a['coords'] is not None and a['coords'].shape==(len(a['y']),3) and b['coords'].shape==(len(b['y']),3):
            distance=np.linalg.norm(a['coords'][:,None,:]-b['coords'][None,:,:],axis=-1)
            close=np.isfinite(distance)&(distance<=.001);unique=close&(close.sum(axis=0)[None,:]==1)&(close.sum(axis=1)[:,None]==1)
            i,j=np.where(unique);row.update(unique_coordinate_matches=len(i),label_disagreements_in_matches=int((a['y'][i]!=b['y'][j]).sum()),
               ambiguous_candidate_pairs=int(close.sum()-unique.sum()))
        else:row['unique_coordinate_matches']=None;row['mapping_limit']='Saved coordinates unavailable or wrong shape'
        rows.append(row)
    source_data=project/'vendor/norris2026_hat/data';ignored=source_data/'ignore_X_ccepAgeUMCU47.npz'
    shape,dtype=header(ignored);ignored_y=load_array(source_data/'y_ccepAgeUMCU47.npz')
    result={'scope':'Nested cohort-quality check supporting the existing research reports; no extra report/dashboard',
      'grain':'One contact row per patient per processed-data version, using actual first-repeat outer scoring files',
      'old':{'patients':len(old),'contacts':sum(len(r['y']) for r in old.values()),'SOZ':sum(int(r['y'].sum()) for r in old.values()),'patient_ids':sorted(old)},
      'new':{'patients':len(new),'contacts':sum(len(r['y']) for r in new.values()),'SOZ':sum(int(r['y'].sum()) for r in new.values()),'patient_ids':sorted(new)},
      'shared_patients':len(shared),'old_only_patients':sorted(set(old)-set(new)),'new_only_patients':sorted(set(new)-set(old)),
      'unique_coordinate_matches_in_shared_patients':sum(r.get('unique_coordinate_matches') or 0 for r in rows),
      'label_disagreements_in_unique_matches':sum(r.get('label_disagreements_in_matches',0) for r in rows),
      'coordinate_mapping_limit':'Only unique reciprocal matches within 0.001 coordinate units. Coordinate reference frames were not independently re-registered. Unmatched contacts are unresolved, not proven distinct.',
      'ignored_author_file':{'name':ignored.name,'shape':shape,'dtype':dtype,'labels':len(ignored_y),'SOZ':int(ignored_y.sum()),
           'handling':'Not included: author default X_* discovery excludes ignore_X_*. Reason for the ignore prefix has not been established; labels contain both classes.'},
      'findings':[{'finding':'Patient composition and contact denominators differ between versions','severity':'high for cross-version score comparisons',
          'evidence':'Direct rosters and scored labels; source hashes retained','impact':'Old/new AUROCs cannot be interpreted as an isolated architecture effect.',
          'remediation':'Keep model/policy comparisons within one version; disclose cohort differences; do not pool or reindex contacts across versions.'},
         {'finding':'Within the new experiments, all scorers use the same current cohort','severity':'no cohort-integrity defect found',
          'evidence':'Existing independent v4/v5/v6/v7 label-alignment and patient-isolation audits','impact':'Current same-version comparisons remain interpretable within their exploratory scope.'}],
      'rows':rows,'source_sha256':sources,'source_script_sha256':sha(Path(__file__))}
    write_json(project/'outputs/spes_cohort_versions_audit.json',result)
    write_json(project/'examples/spes_cohort_versions_summary.json',{k:v for k,v in result.items() if k not in ['rows','source_sha256']})
    print({k:v for k,v in result.items() if k not in ['rows','source_sha256','findings']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args();run(a.project)
