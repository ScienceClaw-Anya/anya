"""Map compact response columns to anatomical pairs using public geometry only."""
import numpy as np
from scipy.spatial.distance import cdist


def match_pairs(raw_distance,recording_coords,pair_centers,tolerance=1e-4):
    distance=np.asarray(raw_distance,dtype=float);coords=np.asarray(recording_coords,dtype=float);centers=np.asarray(pair_centers,dtype=float)
    if distance.ndim!=2 or coords.shape!=(len(distance),3) or centers.ndim!=2 or centers.shape[1]!=3 or not np.isfinite(distance).all() or not np.isfinite(coords).all() or not np.isfinite(centers).all():raise ValueError('Invalid geometry inputs')
    expected=cdist(coords,centers);index=np.full((len(coords),len(centers)),-1,dtype=np.int32);cell_pair=np.full(distance.shape,-1,dtype=np.int32)
    valid=distance>1e-12;matched=unique=ambiguous=duplicates=0
    for i in range(len(coords)):
        for col in np.flatnonzero(valid[i]):
            candidates=np.flatnonzero(np.abs(expected[i]-distance[i,col])<=tolerance)
            matched+=int(len(candidates)>0)
            if len(candidates)==1:
                pair=int(candidates[0]);cell_pair[i,col]=pair;unique+=1
                if index[i,pair]==-1:index[i,pair]=col
                else:index[i,pair]=-2;duplicates+=1
            elif len(candidates)>1:cell_pair[i,col]=-2;ambiguous+=1
    # Multiple raw columns with the same anatomical pair cannot be resolved from
    # geometry into stimulation conditions. Reject that recording/pair cell.
    columns=[]
    for col in range(distance.shape[1]):
        values=cell_pair[:,col];unambiguous=values[values>=0];pairs=set(map(int,unambiguous))
        columns.append({'column':col,'valid_cells':int(valid[:,col].sum()),'unique_matches':len(unambiguous),'distinct_pairs':len(pairs),
          'conflicting_verified_pairs':len(pairs)>1})
    return index,{'valid_cells':int(valid.sum()),'distance_matched_cells':matched,'unambiguous_cells':unique,'ambiguous_cells':ambiguous,
      'duplicate_recording_pair_assignments':duplicates,'usable_recording_pair_cells':int((index>=0).sum()),
      'usable_anatomical_pairs':int((index>=0).any(axis=0).sum()),'columns':columns,'tolerance_mm':tolerance}


def physical_input(raw,index,selected):
    """One selected anatomical pair may have DIFFERENT raw columns per contact."""
    raw=np.asarray(raw);index=np.asarray(index);selected=list(map(int,selected))
    if raw.ndim!=4 or index.ndim!=2 or len(index)!=len(raw) or len(set(selected))!=len(selected) or any(q<0 or q>=index.shape[1] for q in selected):raise ValueError('Invalid physical query mapping')
    if ((index>=raw.shape[2])|(index< -2)).any():raise ValueError('Mapping column outside source')
    output=np.zeros((raw.shape[0],raw.shape[1],len(selected),raw.shape[3]),dtype=raw.dtype)
    for j,pair in enumerate(selected):
        columns=index[:,pair];rows=np.flatnonzero(columns>=0)
        geometry=raw[rows,:,columns[rows],0];available=geometry>1e-12
        highest=np.max(np.where(available,geometry,-np.inf),axis=1);lowest=np.min(np.where(available,geometry,np.inf),axis=1)
        if np.any((highest-lowest)[available.any(axis=1)]>1e-4):raise ValueError('A row-only map cannot combine trials with changing physical geometry')
        output[rows,:,j,:]=raw[rows,:,columns[rows],:]
    return output
