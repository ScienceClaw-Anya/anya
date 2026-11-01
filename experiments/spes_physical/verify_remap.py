"""Actual HAT check: remapping equals the same retained response set."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import load_array
from experiments.spes_agent_v8.crossfit import write_json
from experiments.spes_agent_v8.targets import score_subset
from .mapping import physical_input
from .environment import PhysicalAcquisition


def verify(project,output):
    import torch
    project,output=Path(project),Path(output);torch.set_num_threads(2);case=read_json(output/'two_query_smoke.json')['case'];env=PhysicalAcquisition(project,case['outer'],case['subject'])
    raw=load_array(project/f'vendor/norris2026_hat/data/X_{case["subject"]}.npz')[:,:20];pruned=np.zeros_like(raw)
    for row in range(len(raw)):
        columns=env.index[row][env.index[row]>=0];pruned[row,:,columns,:]=raw[row,:,columns,:]
    aligned=physical_input(raw,env.index,list(range(env.index.shape[1])))
    a=score_subset(env.models['hat'],pruned,env.stats,list(range(pruned.shape[2])))
    b=score_subset(env.models['hat'],aligned,env.stats,list(range(aligned.shape[2])))
    error=float(np.abs(a-b).max());np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)
    result={'status':'verified','case':case,'maximum_absolute_HAT_score_difference':error,'tolerance':1e-5,
      'comparison':'Original compact arrays with unresolved cells zeroed versus physically aligned arrays containing exactly the same retained cells.',
      'clinical_labels_used':False,'agent_exposure_to_full_predictions':False,'API_calls':0,'source_sha256':sha(project/'experiments/spes_physical/verify_remap.py')}
    write_json(output/'remap_invariance.json',result);write_json(project/'examples/spes_physical_remap_summary.json',result);print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.project,a.output)
