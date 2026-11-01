"""Compact GCP report/figure from completed outputs; no patient data export."""
import json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
os.chdir(ROOT);out=ROOT/'outputs/pilot-v1';reports=ROOT/'reports';reports.mkdir(exist_ok=True)
summary=json.loads((out/'summary.json').read_text())
controls=json.loads((out/'controls.json').read_text())
replay=json.loads((out/'replay.json').read_text())
rows=[json.loads(x) for x in (out/'rows.jsonl').read_text().splitlines()]
export=[]
for row in rows:
    record={k:v for k,v in row.items() if k not in ['onset_generator_burden','nononset_generator_burden']}
    record['generator_x0_gt_minus2p1_burden']=row['onset_generator_burden']
    record['generator_x0_le_minus2p1_burden']=row['nononset_generator_burden']
    export.append(record)
aggregate={'summary':summary,'controls':controls,'replay':replay,'trajectories':export,
           'partition_caveat':'x0 threshold partitions susceptibility; it is not an independently verified onset/propagation split. Legacy raw output field names are corrected here.'}
(reports/'pilot-v1-summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')
policies=['SOZ_proxy','network_heuristic','personal_oracle','donor_oracle']
names=['SOZ proxy','Network heuristic','Personal oracle','Donor oracle']
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
for j,seed in enumerate([201,202]):
    values=[100*next(r['R'] for r in summary['results'] if r['policy']==p and r['scenario']['seed']==seed) for p in policies]
    ax.bar(np.arange(4)+(j-.5)*.35,values,width=.35,label=['Fresh seed','Fresh seed + parameter shift'][j],color=['#345b86','#b97635'][j])
ax.set_xticks(np.arange(4),names);ax.set_ylabel('Simulated burden reduction (%)');ax.axhline(0,color='black',linewidth=.7)
ax.legend(frameon=False,fontsize=9);ax.spines[['top','right']].set_visible(False)
ax.set_title('One development patient • one-parcel proxy budget',loc='left',fontsize=13)
fig.supxlabel('Oracle planners know generator parameters. No patient-level uncertainty or clinical benefit established.',fontsize=9)
fig.savefig(reports/'pilot-v1.png',dpi=170)
print(json.dumps({'results':summary['results'],'runtime_seconds':summary['runtime_seconds'],'controls':controls,'replay':replay},indent=2))
