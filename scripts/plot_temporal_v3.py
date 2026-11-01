"""Plot aggregate development diagnostics on GCP, retaining raw arrays there."""
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v3'
selection=json.loads((OUT/'selection.json').read_text())
evaluation=json.loads((OUT/'evaluation.json').read_text())
replay=json.loads((OUT/'replay.json').read_text()); assert replay['passed']
f=np.load(OUT/'features.npz'); times=f['times']; j=int(f['j']); i=int(f['i'])
fig=plt.figure(figsize=(12,9),layout='constrained'); grid=fig.add_gridspec(2,2)
ax=fig.add_subplot(grid[0,:]); bars=fig.add_subplot(grid[1,0]); profiles=fig.add_subplot(grid[1,1])
colors={'personal':'#2166ac','donor':'#b35806','zero':'#777777'}
ax.plot(times,f['observed'].mean(axis=1),color='black',lw=1.8,label='Observed activity proxy')
for name in ['personal','zero']:
    a=np.load(OUT/f'{name}-reference.npz')
    curve=np.concatenate([a['prefix_prediction'],a['prediction']]).mean(axis=1)
    ax.plot(times,curve,color=colors[name],lw=1.2,label=f'{name.title()} model, aligned seed 401')
baseline=selection['selected_baseline']
bp=np.load(OUT/f'baseline-{baseline}.npz')['prediction'].mean(axis=1)
ax.plot(times[j:],bp,color='#c51b7d',ls='--',lw=2,label=f'Selected baseline: {baseline}')
ax.axhline(float(f['threshold']),color='#999999',ls=':',label='High-activity proxy threshold')
ax.axvspan(times[i],times[j],color='#efdd9e',alpha=.3)
ax.axvspan(times[j],times[-1],color='#a6dba0',alpha=.2)
ax.axvline(times[j],color='black',lw=.8)
ax.set(xlabel='Exported sample / model units (not verified biological time)',ylabel='Mean normalized log envelope',
       title='sub-010 run-01: fit first 60%, select on next 20%, forecast final 20%')
ax.legend(frameon=False,fontsize=8,ncol=2)
for k,name in enumerate(['personal','donor','zero']):
    rows=[r for r in evaluation['rows'] if r['method']==name]
    mean=next(r['mse_ratio_to_selected_baseline'] for r in rows if r['kind']=='fresh_seed_mean_prediction')
    bars.bar(k,mean,color=colors[name],alpha=.4,width=.6)
    for delta,kind,marker in [(-.15,'reference_seed401','^'),(0,'fresh_seed701','o'),(.15,'fresh_seed702','s')]:
        value=next(r['mse_ratio_to_selected_baseline'] for r in rows if r['kind']==kind)
        bars.scatter(k+delta,value,color=colors[name],marker=marker,s=40,
                     label={'reference_seed401':'Aligned seed 401','fresh_seed701':'Fresh seed 701','fresh_seed702':'Fresh seed 702'}[kind] if k==0 else None)
    a=np.load(OUT/f'{name}-reference.npz')
    profiles.plot(a['offsets'],a['prefix_losses'],color=colors[name],label=name.title())
bars.axhline(1,color='black',ls='--',lw=1)
bars.set(xticks=[0,1,2],xticklabels=['Personal','Donor','Zero'],ylabel='Final envelope MSE / selected baseline MSE',
         title='Prediction errors; bars = fresh-seed mean prediction')
bars.legend(frameon=False,fontsize=8)
profiles.set(xlabel='Trajectory time offset (samples)',ylabel='First-80% envelope MSE',title='Phase profiles for selected candidates')
profiles.legend(frameon=False)
for panel in [ax,bars,profiles]:
    panel.spines[['top','right']].set_visible(False); panel.grid(axis='y',alpha=.15); panel.set_axisbelow(True)
fig.suptitle('CPU development v3 — reused development recording; activity proxies, not clinical seizure labels',fontsize=12)
fig.savefig(OUT/'development-v3.png',dpi=160); plt.close(fig)
budget=json.loads((OUT/'budget.json').read_text())
resource=dict(host='scienceclaw-gcp',workers=1,threads=1,nice=10,gpu_used=False,new_data_bytes=0,
              new_simulator_calls=evaluation['new_simulator_calls'],reused_fit_trajectories=12,
              experiment_wall_seconds_to_report=time.time()-budget['start_epoch'],
              stage_compute_seconds=selection['runtime_seconds']+evaluation['evaluation_runtime_seconds']+replay['runtime_seconds'],
              max_wall_seconds=budget['max_wall_seconds'],held_out_patients_inspected=0)
assert resource['experiment_wall_seconds_to_report']<resource['max_wall_seconds']
(OUT/'resources.json').write_text(json.dumps(resource,indent=2)+'\n')
