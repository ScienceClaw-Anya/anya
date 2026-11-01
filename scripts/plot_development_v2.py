"""Compact development diagnostics; execute plot calculations only on GCP."""
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
assert str(ROOT)=='/home/devstar3022/anya-intervention-research'
OUT=ROOT/'outputs/development-v2'
replay=json.loads((OUT/'replay.json').read_text())
forecast=json.loads((OUT/'forecast/summary.json').read_text())
evaluation=json.loads((OUT/'epileptor-fit/evaluation-summary.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
colors=['#2166ac','#b35806']
for j,subject in enumerate(['sub-003','sub-010']):
    rows=[r for r in replay['paired_network_comparisons'] if r['subject']==subject]
    zero=np.mean([r['paired_zero_mask_disagreement'] for r in rows if 'seed' in r])*100
    noise=next(r['personal_seed_change_mask_disagreement'] for r in rows if 'personal_seed_change_mask_disagreement' in r)*100
    axes[0].bar(np.arange(2)+j*.32,[zero,noise],width=.3,color=colors[j],label=subject)
    row=next(r for r in forecast['results'] if r['subject']==subject and r['network']=='personal')
    ratios=[row['test_errors'][str(h)]['mse']/row['test_errors'][str(h)]['persistence_mse'] for h in [1,10,50]]
    axes[1].plot([1,10,50],ratios,'o-',color=colors[j],label=subject)
axes[0].set(xticks=[.16,1.16],xticklabels=['Remove all edges','Change noise seed'],ylabel='Ictal-mask disagreement (%)',title='Network sensitivity')
axes[0].legend(frameon=False)
axes[1].axhline(1,color='black',ls='--',lw=1)
axes[1].set(xscale='log',yscale='log',xticks=[1,10,50],xticklabels=['1','10','50'],xlabel='Forecast horizon (samples)',ylabel='MSE / last-observation MSE',title='Linear prediction: lower is better')
axes[1].legend(frameon=False)
labels=['Personal','Donor','Zero','Training\nfeatures']
values=[]
for name in ['personal','donor','zero']:
    values.append(np.mean([r['errors_against_development_test_block']['covariance_relative_error'] for r in evaluation['rows'] if r['network']==name]))
values.append(evaluation['training_feature_baseline']['covariance_relative_error'])
axes[2].bar(np.arange(4),values,color=['#2166ac','#888888','#bbbbbb','#b35806'])
axes[2].axhline(1,color='black',ls='--',lw=1,label='Zero covariance reference')
axes[2].legend(frameon=False,fontsize=8)
axes[2].set(xticks=np.arange(4),xticklabels=labels,ylabel='Relative sensor covariance error',title='Epileptor fit: sub-010 development')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
fig.suptitle('CPU development checks — two development cases; no intervention-benefit claim',fontsize=14)
fig.savefig(OUT/'development-v2.png',dpi=160)
plt.close(fig)
budget=json.loads((OUT/'budget.json').read_text())
resource={'experiment_wall_seconds_to_report':time.time()-budget['start_epoch'],
          'max_wall_seconds':budget['max_wall_seconds'],'workers':1,'threads':1,'nice':10,
          'host':'scienceclaw-gcp','new_simulator_calls':35,'new_network_calls':15,
          'new_fit_library_calls':2,'new_fit_candidate_calls':12,'new_fit_evaluation_calls':6,
          'held_out_patients_inspected':0,'gpu_used':False}
if not (OUT/'resource-summary.json').exists():
    (OUT/'resource-summary.json').write_text(json.dumps(resource,indent=2)+'\n')
