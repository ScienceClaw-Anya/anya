"""Observation-gated retrospective SPES acquisition environment.

Querying a stimulation column reveals real released responses at every recording
contact. Unqueried waveforms are zeroed before EVERY scorer call. Geometry and
the released availability map are assumed known in advance; no outcome, SOZ
labels, full-response predictions or unobserved response summaries are public.
This is a new task, not the full-response paper benchmark or physical stimulation.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from experiments.spes_agent_v3.common import read_json
from .neural import FAMILIES,upstream_models,load_array

def case_id(subject,outer):return hashlib.sha256(f'v4:active:{outer}:{subject}'.encode()).hexdigest()[:16]

def gated_input(x, selected):
    """Only selected columns survive. Return a copy, never modify hidden data."""
    selected=list(selected)
    if len(selected)!=len(set(selected)) or any(type(i) is not int or i<0 or i>=x.shape[2] for i in selected):
        raise ValueError('Invalid or repeated measurement')
    return np.array(x[:,:,selected,:],copy=True)

class Acquisition:
    def __init__(self,project,training,outer,subject,device='cuda'):
        import torch
        self.project=Path(project);self.training=Path(training);self.outer=outer;self.subject=subject;self.device=device
        protocol=read_json(self.training/'neural_protocol.json');row=protocol['jobs'][outer]
        if subject not in row['predict'] or subject in row['train']+row['early_stop']:raise ValueError('Patient is not held out')
        root=self.project/'vendor/norris2026_hat/data';self.__raw=load_array(root/f'X_{subject}.npz')[:,:20].copy()
        self.__labels=load_array(root/f'y_{subject}.npz').astype(int)
        self.n=len(self.__labels);self.selected=[];self.history=[];self.models={}
        self.predictions={f:np.full(self.n,.5) for f in FAMILIES}
        # The pre-existing structural availability map is an explicit assumption:
        # which stimulation-recording pairs/trials survived author preprocessing.
        distance=self.__raw[...,0];valid=np.abs(distance)>1e-12
        self.dist=np.divide(distance.sum(axis=1),valid.sum(axis=1),out=np.zeros(distance.shape[::2]),where=valid.sum(axis=1)>0)
        self.available=[int(i) for i in np.flatnonzero((self.dist>0).any(axis=0))]
        if len(self.available)<8:raise ValueError('Fewer than eight eligible measurement sites')
        self.kernel=np.where(self.dist>0,np.exp(-self.dist/40.),0.)
        receipt=read_json(self.training/f'neural/outer{outer}/hat/receipt.json')
        self.stats=receipt['normalisation'];mods=upstream_models(self.project)
        for family in FAMILIES:
            record=read_json(self.training/f'neural/outer{outer}/{family}/receipt.json')
            if record['normalisation']!=self.stats:raise ValueError('Scorer preprocessing differs')
            model,_,_=mods.create_model(family)
            saved=torch.load(self.training/f'neural/outer{outer}/{family}/checkpoint.pth',map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model_state_dict']);self.models[family]=model.to(device).eval()

    def normalized_observed(self):
        x=gated_input(self.__raw,self.selected)
        for segment,stat in zip([slice(0,1),slice(1,129),slice(129,None)],self.stats):
            view=x[...,segment];mask=np.abs(view)>1e-12;view[mask]=(view[mask]-stat['mean'])/stat['std']
        return x

    def observe(self,stimulus):
        import torch
        if stimulus not in self.available or stimulus in self.selected:raise ValueError('Invalid or repeated query')
        if len(self.selected)>=8:raise ValueError('Measurement budget exhausted')
        old=self.predictions['hat'].copy();self.selected.append(int(stimulus));x=self.normalized_observed()
        with torch.no_grad():
            for name,model in self.models.items():
                values=[]
                for begin in range(0,self.n,8):
                    batch=torch.from_numpy(x[begin:begin+8]).to(self.device)
                    value=torch.sigmoid(model(batch)[:,0]).cpu().numpy()
                    if not np.isfinite(value).all():raise ValueError('Nonfinite partial-response score')
                    values.append(value)
                self.predictions[name]=np.concatenate(values)
        observed=self.__raw[:,:,stimulus,1:];mask=np.abs(self.__raw[:,:,stimulus,0])>1e-12
        valid_values=observed[mask]
        event={'measurement':len(self.selected),'stimulus':f's{stimulus:03d}',
          'observed_valid_pairs':int(mask.sum()),'median_trial_rms':float(np.median(np.sqrt(np.mean(valid_values**2,axis=-1)))) if len(valid_values) else 0.,
          'mean_absolute_hat_score_change':float(np.abs(self.predictions['hat']-old).mean()),
          'top_contacts_after_query':[f'c{i:03d}' for i in np.argsort(self.predictions['hat'])[-5:][::-1]]}
        self.history.append(event);return event

    def candidate_table(self):
        current=self.kernel[:,self.selected].max(axis=1) if self.selected else np.zeros(self.n)
        scores=self.predictions['hat'];rank=rankdata(scores)/(self.n+1)
        matrix=np.column_stack([self.predictions[f] for f in FAMILIES])
        disagreement=matrix.std(axis=1);uncertain=disagreement+.1*(1-np.abs(scores-.5)*2)
        def weighted(x,w):return float(np.average(x,weights=w))
        rows=[]
        for i in self.available:
            if i in self.selected:continue
            gain=np.maximum(self.kernel[:,i]-current,0.)
            rows.append({'id':f's{i:03d}','coverage_gain':float(gain.mean()),
                'score_focus_gain':weighted(gain,.25+rank),'uncertainty_focus_gain':weighted(gain,.05+uncertain),
                'available_recording_contacts':int((self.dist[:,i]>0).sum()),
                'nearest_recording_contacts':[f'c{k:03d}' for k in np.argsort(np.where(self.dist[:,i]>0,self.dist[:,i],np.inf))[:5]]})
        return rows

    def public(self):
        matrix=np.column_stack([self.predictions[f] for f in FAMILIES]);disagreement=matrix.std(axis=1)
        return {'task':'Choose the next diagnostic SPES measurement in a retrospective replay. This does not deliver treatment.',
            'budget':{'maximum_measurements':8,'used':len(self.selected)},'evidence':{
              'geometry':{'definition':'Known distances and released structural availability; coverage uses exp(-distance_mm/40). Gains are geometric heuristics, not predicted SOZ-AUROC gains.',
                  'candidates':self.candidate_table()},
              'observed':{'queries':self.history,'selected':[f's{i:03d}' for i in self.selected]},
              'model_uncertainty':{'definition':'Scores recomputed using ONLY observed stimulation columns. Scorers trained without this patient. Disagreement is descriptive, not calibrated error probability.',
                'model_order':FAMILIES,'contacts':[{'id':f'c{i:03d}','scores':[round(float(v),6) for v in matrix[i]],'disagreement':round(float(disagreement[i]),6)} for i in range(self.n)]},
              'limitations':['Unobserved response waveforms and full-response predictions are inaccessible.',
                  'Clinical SOZ labels, outcome and test accuracy are hidden until all eight queries finish.',
                  'Known released availability is an assumption; real-world acquisition cost and safety are not modeled.',
                  'The final ranking is always the HAT applied to observed measurements; agents cannot write probabilities.']}}

    def final_judge(self):
        if len(self.selected)!=8:raise ValueError('Scoring requires a complete eight-measurement trajectory')
        from sklearn.metrics import roc_auc_score,average_precision_score
        return {'patient':int(self.subject[-2:]),'outer':self.outer,'AUROC':float(roc_auc_score(self.__labels,self.predictions['hat'])),
                'average_precision':float(average_precision_score(self.__labels,self.predictions['hat'])),
                'prediction':self.predictions['hat'].tolist(),'queries':[f's{i:03d}' for i in self.selected],
                'contacts':self.n,'SOZ':int(self.__labels.sum())}

def numeric_choice(public,policy,rng=None):
    rows=public['evidence']['geometry']['candidates']
    if not rows:raise ValueError('No available queries')
    if policy.startswith('random'):
        if rng is None:raise ValueError('Random policy requires seeded generator')
        return rows[int(rng.integers(len(rows)))]['id']
    key={'coverage':'coverage_gain','score_focus':'score_focus_gain','uncertainty':'uncertainty_focus_gain'}[policy]
    return min(rows,key=lambda r:(-r[key],r['id']))['id']
