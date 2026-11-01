"""Observation-gated replay using verified anatomical-pair actions."""
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from experiments.spes_agent_v3.common import read_json,sha
from experiments.spes_agent_v4.neural import FAMILIES,load_array,upstream_models
from experiments.spes_agent_v8.audit import independent_auc
from .mapping import physical_input


class PhysicalAcquisition:
    def __init__(self,project,outer,subject,device='cuda'):
        import torch
        self.project=Path(project);self.outer=outer;self.subject=subject;self.device=device;self.selected=[];self.history=[];self.models={}
        training=self.project/'outputs/spes_agent_v4/multitrial';p=read_json(training/'neural_protocol.json');job=p['jobs'][outer]
        if subject not in job['predict'] or subject in job['train']+job['early_stop']:raise ValueError('Patient not held out')
        folder=self.project/'outputs/spes_physical/maps'/subject;self.mapping=read_json(folder/'receipt.json');data=self.project/'vendor/norris2026_hat/data'
        if sha(folder/'index.npz')!=self.mapping['index_sha256'] or sha(data/f'X_{subject}.npz')!=self.mapping['raw_sha256']:raise ValueError('Verified mapping/source differs')
        self.index=load_array(folder/'index.npz');self.__raw=load_array(data/f'X_{subject}.npz')[:,:20];self.__labels=load_array(data/f'y_{subject}.npz');self.n=len(self.__labels)
        self.pairs=self.mapping['pair_rows'];centers=np.array([r['center_mm'] for r in self.pairs]);coords=load_array(data/f'coords_{subject}.npz')
        self.distance=cdist(coords,centers);self.kernel=np.where(self.index>=0,np.exp(-self.distance/40),0.)
        self.available=[q for q in range(len(self.pairs)) if (self.index[:,q]>=0).any() and len(self.pairs[q]['recorded_conditions'])==1]
        if len(self.available)<8:raise ValueError('Fewer than eight geometry-verified single-condition anatomical pairs')
        self.predictions={name:np.full(self.n,.5) for name in FAMILIES};mods=upstream_models(self.project)
        for family in FAMILIES:
            folder=training/f'neural/outer{outer}/{family}';receipt=read_json(folder/'receipt.json')
            if sha(folder/'checkpoint.pth')!=receipt['checkpoint_sha256']:raise ValueError('Trained model differs')
            if family==FAMILIES[0]:self.stats=receipt['normalisation']
            elif self.stats!=receipt['normalisation']:raise ValueError('Scorer preprocessing differs')
            model,_,_=mods.create_model(family);model.load_state_dict(torch.load(folder/'checkpoint.pth',map_location='cpu',weights_only=True)['model_state_dict']);self.models[family]=model.to(device).eval()

    def normalized_observed(self):
        x=physical_input(self.__raw,self.index,self.selected)
        for segment,stat in zip([slice(0,1),slice(1,129),slice(129,None)],self.stats):
            view=x[...,segment];mask=np.abs(view)>1e-12;view[mask]=(view[mask]-stat['mean'])/stat['std']
        return x

    def observe(self,pair):
        import torch
        pair=int(pair)
        if pair not in self.available or pair in self.selected or len(self.selected)>=8:raise ValueError('Invalid, repeated or over-budget physical query')
        old=self.predictions['hat'].copy();self.selected.append(pair);x=self.normalized_observed()
        with torch.no_grad():
            for name,model in self.models.items():
                values=[torch.sigmoid(model(torch.from_numpy(x[b:b+8]).to(self.device))[:,0]).cpu().numpy() for b in range(0,self.n,8)]
                self.predictions[name]=np.concatenate(values)
                if not np.isfinite(self.predictions[name]).all():raise ValueError('Nonfinite physical-response prediction')
        revealed=physical_input(self.__raw,self.index,[pair]);valid=revealed[:,:,0,0]>1e-12;rms=np.sqrt(np.square(revealed[:,:,0,-512:].astype(float)).mean(axis=-1))
        event={'measurement':len(self.selected),'anatomical_pair':self.pairs[pair]['id'],'contacts':self.pairs[pair]['contacts'],
          'revealed_recording_contacts':int((self.index[:,pair]>=0).sum()),'median_trial_rms':float(np.median(rms[valid])) if valid.any() else 0.,
          'mean_absolute_hat_score_change':float(np.abs(self.predictions['hat']-old).mean())}
        self.history.append(event);return event

    def public(self):
        current=self.kernel[:,self.selected].max(axis=1) if self.selected else np.zeros(self.n);hat=self.predictions['hat'];rank=rankdata(hat)/(self.n+1)
        matrix=np.column_stack([self.predictions[n] for n in FAMILIES]);uncertain=matrix.std(axis=1)+.1*(1-2*np.abs(hat-.5));rows=[]
        for q in self.available:
            if q in self.selected:continue
            gain=np.maximum(self.kernel[:,q]-current,0.);r=self.pairs[q]
            rows.append({'id':r['id'],'contacts':r['contacts'],'declared_anatomy':r['declared_anatomy'],'center_mm':r['center_mm'],
              'coverage_gain':float(gain.mean()),'score_focus_gain':float(np.average(gain,weights=.25+rank)),
              'uncertainty_focus_gain':float(np.average(gain,weights=.05+uncertain)),'available_recording_contacts':int((self.index[:,q]>=0).sum())})
        return {'task':'Choose a real recorded anatomical stimulation pair in retrospective diagnostic replay. No physical stimulation is delivered.',
          'budget':{'used':len(self.selected),'maximum_measurements':8},'evidence':{'geometry':{'candidates':rows},
            'anatomy':{'frame':self.mapping['coordinate_frame'],'source':'Whitelisted public electrode labels; no SOZ, resection, outcomes or clinical notes.'},
            'observed':{'selected':[self.pairs[q]['id'] for q in self.selected],'queries':self.history},
            'model_uncertainty':{'model_order':FAMILIES,'contacts':[{'id':f'c{i:03d}','scores':matrix[i].tolist()} for i in range(self.n)],'calibration':'Descriptive scores and disagreement, not clinical risk or calibrated error.'},
            'limitations':['Only first twenty trials; distance-ambiguous cells excluded; released availability assumed known.',
              'Pairs with multiple recorded parameter tuples excluded. No treatment, safety or physiological state carry-over model.',
              'Labels and unobserved waveforms are hidden; all final scores use actual revealed physical-pair responses.']}}

    def final_judge(self):
        if len(self.selected)!=8:raise ValueError('Eight complete physical queries required')
        return {'patient':int(self.subject[-2:]),'outer':self.outer,'AUROC':independent_auc(self.__labels,self.predictions['hat']),
          'prediction':self.predictions['hat'].tolist(),'queries':[self.pairs[q]['id'] for q in self.selected],
          'query_contacts':[self.pairs[q]['contacts'] for q in self.selected],'contacts':self.n,'SOZ':int(self.__labels.sum())}
