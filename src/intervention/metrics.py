import numpy as np
from scipy.ndimage import uniform_filter1d


def ictal_mask(states,action=(),window=100,threshold=-1.0):
    # Slow mean of x1; development detector, not a clinical seizure label.
    mask=uniform_filter1d(states[:,0,:],size=window,axis=0,mode='nearest')>threshold
    if action:
        mask[:,list(action)]=False
    return mask


def burden(states,action=(),transient=1000,period=1.):
    mask=ictal_mask(states,action)
    return float(mask[int(transient/period):].sum()*period)


def reduction(b,untreated):
    if untreated<=0:
        raise ValueError('Ineligible zero untreated burden')
    return 1-b/untreated
