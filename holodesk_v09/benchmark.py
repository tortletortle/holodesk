"""V0.9 motion-compensated Gray-code benchmark.
Synthetic 1-D projector coordinate model; flow is estimated from noisy temporal tracks.
"""
from pathlib import Path
import csv
import numpy as np

N=256
TAU=np.arange(8,dtype=float)-3.5

def gray_book():
    x=np.arange(N,dtype=np.uint16); g=x^(x>>1)
    return ((g[:,None]>>np.arange(7,-1,-1))&1).astype(np.uint8)

def decode(bits):
    g=np.zeros(len(bits),dtype=np.uint16)
    for j in range(bits.shape[1]): g |= bits[:,j].astype(np.uint16) << (7-j)
    b=g.copy()
    for shift in (1,2,4): b ^= b >> shift
    return np.clip(b.astype(int),0,N-1)

def regress_motion(track):
    # Fit per-track linear image motion across the eight pattern frames.
    t=TAU-TAU.mean()
    slope=np.sum(track*t[None,:],axis=1)/np.sum(t*t)
    intercept=track.mean(axis=1)
    return intercept[:,None]+slope[:,None]*TAU[None,:]

def metrics(pred,truth):
    e=np.abs(pred-truth)
    return [float(e.mean()),float(np.median(e)),float(np.percentile(e,95)),
            float(np.mean(e<=1)),float(np.mean(e>8))]

def trial(seed,velocity,flow_sigma,n=10000):
    rng=np.random.default_rng(seed); book=gray_book()
    x0=rng.uniform(4,251,n)
    true_disp=velocity*TAU[None,:]
    observed=x0[:,None]+true_disp+rng.normal(0,.15,(n,8))
    flow_observation=true_disp+rng.normal(0,flow_sigma,(n,8))
    estimated_disp=regress_motion(flow_observation)
    raw=np.empty((n,8),dtype=np.uint8); corrected=np.empty_like(raw)
    for k in range(8):
        raw_idx=np.clip(np.rint(observed[:,k]).astype(int),0,N-1)
        corr_idx=np.clip(np.rint(observed[:,k]-estimated_disp[:,k]).astype(int),0,N-1)
        raw_signal=.1+.8*book[raw_idx,k]+rng.normal(0,.023,n)
        corr_signal=.1+.8*book[corr_idx,k]+rng.normal(0,.023,n)
        raw[:,k]=raw_signal>=.5
        corrected[:,k]=corr_signal>=.5
    return metrics(decode(raw),x0),metrics(decode(corrected),x0),float(np.sqrt(np.mean((estimated_disp-true_disp)**2)))

def main():
    rows=[]
    for velocity in (0,.25,.5,1,2,4):
      for flow_sigma in (0,.1,.25,.5,1):
        vals=[trial(seed+1200,velocity,flow_sigma) for seed in range(5)]
        for idx,name in ((0,'gray8_uncompensated'),(1,'gray8_linear_flow_compensated')):
            m=np.mean([v[idx] for v in vals],axis=0)
            rows.append([velocity,flow_sigma,name,*m,np.mean([v[2] for v in vals])])
    out=Path(__file__).parent/'results.csv'
    with out.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['motion_px_per_frame','flow_observation_noise_sigma_px','method','mae_px','median_error_px','p95_error_px','within_1px_rate','catastrophic_gt8px_rate','flow_fit_rmse_px']); w.writerows(rows)
    print(out.read_text())

if __name__=='__main__': main()
