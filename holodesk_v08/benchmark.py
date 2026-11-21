"""HoloDesk V0.8: temporal structured-light code sensitivity benchmark.
Synthetic 1-D projector coordinate decoder; not a physical camera simulation.
"""
from pathlib import Path
import csv
import numpy as np
NPIX=256

def gray_codebook():
    x=np.arange(NPIX,dtype=np.uint16); g=x^(x>>1)
    return ((g[:,None] >> np.arange(7,-1,-1)) & 1).astype(float)

def random_codebook(seed=2026):
    r=np.random.default_rng(seed); words=np.arange(NPIX,dtype=np.uint16); r.shuffle(words)
    return ((words[:,None] >> np.arange(7,-1,-1)) & 1).astype(float)

def phase_codebook():
    x=np.arange(NPIX); frames=[]
    for period in (32.,256.):
        for k in range(3): frames.append(.5+.5*np.cos(2*np.pi*x/period+2*np.pi*k/3))
    return np.stack(frames,axis=1)

def decode_gray(obs):
    bits=(obs>=.5).astype(np.uint16); g=np.zeros(len(obs),dtype=np.uint16)
    for j in range(8): g |= bits[:,j] << (7-j)
    b=g.copy()
    for shift in (1,2,4): b ^= b >> shift
    return np.clip(b.astype(int),0,NPIX-1)

def decode_random(obs,book):
    bits=(obs>=.5).astype(np.uint8)
    dist=np.count_nonzero(bits[:,None,:] != book[None,:,:],axis=2)
    order=np.partition(dist,1,axis=1)[:,:2]; pred=np.argmin(dist,axis=1)
    return pred,order[:,1] != order[:,0]

def decode_phase(obs):
    phases=[]
    for j in (0,3):
        a,b,c=obs[:,j:j+3].T
        phases.append(np.mod(np.arctan2(np.sqrt(3)*(c-b),2*a-b-c),2*np.pi))
    # 256-pixel phase period spans the full projector coordinate range.
    return np.clip(np.rint(phases[1]*256/(2*np.pi)).astype(int),0,NPIX-1)

def observe(book,truth,contrast,ambient,noise,drift,rng):
    n=len(truth); frames=book.shape[1]; obs=np.empty((n,frames))
    for k in range(frames):
        x=np.clip(np.rint(truth+drift*(k-(frames-1)/2)).astype(int),0,NPIX-1)
        sigma_raw=np.sqrt(noise**2 + 0.01*ambient)\n        obs[:,k]=ambient+contrast*book[x,k]+rng.normal(0,sigma_raw,n)
    return np.clip((obs-ambient)/max(contrast,1e-6),-1,2)

def score(pred,valid,truth):
    e=np.abs(pred-truth); ok=valid
    return [float(np.mean(e[ok])) if np.any(ok) else float('nan'),
            float(np.percentile(e[ok],95)) if np.any(ok) else float('nan'),
            float(np.mean(e[ok]<=1)) if np.any(ok) else float('nan'),
            float(1-np.mean(ok)),float(np.mean(e[ok]>8)) if np.any(ok) else float('nan')]

def run(seed,contrast,ambient,noise,drift,n=800):
    rng=np.random.default_rng(seed); truth=rng.integers(0,NPIX,n)
    books={'gray8':gray_codebook(),'random8':random_codebook(),'phase6':phase_codebook()}; out=[]
    for name,book in books.items():
        obs=observe(book,truth,contrast,ambient,noise,drift,rng)
        if name=='gray8': pred=decode_gray(obs); valid=np.ones(n,bool)
        elif name=='random8': pred,valid=decode_random(obs,book)
        else: pred=decode_phase(obs); valid=np.ones(n,bool)
        out.append((name,score(pred,valid,truth),book.shape[1]))
    return out

def main():
    rows=[]
    for contrast in (.2,.6):
      for ambient in (.1,1.0):
       for noise in (.03,.08):
        for drift in (0,1.0):
         agg={k:[] for k in ('gray8','random8','phase6')}
         for seed in range(2):
          for name,metrics,frames in run(900+seed,contrast,ambient,noise,drift): agg[name].append(metrics)
         for name,vals in agg.items():
          m=np.nanmean(vals,axis=0); frame_count={'gray8':8,'random8':8,'phase6':6}[name]
          rows.append([contrast,ambient,noise,drift,name,frame_count,*m])
    out=Path(__file__).parent/'results.csv'
    with out.open('w',newline='') as f:
      w=csv.writer(f); w.writerow(['contrast','ambient_offset','gaussian_noise_sigma','motion_px_per_frame','code','frames','mae_px','p95_error_px','within_1px_rate','rejection_rate','catastrophic_gt8px_rate']); w.writerows(rows)
    print(out)
if __name__=='__main__': main()
