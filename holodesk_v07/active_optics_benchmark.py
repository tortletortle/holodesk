"""V0.7 active optical correspondence sensitivity study.
Synthetic pinhole camera/projector model only; not real-device evidence.
"""
from pathlib import Path
import csv
import numpy as np

def unit(v):
    n=np.linalg.norm(v,axis=-1,keepdims=True)
    return v/np.maximum(n,1e-12)

def look_at(center,target):
    z=unit(target-center)
    up=np.array([0.,0.,1.])
    x=unit(np.cross(z,up))
    y=np.cross(z,x)
    return np.stack([x,y,z],axis=1)

def project(points,center,R,f):
    q=(points-center)@R
    return f*q[:,:2]/q[:,2:3]

def rays(pixels,R,f):
    return unit(np.column_stack([pixels/f,np.ones(len(pixels))])@R.T)

def triangulate(c,uc,p,up,f):
    a=rays(uc,RC,f); b=rays(up,RP,f)
    w=c-p; ab=np.sum(a*b,axis=1); d=a@w; e=b@w
    den=np.maximum(1-ab*ab,1e-9)
    ta=(ab*e-d)/den; tb=(e-ab*d)/den
    pa=c+ta[:,None]*a; pb=p+tb[:,None]*b
    return (pa+pb)/2

# Fixed, documented synthetic rig geometry in millimetres.
C=np.array([-300.,-650.,650.]); P=np.array([320.,-120.,520.])
RC=look_at(C,np.array([0.,0.,0.])); RP=look_at(P,np.array([0.,0.,0.]))
F=850.

def run(seed,n,ambient,pattern_contrast,base_noise):
    rng=np.random.default_rng(seed)
    xyz=np.column_stack([rng.uniform(-180,180,n),rng.uniform(-130,130,n),rng.uniform(0,40,n)])
    uc=project(xyz,C,RC,F); up=project(xyz,P,RP,F)
    # Detection noise increases as active pattern contrast is diluted by ambient light.
    sigma=base_noise*np.sqrt(1+ambient)/(max(pattern_contrast,0.03))
    uc_n=uc+rng.normal(0,sigma,(n,2))
    up_n=up+rng.normal(0,sigma,(n,2))
    est=triangulate(C,uc_n,P,up_n,F)
    err=np.linalg.norm(est-xyz,axis=1)
    h_err=np.abs(est[:,2]-xyz[:,2])
    return float(np.median(err)),float(np.percentile(err,95)),float(np.median(h_err)),float(np.percentile(h_err,95)),float(sigma)

def main():
    out=Path(__file__).parent/'results.csv'
    rows=[]
    for ambient in [0.25,1.,4.,9.]:
        for contrast in [0.15,0.3,0.6,1.0]:
            vals=np.mean([run(seed,10000,ambient,contrast,.25) for seed in range(5)],axis=0)
            rows.append([ambient,contrast,*vals])
    with out.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['ambient_to_reference','pattern_contrast','median_3d_error_mm','p95_3d_error_mm','median_height_error_mm','p95_height_error_mm','effective_pixel_sigma'])
        w.writerows(rows)
    print(out.read_text())

if __name__=='__main__':
    main()
