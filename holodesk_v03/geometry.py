import numpy as np

def unit(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)
def basis(c):
    z=unit(-c); x=unit(np.cross(z,[0,1,0])); y=np.cross(z,x); return np.stack([x,y,z],axis=1)
def project(p,c,R,f):
    q=(p-c)@R; return f*q[:,:2]/q[:,2:]
def ray(p,R,f): return unit(np.column_stack([p/f,np.ones(len(p))])@R.T)
def plane(c,u): return c-u*(c[2]/u[:,2])[:,None]
def reconstruct(c,l,u,s):
    v=unit(s-l); w=c-l; b=np.sum(u*v,axis=1); d=u@w; e=v@w; den=1-b*b
    t=(b*e-d)/den; k=(e-b*d)/den
    return (c+t[:,None]*u+l+k[:,None]*v)/2

def run(seed,noise,angle):
    rng=np.random.default_rng(seed); n=5000; a=np.deg2rad(angle); c=np.array([0,-800*np.cos(a),800*np.sin(a)]); l=np.array([350,250,650]); R=basis(c); f=900
    h=rng.uniform(0,50,n); h[:n//2]=0; p=np.column_stack([rng.uniform(-200,200,n),rng.uniform(-150,150,n),h]); s=l+(p-l)*(l[2]/(l[2]-h))[:,None]
    u=ray(project(p,c,R,f)+rng.normal(0,noise,(n,2)),R,f); sh=plane(c,ray(project(s,c,R,f)+rng.normal(0,noise,(n,2)),R,f)); q=plane(c,u)
    # Hypothetical relative-depth estimator; not measured MediaPipe output.
    A=h+rng.normal(0,8,n)+rng.normal(0,6)
    # Constant angle approximation using central differential parallax.
    center=np.array([[0.,0.,1.]]); sc=l+(center-l)*(l[2]/(l[2]-1)); qc=plane(c,unit(center-c)); scale=np.linalg.norm(sc-qc)
    B=np.linalg.norm(sh-q,axis=1)/scale; P=reconstruct(c,l,u,sh)[:,2]
    out=[]
    for pred in [A,B,P]:
        y=h==0; yh=pred<=2; tp=np.sum(y&yh); fp=np.sum(~y&yh); fn=np.sum(y&~yh); out.append([np.mean(abs(pred-h)),2*tp/(2*tp+fp+fn),fp/np.sum(~y)])
    return np.array(out)
if __name__=='__main__':
    print('angle,noise,method,MAE_mm,F1,hover_FPR')
    for angle in [45,70,90]:
        for noise in [.5,1,2]:
            r=np.mean([run(s,noise,angle) for s in range(10)],axis=0)
            for name,row in zip(['synthetic_depth','fixed_angle','PACT'],r): print(f'{angle},{noise},{name},'+','.join(f'{x:.5f}' for x in row))
