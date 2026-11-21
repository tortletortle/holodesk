import numpy as np
from scipy.special import ndtr
from geometry import *
import csv

def experiment(seed,noise,angle):
 r=np.random.default_rng(seed); n=1200; T=12; a=np.deg2rad(angle); c=np.array([0,-800*np.cos(a),800*np.sin(a)]); l=np.array([350,250,650]); R=basis(c); f=900
 h=np.r_[np.zeros(n//2),r.uniform(.3,5,n//4),r.uniform(5,50,n//4)]
 p=np.column_stack([r.uniform(-200,200,n),r.uniform(-150,150,n),h]); p=np.repeat(p,T,axis=0); y=h==0
 s=l+(p-l)*(l[2]/(l[2]-p[:,2]))[:,None]; A=project(p,c,R,f)+r.normal(0,noise,(n*T,2)); B=project(s,c,R,f)+r.normal(0,noise,(n*T,2)); bad=r.random(n*T)<.08; B[bad]+=r.normal(0,30,(bad.sum(),2))
 def est(a,b):
  u=ray(a,R,f); sh=plane(c,ray(b,R,f)); v=unit(sh-l); P=reconstruct(c,l,u,sh); cross=np.cross(u,v); residual=np.abs(cross@(c-l))/np.maximum(np.linalg.norm(cross,axis=1),1e-9); return P[:,2],residual
 z,res=est(A,B); jac=[]
 for k in range(4):
  aa=A.copy(); bb=B.copy(); (aa if k<2 else bb)[:,k%2]+=.05; jac.append((est(aa,bb)[0]-z)/.05)
 sig=np.maximum(noise*np.linalg.norm(jac,axis=0),.05); valid=res<3*noise*1.5
 Z=z.reshape(n,T); S=sig.reshape(n,T); V=valid.reshape(n,T)
 def metric(pred,known):
  tp=np.sum(y&pred&known); fp=np.sum(~y&pred&known); fn=np.sum(y&~(pred&known)); return [2*tp/max(2*tp+fp+fn,1),fp/(~y).sum(),known.mean(),tp/y.sum()]
 base=metric(Z[:,-1]<=2,np.ones(n,bool)); prob=ndtr((.8-Z[:,-1])/S[:,-1]); single=metric(prob>=.9,V[:,-1] & ((prob>=.9)|(prob<=.1)))
 w=V/S**2; sw=w.sum(1); mean=(w*Z).sum(1)/np.maximum(sw,1e-9); sd=1/np.sqrt(np.maximum(sw,1e-9)); pr=ndtr((.8-mean)/sd); known=(V.sum(1)>=4)&((pr>=.9)|(pr<=.1)); fused=metric(pr>=.9,known)
 return np.array([base,single,fused])
if __name__=='__main__':
 with open('results.csv','w') as f:
  w=csv.writer(f); w.writerow(['angle','noise','method','F1_all','hover_FPR','coverage','contact_recall'])
  for angle in [45,70,90]:
   for noise in [.5,1,2]:
    a=np.mean([experiment(s,noise,angle) for s in range(100,105)],axis=0)
    for name,row in zip(['V01','V02_single','V02_fused'],a): w.writerow([angle,noise,name,*row])
