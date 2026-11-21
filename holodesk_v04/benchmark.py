import numpy as np, csv
from geometry import *
from scipy.special import ndtr

def run(seed,noise,angle):
 r=np.random.default_rng(seed); n=600; T=60; dt=1/30; a=np.deg2rad(angle); c=np.array([0,-800*np.cos(a),800*np.sin(a)]); l=np.array([350,250,650]); R=basis(c); f=900
 t=np.arange(T)[None,:]; start=r.integers(12,22,(n,1)); dur=r.integers(5,20,(n,1)); end=start+dur; touch=np.arange(n)[:,None]<n//2; floor=np.where(touch,0,r.uniform(.3,5,(n,1))); speed=r.uniform(1,5,(n,1)); h=floor+np.minimum(40,speed*np.maximum(np.maximum(start-t,t-end+1),0)); y=touch&(t>=start)&(t<end)
 p=np.stack([np.broadcast_to(r.uniform(-200,200,(n,1)),h.shape),np.broadcast_to(r.uniform(-150,150,(n,1)),h.shape),h],-1).reshape(-1,3); s=l+(p-l)*(l[2]/(l[2]-p[:,2]))[:,None]; A=project(p,c,R,f)+r.normal(0,noise,(n*T,2)); B=project(s,c,R,f)+r.normal(0,noise,(n*T,2)); bad=r.random(n*T)<.08; B[bad]+=r.normal(0,30,(bad.sum(),2))
 def est(a,b):
  u=ray(a,R,f); sh=plane(c,ray(b,R,f)); v=unit(sh-l); cross=np.cross(u,v); return reconstruct(c,l,u,sh)[:,2],np.abs(cross@(c-l))/np.maximum(np.linalg.norm(cross,axis=1),1e-9)
 z,res=est(A,B); jac=[]
 for k in range(4):
  aa=A.copy(); bb=B.copy(); (aa if k<2 else bb)[:,k%2]+=.05; jac.append((est(aa,bb)[0]-z)/.05)
 Z=z.reshape(n,T); S=np.maximum(noise*np.linalg.norm(jac,axis=0),.05).reshape(n,T); V=(res<4.5*noise).reshape(n,T); preds=[Z<=2,np.zeros_like(y),np.zeros_like(y),np.zeros_like(y),np.zeros_like(y),np.zeros_like(y)]; x=np.zeros((n,2)); x[:,0]=np.where(V[:,0],Z[:,0],30); P=np.tile(np.diag([100.,10000.]),(n,1,1)); F=np.array([[1,dt],[0,1]]); g=np.array([dt**2/2,dt]); Q=900**2*np.outer(g,g); active=np.zeros(n,bool); evidence=np.zeros(n); active2=active.copy(); ev2=evidence.copy()
 contact=np.full(n,.01); free_h=np.maximum(Z[:,0],0); free_v=np.zeros(n); free_var=np.full(n,25.); on=np.zeros(n,bool); on2=on.copy()
 for k in range(T):
  j=max(0,k-11); w=V[:,j:k+1]/S[:,j:k+1]**2; sw=w.sum(1); m=(w*Z[:,j:k+1]).sum(1)/np.maximum(sw,1e-9); preds[1][:,k]=(V[:,j:k+1].sum(1)>=4)&(ndtr((.8-m)*np.sqrt(sw))>=.9)
  x=x@F.T; P=F@P@F.T+Q; innov=Z[:,k]-x[:,0]; var=P[:,0,0]+S[:,k]**2; valid=V[:,k]&(abs(innov)<5*np.sqrt(var)); K=P[:,:,0]/var[:,None]*valid[:,None]; x+=K*innov[:,None]; P-=K[:,:,None]*P[:,None,0,:].copy(); sd=np.sqrt(np.maximum(P[:,0,0],.02)); prob=ndtr((1.3-x[:,0])/sd); ll=np.clip(np.log(np.clip(prob,1e-5,1-1e-5)/np.clip(1-prob,1e-5,1)), -3,3); evidence=np.clip(.65*evidence+np.where(valid,ll,0),-5,5); active=np.where(active,(evidence>-.7)&(x[:,0]<3), (evidence>1.6)&(abs(x[:,1])<85)&valid); preds[2][:,k]=active
  pr=ndtr((1.3-Z[:,k])/S[:,k]); ll2=np.clip(np.log(np.clip(pr,1e-5,1-1e-5)/np.clip(1-pr,1e-5,1)),-3,3); ev2=np.clip(.65*ev2+np.where(V[:,k],ll2,0),-5,5); active2=np.where(active2,ev2>-.7,(ev2>1.6)&V[:,k]); preds[3][:,k]=active2
  prior_h=np.maximum(free_h+free_v*dt,0.35); fv=free_var+2.0; vc=S[:,k]**2+.20**2; vf=S[:,k]**2+fv
  lc=np.exp(-.5*Z[:,k]**2/vc)/np.sqrt(vc); lf=np.exp(-.5*(Z[:,k]-prior_h)**2/vf)/np.sqrt(vf); lc=.97*lc+.03/80; lf=.97*lf+.03/80
  enter=np.where((prior_h<5)&(free_v<30),.20,.005); prior=contact*.90+(1-contact)*enter; contact=np.where(V[:,k],prior*lc/(prior*lc+(1-prior)*lf),prior)
  gain=fv/vf; update=np.where(V[:,k],gain*(Z[:,k]-prior_h),0); new_h=np.maximum(prior_h+update,.35); free_v=np.clip(.65*free_v+.35*(new_h-free_h)/dt,-160,160); free_h=new_h; free_var=np.where(V[:,k],(1-gain)*fv,fv)
  on=np.where(on,(contact>.30)&(Z[:,k]<4), (contact>.72)&V[:,k]); preds[4][:,k]=on; on2=np.where(on2,(contact>.45)&(Z[:,k]<4),(contact>.90)&V[:,k]); preds[5][:,k]=on2
 out=[]
 for b in preds:
  tp=(b&y).sum(); fp=(b&~y).sum(); fn=(~b&y).sum(); detected=(b&y).any(1)[:n//2]; first=np.argmax(b&y,axis=1)[:n//2]; lag=(first-start[:n//2,0])[detected]*1000/30; hover=b[n//2:]&(h[n//2:]<=5); early=(b[:n//2]&(t<start[:n//2])).any(1); clean=detected&~early; cleanlag=(first-start[:n//2,0])[clean]*1000/30; hoverevent=b[n//2:].any(1).mean(); out.append([early.mean(),clean.mean(),np.mean(cleanlag) if len(cleanlag) else np.nan,hoverevent,2*tp/(2*tp+fp+fn),fp/(~y).sum(),hover.sum()/max((h[n//2:]<=5).sum(),1),tp/y.sum(),detected.mean(),np.mean(lag) if len(lag) else np.nan])
 return out
if __name__=='__main__':
 with open('results.csv','w') as f:
  w=csv.writer(f); w.writerow(['angle','noise','method','early_event_rate','clean_event_recall','clean_latency_ms','hover_event_FPR','frame_F1','FPR','near_hover_FPR','recall','event_recall','detected_latency_ms'])
  for a in [45,70,90]:
   for s in [.5,1,2]:
    x=np.mean([run(i,s,a) for i in range(410,415)],0)
    for name,row in zip(['V01','V02_rolling','V03_KF','V03_noKF','V04_competing','V04_conservative'],x): w.writerow([a,s,name,*row])
