import cv2 as cv, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from pathlib import Path
P=Path(__file__).parent

def sample(r, y, strength, noise):
    # Synthetic side-view crop, not a physical hand renderer.
    H=W=96; yy,xx=np.mgrid[:H,:W]; cx=r.uniform(43,53); cy=r.uniform(43,51); a=r.uniform(13,22); b=r.uniform(23,32)
    flatten=r.uniform(0,.10)+y*strength*r.uniform(.5,1.5)
    mask=((xx-cx)/a)**2+((yy-cy)/b)**2<=1
    mask &= yy<=cy+b*(1-flatten)
    bg=np.array([r.uniform(105,175)]*3); skin=np.array([r.uniform(65,105),r.uniform(125,170),r.uniform(175,225)])
    im=np.where(mask[...,None],skin,bg).astype(float); im+=r.uniform(-25,25)*(xx[...,None]/W-.5)
    M=cv.getRotationMatrix2D((48,48),r.uniform(-15,15),r.uniform(.8,1.1)); im=cv.warpAffine(im,M,(W,H),borderMode=cv.BORDER_REFLECT)
    im=cv.GaussianBlur(im,(5,5),r.uniform(.5,1.5)); im+=r.normal(0,noise,im.shape); return np.uint8(np.clip(im,0,255))

def features(im):
    z=im.astype(float); m=np.uint8((z[:,:,2]-z[:,:,0]>42)*255); m=cv.morphologyEx(m,cv.MORPH_OPEN,np.ones((3,3),np.uint8)); m=cv.morphologyEx(m,cv.MORPH_CLOSE,np.ones((3,3),np.uint8)); cs,_=cv.findContours(m,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    if not cs:return np.zeros(11)
    c=max(cs,key=cv.contourArea); area=cv.contourArea(c); x,y,w,h=cv.boundingRect(c); hull=cv.contourArea(cv.convexHull(c)); peri=cv.arcLength(c,True)
    # align using principal axis; sign chosen toward image bottom
    pts=c[:,0,:].astype(float); _,_,v=np.linalg.svd(pts-pts.mean(0),full_matrices=False); axis=v[0]*np.sign(v[0,1]); q=(pts-pts.mean(0))@axis; t=(pts-pts.mean(0))@np.array([-axis[1],axis[0]]); span=np.ptp(q); width=max(np.ptp(t),1)
    widths=[]
    for z0 in [.02,.08,.15,.25,.4]:
        sel=t[q>=q.max()-span*z0]; widths.append(np.ptp(sel)/width if len(sel)>1 else 0)
    return np.array([area/(w*h),area/max(hull,1),4*np.pi*area/max(peri**2,1),w/h,span/width,*widths,1])

def dataset(seed,n,strength,noise):
    r=np.random.default_rng(seed); y=np.arange(n)%2; r.shuffle(y); X=np.array([features(sample(r,i,strength,noise)) for i in y]); return X,y

def main():
    X,y=dataset(601,2400,.18,5); model=RandomForestClassifier(n_estimators=100,max_depth=7,random_state=1,n_jobs=2).fit(X,y)
    rows=[]
    for s in [0,.05,.10,.18,.28]:
      for noise in [5,15,25]:
        X,y=dataset(800+int(s*1000)+noise,800,s,noise); p=model.predict_proba(X)[:,1]; pred=p>=.5
        rows.append(dict(strength=s,noise=noise,f1=f1_score(y,pred),recall=recall_score(y,pred),fpr=pred[y==0].mean(),auc=roc_auc_score(y,p),segmentation_success=X[:,-1].mean()))
    df=pd.DataFrame(rows); df.to_csv(P/'results.csv',index=False); print(df.to_string(index=False))
    r=np.random.default_rng(888); cv.imwrite(str(P/'synthetic_examples.png'),np.vstack([np.hstack([sample(r,y,.18,5) for _ in range(6)]) for y in [0,1]]))
if __name__=='__main__':main()
