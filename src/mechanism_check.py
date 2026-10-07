"""Leakage mechanism, group-weighted accuracy, per-fold composition and readout counts (v6, Stage 2).
Run from the folder holding the cv.py outputs (oof_*.npy, labels.npy, paths.npy)."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import cv, numpy as np, collections, struct, pandas as pd
L=cv.labels; G=cv.groups; paths=np.load('paths.npy')
size=collections.Counter(G); multi=np.array([size[g]>1 for g in G])
acc=lambda P,m:[100*(P[s].argmax(1)==L)[m].mean() for s in range(3)]
for t in ['resnet50_crop_imagelevel','resnet50_crop_RANDOMgroups','resnet50_crop_grouped']:
    P=np.load(f'oof_{t}.npy'); a,b=acc(P,multi),acc(P,~multi)
    gw=[100*np.mean([(P[s].argmax(1)==L)[G==g].mean() for g in np.unique(G)]) for s in range(3)]
    print(f"{t}: multi-image groups {np.mean(a):.1f}±{np.std(a,ddof=1):.1f}; single-image {np.mean(b):.1f}±{np.std(b,ddof=1):.1f}; group-weighted {np.mean(gw):.1f}±{np.std(gw,ddof=1):.1f}")
rows=[]
for s in cv.SEEDS:
    for k,tr,va,te in cv.splits(s,True):
        for part,idx in [('train',tr),('val',va),('test',te)]:
            for c in range(6):
                m=idx[L[idx]==c]; rows.append(dict(seed=s,fold=k,part=part,cls=cv.CLASSES[c],images=int(len(m)),groups=int(len(set(G[m])))))
pd.DataFrame(rows).to_csv('fold_composition_leaf_grouped.csv',index=False)
def has_measurement(f):  # FLIR FFF record type 33 (measurement tools: spot meter/readout)
    b=open(f,'rb').read(); i=2; segs=[]
    while i<len(b)-4:
        m=b[i+1]
        if m==0xDA: break
        n=struct.unpack('>H',b[i+2:i+4])[0]; seg=b[i+4:i+2+n]
        if m==0xE1 and seg[:5]==b'FLIR\x00': segs.append(seg)
        i+=2+n
    d=b''.join(s[8:] for s in segs); _,off,n=struct.unpack('>III',d[20:32])
    return any(struct.unpack('>H',d[off+k*32:off+k*32+2])[0]==33 and struct.unpack('>I',d[off+k*32+16:off+k*32+20])[0]>0 for k in range(n))
ro=np.array([has_measurement(str(p)) for p in paths]); print('readout/reticle images:',int(ro.sum()),dict(zip(cv.CLASSES,[int(ro[L==c].sum()) for c in range(6)])))
