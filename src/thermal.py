"""Run from the repository root after src/extract.py (needs paths.npy, labels.npy and the ResNet50 weights in weights/).
Results (6 Oct 2026): absolute temperature 77.5 ± 6.1% leaf-grouped, 93.0 ± 2.3% image-level; relative 74.4 ± 8.0% / 91.7 ± 2.6%;
deviation from median 74.0 ± 5.7%; temperature-level probe: random forest 67.8 ± 7.2%, logistic regression 45.6 ± 6.8% (leaf-grouped)."""
"""Stage 3 (v6): models trained on the radiometric temperature data embedded in the FLIR JPEGs.
Decodes the raw 16-bit thermal image of every file to temperature (Planck constants from the file),
builds three 3-channel inputs for the same frozen ResNet50 + head pipeline, and runs the leaf-grouped
(and image-level) protocol. Also runs a temperature-level-only confound probe."""
import sys,os,io,struct,json,time; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from PIL import Image
paths=np.load('paths.npy'); labels=np.load('labels.npy')
def flir(f):
    b=open(f,'rb').read(); i=2; segs=[]
    while i<len(b)-4:
        m=b[i+1]
        if m==0xDA: break
        L=struct.unpack('>H',b[i+2:i+4])[0]; seg=b[i+4:i+2+L]
        if m==0xE1 and seg[:5]==b'FLIR\x00': segs.append(seg)
        i+=2+L
    d=b''.join(s[8:] for s in segs); ver,off,n=struct.unpack('>III',d[20:32]); rec={}
    for k in range(n):
        typ,sub,v,idn,o,l=struct.unpack('>HHIIII',d[off+k*32:off+k*32+20])
        if typ: rec[typ]=d[o:o+l]
    return rec
def temperature(f):
    r=flir(f); raw=np.array(Image.open(io.BytesIO(r[1][32:]))).byteswap().astype(np.float64); ci=r[32]
    g=lambda o: struct.unpack('<f',ci[o:o+4])[0]
    R1,B,F,R2=g(0x58),g(0x5c),g(0x60),g(0x30c); O=struct.unpack('<i',ci[0x308:0x30c])[0]; E=g(0x20); Tr=g(0x2c)
    raw_refl=R1/(R2*(np.exp(B/Tr)-F))-O
    raw_obj=(raw-(1-E)*raw_refl)/E
    return (B/np.log(R1/(R2*(raw_obj+O))+F)-273.15).astype(np.float32)
t0=time.time()
if not os.path.exists('temps.npy'):
    T=np.stack([temperature(str(p)) for p in paths]); np.save('temps.npy',T)
T=np.load('temps.npy'); print('temps',T.shape,f'{time.time()-t0:.0f}s',flush=True)
lo,hi=np.floor(np.percentile(T,0.5)),np.ceil(np.percentile(T,99.5)); print('global range',lo,hi,flush=True)
def to_img(a):  # a in [0,1] -> 224x224x3 float (0-255), bilinear resize as for the renderings
    x=tf.image.resize(tf.convert_to_tensor(np.clip(a,0,1)[...,None]*255.0),(224,224)).numpy()
    return np.repeat(x,3,axis=-1).astype(np.float32)
inputs={
 'Tabs': lambda t: (t-lo)/(hi-lo),                                                   # absolute temperature, fixed scale
 'Trel': lambda t: (t-np.percentile(t,1))/max(np.percentile(t,99)-np.percentile(t,1),1e-3),  # per-image min-max (like auto-scaling)
 'Tdev': lambda t: (t-np.median(t)+4.0)/8.0,                                          # deviation from image median, fixed ±4 °C span
}
W='weights/'
def resnet_backbone():
    m=resnet50_imagenet(W)   # original project model if present in weights/, otherwise the Keras ImageNet ResNet50 (see common.py)
    out=keras.layers.GlobalAveragePooling2D()(m.get_layer('conv5_block3_out').output); return keras.Model(m.inputs,out)
bb=None
for name,fn in inputs.items():
    out=f'feat_resnet50_{name}.npy'
    if os.path.exists(out): continue
    if bb is None: bb=resnet_backbone()
    X=np.stack([to_img(fn(t)) for t in T]); np.save(f'X224_{name}_sample.npy',X[:12])
    Fe=bb.predict(keras.applications.resnet50.preprocess_input(X),batch_size=16,verbose=0); np.save(out,Fe.astype(np.float32))
    print('features',name,Fe.shape,f'{time.time()-t0:.0f}s',flush=True)
del bb; keras.backend.clear_session()
import cv
for name in inputs: cv.run(f'resnet50_{name}',True)
cv.run('resnet50_Tabs',False); cv.run('resnet50_Trel',False)
# confound probe: temperature level only (1st percentile, median, 99th percentile of each image)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
Q=np.stack([np.percentile(T.reshape(len(T),-1),q,axis=1) for q in (1,50,99)],1)
res={}
for nm,mk in [('logreg',lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))),('rf',lambda: RandomForestClassifier(300,random_state=0))]:
    for grouped in (True,False):
        acc=[]
        for s in cv.SEEDS:
            for k,tr,va,te in cv.splits(s,grouped):
                trv=np.concatenate([tr,va]); m=mk().fit(Q[trv],labels[trv]); acc.append(float((m.predict(Q[te])==labels[te]).mean()))
        res[f'{nm}_{"grouped" if grouped else "imagelevel"}']=(100*np.mean(acc),100*np.std(acc,ddof=1))
print('confound probe (temperature level only):',json.dumps(res),flush=True)
json.dump({'range':[float(lo),float(hi)],'probe':res,'class_median_temp':{cv_c:float(np.median(Q[labels==i,1])) for i,cv_c in enumerate(CLASSES)}},open('thermal_summary.json','w'),indent=1)
print('DONE',f'{time.time()-t0:.0f}s',flush=True)
