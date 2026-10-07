"""Figures 4 (confusion matrix and ROC curves, seed 0) and 5 (Grad-CAM), and perclass.json (Table 5; copied to results/).
Figures 1-3 and 6 are drawn by the scripts in figures/. Run from the repository root after src/cv.py."""
import sys,json; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from common import *
from cv import head, splits, labels, groups
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
OUT='figs/'; os.makedirs(OUT,exist_ok=True); W='weights/'
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans','axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
NAMES=['BLB','Blast','Healthy','Hispa','Leaf folder','Leaf spot']
PAL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']; LS=['-','--','-.',':','-','--']
# ---------- per-class table over 3 seeds (pooled OOF) + ROC/CM (seed 0)
P=np.load('oof_resnet50_crop_grouped.npy')
pcs=[precision_recall_fscore_support(labels,P[s].argmax(1),labels=range(6),zero_division=0) for s in range(3)]
aucs=[[auc(*roc_curve(labels==c,P[s][:,c])[:2]) for c in range(6)] for s in range(3)]
per={}
for c in range(6):
    per[NAMES[c]]=dict(n=int((labels==c).sum()),groups=int(len(set(groups[labels==c]))),
      P=(np.mean([p[0][c] for p in pcs]),np.std([p[0][c] for p in pcs],ddof=1)),
      R=(np.mean([p[1][c] for p in pcs]),np.std([p[1][c] for p in pcs],ddof=1)),
      F=(np.mean([p[2][c] for p in pcs]),np.std([p[2][c] for p in pcs],ddof=1)),
      AUC=(np.mean([a[c] for a in aucs]),np.std([a[c] for a in aucs],ddof=1)))
json.dump(per,open('perclass.json','w'),indent=1)
fig,ax=plt.subplots(1,2,figsize=(7.2,3.1),dpi=300)
cm=confusion_matrix(labels,P[0].argmax(1)); cmn=cm/cm.sum(1,keepdims=True)
im=ax[0].imshow(cmn,cmap='Blues',vmin=0,vmax=1)
for i in range(6):
    for j in range(6):
        ax[0].text(j,i,f"{cm[i,j]}",ha='center',va='center',fontsize=6.5,color='white' if cmn[i,j]>0.55 else '#0b0b0b')
ax[0].set_xticks(range(6),NAMES,rotation=35,ha='right'); ax[0].set_yticks(range(6),NAMES)
ax[0].set_xlabel('Predicted class'); ax[0].set_ylabel('True class'); ax[0].set_title('(a) Pooled confusion matrix (seed 0, n = 636)',fontsize=7)
cb=plt.colorbar(im,ax=ax[0],fraction=0.046,pad=0.03); cb.set_label('Row-normalized recall',fontsize=6); cb.ax.tick_params(labelsize=6)
for c in range(6):
    f,t,_=roc_curve(labels==c,P[0][:,c]); ax[1].plot(f,t,color=PAL[c],ls=LS[c],lw=1.2,label=f"{NAMES[c]} (AUC {auc(f,t):.3f})")
ax[1].plot([0,1],[0,1],color='#b0afaa',lw=0.8,ls=':')
ax[1].set_xlabel('False positive rate'); ax[1].set_ylabel('True positive rate'); ax[1].set_title('(b) One-vs-rest ROC curves (seed 0)',fontsize=7)
ax[1].legend(fontsize=5.8,frameon=False,loc='lower right'); ax[1].grid(color='#e6e5e0',lw=0.5); ax[1].set_xlim(0,1); ax[1].set_ylim(0,1.01)
for s in ['top','right']: ax[1].spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+'fig4_cm_roc.png',dpi=300); plt.close()
# Figure 3 (evaluation protocols) is drawn by figures/fig3_inflation.py.
# ---------- Grad-CAM (seed 0, fold 0 model)
m0=resnet50_imagenet(W)
conv=m0.get_layer('conv5_block3_out').output; bb=keras.Model(m0.inputs,[conv,keras.layers.GlobalAveragePooling2D()(conv)])
F=np.load('feat_resnet50_crop.npy'); Y=keras.utils.to_categorical(labels,6); X=np.load('X224_crop.npy')
k,tr,va,te=next(splits(0,True)); keras.utils.set_random_seed(0)
h=head(F.shape[1]); h.fit(F[tr],Y[tr],validation_data=(F[va],Y[va]),epochs=100,batch_size=32,verbose=0,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)])
d1,d2=h.layers[2],h.layers[4]; W2,b2=d2.get_weights()
def logits_from_pool(g): return tf.matmul(tf.nn.relu(tf.matmul(g,d1.kernel)+d1.bias),W2)+b2
pp=keras.applications.resnet50.preprocess_input
prob=h.predict(F[te],verbose=0); pred=prob.argmax(1); yt=labels[te]
sel=[]
for c in range(6):
    idx=[i for i in range(len(te)) if yt[i]==c and pred[i]==c]
    if idx: sel.append((te[max(idx,key=lambda i:prob[i,c])],c,c,prob[max(idx,key=lambda i:prob[i,c]),c]))
wrong=[i for i in range(len(te)) if pred[i]!=yt[i]]; wrong.sort(key=lambda i:-prob[i,pred[i]])
used=set()
for i in wrong:
    if yt[i] in used: continue
    sel.append((te[i],yt[i],pred[i],prob[i,pred[i]])); used.add(yt[i])
    if len(used)==2: break
def gradcam(img,c):
    x=tf.convert_to_tensor(pp(img[None].copy()))
    with tf.GradientTape() as tape:
        A,g=bb(x,training=False); tape.watch(A); g=tf.reduce_mean(A,axis=(1,2)); z=logits_from_pool(g)[:,c]
    gr=tape.gradient(z,A)[0]; a=tf.reduce_mean(gr,axis=(0,1)); L=tf.nn.relu(tf.reduce_sum(A[0]*a,-1)).numpy()
    L=L/(L.max()+1e-8); return tf.image.resize(L[...,None],(224,224),'bilinear').numpy()[...,0]
fig,ax=plt.subplots(2,len(sel),figsize=(7.2,2.35),dpi=300)
for j,(i,t,p,pr) in enumerate(sel):
    L=gradcam(X[i],p)
    ax[0,j].imshow(np.clip(X[i],0,255).astype('uint8')); ax[0,j].contour(L,levels=[0.5],colors='white',linewidths=0.7)
    ax[1,j].imshow(L,cmap='Blues',vmin=0,vmax=1)
    tag=f"True: {NAMES[t]}\nPred: {NAMES[p]} ({pr*100:.0f}%)"
    ax[0,j].set_title(tag,fontsize=5.3,color='#0b0b0b' if t==p else '#d03b3b')
    for a in ax[:,j]: a.axis('off')
    print(os.path.basename(str(np.load('paths.npy')[i])), NAMES[t],NAMES[p],round(float(pr),3))
plt.tight_layout(pad=0.2); plt.savefig(OUT+'fig5_gradcam.png',dpi=300); plt.close()
json.dump([[int(i),int(t),int(p),float(pr)] for i,t,p,pr in sel],open('gradcam_sel.json','w'))
print("fig5 done")
