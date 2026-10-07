import sys,time,json; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from common import *
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
labels=np.load('labels.npy'); paths=np.load('paths.npy')
groups=np.array([group_id(p,l) for p,l in zip(paths,labels)])
print("groups:",len(set(groups)))
SEEDS=[0,1,2]
def head(d):
    i=keras.Input((d,)); x=keras.layers.Dropout(0.5)(i); x=keras.layers.Dense(256,activation='relu')(x)
    x=keras.layers.Dropout(0.5)(x); o=keras.layers.Dense(6,activation='softmax')(x)
    m=keras.Model(i,o); m.compile(optimizer=keras.optimizers.Adam(1e-3),loss='categorical_crossentropy',metrics=['accuracy']); return m
def splits(seed,grouped):
    if grouped:
        outer=StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(labels)),labels,groups)
    else:
        outer=StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(labels)),labels)
    for k,(tr,te) in enumerate(outer):
        if grouped:
            itr,iva=next(StratifiedGroupKFold(6,shuffle=True,random_state=seed).split(np.zeros(len(tr)),labels[tr],groups[tr]))
        else:
            itr,iva=next(StratifiedKFold(6,shuffle=True,random_state=seed).split(np.zeros(len(tr)),labels[tr]))
        yield k,tr[itr],tr[iva],te
def run(feat,grouped,cw=False,tag=None,testval=False):
    F=np.load(f'feat_{feat}.npy'); Y=keras.utils.to_categorical(labels,6)
    rows=[]; oof={}
    for s in SEEDS:
        P=np.zeros((len(labels),6))
        for k,tr,va,te in splits(s,grouped):
            keras.utils.set_random_seed(1000*s+k)
            m=head(F.shape[1])
            kw={}
            if cw:
                w=compute_class_weight('balanced',classes=np.arange(6),y=labels[tr]); kw['class_weight']=dict(enumerate(w))
            es=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
            vtr,vva=(np.concatenate([tr,va]),te) if testval else (tr,va)
            h=m.fit(F[vtr],Y[vtr],validation_data=(F[vva],Y[vva]),epochs=100,batch_size=32,callbacks=[es],verbose=0,**kw)
            p=m.predict(F[te],verbose=0); P[te]=p; yp=p.argmax(1); yt=labels[te]
            rows.append(dict(seed=s,fold=k,n_test=len(te),acc=accuracy_score(yt,yp),bacc=balanced_accuracy_score(yt,yp),
                 f1=f1_score(yt,yp,average='macro'),auc=roc_auc_score(yt,p,multi_class='ovr',average='macro',labels=np.arange(6)),
                 epochs=len(h.history['loss']),best_epoch=int(np.argmin(h.history['val_loss']))+1,
                 group_overlap=int(len(set(groups[te])&set(groups[tr])))))
            keras.backend.clear_session()
        oof[s]=P
    tag=tag or f"{feat}_{'grouped' if grouped else 'imagelevel'}{'_cw' if cw else ''}{'_testval' if testval else ''}"
    np.save(f'oof_{tag}.npy',np.stack([oof[s] for s in SEEDS]))
    json.dump(rows,open(f'rows_{tag}.json','w'))
    a=lambda k: (np.mean([r[k] for r in rows]),np.std([r[k] for r in rows],ddof=1))
    print(f"{tag:40s} acc {a('acc')[0]*100:.2f}±{a('acc')[1]*100:.2f} bacc {a('bacc')[0]*100:.2f}±{a('bacc')[1]*100:.2f} F1 {a('f1')[0]:.3f}±{a('f1')[1]:.3f} AUC {a('auc')[0]:.3f}±{a('auc')[1]:.3f} ep {a('best_epoch')[0]:.0f} overlap {a('group_overlap')[0]:.1f}",flush=True)
if __name__=='__main__':
  for j in sys.argv[1:]:
    feat,mode=j.split(':'); t=time.time()
    run(feat, mode.startswith('grouped'), cw=mode.endswith('cw'), testval=mode.endswith('testval'))
    print("  time",f"{time.time()-t:.0f}s",flush=True)
