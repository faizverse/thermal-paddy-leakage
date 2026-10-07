"""Test-fold early stopping with the SAME training set as the main protocol (inner validation groups withheld).
Isolates the effect of selecting the stopping epoch on test data (the cv.py '_testval' modes also return the
inner validation groups to training, as in the original project code). Run from the repository root after src/extract.py.
Results (6 Oct 2026): leaf-grouped 81.71 ± 4.44%, image-level 91.67 ± 2.81%."""
import sys,os,json; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, keras, cv
def run_same(feat,grouped,tag):
    F=np.load(f'feat_{feat}.npy'); Y=keras.utils.to_categorical(cv.labels,6); rows=[]
    for s in cv.SEEDS:
        for k,tr,va,te in cv.splits(s,grouped):
            keras.utils.set_random_seed(1000*s+k); m=cv.head(F.shape[1])
            es=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
            m.fit(F[tr],Y[tr],validation_data=(F[te],Y[te]),epochs=100,batch_size=32,callbacks=[es],verbose=0)
            yp=m.predict(F[te],verbose=0).argmax(1); rows.append(dict(seed=s,fold=k,acc=float((yp==cv.labels[te]).mean())))
            keras.backend.clear_session()
    json.dump(rows,open(f'rows_{tag}.json','w')); a=[r['acc'] for r in rows]
    print(tag, f"{100*np.mean(a):.2f} ± {100*np.std(a,ddof=1):.2f}",flush=True)
if __name__=='__main__':
    run_same('resnet50_crop',True,'resnet50_crop_grouped_testval_SAMEtrain')
    run_same('resnet50_crop',False,'resnet50_crop_imagelevel_testval_SAMEtrain')
