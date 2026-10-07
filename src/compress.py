import sys,time,json; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from common import *
from cv import head, splits, labels, groups   # same splits as CV
from sklearn.metrics import accuracy_score,f1_score
W='weights/'
paths=np.load('paths.npy')
def backbone(name):
    if name=='resnet50':
        m=resnet50_imagenet(W)
        return keras.Model(m.inputs,keras.layers.GlobalAveragePooling2D()(m.get_layer('conv5_block3_out').output)), keras.applications.resnet50.preprocess_input
    b=keras.applications.MobileNetV2(include_top=False,weights=W+'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',input_shape=(224,224,3),pooling='avg')
    return b, keras.applications.mobilenet_v2.preprocess_input
def tflite_eval(buf,X,y,lat_n=0):
    it=tf.lite.Interpreter(model_content=buf,num_threads=1); it.allocate_tensors()
    inp=it.get_input_details()[0]; out=it.get_output_details()[0]
    preds=[]
    for x in X:
        it.set_tensor(inp['index'],x[None].astype(inp['dtype'])); it.invoke(); preds.append(it.get_tensor(out['index'])[0])
    preds=np.array(preds); lat=None
    if lat_n:
        ts=[]
        for _ in range(lat_n):
            t=time.perf_counter(); it.set_tensor(inp['index'],X[:1].astype(inp['dtype'])); it.invoke(); ts.append(time.perf_counter()-t)
        lat=float(np.median(ts)*1000)
    # largest non-constant (activation) tensor in bytes
    mx=0
    for d in it.get_tensor_details():
        try:
            if d['index'] in (inp['index'],): pass
            shp=d['shape']; 
            if len(shp)==4 and shp[0]==1 and shp[1]>1:
                mx=max(mx,int(np.prod(shp))*np.dtype(d['dtype']).itemsize)
        except Exception: pass
    return preds,lat,mx
name=sys.argv[1]; FOLDS=int(sys.argv[2]) if len(sys.argv)>2 else 5
F=np.load(f'feat_{name}_crop.npy'); Y=keras.utils.to_categorical(labels,6)
bb,pp=backbone(name)
Ximg=np.load('X224_crop.npy')   # cropped 224 images (same loader as CV features)
res=[]
for k,tr,va,te in splits(0,True):
    if k>=FOLDS: break
    keras.utils.set_random_seed(1000*0+k)
    h=head(F.shape[1]); es=keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
    h.fit(F[tr],Y[tr],validation_data=(F[va],Y[va]),epochs=100,batch_size=32,callbacks=[es],verbose=0)
    full=keras.Model(bb.input,h(bb.output))
    Xte=pp(Ximg[te].copy()); yte=labels[te]
    t=time.perf_counter(); kp=full.predict(Xte,verbose=0); 
    ref=h.predict(F[te],verbose=0)
    assert np.allclose(kp,ref,atol=1e-3), "full model != feature head"
    rep=pp(Ximg[tr][np.random.RandomState(k).choice(len(tr),100,replace=False)].copy())
    def rds():
        for x in rep: yield [x[None].astype(np.float32)]
    variants={}
    c=tf.lite.TFLiteConverter.from_keras_model(full); variants['float32']=c.convert()
    c=tf.lite.TFLiteConverter.from_keras_model(full); c.optimizations=[tf.lite.Optimize.DEFAULT]; variants['dynamic']=c.convert()
    c=tf.lite.TFLiteConverter.from_keras_model(full); c.optimizations=[tf.lite.Optimize.DEFAULT]; c.target_spec.supported_types=[tf.float16]; variants['float16']=c.convert()
    c=tf.lite.TFLiteConverter.from_keras_model(full); c.optimizations=[tf.lite.Optimize.DEFAULT]; c.representative_dataset=rds
    c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; c.inference_input_type=tf.int8; c.inference_output_type=tf.int8
    variants['int8']=c.convert()
    row=dict(fold=k,n=len(te),keras_acc=accuracy_score(yte,kp.argmax(1)),keras_f1=f1_score(yte,kp.argmax(1),average='macro'))
    # keras latency, batch 1
    ts=[]
    for _ in range(20):
        t=time.perf_counter(); full(Xte[:1],training=False); ts.append(time.perf_counter()-t)
    row['keras_lat_ms']=float(np.median(ts)*1000)
    for v,buf in variants.items():
        Xin=Xte
        if v=='int8':
            it=tf.lite.Interpreter(model_content=buf); s,z=it.get_input_details()[0]['quantization']
            Xin=np.clip(np.round(Xte/s+z),-128,127)
        p,lat,mx=tflite_eval(buf,Xin,yte,lat_n=30 if k==0 else 0)
        row[f'{v}_acc']=accuracy_score(yte,p.argmax(1)); row[f'{v}_f1']=f1_score(yte,p.argmax(1),average='macro')
        row[f'{v}_agree']=float(np.mean(p.argmax(1)==kp.argmax(1)))
        row[f'{v}_MB']=len(buf)/2**20
        if lat: row[f'{v}_lat_ms']=lat
        if k==0: row[f'{v}_max_act_bytes']=mx
    print(json.dumps({a:(round(b,4) if isinstance(b,float) else b) for a,b in row.items()}),flush=True)
    res.append(row); keras.backend.clear_session()
json.dump(res,open(f'compress_{name}.json','w'))
print("DONE")
