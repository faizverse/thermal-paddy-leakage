import sys,time; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from common import *
paths,labels=list_files()
W='weights/'
def resnet_backbone():
    m=resnet50_imagenet(W)
    out=keras.layers.GlobalAveragePooling2D()(m.get_layer('conv5_block3_out').output)
    return keras.Model(m.inputs,out)
cfg={
 'resnet50':   (resnet_backbone, 224, keras.applications.resnet50.preprocess_input),
 'mobilenetv2':(lambda: keras.applications.MobileNetV2(include_top=False,weights=W+'mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',input_shape=(224,224,3),pooling='avg'),224,keras.applications.mobilenet_v2.preprocess_input),
 'inceptionv3':(lambda: keras.applications.InceptionV3(include_top=False,weights=W+'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',input_shape=(299,299,3),pooling='avg'),299,keras.applications.inception_v3.preprocess_input),
 'efficientnetb7':(lambda: keras.applications.EfficientNetB7(include_top=False,weights=W+'efficientnetb7_notop.h5',input_shape=(224,224,3),pooling='avg'),224,keras.applications.efficientnet.preprocess_input),
}
jobs=[('resnet50',True),('resnet50',False),('mobilenetv2',True),('inceptionv3',True),('efficientnetb7',True)]
for name,crop in jobs:
    out=f'feat_{name}_{"crop" if crop else "nocrop"}.npy'
    if os.path.exists(out): continue
    t=time.time(); fn,s,pp=cfg[name]; bb=fn()
    X=np.stack([load_image(p,(s,s),crop) for p in paths])
    F=bb.predict(pp(X),batch_size=16,verbose=0)
    if name=='resnet50' and crop: np.save('X224_crop.npy',X)
    np.save(out,F.astype(np.float32)); print(name,crop,F.shape,f"{time.time()-t:.0f}s",flush=True)
    del bb; keras.backend.clear_session()
np.save('labels.npy',labels); np.save('paths.npy',paths)
print("DONE")
