import os, re, glob, numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf, keras
DATA=os.environ.get('PADDY_DATA','data')  # folder containing the six class subfolders
CLASSES=['BLB','blast','healthy','hispa','leaf folder','leaf spot']
NAMES=['BLB','Blast','Healthy','Hispa','Leaf folder','Leaf spot']
def list_files():
    paths,labels=[],[]
    for i,c in enumerate(CLASSES):
        fs=sorted(f for f in os.listdir(os.path.join(DATA,c)) if f.lower().endswith(('.jpg','.jpeg')))
        paths+= [os.path.join(DATA,c,f) for f in fs]; labels+=[i]*len(fs)
    return np.array(paths), np.array(labels)
def group_id(path,label):
    f=os.path.basename(path); m=re.match(r'(?i)thermalimage\s*(\d+)',f)
    return f"{label}_{m.group(1) if m else 'NA_'+f}"
def load_image(path,size,crop=True):
    raw=tf.io.read_file(path); img=tf.io.decode_image(raw,channels=3,expand_animations=False)
    img=tf.image.resize(tf.cast(img,tf.float32),size)            # as image_dataset_from_directory (bilinear)
    if crop:
        h,w=size; img=img[0:h-int(h*0.15),0:w-int(w*0.20),:]       # notebook crop: bottom 15%, right 20%
        img=tf.image.resize(img,size)
    return img.numpy().astype(np.float32)

def resnet50_imagenet(weights_dir='weights/'):
    """Frozen ImageNet ResNet50 (no top). Uses the original project model if present (its backbone was frozen,
    hence identical to ImageNet weights); otherwise the Keras ImageNet weights."""
    j=os.path.join(weights_dir,'paddy_224.json')
    if os.path.exists(j):
        m=keras.models.model_from_json(open(j).read()); m.load_weights(os.path.join(weights_dir,'paddy_224.weights.h5')); return m
    f=os.path.join(weights_dir,'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    return keras.applications.ResNet50(include_top=False,input_shape=(224,224,3),weights=f if os.path.exists(f) else 'imagenet')
