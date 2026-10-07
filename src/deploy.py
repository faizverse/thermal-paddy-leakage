import sys; sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from common import *
from cv import head, labels, groups
from sklearn.model_selection import StratifiedGroupKFold
F=np.load('feat_resnet50_crop.npy'); Y=keras.utils.to_categorical(labels,6)
tr,va=next(StratifiedGroupKFold(6,shuffle=True,random_state=0).split(F,labels,groups))
keras.utils.set_random_seed(0); h=head(F.shape[1])
h.fit(F[tr],Y[tr],validation_data=(F[va],Y[va]),epochs=100,batch_size=32,verbose=0,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)])
print("val acc", (h.predict(F[va],verbose=0).argmax(1)==labels[va]).mean())
h.save('paddy_head.keras')
