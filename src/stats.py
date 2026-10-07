import json,numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import classification_report,confusion_matrix
L=lambda t: json.load(open(f'rows_{t}.json'))
base=L('resnet50_crop_grouped')
for o in ['mobilenetv2_crop_grouped','inceptionv3_crop_grouped','efficientnetb7_crop_grouped','resnet50_nocrop_grouped','resnet50_crop_grouped_cw']:
    b=L(o)
    for m in ['acc','f1']:
        d=np.array([x[m] for x in base])-np.array([x[m] for x in b])
        print(f"R50 vs {o:30s} {m}: mean diff {d.mean()*100:+.2f} pts, wins {int((d>0).sum())}/15, Wilcoxon p={wilcoxon(d).pvalue:.4f}")
labels=np.load('labels.npy'); P=np.load('oof_resnet50_crop_grouped.npy')
names=['BLB','Blast','Healthy','Hispa','Leaf folder','Leaf spot']
for s in range(3):
    yp=P[s].argmax(1); print(f"seed {s} pooled acc {(yp==labels).mean()*100:.2f}")
yp=P[0].argmax(1)
print(classification_report(labels,yp,target_names=names,digits=3))
print(confusion_matrix(labels,yp))
