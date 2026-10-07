"""Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio 2003): variance x (1/n + n_test/n_train),
n = 15 folds (5 folds x 3 seeds); Holm (1979) correction over the three backbone comparisons; significance level 0.05.
n_test/n_train = 0.32: mean over the 15 leaf-grouped folds of (test images)/(images used to fit the head), i.e. the
training part without the inner validation groups (127.2 / 398.4 on average; range 0.28-0.39). If labels.npy and
paths.npy are present (repository root, after src/extract.py) the ratio is recomputed from the folds.
Run from the repository root. Writes results/corrected_tests.json."""
import sys,os,json,numpy as np
from scipy import stats
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
RATIO=0.32
if os.path.exists('labels.npy') and os.path.exists('paths.npy'):
    import cv
    r=[len(te)/len(tr) for s in cv.SEEDS for k,tr,va,te in cv.splits(s,True)]
    RATIO=round(float(np.mean(r)),2); print(f"ratio recomputed from the folds: {np.mean(r):.4f} (range {min(r):.3f}-{max(r):.3f})")
R=lambda t:[r['acc'] for r in sorted(json.load(open(f'results/rows_{t}.json')),key=lambda r:(r['seed'],r['fold']))]
def corr_t(x,y,ratio=RATIO):
    d=np.array(x)-np.array(y); n=len(d); t=d.mean()/np.sqrt(d.var(ddof=1)*(1/n+ratio)); return 100*d.mean(),t,2*stats.t.sf(abs(t),n-1)
base=R('resnet50_crop_grouped'); out={'ratio':RATIO,'alpha':0.05,'backbones':{},'other':{}}
res={t:corr_t(base,R(t)) for t in ['mobilenetv2_crop_grouped','inceptionv3_crop_grouped','efficientnetb7_crop_grouped']}
order=sorted(res,key=lambda k:res[k][2]); prev=0
for i,k in enumerate(order):
    adj=max(prev,min(1,(len(order)-i)*res[k][2])); prev=adj
    out['backbones'][k]=dict(diff_points=round(res[k][0],2),t=round(res[k][1],3),p=round(res[k][2],4),p_holm=round(adj,4),
                             resnet50_better_folds=int(sum(a>b for a,b in zip(base,R(k)))))
    print(f"ResNet50 vs {k}: diff {res[k][0]:.2f} pts, t {res[k][1]:.2f}, p {res[k][2]:.4f}, Holm {adj:.4f}")
for t in ['resnet50_nocrop_grouped','resnet50_crop_grouped_cw','resnet50_Tabs_grouped','resnet50_Trel_grouped','resnet50_Tdev_grouped']:
    d,tt,p=corr_t(base,R(t)); out['other'][t]=dict(diff_points=round(d,2),t=round(tt,3),p=round(p,4))
    print(f"ResNet50 renderings (cropped) vs {t}: diff {d:.2f} pts, p {p:.4f}")
os.makedirs('results',exist_ok=True); json.dump(out,open('results/corrected_tests.json','w'),indent=1)
