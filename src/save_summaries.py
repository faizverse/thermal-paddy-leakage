"""Save the numbers reported in the manuscript that are computed from the out-of-fold predictions or the group files
but were only printed before (v7, 7 Oct 2026). Run from the repository root after src/cv.py, src/randgroup.py,
src/grouping_sensitivity.py and src/thermal.py (needs oof_*.npy, labels.npy, paths.npy and groups_*.npy). Writes results/summary_values.json and results/coarse_groupings.csv."""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import cv
L = cv.labels; G = cv.groups; C = cv.CLASSES; paths = np.load('paths.npy')
out = {}
mean_sd = lambda v: [round(float(np.mean(v)), 2), round(float(np.std(v, ddof=1)), 2)]

# 1. Leakage mechanism: accuracy for images in multi-image vs single-image file-name groups; group-weighted accuracy
size = collections.Counter(G); multi = np.array([size[g] > 1 for g in G])
mech = {'n_images_multi': int(multi.sum()), 'n_images_single': int((~multi).sum()), 'n_single_image_groups': int((~multi).sum())}
for t in ['resnet50_crop_imagelevel', 'resnet50_crop_RANDOMgroups', 'resnet50_crop_grouped']:
    P = np.load(f'oof_{t}.npy'); ok = [(P[s].argmax(1) == L) for s in range(3)]
    mech[t] = {'acc_multi_image_groups_%': mean_sd([100 * o[multi].mean() for o in ok]),
               'acc_single_image_groups_%': mean_sd([100 * o[~multi].mean() for o in ok]),
               'group_weighted_acc_%': mean_sd([100 * np.mean([o[G == g].mean() for g in np.unique(G)]) for o in ok]),
               'note': 'mean and SD over the 3 seeds of pooled out-of-fold accuracy'}
out['mechanism'] = mech

# 2. Images with a spot-meter readout/reticle (FLIR measurement record, see mechanism_check.py) and their accuracy
def has_measurement(f):  # FLIR FFF record type 33 (measurement tools: spot meter/readout), as in mechanism_check.py
    import struct
    b=open(f,'rb').read(); i=2; segs=[]
    while i<len(b)-4:
        m=b[i+1]
        if m==0xDA: break
        n=struct.unpack('>H',b[i+2:i+4])[0]; seg=b[i+4:i+2+n]
        if m==0xE1 and seg[:5]==b'FLIR\x00': segs.append(seg)
        i+=2+n
    d=b''.join(s[8:] for s in segs); _,off,n=struct.unpack('>III',d[20:32])
    return any(struct.unpack('>H',d[off+k*32:off+k*32+2])[0]==33 and struct.unpack('>I',d[off+k*32+16:off+k*32+20])[0]>0 for k in range(n))
ro = (np.load('readout_mask.npy') if os.path.exists('readout_mask.npy') else np.array([has_measurement(str(p)) for p in paths])).astype(bool)
P = np.load('oof_resnet50_crop_grouped.npy'); ok = np.stack([(P[s].argmax(1) == L) for s in range(3)])
rd = {'n_images': int(ro.sum()), 'per_class': {C[c]: int(ro[L == c].sum()) for c in range(6)}, 'leaf_grouped_accuracy': {}}
for c in range(6):
    m1 = (L == c) & ro; m0 = (L == c) & ~ro
    if m1.sum():
        rd['leaf_grouped_accuracy'][C[c]] = {'with_readout_%': round(100 * ok[:, m1].mean(), 1), 'n_with': int(m1.sum()),
                                             'without_readout_%': round(100 * ok[:, m0].mean(), 1), 'n_without': int(m0.sum()),
                                             'with_readout_correct_per_seed': [int(ok[s, m1].sum()) for s in range(3)]}
rd['note'] = 'accuracy pooled over the 3 seeds of the main leaf-grouped protocol (ResNet50, cropped)'
out['readout_images'] = rd

# 3. Per-class F1 with absolute-temperature input vs renderings (mean over seeds of pooled out-of-fold F1)
def perclass_f1(tag):
    P = np.load(f'oof_{tag}.npy')
    f = np.array([f1_score(L, P[s].argmax(1), average=None, labels=range(6)) for s in range(3)])
    return {C[c]: [round(float(f[:, c].mean()), 3), round(float(f[:, c].std(ddof=1)), 3)] for c in range(6)}
out['perclass_f1'] = {'renderings_resnet50_crop_grouped': perclass_f1('resnet50_crop_grouped'),
                      'absolute_temperature_resnet50_Tabs_grouped': perclass_f1('resnet50_Tabs_grouped'),
                      'note': 'mean and SD over the 3 seeds'}

# 4. Confusions (main protocol): row = true class, column = predicted class
P = np.load('oof_resnet50_crop_grouped.npy')
cms = [confusion_matrix(L, P[s].argmax(1), labels=range(6)) for s in range(3)]
out['confusion_matrix'] = {'classes': C, 'seed0': cms[0].tolist(), 'sum_over_3_seeds': sum(cms).tolist()}

# 5. Coarser groupings (group counts); the group labels are saved in coarse_groupings.csv
cg = {'file_name_groups': G}
for f, k in [('groups_sim098', 'similarity_0.98'), ('groups_sim097', 'similarity_0.97'), ('groups_sim096', 'similarity_0.96'),
             ('groups_block2', 'consecutive_pairs')]:
    cg[k] = np.load(f + '.npy', allow_pickle=True)
out['grouping_counts'] = {k: int(len(set(v.tolist()))) for k, v in cg.items()}
os.makedirs('results', exist_ok=True)
pd.DataFrame({'file': [os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p in paths],
              'class': [C[c] for c in L], **{k: v for k, v in cg.items()}}).to_csv('results/coarse_groupings.csv', index=False)

# 6. File names without a numeric identifier (each forms its own group)
out['files_without_numeric_identifier'] = [os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                                           for p in paths if not any(ch.isdigit() for ch in os.path.basename(p))]

# 7. Original project single-split values (not computed here; recorded from the original project notebook)
out['original_project_single_split'] = {'cropped_%': 95.31, 'cropped_correct': '61/64', 'uncropped_%': 98.46, 'uncropped_correct': '64/65',
    'source': "original project (preliminary, unpublished study by the authors), notebook outputs: uncropped value from an earlier model, cropped value from the final model; image-level splits, test set used for early stopping"}

json.dump(out, open('results/summary_values.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:6000])
