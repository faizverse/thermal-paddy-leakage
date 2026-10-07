"""Grouping sensitivity (v6, Stage 2): coarser groupings than the file-name groups, leaf-grouped protocol, ResNet50.
(a) merge file-name groups (same class) when any two of their images have cosine similarity >= 0.98/0.97/0.96
    between frozen ResNet50 features; (b) merge consecutive numeric identifiers (same class) in pairs;
(c) random-group control with the pair-merged group sizes. Run from the repository root after src/extract.py.
Results (6 Oct 2026, accuracy %): 0.98 -> 73.5 ± 7.1 (243 groups); 0.97 -> 72.5 ± 5.6 (212); 0.96 -> 63.8 ± 12.0 (182);
pairs -> 65.6 ± 9.2 (131); random with pair sizes -> 88.6 ± 3.0."""
import sys,os,re; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv
L=cv.labels; G0=cv.groups.copy()
F=np.load('feat_resnet50_crop.npy'); X=F/np.linalg.norm(F,axis=1,keepdims=True); S=X@X.T
ug={g:i for i,g in enumerate(np.unique(G0))}; gi=np.array([ug[g] for g in G0])
def sim_groups(th):
    par=list(range(len(ug)))
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    I,J=np.where((S>th)&(L[:,None]==L[None,:]))
    for i,j in zip(I,J):
        a,b=f(gi[i]),f(gi[j])
        if a!=b: par[a]=b
    return np.array([f"m{f(x)}" for x in gi])
num=np.array([int(re.search(r'_(\d+)$',g).group(1)) if re.search(r'_(\d+)$',g) else -1 for g in G0])
pairs=np.array([f"{L[i]}_{num[i]//2}" if num[i]>=0 else G0[i] for i in range(len(G0))])
runs=[('sim098',sim_groups(0.98)),('sim097',sim_groups(0.97)),('sim096',sim_groups(0.96)),('block2',pairs)]
rng=np.random.RandomState(123); rnd=pairs.copy()
for c in np.unique(L):
    idx=np.where(L==c)[0]; rnd[idx]=pairs[idx][rng.permutation(len(idx))]
for tag,g in runs:
    np.save(f'groups_{tag}.npy',g)   # group labels, used by src/save_summaries.py
    cv.groups=g; cv.run('resnet50_crop',True,tag=f'resnet50_crop_grouped_{tag}')
cv.groups=rnd; cv.run('resnet50_crop',True,tag='resnet50_crop_RANDOMgroups_block2sizes')
