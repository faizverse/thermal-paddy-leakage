"""Random-group control: permute the file-name group labels among the images of each class
(keeping the number and sizes of groups), then repeat the leaf-grouped protocol for ResNet50.
Run from the repository root after src/extract.py. Output: rows_resnet50_crop_RANDOMgroups.json."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv
from cv import labels
rng=np.random.RandomState(123)
G=cv.groups.copy(); newG=G.copy()
for c in np.unique(labels):           # shuffle group membership within each class, keeping the group sizes
    idx=np.where(labels==c)[0]; newG[idx]=G[idx][rng.permutation(len(idx))]
cv.groups=newG
cv.run('resnet50_crop',True,tag='resnet50_crop_RANDOMgroups')
