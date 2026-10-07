import json,numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans','axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
import os; OUT='figs/'; os.makedirs(OUT,exist_ok=True)
R=lambda t: json.load(open(f'results/rows_{t}.json'))
GRP,CTL,IMG,ORG='#2a78d6','#9a9893','#eb6834','#f5b48f'
rows=[('Adjacent identifiers merged in pairs (sensitivity)','resnet50_crop_grouped_block2',GRP),
      ('Leaf-grouped, separate validation (main protocol)','resnet50_crop_grouped',GRP),
      ('Leaf-grouped, test fold for early stopping','resnet50_crop_grouped_testval',GRP),
      ('Random groups of the same sizes (control)','resnet50_crop_RANDOMgroups',CTL),
      ('Image-level, separate validation','resnet50_crop_imagelevel',IMG),
      ('Image-level, test fold for early stopping','resnet50_crop_imagelevel_testval',IMG)]
lab=[a for a,_,_ in rows]; cols=[c for _,_,c in rows]
vals=[np.mean([r['acc'] for r in R(t)])*100 for _,t,_ in rows]; errs=[np.std([r['acc'] for r in R(t)],ddof=1)*100 for _,t,_ in rows]
lab+=['Single image-level split, test-set selection (original project, cropped)','Same protocol, uncropped images (original project)']
# original project single splits (95.31%, 98.46%): see results/summary_values.json
vals+=[95.31,98.46]; errs+=[None,None]; cols+=[ORG,ORG]
fig,ax=plt.subplots(figsize=(7.2,3.1),dpi=300); y=np.arange(len(vals))[::-1]
ax.barh(y,vals,color=cols,height=0.62)
for yi,v,e in zip(y,vals,errs):
    if e is not None: ax.errorbar(v,yi,xerr=e,fmt='none',ecolor='#33475b',elinewidth=0.8,capsize=2)
    ax.text(v+(e or 0)+1.2,yi,f"{v:.1f}%"+(f" ± {e:.1f}" if e is not None else ""),va='center',fontsize=6.5,color='#0b0b0b')
ax.set_yticks(y,lab,fontsize=6.5); ax.set_xlim(0,112); ax.set_xticks(range(0,101,20)); ax.set_xlabel('ResNet50 test accuracy (%)')
ax.axvline(77.22,color='#52514e',lw=0.6,ls=':'); ax.grid(axis='x',color='#e6e5e0',lw=0.5); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(color=GRP,label='Leaf-grouped splits'),Patch(color=CTL,label='Control'),Patch(color=IMG,label='Image-level splits'),Patch(color=ORG,label='Original project')],
          fontsize=6,frameon=False,loc='lower center',bbox_to_anchor=(0.30,1.0),ncol=4)
plt.tight_layout(); plt.savefig(OUT+'fig3_inflation.png',dpi=300); plt.close(); print('ok')
