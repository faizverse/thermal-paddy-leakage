import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans'})
import os; D='data/'
cls=[('BLB','BLB/thermalimage46a.jpg',220,103),('Blast','blast/Thermalimage10h.jpg',67,12),('Healthy','healthy/thermalimage5d.jpg',93,9),
     ('Hispa','hispa/thermalimage57b.jpg',142,72),('Leaf folder','leaf folder/thermalimage15b.jpg',34,19),('Leaf spot','leaf spot/thermalimage15a.jpg',80,33)]
grp=['8a','8k','8t','8ff','8pp','8uu']
FW,FH=7.2,2.72
fig=plt.figure(figsize=(FW,FH),dpi=300)
w=1.10; h=w*0.75; gap=(FW-0.12-6*w)/5
def put(x,y,path,title,sub=None,tc='#0b0b0b'):
    ax=fig.add_axes([x/FW,y/FH,w/FW,h/FH]); ax.imshow(Image.open(D+path)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color('#52514e'); s.set_linewidth(0.5)
    ax.set_title(title,fontsize=6.6,color=tc,pad=2,fontweight='bold' if sub else None)
    if sub: ax.text(0.5,-0.05,sub,transform=ax.transAxes,ha='center',va='top',fontsize=6.0,color='#52514e')
fig.text(0.04/FW,(FH-0.04)/FH,'(a) One example image per class',fontsize=7.5,fontweight='bold',va='top')
ya=FH-0.27-h
for i,(n,p,ni,ng) in enumerate(cls): put(0.06+i*(w+gap),ya,p,n,f'{ni} images, {ng} groups')
fig.text(0.04/FW,(ya-0.27)/FH,'(b) Six of the 51 images in one file-name group (healthy, identifier 8)',fontsize=7.5,fontweight='bold',va='top')
yb=ya-0.27-0.33-h
for i,g in enumerate(grp): put(0.06+i*(w+gap),yb,f'healthy/thermalimage{g}.jpg',f'thermalimage{g}',tc='#52514e')
os.makedirs('figs',exist_ok=True); plt.savefig('figs/fig2_dataset.png',dpi=300,facecolor='white'); print('saved')
