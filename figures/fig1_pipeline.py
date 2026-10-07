import sys,os; sys.path.insert(0,'src')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from PIL import Image
from common import load_image
plt.rcParams.update({'font.size':7,'font.family':'DejaVu Sans'})
INK='#0b0b0b'; MUTE='#52514e'; LINE='#33475b'; BLUE='#2a78d6'; ORG='#eb6834'; GRN='#1baf7a'; PUR='#7b4bb3'; GRY='#8a8a86'
P='data/BLB/thermalimage56a.jpg'
raw=np.array(Image.open(P).convert('RGB'))
r224=load_image(P,(224,224),crop=False).clip(0,255).astype(np.uint8)
c224=load_image(P,(224,224),crop=True).clip(0,255).astype(np.uint8)
FW,FH=7.2,5.5
fig=plt.figure(figsize=(FW,FH),dpi=300)
def ax_in(x,y,w,h): return fig.add_axes([x/FW,y/FH,w/FW,h/FH])
def box(ax_,x,y,w,h,txt,ec=LINE,fc='#eef1f5',fs=6.6,bold=None):
    ax_.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.012',lw=0.9,ec=ec,fc=fc,transform=ax_.transAxes))
    if bold:
        ax_.text(x+w/2,y+h*0.74,bold,ha='center',va='center',fontsize=fs+0.3,fontweight='bold',color=INK,transform=ax_.transAxes)
        ax_.text(x+w/2,y+h*0.38,txt,ha='center',va='center',fontsize=fs,color=INK,transform=ax_.transAxes,linespacing=1.25)
    else:
        ax_.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=INK,transform=ax_.transAxes,linespacing=1.25)
def arrow(ax_,x0,y0,x1,y1,c=LINE):
    ax_.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle='-|>',mutation_scale=8,lw=0.9,color=c,transform=ax_.transAxes))
fig.text(0.05/FW,1-0.05/FH,'(a) Preprocessing (example: BLB image)',fontsize=7.5,fontweight='bold',color=INK,va='top')
IB,IH=3.30,1.50   # image bottom, height (inches)
axA=ax_in(0.25,IB,IH*4/3,IH); axA.imshow(raw); axA.set_xticks([]); axA.set_yticks([])
H,W=raw.shape[:2]
def lab(ax,rect,txt,c,tx,ty,ha='left'):
    x,y,w,h=rect; ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=c,lw=1.1))
    ax.annotate(txt,xy=(x+w/2,y+(h if ty>y else 0)),xytext=(tx,ty),textcoords='data',fontsize=6.4,color=c,ha=ha,va='center',fontweight='bold',
                arrowprops=dict(arrowstyle='-',color=c,lw=0.7,shrinkA=1,shrinkB=0),annotation_clip=False)
lab(axA,(1,224,58,15),'Camera logo',ORG,0,262)
lab(axA,(100,80,122,90),'Spot-meter reticle*',PUR,150,-12)
lab(axA,(272,8,46,228),'Temperature scale bar',ORG,322,262,ha='right')
lab(axA,(2,8,96,22),'Readout*',PUR,0,-12)
axA.set_xlim(0,W); axA.set_ylim(H,0)
axA.set_title('(i) Original FLIR E8 image, 320 × 240',fontsize=7,color=INK,pad=13)
axB=ax_in(2.82,IB,IH,IH); axB.imshow(r224); axB.set_xticks([]); axB.set_yticks([])
ch=224-int(224*0.15); cw=224-int(224*0.20)
axB.add_patch(Rectangle((cw,0),224-cw,224,fc='black',alpha=0.55,ec='none')); axB.add_patch(Rectangle((0,ch),cw,224-ch,fc='black',alpha=0.55,ec='none'))
axB.plot([cw,cw],[0,224],color='white',lw=0.9,ls='--'); axB.plot([0,cw],[ch,ch],color='white',lw=0.9,ls='--')
axB.text(cw+(224-cw)/2,112,'right\n20%',color='white',fontsize=6.3,ha='center',va='center',fontweight='bold')
axB.text(cw/2,ch+(224-ch)/2,'bottom 15%',color='white',fontsize=6.3,ha='center',va='center',fontweight='bold')
axB.set_xlim(-0.5,223.5); axB.set_ylim(223.5,-0.5)
axB.set_title('(ii) Resized to 224 × 224;\nshaded area removed',fontsize=7,color=INK,pad=4)
axC=ax_in(4.82,IB,IH,IH); axC.imshow(c224); axC.set_xticks([]); axC.set_yticks([])
axC.set_title('(iii) Cropped, resized to 224 × 224\n(299 × 299 for InceptionV3)',fontsize=7,color=INK,pad=4)
for a in (axA,axB,axC):
    for sp in a.spines.values(): sp.set_color(MUTE); sp.set_linewidth(0.6)
ov=fig.add_axes([0,0,1,1],facecolor='none'); ov.set_axis_off(); ov.set_xlim(0,1); ov.set_ylim(0,1)
yA=(IB+IH/2)/FH
arrow(ov,2.33/FW,yA,2.76/FW,yA); arrow(ov,4.36/FW,yA,4.76/FW,yA); arrow(ov,6.36/FW,yA,6.56/FW,yA)
ov.text(6.60/FW,yA,'Model-\nspecific\ninput\nscaling',fontsize=6.5,ha='left',va='center',color=INK,linespacing=1.2)
fig.text(0.25/FW,2.98/FH,'* Temperature readout and spot-meter reticle: present in some images only and not removed by the crop.',fontsize=6.3,color=PUR,va='top')
# ---------------- (b) evaluation and analyses
fig.text(0.05/FW,2.72/FH,'(b) Evaluation protocol and analyses',fontsize=7.5,fontweight='bold',color=INK,va='top')
bx=fig.add_axes([0.0,0.0,1,2.55/FH]); bx.set_axis_off(); bx.set_xlim(0,1); bx.set_ylim(0,1)
# grouped split schematic
gx,gy=0.015,0.47; gw,gh=0.335,0.50
bx.add_patch(FancyBboxPatch((gx,gy),gw,gh,boxstyle='round,pad=0.004,rounding_size=0.012',lw=0.9,ec=LINE,fc='#eef1f5'))
bx.text(gx+gw/2,gy+gh-0.06,'Leaf-grouped stratified 5-fold CV × 3 seeds',ha='center',va='center',fontsize=6.6,fontweight='bold')
rng=np.random.RandomState(3); cols=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#86b6ef','#008300','#b07cc6','#9c6b30','#5aa0a0']
fw=(gw-0.036-4*0.008)/5
for f in range(5):
    fx=gx+0.018+f*(fw+0.008); fy=gy+0.06
    is_test=(f==2)
    bx.add_patch(Rectangle((fx,fy+0.05),fw,0.20,fc='#fff4e8' if is_test else 'white',ec=ORG if is_test else MUTE,lw=0.9 if is_test else 0.5))
    bx.text(fx+fw/2,fy+0.27,'Test' if is_test else f'Fold {f+1}',ha='center',va='bottom',fontsize=6.0,color=ORG if is_test else MUTE,fontweight='bold' if is_test else None)
    gids=[f*2,f*2+1]
    k=0
    for g in gids:
        n=rng.randint(2,5)
        for j in range(n):
            r,c_=divmod(k,3); k+=1
            bx.add_patch(Rectangle((fx+0.006+c_*(fw-0.012)/3,fy+0.20-r*0.05),(fw-0.012)/3-0.003,0.04,fc=cols[g%10],ec='white',lw=0.3))
bx.text(gx+gw/2,gy+0.055,'Same colour = one leaf group; a group never spans\nfolds. 1/6 of training groups held out for early stopping.',ha='center',va='center',fontsize=5.9,color=MUTE,linespacing=1.2)
# model
box(bx,0.385,0.56,0.235,0.31,'ResNet50 · MobileNetV2 ·\nInceptionV3 · EfficientNetB7\n→ global average pooling',bold='Frozen ImageNet backbone')
box(bx,0.66,0.56,0.15,0.31,'Dropout 0.5\nDense 256 (ReLU)\nDropout 0.5\nSoftmax (6 classes)',bold='Dense head',fs=6.2)
box(bx,0.85,0.56,0.14,0.31,'Accuracy, balanced\naccuracy, macro-F1,\nmacro AUC\n(mean ± SD);\ncorrected t-test',bold='Evaluation',fs=6.0)
arrow(bx,0.352,0.715,0.381,0.715); arrow(bx,0.622,0.715,0.656,0.715); arrow(bx,0.812,0.715,0.846,0.715)
# analyses
ys=0.06; hh=0.30
items=[('Ablation','image-level vs leaf-grouped;\ntest fold vs validation for\nearly stopping; crop vs no\ncrop; random-group control',ORG,'#fdf2e9'),
       ('Temperature inputs','raw radiometric data → °C;\nabsolute, relative and\nmedian-deviation inputs;\ntemperature-level probe',PUR,'#f1ebf7'),
       ('Grad-CAM','last convolutional block;\ncorrect and incorrect\npredictions',GRN,'#e9f6ef'),
       ('Web prototype','TensorFlow.js in the\nbrowser; same\npreprocessing as training',GRY,'#f2f2f1')]
xs=[0.02,0.265,0.51,0.755]; ww=0.225
for (t,d,ec,fc),x in zip(items,xs):
    box(bx,x,ys,ww,hh,d,ec=ec,fc=fc,bold=t,fs=6.5)
bx.plot([0.13,0.87],[0.44,0.44],color=LINE,lw=0.8)
bx.plot([0.5025,0.5025],[0.56,0.44],color=LINE,lw=0.8)
for x in xs: arrow(bx,x+ww/2,0.44,x+ww/2,ys+hh+0.005)
os.makedirs('figs',exist_ok=True); plt.savefig('figs/fig1_pipeline.png',dpi=300,facecolor='white')
print('saved')
