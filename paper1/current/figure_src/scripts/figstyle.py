"""Shared visual system for Figures 1-5 (CVPR text width 6.875 in = 495 pt).
One accent (blue) = the shared response directions and their restoration. Controls: orange = clean-PCA basis,
gray = random basis / references. Red = lost success. Numbers and labels are always set in ink, never in a series colour."""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
W_PT=495.0; W_IN=W_PT/72
INK='#1b2530'; SUB='#55616c'; MUTE='#8b96a0'; RULE='#dfe4e8'; PAPER='#ffffff'
K='#222222'; REF='#7a7a7a'; DASH=(0,(2,1.6))      # axis lines and tick labels; reference lines and their dash pattern
ACC='#1f6fb5'; ACC_T='#e4eef8'; ACC_M='#9fc3e6'
CLEANC='#d06a1f'; RAND='#6f7b85'
LOST='#c2452d'; GAIN='#3f5568'
COOL='#eef2f5'; COOL_E='#b9c5ce'; WARM='#fbf2e2'; WARM_E='#d3ba8a'; BAND='#f3f5f7'
# backwards-compatible names used by older scripts
RESP=ACC; RESTORE=ACC; GRAY=SUB; LIGHT='#b9c2ca'; GAINED=GAIN
FS=dict(letter=9.5,title=8.2,label=7.4,tick=7.0,note=7.0)
def apply():
    mpl.rcParams.update({
        'font.family':'sans-serif','font.sans-serif':['Liberation Sans','Arial','Helvetica','DejaVu Sans'],
        'mathtext.fontset':'stix','font.size':FS['label'],'axes.labelsize':FS['label'],'axes.titlesize':FS['title'],
        'xtick.labelsize':FS['tick'],'ytick.labelsize':FS['tick'],'legend.fontsize':FS['note'],
        'axes.edgecolor':K,'axes.labelcolor':K,'text.color':INK,'xtick.color':K,'ytick.color':K,
        'axes.linewidth':0.7,'xtick.major.width':0.7,'ytick.major.width':0.7,'xtick.major.size':2.6,'ytick.major.size':2.6,
        'xtick.major.pad':2,'ytick.major.pad':2,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':400,'axes.unicode_minus':True,'hatch.linewidth':0.5,
        'lines.solid_capstyle':'round','lines.solid_joinstyle':'round'})
class Fig:
    """Figure addressed in points, origin at the top-left corner."""
    def __init__(self,h_pt):
        self.H=h_pt; self.fig=plt.figure(figsize=(W_IN,h_pt/72)); self.cv=self.fig.add_axes([0,0,1,1]); self.cv.set_xlim(0,W_PT); self.cv.set_ylim(h_pt,0); self.cv.axis('off'); self.cv.set_zorder(1); self.cv.patch.set_alpha(0)
    def axes(self,x,y,w,h,**k):
        a=self.fig.add_axes([x/W_PT,1-(y+h)/self.H,w/W_PT,h/self.H],**k); a.set_zorder(2); a.patch.set_alpha(0); return a
    def text(self,x,y,s,size=None,color=INK,ha='left',va='center',weight='normal',style='normal',**k):
        return self.cv.text(x,y,s,fontsize=size or FS['label'],color=color,ha=ha,va=va,fontweight=weight,fontstyle=style,**k)
    def panel(self,x,y,letter,title,sub=None,sub_dx=None):
        self.text(x,y,letter,size=FS['letter'],weight='bold'); self.text(x+11,y,title,size=FS['title'],weight='bold')
        if sub: self.text(x+11,y+10.5,sub,size=FS['note'],color=SUB)
    def box(self,x,y,w,h,fc=PAPER,ec=COOL_E,lw=0.7,r=3,z=2,ls='-'):
        p=FancyBboxPatch((x+r,y+r),w-2*r,h-2*r,boxstyle=f'round,pad={r},rounding_size={r}',fc=fc,ec=ec,lw=lw,zorder=z,ls=ls,mutation_aspect=1); self.cv.add_patch(p); return p
    def line(self,pts,color=INK,lw=0.8,ls='-',z=3):
        pts=np.asarray(pts,float); self.cv.plot(pts[:,0],pts[:,1],color=color,lw=lw,ls=ls,zorder=z,solid_capstyle='butt',solid_joinstyle='round',dash_capstyle='butt')
    def arrow(self,pts,color=INK,lw=0.8,ls='-',z=3,hs=3.2,r=3.0):
        """Poly-line with rounded elbows and a filled triangular head on the final segment."""
        pts=[np.asarray(p,float) for p in pts]; out=[pts[0]]
        for i in range(1,len(pts)-1):
            a,b,c=pts[i-1],pts[i],pts[i+1]; d1=(b-a); d2=(c-b); l1=np.linalg.norm(d1); l2=np.linalg.norm(d2); rr=min(r,l1/2,l2/2)
            p1=b-d1/l1*rr; p2=b+d2/l2*rr
            t=np.linspace(0,1,7)[:,None]; out+=list((1-t)**2*p1+2*(1-t)*t*b+t**2*p2)
        out.append(pts[-1]); out=np.array(out)
        d=out[-1]-out[-2]; d=d/np.linalg.norm(d); n=np.array([-d[1],d[0]]); tip=out[-1]; base=tip-d*hs*1.35
        body=out.copy(); body[-1]=base+d*0.4
        self.cv.plot(body[:,0],body[:,1],color=color,lw=lw,ls=ls,zorder=z,solid_capstyle='butt',solid_joinstyle='round',dash_capstyle='butt')
        self.cv.add_patch(Polygon([tip,base+n*hs*0.5,base-n*hs*0.5],closed=True,fc=color,ec='none',zorder=z+0.1))
    def image(self,x,y,w,h,path,ec=COOL_E,lw=0.8):
        import matplotlib.image as mi
        self.cv.imshow(mi.imread(path),extent=(x,x+w,y+h,y),zorder=2,aspect='auto',interpolation='lanczos'); self.cv.set_xlim(0,W_PT); self.cv.set_ylim(self.H,0)
        self.cv.add_patch(plt.Rectangle((x,y),w,h,fc='none',ec=ec,lw=lw,zorder=2.5))
    def save(self,stem):
        for ext in ('pdf','svg','png'): self.fig.savefig(f'{stem}.{ext}')
        plt.close(self.fig)
def clean_axis(a,left=False,bottom=True):
    a.spines['left'].set_visible(left); a.spines['bottom'].set_visible(bottom)
    if not left: a.set_yticks([])
    if not bottom: a.set_xticks([])
def save(fig,stem):
    for ext in ('pdf','svg','png'): fig.savefig(f'{stem}.{ext}')
