import pandas as pd, numpy as np, json
from scipy.stats import binomtest
out={}
# Fig1 links
R=pd.read_csv('M3_links.csv')
order=['V->H','H->JS','H->Att','H->T','JS->T','Att->T','T->R','H->dY','Att->dY','T->dY','R->dY','G->dY']
lab={'V->H':'像素变化 → L18隐藏','H->JS':'L18 → attention分布变化(JS)','H->Att':'L18 → 主相机attention质量','H->T':'L18 → 平移动作','JS->T':'attention分布 → 平移动作','Att->T':'attention质量 → 平移动作','T->R':'平移 → 旋转','H->dY':'L18 → 成败变化','Att->dY':'attention质量 → 成败变化','T->dY':'平移动作 → 成败变化','R->dY':'旋转动作 → 成败变化','G->dY':'夹爪动作 → 成败变化'}
L=[]
for k in order:
    r=R[R.link==k].set_index('level')
    L.append(dict(link=k,label=lab[k],cond=round(r.loc['condition','rho_total'],3),cell=round(r.loc['task×cond','rho_total'],3),within=round(r.loc['episode','rho_within_cond'],3)))
out['links']=L
# Fig2 variance decomposition
M2=pd.read_csv('M2_vardecomp.csv')
nm={'V_main':'主相机像素','H18':'L18隐藏响应','JS':'attention分布(JS)','Att_main':'主相机attention质量','Att_wrist':'腕相机attention质量','Trans':'平移动作','Rot':'旋转动作','Grip':'夹爪动作'}
VD=[]
for _,r in M2.iterrows():
    if r['var'] in nm: VD.append(dict(name=nm[r['var']],condition=r.condition,task=r.task,cxt=r.cond_x_task,state=r.init_state,noise=r.noise_resid))
VD.append(dict(name='受扰episode成败',condition=0.0137,task=0.2974,cxt=0.2141,state=0.0145,noise=0.4603))
out['vardecomp']=VD
# Fig3 time course (condition share)
out['timecourse']={'L18隐藏响应':[0.85,0.58,0.47],'attention分布(JS)':[0.87,0.75,0.65],'平移动作':[0.69,0.43,0.34],'主相机attention质量':[0.61,0.06,0.04],'夹爪动作':[0.26,0.03,0.03]}
out['timecourse_steps']=['q0 (第0步)','锚点1 (约第31步)','锚点2 (约第64步)']
# Fig4 outcomes
M1=pd.read_csv('M1_outcomes.csv')
cn={'camera_minus30':'相机 −30°','camera_plus30':'相机 +30°','camera_minus15':'相机 −15°','camera_plus15':'相机 +15°','camera_minus15_surface_cool':'−15° + 冷材质','camera_plus15_surface_warm':'+15° + 暖材质','headlight_off':'关头灯','matched_global':'像素匹配调暗','surface_warm':'暖材质','surface_cool':'冷材质','camera_plus15_q0_only':'+15° 仅首帧','headlight_off_q0_only':'关头灯 仅首帧','surface_warm_q0_only':'暖材质 仅首帧'}
out['outcomes']=[dict(cond=cn[r.cond],key=r.cond,n=int(r.n),harm=int(r.harm),recov=int(r.recov),net=round(r.net_pp,2),lo=round(r.ci_lo,2),hi=round(r.ci_hi,2),p=round(r.mcnemar_p,4)) for _,r in M1.iterrows()]
out['cam30_pooled_p']=binomtest(44,61,0.5).pvalue
# Fig5 dose
T=pd.read_csv('M4_condition_table.csv',index_col=0)
x=T.H0.values; y=T.dSR.values*100
best=None
for t0 in np.linspace(x.min(),x.max(),200):
    z=np.maximum(0,x-t0); A=np.c_[np.ones_like(z),z]; c,*_=np.linalg.lstsq(A,y,rcond=None); sse=((A@c-y)**2).sum()
    if best is None or sse<best[0]: best=(sse,t0,c)
_,t0,c=best
out['dose']=dict(points=[dict(cond=cn[i],key=i,H=round(T.loc[i,'H0'],4),V=round(T.loc[i,'V0'],4),T=round(T.loc[i,'T0'],4),dSR=round(T.loc[i,'dSR']*100,2),lo=float(M1.set_index('cond').loc[i,'ci_lo']),hi=float(M1.set_index('cond').loc[i,'ci_hi'])) for i in T.index],hinge=dict(knot=round(t0,4),intercept=round(c[0],3),slope=round(c[1],2)))
# Fig6 task replication
F=pd.read_csv('M6_task_features.csv',index_col=0)
out['tasks']=[dict(task=i,D3=round(r.D3_30*100,1),Sev=round(r.Sev_30*100,1),F15=round(r.Fresh_15*100,1),H=round(r.H,4),T=round(r['T'],4)) for i,r in F.iterrows()]
# Fig7 AUC forest
out['auc']=[
 dict(cohort='D3',name='L18隐藏响应 · q0',auc=.462,lo=.35,hi=.55),dict(cohort='D3',name='平移动作 · q0',auc=.495,lo=.42,hi=.57),dict(cohort='D3',name='平移/clean更新 · q0',auc=.443,lo=.33,hi=.55),dict(cohort='D3',name='旋转动作 · q0',auc=.517,lo=.44,hi=.62),dict(cohort='D3',name='attention分布(JS) · q0',auc=.456,lo=.39,hi=.53),
 dict(cohort='D3',name='L18隐藏响应 · 第31步',auc=.467,lo=.40,hi=.54),dict(cohort='D3',name='平移动作 · 第31步',auc=.436,lo=.37,hi=.51),dict(cohort='D3',name='L18隐藏响应 · 第64步',auc=.504,lo=.45,hi=.57),dict(cohort='D3',name='平移动作 · 第64步',auc=.486,lo=.41,hi=.57),
 dict(cohort='D3',name='末步更新/累积比 s',auc=.536,lo=.45,hi=.64),dict(cohort='D3',name='最终/累积比 r',auc=.561,lo=.49,hi=.64),dict(cohort='D3',name='L18方向时间持续性',auc=.571,lo=.49,hi=.63),
 dict(cohort='Fresh',name='读出响应 R',auc=.466,lo=.35,hi=.60),dict(cohort='Fresh',name='原生flow差',auc=.499,lo=.37,hi=.63),dict(cohort='Fresh',name='动作头可读比例',auc=.410,lo=.28,hi=.54),dict(cohort='Fresh',name='末步更新/累积比 s',auc=.652,lo=.56,hi=.77),dict(cohort='Fresh',name='最终/累积比 r',auc=.600,lo=.49,hi=.73),dict(cohort='Fresh',name='累积差 Δx9',auc=.561,lo=.45,hi=.67)]
# Fig8 E1 vs native
E1=pd.DataFrame({'clean':[.95,1,.9,.95,1,.95,.85,1,.7,1],'r10':[.7,.9,.7,.9,.4,.8,.6,.8,.6,.25]},index=[f'libero_10-t{i:02d}' for i in range(10)])
j=E1.join(F)
out['e1']=[dict(task=i.replace('libero_10-',''),H=round(r.H,4),fgm=round((r.r10-r.clean)*100,1),native=round(((r.D3_30+r.Sev_30)/2)*100,1)) for i,r in j.iterrows()]
json.dump(out,open('report_data.json','w'),ensure_ascii=False)
print(json.dumps(out['dose']['hinge']), out['cam30_pooled_p'])
print(len(json.dumps(out,ensure_ascii=False)))
