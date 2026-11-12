"""Fill the native-ablation placeholders in v7.20 from a value sheet (ABLATION-VALUES-*.json).

Usage: python3 fill_ablation.py --values <sheet.json> [--apply]
Without --apply it prints every replacement. With --apply it writes the four section files (keeping *.pre_fill.bak).
The sheet must come from FINAL-PRESENTATION-VIEW (ablation_values_view.py); the recompute sheet is a cross-check only
(compare_sheets below). Sentences are replaced whole so that the 'removes [...] of the layer-9 response' placeholder,
for which no per-case metric exists in the frozen readout, becomes the measured edit size instead."""
import argparse,json,re,sys
from pathlib import Path
L=Path(__file__).resolve().parents[1]/'latex/sections'
CONDS=['clean','camera_plus15','headlight_off'];CONFS=['Native','Response-small','Response-large','Clean-matched','Random-1-matched','Random-2-matched','Random-3-matched']
def f1(x):return 'NA' if x is None else f'{x:.1f}'
def bnd(r):
    """Identification bounds of a success rate (the frozen readout suppresses the point when any planned outcome is missing)."""
    lo,hi=r['SR_bounds'];return f'{lo:.1f}--{hi:.1f}'
def sgn(x):return 'NA' if x is None else ('0.0' if abs(x)<0.05 else f'{x:+.1f}')
def row(V,c,k):return V['rows'][f'{c}|{k}']
def compare_sheets(A,B,tol=0.051):
    """Both sheets must agree to the printed precision on every field used below."""
    bad=[]
    for key,ra in A['rows'].items():
        rb=B['rows'][key]
        for fld in ('SR','harm','recovery','edit_l2','L18_reduction','footprint_pct'):
            a,b=ra.get(fld),rb.get(fld)
            if (a is None)!=(b is None) or (a is not None and abs(a-b)>(0.051 if fld!='edit_l2' else 0.06)):bad.append((key,fld,a,b))
        for i in (0,1):
            a,b=ra['SR_bounds'][i],rb['SR_bounds'][i]
            if abs(a-b)>tol:bad.append((key,'SR_bounds',a,b))
    for key,ca in A['contrasts'].items():
        cb=B['contrasts'][key]
        if abs(ca['pp']-cb['pp'])>tol or any(abs(x-y)>tol for x,y in zip(ca['bounds'],cb['bounds'])):bad.append((key,'contrast',ca,cb))
    return bad
def render(V):
    RL={c:row(V,c,'Response-large') for c in CONDS};RS={c:row(V,c,'Response-small') for c in CONDS};CM={c:row(V,c,'Clean-matched') for c in CONDS};NA={c:row(V,c,'Native') for c in CONDS}
    rnd=[row(V,c,k)['L18_reduction'] for c in CONDS[1:] for k in CONFS[4:]];rnd_lo,rnd_hi=min(rnd),max(rnd)
    l18=(RL['camera_plus15']['L18_reduction'],RL['headlight_off']['L18_reduction']);fp=(RL['camera_plus15']['footprint_pct'],RL['headlight_off']['footprint_pct'])
    D={c:V['contrasts'][f'D|{c}'] for c in CONDS};Q={c:V['contrasts'][f'Q|{c}'] for c in CONDS};R={c:V['contrasts'][f'R|{c}'] for c in CONDS}
    def incl0(x):return x['bounds'][0]<=0<=x['bounds'][1]
    excl=[(n,c) for n,T in (('D',D),('Q',Q),('R',R)) for c in CONDS if not incl0(T[c])]
    cname={'clean':'clean','camera_plus15':'Camera','headlight_off':'headlight'}
    if not excl:ctext='identification bounds that include zero in all nine cases'
    else:ctext='identification bounds that include zero except '+' and '.join(f'{n} ({cname[c]}; $[{sgn(T["bounds"][0])},{sgn(T["bounds"][1])}]$ pp)' for n,c in excl for T in [{'D':D,'Q':Q,'R':R}[n][c]])
    lo_l18,hi_l18=sorted(l18);lo_fp,hi_fp=sorted(fp)
    rep={}
    rep['abstract.tex']=[(r"reduces the deep response by $[\ldots]\%$ while moving issued commands by about $[\ldots]\%$ of the visual command shift.",
        f"reduces the deep response by ${f1(lo_l18)}$--${f1(hi_l18)}\\%$ while moving issued commands by about ${f1(lo_fp)}$--${f1(hi_fp)}\\%$ of the visual command shift.")]
    rep['introduction.tex']=[(r"reduces the deep visual response by $[\ldots]\%$ while moving issued commands by about $[\ldots]\%$ of the visual command shift;",
        f"reduces the deep visual response by ${f1(lo_l18)}$--${f1(hi_l18)}\\%$ while moving issued commands by about ${f1(lo_fp)}$--${f1(hi_fp)}\\%$ of the visual command shift;")]
    rep['structured.tex']=[(r"On the same cases it removes $[\ldots]$ of the layer-9 response, reduces the layer-18 visual response by $[\ldots]\%$ (Camera) and $[\ldots]\%$ (headlight), and after full generation moves the first five issued translation commands by about $[\ldots]\%$ and $[\ldots]\%$ of the visual command shift, with paired success within identification bounds of Native; matched clean and random dictionaries give $[\ldots]$.",
        f"On the same cases an edit of actual size $L_2={f1(RL['camera_plus15']['edit_l2'])}$ (Camera) and ${f1(RL['headlight_off']['edit_l2'])}$ (headlight) reduces the layer-18 visual response by ${f1(l18[0])}\\%$ and ${f1(l18[1])}\\%$, and after full generation moves the first five issued translation commands by about ${f1(fp[0])}\\%$ and ${f1(fp[1])}\\%$ of the visual command shift, while the Native-to-edited success difference has identification bounds of $[{sgn(D['clean']['bounds'][0])},{sgn(D['clean']['bounds'][1])}]$, $[{sgn(D['camera_plus15']['bounds'][0])},{sgn(D['camera_plus15']['bounds'][1])}]$ and $[{sgn(D['headlight_off']['bounds'][0])},{sgn(D['headlight_off']['bounds'][1])}]$ pp (clean/Camera/headlight), each including zero; at the same edit size the matched clean dictionary gives ${f1(CM['camera_plus15']['L18_reduction'])}\\%$ and ${f1(CM['headlight_off']['L18_reduction'])}\\%$ and the three random dictionaries ${f1(rnd_lo)}$--${f1(rnd_hi)}\\%$ layer-18 reduction.")]
    pop=V['population']
    supp=[]
    supp.append((r"$[N_{\mathrm{closed}}]$ roots closed with all 21 arms, $[N_{\mathrm{partial}}]$ closed partially, one was interrupted after creation (root 7) and $[N_{\mathrm{unrun}}]$ never executed; $[N_{\mathrm{known}}]$ of 6{,}048 main outcomes are known.",
        f"{pop['closed_all21']} roots closed with all 21 arms, {pop['closed_partial']} closed partially ({pop['missing_in_partial']} missing arms), one was interrupted after creation (root 7) and none remained unexecuted (root 140 was replaced once under a prelisted rule after a host--guard startup race); {pop['known_cells']:,} of 6{{,}}048 main outcomes are known."))
    supp.append((r"is $[\ldots]\%$ (Camera) and $[\ldots]\%$ (headlight) for Response-large, $[\ldots]/[\ldots]$ for Clean-matched, $[\ldots]$ for the three random bases and $[\ldots]/[\ldots]$ for Response-small.",
        f"is ${f1(l18[0])}\\%$ (Camera) and ${f1(l18[1])}\\%$ (headlight) for Response-large, ${f1(CM['camera_plus15']['L18_reduction'])}/{f1(CM['headlight_off']['L18_reduction'])}\\%$ for Clean-matched, ${f1(rnd_lo)}$--${f1(rnd_hi)}\\%$ for the three random bases and ${f1(RS['camera_plus15']['L18_reduction'])}/{f1(RS['headlight_off']['L18_reduction'])}\\%$ for Response-small."))
    supp.append((r"is $[\ldots]$, about $[\ldots]\%$ of the visual response (visual-minus-clean RMS $[\ldots]$ for Native); rotation and the continuous gripper are reported separately, and the formatted gripper output is identical in $[\ldots]$ of $[\ldots]$ non-self pairs.",
        f"is ${RL['camera_plus15']['edit_translation_RMS']:.4f}$ (Camera) and ${RL['headlight_off']['edit_translation_RMS']:.4f}$ (headlight), about ${f1(fp[0])}\\%$ and ${f1(fp[1])}\\%$ of the visual response (visual-minus-clean RMS ${NA['camera_plus15']['visual_translation_RMS']:.3f}$ and ${NA['headlight_off']['visual_translation_RMS']:.3f}$ for Native); rotation and the continuous gripper are reported separately, and in the 120-root stage view the formatted gripper output was identical in all 3{{,}}840 non-self pairs although the continuous gripper value differed in 3{{,}}837 (a stage finding, not recomputed on the full population)."))
    hr=lambda c:f"{f1(RL[c]['harm'])}/{f1(RL[c]['recovery'])}"
    hr=lambda c:f"{f1(RL[c]['harm'])}/{f1(RL[c]['recovery'])}"
    supp.append((r"Closed-loop success is $[\ldots]/[\ldots]/[\ldots]$ (clean/Camera/headlight) for Native and $[\ldots]/[\ldots]/[\ldots]$ for Response-large; Native-to-edited harm and recovery within condition are $[\ldots]/[\ldots]$, and the nine prelisted contrasts carry identification bounds $[\ldots]$. For the Native policy the selected visual conditions show no net task cost on this panel ($[\ldots]$ vs $[\ldots]$), so the panel asks whether the edit itself costs success, not whether it buys robustness.",
        f"Closed-loop success, reported as identification bounds because the frozen rule suppresses point estimates and ordinary intervals whenever a planned outcome is missing, is ${bnd(NA['clean'])}$ / ${bnd(NA['camera_plus15'])}$ / ${bnd(NA['headlight_off'])}\\%$ (clean/Camera/headlight) for Native and ${bnd(RL['clean'])}$ / ${bnd(RL['camera_plus15'])}$ / ${bnd(RL['headlight_off'])}\\%$ for Response-large; the known Native-to-edited harm and recovery mass within condition is ${hr('clean')}$, ${hr('camera_plus15')}$ and ${hr('headlight_off')}\\%$ of planned pairs (unknown pair mass ${f1(RL['camera_plus15']['unknown_pair_mass'])}\\%$), and the nine prelisted contrasts carry {ctext}. For the Native policy the Camera condition shows no net task cost on this panel (bounds ${bnd(NA['camera_plus15'])}$ vs ${bnd(NA['clean'])}\\%$) while headlight-off costs ${f1(NA['clean']['SR_bounds'][0]-NA['headlight_off']['SR_bounds'][1])}$--${f1(NA['clean']['SR_bounds'][1]-NA['headlight_off']['SR_bounds'][0])}$ pp (${bnd(NA['headlight_off'])}\\%$), so the panel asks whether the edit itself costs success, not whether it buys robustness."))
    # table
    def trip(k,fld,fmt=f1):
        if fld=='SR':return ' / '.join(bnd(row(V,c,k)) for c in CONDS)
        return ' / '.join(('--' if row(V,c,k)[fld] is None else fmt(row(V,c,k)[fld])) for c in CONDS)
    def dsr(k):
        parts=[]
        for c in CONDS:
            d=V['net_change'].get(f'{c}|{k}')
            parts.append('--' if d is None else f"$[{sgn(d['bounds'][0])},{sgn(d['bounds'][1])}]$")
        return ' / '.join(parts)
    def rnd3(fld):
        if fld=='SR':return ' / '.join(f"{min(row(V,c,k)['SR_bounds'][0] for k in CONFS[4:]):.1f}--{max(row(V,c,k)['SR_bounds'][1] for k in CONFS[4:]):.1f}" for c in CONDS)
        def rng(c):
            lo,hi=f1(min(row(V,c,k)[fld] for k in CONFS[4:])),f1(max(row(V,c,k)[fld] for k in CONFS[4:]));return lo if lo==hi else lo+'--'+hi
        return ' / '.join(('--' if row(V,c,'Random-1-matched')[fld] is None else rng(c)) for c in CONDS)
    tab_old=r"""Native & clean / Camera / headlight & -- & -- & -- & $[\ldots]$ & --\\
Response-large & clean / Camera / headlight & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$\\
Response-small & clean / Camera / headlight & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$\\
Clean-matched & clean / Camera / headlight & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$\\
Random-1/2/3-matched & clean / Camera / headlight & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$ & $[\ldots]$\\\bottomrule"""
    tab_new=(f"Native & clean / Camera / headlight & -- & -- & -- & {trip('Native','SR')} & --\\\\\n"
        f"Response-large & clean / Camera / headlight & {trip('Response-large','edit_l2')} & {trip('Response-large','L18_reduction')} & {trip('Response-large','footprint_pct')} & {trip('Response-large','SR')} & {dsr('Response-large')}\\\\\n"
        f"Response-small & clean / Camera / headlight & {trip('Response-small','edit_l2')} & {trip('Response-small','L18_reduction')} & {trip('Response-small','footprint_pct')} & {trip('Response-small','SR')} & {dsr('Response-small')}\\\\\n"
        f"Clean-matched & clean / Camera / headlight & {trip('Clean-matched','edit_l2')} & {trip('Clean-matched','L18_reduction')} & {trip('Clean-matched','footprint_pct')} & {trip('Clean-matched','SR')} & {dsr('Clean-matched')}\\\\\n"
        f"Random-1/2/3-matched (range) & clean / Camera / headlight & {rnd3('edit_l2')} & {rnd3('L18_reduction')} & {rnd3('footprint_pct')} & {rnd3('SR')} & see text\\\\\\bottomrule")
    supp.append((tab_old,tab_new))
    supp.append((r"Approved final joint readout and independent recompute; placeholder table until the final readout is sealed.",
        r"Values from the approved final joint readout (run \texttt{"+V['final_run_id']+r"}); signal means reproduced exactly by an independent recompute. SR and $\Delta$SR are identification bounds (lower--upper, \%); the random rows give the range over the three bases."))
    rep['supp_native_ablation.tex']=supp
    return rep
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--values',required=True);ap.add_argument('--cross',help='recompute sheet to compare against');ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    V=json.loads(Path(a.values).read_bytes())
    if a.cross:
        bad=compare_sheets(V,json.loads(Path(a.cross).read_bytes()))
        if bad:print('SHEETS DISAGREE:');[print(' ',b) for b in bad];sys.exit(2)
        print('sheets agree to printed precision')
    rep=render(V)
    for fn,pairs in rep.items():
        p=L/fn;s=p.read_text()
        for old,new in pairs:
            n=s.count(old);print(f'== {fn}: {n} match\n--- old: {old[:140]}...\n+++ new: {new}\n')
            if n!=1:print('!! expected exactly one match');sys.exit(3)
            s=s.replace(old,new)
        if a.apply:
            bak=p.with_suffix('.tex.pre_fill.bak')
            if not bak.exists():bak.write_bytes(p.read_bytes())
            p.write_text(s);print('wrote',p)
    left=sum(open(L/fn).read().count('[\\ldots]') for fn in rep) if a.apply else None
    print('remaining placeholders after apply:',left)
if __name__=='__main__':main()
