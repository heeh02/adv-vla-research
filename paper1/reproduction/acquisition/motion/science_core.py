"""Pure fixed-coordinate Plus reductions; source identity is checked by worker I/O."""
from __future__ import annotations
import hashlib
import json
import math
import numpy as np

LAYERS=(0,8,17)
CALLS=(0,5,9)

def require(ok,code):
    if not ok:raise ValueError(code)

def decode_field(stored,meta):
    """Check the producer's typed hash before interpreting BF16 uint16 bits."""
    a=np.asarray(stored)
    require(list(a.shape)==meta['shape'] and a.dtype.str==meta['stored_dtype'] and a.nbytes==meta['bytes'],
            'stored_field_shape_dtype_bytes')
    raw=np.ascontiguousarray(a).tobytes()
    hdr={k:meta[k]for k in ('shape','dtype','logical_dtype','bytes')}
    digest=hashlib.sha256(json.dumps(hdr,sort_keys=True).encode()+b'\0'+raw).hexdigest()
    require(digest==meta['sha256'],'producer_typed_field_SHA')
    if meta['logical_dtype']=='bfloat16':
        require(a.dtype==np.dtype('<u2') and meta['dtype']=='<V2','BF16_bitview_contract')
        result=(a.astype(np.uint32)<<16).view(np.float32)
    else:
        require(a.dtype.str==meta['dtype'] and a.dtype.name==meta['logical_dtype'],'nonBF_logical_dtype')
        result=a
    require(bool(np.isfinite(result).all())==meta['finite'],'source_finiteness_receipt')
    return result

def scalar(x):
    y=float(x);return y if math.isfinite(y) else None

def ratio(num,den):return scalar(num/den)if den>0 else None

def participation(energy):
    e=np.asarray(energy,dtype=np.float64)
    require(e.ndim==1 and np.isfinite(e).all() and np.all(e>=0),'nonnegative_axis_energy')
    total=float(e.sum());squares=float(np.square(e).sum())
    return ratio(total*total,squares)

def vector_stats(reference,treatment):
    r,v=np.asarray(reference,dtype=np.float32),np.asarray(treatment,dtype=np.float32)
    require(r.shape==v.shape and r.ndim>=1 and r.size>0 and np.isfinite(r).all()and np.isfinite(v).all(),'same_finite_vector_ABI')
    d=v-r;re=float(np.square(r.astype(np.float64)).sum());ve=float(np.square(v.astype(np.float64)).sum())
    de=float(np.square(d.astype(np.float64)).sum())
    return {'n':r.size,'reference_rms':math.sqrt(re/r.size),'treatment_rms':math.sqrt(ve/r.size),
            'delta_rms':math.sqrt(de/r.size),'relative_l2':ratio(math.sqrt(de),math.sqrt(re)),
            'reference_zero_energy':re==0,'delta_energy':de}

def fixed_axis_energy(delta):
    """[token,channel] fixed coordinates; PR is no manifold dimension."""
    x=np.asarray(delta,dtype=np.float32)
    require(x.ndim==2 and x.shape[0]>0 and x.shape[1]>0 and np.isfinite(x).all(),'token_channel_delta')
    e=np.square(x.astype(np.float64))
    token=e.sum(axis=1);channel=e.sum(axis=0)
    per_token_channel_pr=np.array([participation(row) or np.nan for row in e],np.float64)
    return {'token_energy':token,'channel_energy':channel,
            'token_participation_ratio':participation(token),'channel_participation_ratio':participation(channel),
            'channel_PR_per_token':per_token_channel_pr,
            'coordinate_definition':'squared_difference_in_saved_native_axes'}

def partition_prefix(valid,defined,spans):
    valid=np.asarray(valid,bool).reshape(-1);defined=np.asarray(defined,bool).reshape(-1)
    require(valid.shape==defined.shape and valid.size>0,'prefix_mask_ABI')
    regions={};claimed=np.zeros(valid.size,bool)
    for name,key in (('main','image:base_0_rgb'),('wrist','image:left_wrist_0_rgb'),('text_state','language_tokens')):
        lo,hi=map(int,spans[key]);require(0<=lo<=hi<=valid.size,'prefix_span_bounds')
        selector=np.zeros(valid.size,bool);selector[lo:hi]=True
        require(not np.any(claimed&selector),'prefix_spans_disjoint')
        claimed|=selector;regions[name]=valid&defined&selector
    regions['other_valid_prefix']=valid&defined&~claimed
    regions['all_valid_prefix']=valid&defined
    return regions,{'saved_slots':valid.size,'valid_count':int(valid.sum()),
                    'query_defined_count':int(defined.sum()),'usable_count':int((valid&defined).sum()),
                    'padding_count':int((~valid).sum()),'undefined_valid_count':int((valid&~defined).sum())}

def pg_reduction(reference,treatment,regions,field):
    r,v=np.asarray(reference),np.asarray(treatment)
    require(r.shape==v.shape and r.ndim==3 and r.shape[0]==3 and np.isfinite(r).all()and np.isfinite(v).all(),
            'PG_three_layer_prefix_ABI')
    rows=[];derived={}
    for slot,layer in enumerate(LAYERS):
        for region,mask in regions.items():
            indices=np.flatnonzero(mask)
            if not len(indices):
                rows.append({'metric':field,'layer':layer,'region':region,'status':'no_valid_tokens','n_tokens':0})
                continue
            rr=r[slot,indices,:];vv=v[slot,indices,:]
            stats=vector_stats(rr,vv);axis=fixed_axis_energy(vv-rr)
            rows.append({'metric':field,'layer':layer,'region':region,'status':'measured','n_tokens':len(indices),
                         **stats,'channel_PR':axis['channel_participation_ratio'],
                         'token_PR':axis['token_participation_ratio']})
            key=f'{field}_L{layer}_{region}'
            derived[key+'_token_index']=indices.astype(np.int32)
            derived[key+'_token_energy']=axis['token_energy'].astype(np.float32)
            derived[key+'_channel_energy']=axis['channel_energy'].astype(np.float32)
            derived[key+'_channel_PR_per_token']=axis['channel_PR_per_token'].astype(np.float32)
    return rows,derived

def ae_reduction(reference,treatment):
    r,v=np.asarray(reference),np.asarray(treatment)
    require(r.shape==v.shape and r.shape[0]==10 and r.shape[1]==3 and r.ndim==4
            and r.shape[2]==10 and np.isfinite(r).all()and np.isfinite(v).all(),'T_native_AE_10x3x10xD')
    rows=[];derived={}
    for j in range(10):
        for slot,layer in enumerate(LAYERS):
            diff=v[j,slot]-r[j,slot]
            axis=fixed_axis_energy(diff)
            key=f'AE_j{j:02d}_L{layer}'
            derived[key+'_position_energy']=axis['token_energy'].astype(np.float32)
            derived[key+'_channel_energy']=axis['channel_energy'].astype(np.float32)
            derived[key+'_channel_PR_per_position']=axis['channel_PR_per_token'].astype(np.float32)
            overall=vector_stats(r[j,slot],v[j,slot])
            rows.append({'metric':'T_native_AE_H','generation_call':j,'layer':layer,'position':'all10',
                         **overall,'channel_PR':axis['channel_participation_ratio'],
                         'position_PR':axis['token_participation_ratio']})
            for pos in range(10):
                one=vector_stats(r[j,slot,pos],v[j,slot,pos])
                rows.append({'metric':'T_native_AE_H','generation_call':j,'layer':layer,'position':pos,
                             **one,'channel_PR':scalar(axis['channel_PR_per_token'][pos])})
    return rows,derived

def probability_rows(reference,treatment,visible_ref,visible_treat,query_ref,query_treat,spans,n_prefix,j):
    """P is O-shadow full-key probability, not T-native or a value contribution."""
    r,v=np.asarray(reference,dtype=np.float32),np.asarray(treatment,dtype=np.float32)
    mr,mv=np.asarray(visible_ref,bool),np.asarray(visible_treat,bool)
    qr,qv=np.asarray(query_ref,bool).reshape(-1),np.asarray(query_treat,bool).reshape(-1)
    require(r.shape==v.shape and r.shape[:2]==(3,8)and r.shape[2]==10
            and mr.shape==mv.shape==(10,r.shape[-1])and qr.shape==qv.shape==(10,),
            'O_shadow_probability_axes')
    require(n_prefix+10==r.shape[-1],'prefix_suffix_key_ABI')
    require(np.isfinite(r).all()and np.isfinite(v).all()and np.all(r>=0)and np.all(v>=0),'P_finite_nonnegative')
    region_masks={};claimed=np.zeros(r.shape[-1],bool)
    for name,key in (('main','image:base_0_rgb'),('wrist','image:left_wrist_0_rgb'),('text_state','language_tokens')):
        lo,hi=map(int,spans[key]);m=np.zeros(r.shape[-1],bool);m[lo:hi]=True
        require(not np.any(claimed&m),'P_region_disjoint');claimed|=m;region_masks[name]=m
    m=np.zeros(r.shape[-1],bool);m[:n_prefix]=~claimed[:n_prefix]
    region_masks['other_prefix']=m
    m=np.zeros(r.shape[-1],bool);m[n_prefix:]=True;region_masks['suffix']=m
    rows=[];derived=np.full((3,8,10,len(region_masks),3),np.nan,np.float32)
    shape_tv_array=np.full((3,8,10,len(region_masks)),np.nan,np.float32)
    full_tv_array=np.full((3,8,10),np.nan,np.float32)
    full_mass=np.full((3,8,10,2),np.nan,np.float32)
    names=list(region_masks)
    for slot,layer in enumerate(LAYERS):
        for head in range(8):
            for query in range(10):
                base={'metric':'O_shadow_P','generation_call':j,'layer':layer,'head':head,'action_position':query}
                if not(qr[query]and qv[query])or not np.array_equal(mr[query],mv[query]):
                    rows.append({**base,'region':'full_key','status':'query_or_key_mask_mismatch'})
                    continue
                mask=mr[query]
                pr=np.where(mask,r[slot,head,query],0);pv=np.where(mask,v[slot,head,query],0)
                mrass=float(pr.sum(dtype=np.float64));mvass=float(pv.sum(dtype=np.float64))
                tv=.5*float(np.abs(pv-pr).sum(dtype=np.float64))
                full_tv_array[slot,head,query]=tv
                full_mass[slot,head,query]=[mrass,mvass]
                full={**base,'region':'full_key','status':'measured','mass_reference':mrass,
                      'mass_treatment':mvass,'mass_residual_reference':mrass-1,
                      'mass_residual_treatment':mvass-1,'full_key_TV':tv,
                      'invisible_mass_reference':float(r[slot,head,query,~mask].sum(dtype=np.float64)),
                      'invisible_mass_treatment':float(v[slot,head,query,~mask].sum(dtype=np.float64))}
                rows.append(full)
                region_sum_r=region_sum_v=0.
                for k,(region,where) in enumerate(region_masks.items()):
                    sel=mask&where;ra=pr[sel];va=pv[sel]
                    rm=float(ra.sum(dtype=np.float64));vm=float(va.sum(dtype=np.float64))
                    region_sum_r+=rm;region_sum_v+=vm
                    shape_tv=.5*float(np.abs(va/vm-ra/rm).sum(dtype=np.float64))if rm>0 and vm>0 else None
                    derived[slot,head,query,k]=[rm,vm,vm-rm]
                    if shape_tv is not None:shape_tv_array[slot,head,query,k]=shape_tv
                    rows.append({**base,'region':region,'status':'measured','mass_reference':rm,
                                 'mass_treatment':vm,'net_mass_change':vm-rm,
                                 'within_region_conditional_TV':shape_tv,
                                 'region_visible_keys':int(sel.sum()),
                                 'full_key_TV':tv})
                require(abs(region_sum_r-mrass)<1e-5 and abs(region_sum_v-mvass)<1e-5,
                        'P_region_mass_partition_conservation')
    return rows,{'P_region_mass_ref_treat_delta':derived,'P_region_conditional_shape_TV':shape_tv_array,
                 'P_full_key_TV':full_tv_array,'P_full_key_mass_ref_treat':full_mass,
                 'P_region_names':np.asarray(names)}

def w_reduction(reference,treatment,j):
    r,v=np.asarray(reference),np.asarray(treatment)
    require(r.shape==v.shape and r.shape[:2]==(3,10)and r.ndim==3,'O_pre_gate_W_ABI')
    rows=[];derived={}
    for slot,layer in enumerate(LAYERS):
        d=v[slot]-r[slot];axis=fixed_axis_energy(d)
        key=f'W_j{j:02d}_L{layer}'
        derived[key+'_position_energy']=axis['token_energy'].astype(np.float32)
        derived[key+'_channel_energy']=axis['channel_energy'].astype(np.float32)
        rows.append({'metric':'O_shadow_pre_gate_W','generation_call':j,'layer':layer,'position':'all10',
                     **vector_stats(r[slot],v[slot]),'channel_PR':axis['channel_participation_ratio'],
                     'position_PR':axis['token_participation_ratio']})
        for pos in range(10):
            rows.append({'metric':'O_shadow_pre_gate_W','generation_call':j,'layer':layer,'position':pos,
                         **vector_stats(r[slot,pos],v[slot,pos]),
                         'channel_PR':scalar(axis['channel_PR_per_token'][pos])})
    return rows,derived

def action_reduction(raw_ref,raw_treat,action_ref,action_treat):
    rr,rv=np.asarray(raw_ref),np.asarray(raw_treat)
    ar,av=np.asarray(action_ref),np.asarray(action_treat)
    require(rr.shape==rv.shape==(10,32)and ar.shape==av.shape==(10,7),'canonical_action_ABI')
    rows=[];derived={}
    for scope,end in (('first5',5),('all10',10)):
        for representation,r,v,groups in (
            ('raw32',rr[:end],rv[:end],{'all32':slice(0,32),'adapter_first7':slice(0,7)}),
            ('adapter7',ar[:end],av[:end],{'translation':slice(0,3),'rotation':slice(3,6),'gripper':slice(6,7),'all7':slice(0,7)})):
            for component,sl in groups.items():
                delta=v[:,sl]-r[:,sl];axis=fixed_axis_energy(delta)
                rows.append({'metric':'canonical_C_action','representation':representation,'scope':scope,
                             'component':component,**vector_stats(r[:,sl],v[:,sl]),
                             'channel_PR':axis['channel_participation_ratio'],
                             'position_PR':axis['token_participation_ratio']})
                derived[f'{representation}_{scope}_{component}_position_energy']=axis['token_energy'].astype(np.float32)
    return rows,derived
