"""B-side native targets from the fixed exact common-probe algorithm.

No floating delta addition. Caller supplies source-qualified native uint16 W
and fresh native B baselines; target arrays stay in cloud worker memory/output.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE=Path(__file__).resolve().parent
AUDIT_REF={'bytes':94694,'sha256':'fcf6c7835d01a4e4eaeb12b4b96938e49d70182dc457c348f6bdc5084ee469b1'}
CORE_SHA='f0e205beeb7eeb878677a778793aacfe4704d838bf4813861a08bb08476bd369'
EXACT_SHA='4b7197ed764671979ea5175d930c326427d25baa3f36f4341ef93f9f93690a45'


def need(ok,code):
    if not ok:raise ValueError(code)


def _load(name,filename,sha):
    path=HERE/'frozen/common_probe'/filename;raw=path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==sha,'frozen_common_probe_source_SHA')
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;exec(compile(raw,str(path),'exec'),mod.__dict__);return mod


E=_load('_B_bridge_exact_bf16','exact_bf16.py',EXACT_SHA)
old=sys.modules.get('exact_bf16');sys.modules['exact_bf16']=E
try:C=_load('_B_bridge_common_core','core.py',CORE_SHA)
finally:
    if old is None:sys.modules.pop('exact_bf16',None)
    else:sys.modules['exact_bf16']=old


def identity(a):
    import numpy as np
    a=np.asarray(a)
    return {'shape':list(a.shape),'dtype':a.dtype.str,'bytes':a.nbytes,'sha256':hashlib.sha256(a.tobytes(order='C')).hexdigest()}


def _integer_geometry(g):
    """Compare exact counts, sums, Gram and support, not rounded float printouts."""
    return {'shape':g['shape'],'full_counts':g['full']['counts'],
            'full_energy':g['full']['energy_exact_integer_units_squared'],
            'positions':[{'counts':r['counts'],'energy':r['energy_exact_integer_units_squared'],
                          'support':r['affected_support'],'signed_sum':{
                              k:v for k,v in r['signed_channel_sum'].items() if 'exact' in k}}
                         for r in g['per_position']],
            'cross':{k:v for k,v in g['cross_position'].items() if 'exact_units' in k}}


def qualify_and_build_targets(old_a,old_b,old_b_after,fresh_reference_b,fresh_reference_after,new_b_baseline,
                              audit_bytes,*,synthetic=False,check=lambda:None):
    import numpy as np
    need(synthetic or sys.platform.startswith('linux'),'actual_native_targets_cloud_Linux_only')
    values=[np.asarray(x)for x in(old_a,old_b,old_b_after,fresh_reference_b,fresh_reference_after,new_b_baseline)]
    need(all(a.shape==(10,1024)and a.dtype==np.uint16 for a in values),'all_exact_native_W_uint16_10x1024')
    a,b,after,refb,refafter,newb=values
    need(b.tobytes()==refb.tobytes()==newb.tobytes(),'old_fresh_reference_candidate_B_baseline_exact')
    need(after.tobytes()==refafter.tobytes(),'old_fresh_reference_persistent_target_exact')
    need(type(audit_bytes)is bytes and len(audit_bytes)<=256*1024,'bounded_audit_bytes')
    audit_identity={'bytes':len(audit_bytes),'sha256':hashlib.sha256(audit_bytes).hexdigest()}
    need(synthetic or audit_identity==AUDIT_REF,'fixed_actual_common_geometry_audit_SHA')
    audit=json.loads(audit_bytes)
    need(audit.get('analysis_complete')is True and audit.get('synthetic')is synthetic,'same_actual_or_synthetic_audit_scope')
    geometry=C.analyze_triplet(a,b,after,check=check)
    need(_integer_geometry(geometry)==_integer_geometry(audit['geometry']),'exact_fixed_common_geometry_reproduced')
    counts=geometry['full']['counts']
    if not synthetic:
        need((counts['natural_not_common_representable'],counts['strictly_shrunk_nonzero'],counts['zeroed_reference_nonzero'],counts['reference_nonzero'])
             ==(623,612,11,10091),'fixed_actual_support_623_612_11_10091')
    target_c=np.empty_like(b)
    for p in range(10):
        check()
        for ch in range(1024):
            av,_=E.bits_to_units(a[p,ch]);bv,_=E.bits_to_units(b[p,ch]);tv,_=E.bits_to_units(after[p,ch])
            c=C.project_common_units(av,bv,tv-bv,check=check)
            target_c[p,ch]=C.diagnostic_target_bits(b[p,ch],c)
            used,_=E.bits_to_units(target_c[p,ch]);need(used-bv==c,'native_B_target_actual_increment_exact')
    targets={'none':b[None].copy(),'natural_d':after[None].copy(),'common_c':target_c[None]}
    return targets,{'schema':'B-bridge-fixed-native-targets-v1','synthetic':synthetic,'old_a':identity(a),'old_b':identity(b),
                    'old_b_after':identity(after),'new_b_baseline':identity(newb),'prior_audit':audit_identity,
                    'targets':{k:identity(v)for k,v in targets.items()},'common_counts':counts,
                    'exact_common_geometry_reproduced':True,'single_BF16_delta_cast_used':False,
                    'native_setter_qualified':False,'scientific_rule_changed':False}


def decode_native_target(bits):
    """Bitcast only; this is never float-delta arithmetic."""
    import numpy as np
    import ml_dtypes
    a=np.asarray(bits);need(a.shape==(1,10,1024)and a.dtype==np.uint16,'target_native_storage_ABI')
    return a.view(ml_dtypes.bfloat16).copy()
