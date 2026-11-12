"""The declared d-minus-baseline arithmetic for the archived pulse pair only.

Uses unchanged exact integer units, metrics and two rounding candidates. No
common-c record is fabricated to satisfy the frozen three-arm API.
"""
import quantization_frozen as Q
import numpy as np


def analyze_pair(pulse, persistent, *, check=lambda: None):
    names=("W_used","gate","residual_input","after_attention_residual")
    data=[{k:Q._array(record[k],gate=k=="gate") for k in names}for record in (pulse,persistent)]
    for field in ("gate","residual_input"):
        Q.need(np.array_equal(data[0][field],data[1][field]),"same_archived_B_pre_site_"+field)
    by_position=[]
    for p in range(10):
        check()
        measures={k:[]for k in ("native_W_increment","ideal_gated_increment","native_after_attention_increment",
                                "combined_native_rounding_delta","baseline_combined_residual","treated_combined_residual")}
        for c in range(1024):
            wb,wt=(Q.units(r['W_used'][p,c])for r in data)
            g,r=Q.units(data[0]['gate'][p,c]),Q.units(data[0]['residual_input'][p,c])
            yb,yt=(Q.units(row['after_attention_residual'][p,c])for row in data)
            dw=wt-wb;ideal=dw*g;actual=(yt-yb)<<133
            eb=(yb<<133)-((r<<133)+wb*g);et=(yt<<133)-((r<<133)+wt*g)
            error=actual-ideal
            Q.need(error==et-eb,'exact_combined_rounding_difference_identity')
            for key,value in (("native_W_increment",dw),("ideal_gated_increment",ideal),
                ("native_after_attention_increment",actual),("combined_native_rounding_delta",error),
                ("baseline_combined_residual",eb),("treated_combined_residual",et)):
                measures[key].append(value)
        by_position.append(measures)
    rows=[]
    panels=tuple(("position_"+str(p),p,p+1)for p in range(10))+Q.WINDOWS
    for window,start,end in panels:
        values={k:[v for pos in by_position[start:end]for v in pos[k]]for k in by_position[0]}
        ideal,error,actual=(values[k]for k in ("ideal_gated_increment","combined_native_rounding_delta","native_after_attention_increment"))
        cross=2*sum(a*b for a,b in zip(ideal,error))
        ie,ee,ae=(sum(v*v for v in x)for x in (ideal,error,actual))
        Q.need(ae==ie+ee+cross,'native_after_attention_signed_energy_closure')
        rows.append({'contrast':'archived_persistent_minus_archived_pulse','window':window,'positions':end-start,
            'metrics':{k:Q._metrics(v,-133 if k=='native_W_increment'else -266)for k,v in values.items()},
            'ideal_error_signed_cross_exact_units_squared':str(cross),'ideal_error_signed_cross':Q.scaled(cross,-532),
            'combined_error_over_ideal_energy':None if ie==0 else ee/ie,
            'combined_error_over_actual_energy':None if ae==0 else ee/ae,
            'ideal_nonzero_but_native_zero_coordinates':sum(i!=0 and a==0 for i,a in zip(ideal,actual)),
            'ideal_zero_but_native_nonzero_coordinates':sum(i==0 and a!=0 for i,a in zip(ideal,actual)),
            'exact_closure':True})
    rounding=Q.analyze_rounding_candidates({'archived_pulse':pulse,'archived_persistent':persistent},check=check)
    return {'schema':'B-BRIDGE-archived-natural-d-pair-math-v1','analysis_complete':True,
            'baseline':'archived_pulse','treated':'archived_persistent','rows':rows,
            'rounding_candidates':rounding,'native_product_captured':False,
            'attribution':'combined_only_product_not_captured','metrics_reused_from':'frozen_d_minus_none_branch',
            'common_c_record_created':False,'no_c_or_setter_qualification':True,
            'new_model_calls':0,'new_env_steps':0,'compiled_path_uniquely_identified':False,
            'historical_archive_only':True,'new_graph_qualification':False,'scientific_equivalence_or_GO':False}
