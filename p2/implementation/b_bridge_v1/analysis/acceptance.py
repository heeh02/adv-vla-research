"""Pure CPU B-bridge gates and signed first-velocity readout.

The worker must supply actual native leaves, source-bound call records and
fixed exact targets. This module neither constructs targets nor runs models.
The single-step output is v6 in model coordinates, not adapter7 commands.
"""
import hashlib
import math
import numpy as np

ROOT_ID = "SDRU-CCONF-T06-V00700-I00-PC"
IDENTITY = {"root_id": ROOT_ID, "query_index": 11, "generation_call": 6, "layer_index": 8,
            "receiver": "B_pulse", "prefix_condition": "T", "num_steps": 10}
PRE_SITE = ("W_before", "gate", "residual_input", "query_post_rope_scaled", "probabilities_native",
            "suffix_key_post_rope", "suffix_value", "mask", "suffix_positions", "adarms_condition",
            "actual_x", "actual_time", "time_weak_type", "dt", "prefix_key_used", "prefix_value_used")
OUT_FIELDS = ("W_used", "after_attention_residual", "hidden_post_block", "velocity")
ROLES = ("none", "natural_d", "common_c")
WINDOWS = (("first5", 0, 5), ("tail5", 5, 10), ("full10", 0, 10))
GROUPS = (("translation_index_0_to_2", 0, 3), ("rotation_index_3_to_5", 3, 6), ("gripper_index_6", 6, 7),
          ("padding_index_7_to_31", 7, 32))
SINGLE_GATES = {"original_num_steps_ten", "all_future_and_current_V_edits_off", "only_L9_W_selected",
    "prefix_mask_identical", "prefix_positions_identical", "target_finite", "velocity_finite"}


class AcceptanceError(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise AcceptanceError(code)


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    need(not a.dtype.hasobject and not b.dtype.hasobject, "numeric_native_arrays_only")
    if a.dtype.kind in "fc":
        need(np.isfinite(a).all() and np.isfinite(b).all(), "finite_numeric_native_arrays")
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def identity_bits(a):
    a = np.asarray(a)
    need(not a.dtype.hasobject, "numeric_identity")
    return {"shape": list(a.shape), "storage_dtype": str(a.dtype), "bytes": a.nbytes,
            "payload_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def native_bits(value):
    a = np.asarray(value)
    need(a.dtype.name == "bfloat16" or a.dtype == np.uint16, "BF16_native_or_verified_uint16_storage")
    bits = a.view(np.uint16) if a.dtype.name == "bfloat16" else a
    need(not np.any(((bits.astype(np.uint32) >> 7) & 255) == 255), "finite_native_BF16_leaf")
    return bits


def extract_site(velocity, metadata, *, full=False, legacy_reference=False):
    """Map actual native.query/PVP metadata to the CPU site contract.

    For full Gstar use velocity=None and its saved flow.velocity_native[6].
    legacy_reference=True is only for the source-qualified old reference graph,
    whose W_before and W_used coincide; it is not a selector qualification.
    """
    reads = metadata["reads"]
    need(np.array_equal(np.asarray(metadata["capture_layers"]), (0,8,17)), "original_three_capture_layers")
    if full:
        need(np.array_equal(np.asarray(metadata["capture_calls"]), (5,6,7,8,9)), "original_five_capture_calls")
        get_layer = lambda name: np.asarray(reads[name])[1,1,0]
        get_context = lambda name: np.asarray(reads[name])[1,0]
        x, tau = np.asarray(metadata["flow"]["x"])[6,0], np.asarray(metadata["flow"]["time"])[6]
        weak, dt = np.asarray(metadata["flow"]["time_weak_type"])[6], np.asarray(metadata["flow"]["dt"])[6]
        velocity = np.asarray(metadata["flow"]["velocity_native"])[6,0]
    else:
        need(int(np.asarray(metadata["generation_call"])) == 6 and int(np.asarray(metadata["completed_updates"])) == 0,
             "single_j6_without_Euler_update")
        get_layer = lambda name: np.asarray(reads[name])[1,0]
        get_context = lambda name: np.asarray(reads[name])[0]
        x, tau = np.asarray(metadata["actual_x"])[0], np.asarray(metadata["actual_time"])
        weak, dt = np.asarray(metadata["time_weak_type"]), np.asarray(metadata["sampling_abi"]["actual_dt"])
        velocity = np.asarray(velocity)[0]
        need(set(metadata["gates"]) == SINGLE_GATES and
             all(np.asarray(v).shape == () and np.asarray(v).dtype == bool and bool(np.asarray(v))
                 for v in metadata["gates"].values()), "actual_single_native_gates")
    if legacy_reference:
        need(full, "legacy_reference_is_full_Gstar_only")
        before = used = native_bits(get_layer("attention_output_post_WO_pre_gate"))
        active = np.asarray(False)
    else:
        active_array = np.asarray(reads["bridge_active"])
        need(active_array.dtype == bool and active_array.shape == ((5,3) if full else (3,)), "actual_selector_trace_shape")
        if full:
            need(not active_array.any(), "full_inactive_all_sites")
            active = active_array[1,1]
        else:
            need(not active_array[[0,2]].any(), "non_L9_selector_false")
            active = active_array[1]
        before, used = (native_bits(get_layer(n)) for n in ("bridge_W_before", "bridge_W_used"))
    site = {"W_before":before,"W_used":used,"bridge_active":np.asarray(active),
        "gate":native_bits(get_layer("attention_gate")),"residual_input":native_bits(get_layer("receiver_residual_input")),
        "after_attention_residual":native_bits(get_layer("hidden_after_attention_residual")),
        "hidden_post_block":native_bits(get_layer("hidden_post_block")),"velocity":native_bits(velocity),
        "actual_x":np.asarray(x),"actual_time":np.asarray(tau),"time_weak_type":np.asarray(weak),"dt":np.asarray(dt)}
    for name in ("query_post_rope_scaled","probabilities_native","suffix_key_post_rope","suffix_value","prefix_key_used","prefix_value_used"):
        site[name] = native_bits(get_layer(name))
    for name in ("mask","suffix_positions","adarms_condition"):
        site[name] = get_context(name)
    need(site["W_before"].shape == site["W_used"].shape == (10,1024) and site["velocity"].shape == (10,32), "fixed_B_full_site_shape")
    need(site["actual_x"].dtype == np.float32 and site["actual_time"].dtype == np.float32 and site["dt"].dtype == np.float32,
         "original_FP32_carry_and_dt")
    return site


def require_exact_fields(reference, actual, names):
    for field in names:
        need(field in reference and field in actual, "actual_native_field_required_"+field)
        need(exact(reference[field], actual[field]), "native_exact_"+field)
    return {"fields": list(names), "all_native_bytes_exact": True}


def validate_site(reference, actual, expected_target, *, actual_identity, reference_identity, active):
    """Target selector cannot alter the upstream B state or other cache leaves."""
    need(type(active) is bool, "explicit_activation_bool")
    need(all(actual_identity.get(k) == v == reference_identity.get(k) for k,v in IDENTITY.items()), "fixed_B_only_site_and_solver")
    for name in ("snapshot_identity", "noise_identity", "prefix_identity", "model_identity", "source_manifest_sha256"):
        need(isinstance(actual_identity.get(name), str) and len(actual_identity[name]) == 64 and
             actual_identity[name] == reference_identity.get(name), "same_actual_identity_"+name)
    pre = require_exact_fields(reference, actual, PRE_SITE)
    target = np.asarray(expected_target)
    need(target.shape == (10,1024) and target.dtype == np.uint16, "whole_native_BF16_target_bits")
    need(not np.any(((target.astype(np.uint32) >> 7) & 255) == 255), "finite_native_W_target")
    need(actual["W_used"].shape == target.shape and actual["W_used"].dtype == np.uint16 and exact(actual["W_used"], target),
         "actual_consumed_W_equals_exact_target")
    need(np.asarray(actual["bridge_active"]).shape == () and np.asarray(actual["bridge_active"]).dtype == bool and
         bool(actual["bridge_active"]) is active, "actual_target_selector_value")
    if not active:
        need(exact(actual["W_used"], actual["W_before"]), "inactive_W_is_own_identity")
    return {"schema": "B-BRIDGE-site-qualification-v1", "qualified": True, "upstream": pre,
            "actual_target": identity_bits(target), "active": active, "A_model_calls": 0,
            "scope": "single_site_consumption_only_not_functional_bridge_acceptance"}


def _bf_units(bits):
    bits = int(bits)
    exp, frac = (bits >> 7) & 255, bits & 127
    need(exp != 255, "finite_native_velocity")
    value = frac if exp == 0 else (128+frac) << (exp-1)
    return -value if bits >> 15 else value


def _scaled(value, exponent):
    if not value:
        return 0.
    sign, v = (-1 if value < 0 else 1), abs(value)
    shift = max(0, v.bit_length()-53)
    out = math.ldexp(float(sign*(v >> shift)), exponent+shift)
    need(math.isfinite(out), "finite_velocity_report")
    return out


def velocity_readout(velocities):
    """Exact dyadic signed effect accounting, numeric reports after accumulation."""
    need(set(velocities) == set(ROLES), "exact_B_none_d_c_velocity_roles")
    arrays = {r: np.asarray(velocities[r]) for r in ROLES}
    need(all(a.shape == (10,32) and a.dtype == np.uint16 for a in arrays.values()), "native_BF16_v6_10_by_32")
    ints = {r: [[_bf_units(v) for v in row] for row in arrays[r]] for r in ROLES}
    d = [[ints["natural_d"][p][c]-ints["none"][p][c] for c in range(32)] for p in range(10)]
    c = [[ints["common_c"][p][i]-ints["none"][p][i] for i in range(32)] for p in range(10)]
    rows = []
    for window,start,end in WINDOWS:
        for group,lo,hi in GROUPS:
            dv=[d[p][i] for p in range(start,end) for i in range(lo,hi)]
            cv=[c[p][i] for p in range(start,end) for i in range(lo,hi)]
            err=[y-x for x,y in zip(dv,cv)]
            de,ce,ee,dot=sum(x*x for x in dv),sum(x*x for x in cv),sum(x*x for x in err),sum(x*y for x,y in zip(dv,cv))
            need(ee == de+ce-2*dot, "exact_velocity_bridge_energy_closure")
            rows.append({"window":window,"group":group,"positions":end-start,"coordinates":len(dv),
                "d_effect_energy":_scaled(de,-266),"c_effect_energy":_scaled(ce,-266),"c_minus_d_energy":_scaled(ee,-266),
                "d_effect_signed_sum":_scaled(sum(dv),-133),"c_effect_signed_sum":_scaled(sum(cv),-133),
                "c_minus_d_signed_sum":_scaled(sum(err),-133),
                "d_effect_zero_coordinates":sum(x==0 for x in dv),"c_effect_zero_coordinates":sum(x==0 for x in cv),
                "effect_sign_reversals":sum(x*y<0 for x,y in zip(dv,cv)),
                "relative_bridge_error_norm":None if de==0 else math.sqrt(ee/de),
                "effect_norm_ratio":None if de==0 else math.sqrt(ce/de),
                "effect_cosine":None if de==0 or ce==0 else (-1. if dot<0 else 1.)*math.sqrt((dot*dot)/(de*ce)),
                "exact_integer_units_squared":{"d":str(de),"c":str(ce),"error":str(ee),"dot":str(dot)},
                "native_model_velocity_units":True,"adapter_action_or_issued_claim":False})
    # Per-coordinate signed effects remain small (3x320 values), and prevent
    # a window mean/RMS from hiding reversal or exact no-op coordinates.
    return {"schema":"B-BRIDGE-first-velocity-readout-v1","analysis_complete":True,"rows":rows,
        "signed_effects_model_coordinates":{"natural_d":[[_scaled(v,-133) for v in row] for row in d],
            "common_c":[[_scaled(v,-133) for v in row] for row in c],
            "common_minus_natural":[[_scaled(c[p][i]-d[p][i],-133) for i in range(32)]for p in range(10)]},
        "no_mixed_unit_ranking":True,"padding_is_not_control":True,"adapter7_not_computed":True,
        "meaningful_equivalence_threshold":None,"scientific_GO":False}
