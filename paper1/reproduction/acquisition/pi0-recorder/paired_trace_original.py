"""Reduce paired free-sampling traces before any device-to-host transfer."""

import numpy as np

from openpi.eval.attention_response import attention_summary
from openpi.eval.attention_response import camera_map
from openpi.eval.attention_response import native_mean
from openpi.eval.attention_response import paired_js


def hidden_group_rms(value, *, xp=np):
    """Actual hidden RMS, separately for executed prefix and predicted tail."""
    if value.shape[-2] != 10:
        raise ValueError("Layer scalars require the frozen ten action tokens")
    return xp.stack(
        [
            xp.sqrt(native_mean(value[..., :5, :] ** 2, (-2, -1), xp=xp)),
            xp.sqrt(native_mean(value[..., 5:, :] ** 2, (-2, -1), xp=xp)),
        ],
        axis=-1,
    )


def shadow_recompute_error(native, shadow, *, xp=np):
    """Actual vector residuals in source dtype, distinct from RMS differences.

    A difference of vector RMS values is not an upper bound on vector error.
    Norm/max summaries below operate on shadow-minus-recorded tensors first.
    Nothing in this function updates a flow state or produces a control action.
    """
    delta = {}
    for key in ("hidden", "readout", "velocity"):
        if native[key].shape != shadow[key].shape or native[key].dtype != shadow[key].dtype:
            raise ValueError(f"Native/shadow {key} shape or source dtype differs")
        delta[key] = shadow[key] - native[key]
    return {
        "velocity_delta": delta["velocity"],
        "velocity_rms_error": xp.sqrt(native_mean(delta["velocity"] ** 2, (-2, -1), xp=xp)),
        "velocity_max_abs_error": xp.max(xp.abs(delta["velocity"]), axis=(-2, -1)),
        "hidden_error_rms": hidden_group_rms(delta["hidden"], xp=xp),
        "hidden_error_max_abs": xp.max(xp.abs(delta["hidden"]), axis=(-2, -1)),
        "readout_error_rms": hidden_group_rms(delta["readout"], xp=xp),
        "readout_error_max_abs": xp.max(xp.abs(delta["readout"]), axis=(-2, -1)),
        "layer_rms_delta": hidden_group_rms(shadow["hidden"], xp=xp) - native["layer_rms"],
    }


def summarize_pair(clean, actual, *, save_visual=False, save_attention=False, xp=np):
    """Fixed pi05-LIBERO layout: three 256-token images, then language.

    The masked third camera remains in the model input, but not the visual
    archive. Arrays preserve time, selected-layer and batch axes. Native
    baseline and signed deltas retain the model output dtype; the diagnostic
    path never upgrades BF16 to FP32.
    """
    prefix_mask = clean["prefix_mask"]
    prefix_len = prefix_mask.shape[-1]
    q_count = clean["probabilities"].shape[-2]
    if prefix_len < 768 or q_count != 10:
        raise ValueError("Expected the frozen three-camera, ten-action-token layout")
    keys = xp.arange(prefix_len + q_count)
    regions = {
        "main": keys < 256,
        "wrist": (keys >= 256) & (keys < 512),
        "language": (keys >= 768) & (keys < prefix_len),
        "action": keys >= prefix_len,
    }
    allowed = xp.concatenate([prefix_mask, xp.ones((prefix_mask.shape[0], q_count), dtype=bool)], axis=-1)
    allowed = allowed[None, None, :, None, None, :]

    def flatten_heads(value):
        # [time, layer, batch, KV-head, query-group, query, key]
        return value.reshape((*value.shape[:3], value.shape[3] * value.shape[4], *value.shape[-2:]))

    def groups(value):
        if value.dtype == bool:
            return xp.stack([value[..., :5].all(-1), value[..., 5:].all(-1)], axis=-1)
        return xp.stack([native_mean(value[..., :5], -1, xp=xp), native_mean(value[..., 5:], -1, xp=xp)], axis=-1)

    pc, pa = flatten_heads(clean["probabilities"]), flatten_heads(actual["probabilities"])
    result = {
        "prefix_mask_equal": xp.all(prefix_mask == actual["prefix_mask"]),
        "hidden_actual_rms": hidden_group_rms(actual["hidden"], xp=xp),
    }
    for name, probabilities in (("clean", pc), ("actual", pa)):
        metrics = attention_summary(probabilities, allowed, regions, xp=xp)
        result[name + "_attention"] = {key: groups(value) for key, value in metrics.items()}
        result[name + "_camera_maps"] = xp.stack(
            [camera_map(probabilities, token_start=start, xp=xp) for start in (0, 256)], axis=-3
        )
    result["attention_js"] = groups(paired_js(pc, pa, allowed, xp=xp))
    result["attention_js_raw"] = groups(paired_js(pc, pa, allowed, xp=xp, clamp=False))
    for key in ("hidden", "readout", "velocity"):
        baseline = clean[key]
        if actual[key].dtype != baseline.dtype:
            raise TypeError("Paired trace dtypes differ")
        delta = actual[key] - baseline
        result[key + "_clean"] = baseline
        result[key + "_delta"] = delta
        result[key + "_signed_mean"] = xp.stack(
            [native_mean(delta[..., :5, :], -2, xp=xp), native_mean(delta[..., 5:, :], -2, xp=xp)], axis=-2
        )
        result[key + "_rms_by_dimension"] = xp.stack(
            [
                xp.sqrt(native_mean(delta[..., :5, :] ** 2, -2, xp=xp)),
                xp.sqrt(native_mean(delta[..., 5:, :] ** 2, -2, xp=xp)),
            ],
            axis=-2,
        )
    branch_delta = actual["attention_output"] - clean["attention_output"]
    result["pre_gate_attention_output_delta_rms"] = xp.sqrt(native_mean(branch_delta**2, (-2, -1), xp=xp))
    visual_delta = actual["prefix"][:, :512] - clean["prefix"][:, :512]
    visual_delta = visual_delta.reshape((visual_delta.shape[0], 2, 256, -1))
    result["visual_signed_mean"] = native_mean(visual_delta, -2, xp=xp)
    result["visual_rms_by_dimension"] = xp.sqrt(native_mean(visual_delta**2, -2, xp=xp))
    result["visual_token_rms"] = xp.sqrt(native_mean(visual_delta**2, -1, xp=xp))
    if save_visual:
        result["visual_clean"] = clean["prefix"][:, :512].reshape(visual_delta.shape)
        result["visual_delta"] = visual_delta
    if save_attention:
        result["attention_clean"] = clean["probabilities"]
        result["attention_actual"] = actual["probabilities"]
    return result
