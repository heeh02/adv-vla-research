"""Exact-PVP-derived independent source overlay; never edits a shared model."""
from pathlib import Path
import hashlib

ORIGINAL_SHA = "dfec7510ee7341dc90ef0c58111642bf102fdd24f70129890ecc98b210b61efe"


def replace_once(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError("exact_once_frozen_overlay_anchor")
    return raw.replace(old, new)


def build(raw):
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA:
        raise ValueError("frozen_PVP_Gstar_gemma_SHA")
    text = raw.decode()
    text = replace_once(text,
        "        capture=False, capture_prefix_interface=False, sru_capture=False,\n",
        "        capture=False, capture_prefix_interface=False, sru_capture=False,\n"
        "        bridge_target=None, bridge_active=None,\n")
    text = replace_once(text,
        "        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]\n",
        """        if bridge_target is not None:
            if not sru_capture or bridge_active is None or post_attn[1] is None:
                raise ValueError("B_bridge_requires_actual_suffix_capture_and_selector")
            bridge_before = post_attn[1]
            if (bridge_target.shape != bridge_before.shape or bridge_target.dtype != bridge_before.dtype
                    or bridge_target.dtype != jnp.bfloat16 or bridge_active.shape != ()
                    or bridge_active.dtype != jnp.bool_):
                raise ValueError("B_bridge_native_full_W_boolean_ABI")
            # This is the exact tensor passed to the original gated residual.
            bridge_used = jnp.where(bridge_active, bridge_target, bridge_before)
            post_attn = [post_attn[0], bridge_used]
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
""")
    text = replace_once(text,
        "        if sru_capture:\n            return xs, kv_cache, {**attention_result[2],\n",
        """        if sru_capture:
            bridge_trace = {} if bridge_target is None else {
                "bridge_W_before": bridge_before, "bridge_W_used": bridge_used,
                "bridge_active": bridge_active}
            return xs, kv_cache, {**attention_result[2], **bridge_trace,
""")
    classes = '''

BRIDGE_FIELDS = SCAN_NATIVE_FIELDS + ('bridge_W_before', 'bridge_W_used', 'bridge_active')


class ScanBridgeCaptureBlock(Block):
    """Main setter at post-WO/pre-gate; dynamic layer index comes from scan."""
    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic,
                 layer_index, target, active):
        xs, cache, trace = super().__call__(
            xs, kv_cache, positions, attn_mask, adarms_cond, deterministic,
            sru_capture=True, bridge_target=target, bridge_active=active & (layer_index == 8))
        return xs, (cache, {name: trace[name] for name in BRIDGE_FIELDS})


class IndependentBridgeCaptureBlock(Block):
    """Separate scoped Attention-return replacement; never uses Block's setter."""
    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic,
                 layer_index, target, active):
        def intercept(next_fun, args, kwargs, context):
            result = next_fun(*args, **kwargs)
            if isinstance(context.module, Attention) and context.method_name == '__call__':
                if len(result) != 3 or not isinstance(result[2], dict):
                    raise ValueError('independent_actual_Attention_capture_ABI')
                outputs, next_cache, operands = result
                before = outputs[1]
                if before is None or before.shape != target.shape or before.dtype != target.dtype:
                    raise ValueError('independent_native_target_ABI')
                selected = active & (layer_index == 8)
                used = jax.lax.select(selected, target, before)
                changed = [outputs[0], used]
                operands = {**operands, 'bridge_W_before': before, 'bridge_W_used': used,
                            'bridge_active': selected}
                return changed, next_cache, operands
            return result
        with nn.intercept_methods(intercept):
            xs, cache, trace = super().__call__(
                xs, kv_cache, positions, attn_mask, adarms_cond, deterministic,
                sru_capture=True)  # bridge_target defaults None: main setter is not called.
        return xs, (cache, {name: trace[name] for name in BRIDGE_FIELDS})
'''
    text = replace_once(text, "\n\nclass PrefixCaptureBlock(Block):", classes+"\n\nclass PrefixCaptureBlock(Block):")
    methods = '''

    def _b_bridge_suffix(self, embedded, positions, mask, adarms_cond, *, kv_cache,
                         w_target, w_active, independent):
        if (len(embedded) != 2 or embedded[0] is not None or embedded[1] is None
                or self.dropout or self.configs[0].depth != 18):
            raise ValueError('B_bridge_fixed_action_only_depth18')
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        if (w_target.shape != embedded[1].shape or w_target.dtype != jnp.bfloat16
                or w_active.shape != () or w_active.dtype != jnp.bool_):
            raise ValueError('B_bridge_target_native_BF16_and_scalar_bool')
        block_cls = nn.remat(
            IndependentBridgeCaptureBlock if independent else ScanBridgeCaptureBlock,
            prevent_cse=False, static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable)
        scanned = nn.scan(
            block_cls, variable_axes={'params': 0},
            split_rngs={'params': True, 'dropout': True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast,
                     0, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(configs=self.configs, dropout=self.dropout, dropout_bdims=self.dropout_bdims, parent=None)
        xs, (_unused_next_cache, all_trace) = scanned.apply(
            {'params': self.variables['params']['layers']}, embedded, kv_cache,
            positions, jnp.asarray(mask)[:, None, :, :], adarms_cond, True,
            jnp.arange(18, dtype=jnp.int32), w_target, w_active)
        selected = jnp.asarray((0, 8, 17), jnp.int32)
        trace = jax.tree.map(lambda value: jnp.take(value, selected, axis=0), all_trace)
        trace.update(mask=mask, suffix_positions=positions,
                     adarms_condition=adarms_cond[1], query_defined=mask.any(-1))
        return self.final_norms[1](xs[1], adarms_cond[1])[0], trace

    def scan_b_bridge_suffix(self, embedded, positions, mask, adarms_cond, *, kv_cache, w_target, w_active):
        return self._b_bridge_suffix(embedded, positions, mask, adarms_cond,
                                     kv_cache=kv_cache, w_target=w_target, w_active=w_active, independent=False)

    def scan_b_bridge_independent_suffix(self, embedded, positions, mask, adarms_cond, *, kv_cache, w_target, w_active):
        return self._b_bridge_suffix(embedded, positions, mask, adarms_cond,
                                     kv_cache=kv_cache, w_target=w_target, w_active=w_active, independent=True)
'''
    text = replace_once(text, "\n\n    def scan_native_prefix(self, embedded, positions, mask):", methods+"\n\n    def scan_native_prefix(self, embedded, positions, mask):")
    compile(text, "<B-bridge-independent-overlay>", "exec")
    return text.encode()


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    out = root/'overlay/gemma.py'
    with out.open('xb') as stream:
        stream.write(build((root/'frozen/gemma.py').read_bytes()))
