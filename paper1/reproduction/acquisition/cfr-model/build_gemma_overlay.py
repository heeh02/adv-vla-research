"""Build the cf-repair-v1 Gemma overlay from the pinned native-subspace-ablation overlay.

Provenance: the base is NSA/execution/overlay/src/openpi/models/gemma.py (SHA-256
3b94bba2b2a68c1a4c3b5d6e2e65168c522aa83f564c687c8b3ae4f326e537ea, 31531 bytes), which is
itself the closeout scientific-source gemma.py (737d0299...) plus default-off capture
branches and ``ablate_suffix``. The base file is read only and never modified. This
builder inserts exactly one new method, ``Module.cfr_scan_suffix``, before ``Module.init``.
``cfr_scan_suffix`` adapts the per-layer ``Block.apply`` scan of NSA ``ablate_suffix``
(gemma.py:573-631): the edit consumes H_C and the frozen bases through
``openpi.models.cfr_repair_rules.edit_rule`` and may run on every generation call.

Usage: python -B build_gemma_overlay.py [--check]   (no model, no network)
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = Path("/Users/researcher/Adv-VLA-π0.5/docs/experiments/native-subspace-ablation-20261014-v1/"
            "execution/overlay/src/openpi/models/gemma.py")
BASE_SHA256 = "3b94bba2b2a68c1a4c3b5d6e2e65168c522aa83f564c687c8b3ae4f326e537ea"
BASE_BYTES = 31531
OUTPUT = HERE / "src/openpi/models/gemma.py"
MARKER = b"    def init(self, use_adarms: Sequence[bool]):\n"

METHOD = '''    def cfr_scan_suffix(self, embedded, positions, mask, adarms_cond, *, kv_cache, layers,
                        edit_at=None, edit_inputs=None):
        """cf-repair-v1: action-expert per-layer Block.apply scan over the first ``layers`` layers.

        Adapted from ``ablate_suffix`` (same Block.apply binding of the stacked parameters,
        same lax.scan over (index, params, cache)). ``layers`` is 18 (treated pass, record
        route, clean variant B) or 9 (clean variant A, layers 0..8 with kv_cache[0:9]).
        With ``edit_at == 8`` the L9 post-block action H is replaced by
        ``cfr_repair_rules.edit_rule``; layers 9..17 then continue from H' with the same
        kv_cache (mixed context). Returns (readout or None, trace).
        """
        if (len(embedded) != 2 or embedded[0] is not None or embedded[1] is None
                or self.dropout or self.configs[0].depth != 18):
            raise ValueError("cfr scan requires deterministic action-only depth18")
        if layers not in (9, 18):
            raise ValueError("cfr scan supports 9 (clean partial) or 18 layers")
        if (edit_at is None) != (edit_inputs is None):
            raise ValueError("cfr edit needs both edit_at and edit_inputs")
        if edit_at is not None and (edit_at != 8 or layers != 18):
            raise ValueError("cfr edit site is the L9 post-block output of the 18-layer pass")
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        h = embedded[1]
        if h.ndim != 3 or h.shape[1] != 10:
            raise ValueError("cfr scan requires all ten action tokens")
        block = Block(configs=self.configs, parent=None)
        params = self.variables["params"]["layers"]
        if layers != 18:
            params = jax.tree.map(lambda value: value[:layers], params)
            kv_cache = jax.tree.map(lambda value: value[:layers], kv_cache)
        empty = jnp.zeros_like(h)
        mask4 = jnp.asarray(mask)[:, None, :, :]
        if edit_inputs is not None:
            from openpi.models import cfr_repair_rules as rules  # noqa: PLC0415
            if set(edit_inputs) != set(rules.EDIT_INPUT_KEYS):
                raise ValueError("cfr edit inputs differ from the frozen rule signature")
            aux0 = rules.empty_aux(h, edit_inputs["bases"])
        else:
            rules = None
            aux0 = {}

        def step(carry, inputs):
            xs, h8_before, h8_after, l10_out, l18_out, aux = carry
            index, layer_params, cache = inputs
            xs, _ = block.apply(
                {"params": layer_params}, xs, cache, positions, mask4, adarms_cond, True,
            )
            l10_out = jax.lax.cond(index == 9, lambda: xs[1], lambda: l10_out)
            l18_out = jax.lax.cond(index == 17, lambda: xs[1], lambda: l18_out)
            h8_before = jax.lax.cond(index == 8, lambda: xs[1], lambda: h8_before)
            if rules is not None:
                def edit(state):
                    x, _old = state
                    after, new_aux = rules.edit_rule(x[1], **edit_inputs)
                    return [None, after], new_aux

                xs, aux = jax.lax.cond(index == 8, edit, lambda state: state, (xs, aux))
            h8_after = jax.lax.cond(index == 8, lambda: xs[1], lambda: h8_after)
            return (xs, h8_before, h8_after, l10_out, l18_out, aux), None

        (xs, before, after, l10_out, l18_out, aux), _ = jax.lax.scan(
            step, (embedded, empty, empty, empty, empty, aux0),
            (jnp.arange(layers), params, kv_cache),
        )
        readout = self.final_norms[1](xs[1], adarms_cond[1])[0] if layers == 18 else None
        return readout, {"L9_H_before": before, "L9_H_after": after,
                         "L10_H_after": l10_out, "L18_H_after": l18_out, "edit_aux": aux}


'''


def build(source: bytes) -> bytes:
    if len(source) != BASE_BYTES or hashlib.sha256(source).hexdigest() != BASE_SHA256:
        raise SystemExit("pinned_NSA_overlay_gemma_source_drift")
    if source.count(MARKER) != 1 or b"def cfr_scan_suffix(" in source:
        raise SystemExit("single_insertion_site_required")
    return source.replace(MARKER, METHOD.encode() + MARKER, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed output only")
    args = parser.parse_args(argv)
    rendered = build(BASE.read_bytes())
    if args.check:
        ok = OUTPUT.is_file() and OUTPUT.read_bytes() == rendered
        print({"overlay_matches_build": ok, "sha256": hashlib.sha256(rendered).hexdigest(), "bytes": len(rendered)})
        return 0 if ok else 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(rendered)
    print({"written": str(OUTPUT), "sha256": hashlib.sha256(rendered).hexdigest(), "bytes": len(rendered)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
