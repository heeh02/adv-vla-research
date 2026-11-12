"""Existing exact-source records -> signed transfer rows, CPU only.

No credentials, networking, model imports or new simulation. The original
context-ablation capsule is retained byte-for-byte under legacy/. All native
arrays are decoded on a Linux CPU after whole object and member validation.
"""
from contextlib import contextmanager, nullcontext
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, code):
    if not ok:
        raise ValueError(code)


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@contextmanager
def legacy_modules(package):
    """Isolate original bare imports from the new statistical/transport layer."""
    names = ("contract", "source_admission", "prepare_runtime", "archive_pack",
             "resolver", "ablation", "sources", "source_gate", "science")
    previous = {n: sys.modules.get(n) for n in names}
    old_path = list(sys.path)
    try:
        sys.path[:0] = [str(package / "worker"), str(package / "worker/core")]
        modules = {}
        for name in names:
            parent = package / ("worker/core" if name in ("ablation", "sources") else "worker")
            spec = importlib.util.spec_from_file_location(name, parent / (name + ".py"))
            value = importlib.util.module_from_spec(spec)
            sys.modules[name] = value
            spec.loader.exec_module(value)
            modules[name] = value
        yield modules
    finally:
        sys.path[:] = old_path
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def expand_row(D, model, query, old_row):
    """One query/edit is one sample; position/head axes are feature axes."""
    p = query.p_native.value()[0, 0].astype(np.float64)
    e = query.edit("shared").astype(np.float64)
    n = e.shape[0]
    wo = model.wo.value().astype(np.float64)
    gate = query.gate.value()[0, 0].astype(np.float64)
    pooled = D.P.masked_mean(model.pooled, query.mask.value()[0])
    need(pooled is not None, "defined_original_full_key_pooled_read")
    values = {}
    for name, probabilities in (("current", p), ("pooled", pooled)):
        # Prefix probabilities retain the original full-softmax denominator;
        # they are never renormalized to the prefix or selected modality.
        read = np.einsum("hpn,nd->hpd", probabilities[:, :, :n], e, optimize=True)
        write = np.einsum("hpd,hdf->pf", read, wo, optimize=True) * gate[None, :]
        need(read.shape == (8, 10, 256) and write.shape == (10, 1024), "signed_feature_ABI")
        need(np.isfinite(read).all() and np.isfinite(write).all(), "finite_signed_features")
        # New optimal contraction order can change float64 roundoff. Its
        # equivalence to the exact retained formula is explicitly checked.
        old_write = D.P.local_write(probabilities, e, model.wo, query.gate).reshape(10, 1024)
        need(np.allclose(write, old_write, rtol=2e-12, atol=2e-12), "same_retained_float64_write_algebra")
        # Use the original algebra's actual output to avoid changing write4
        # input arithmetic and to bind all write comparisons to one array.
        values[name + "_gated_write"] = old_write
        values[name + "_headpos_read"] = read
        values[name + "_headpos_norms"] = np.linalg.norm(read, axis=-1).reshape(80)
    features = model.features(query, "shared")
    need(np.array_equal(features["current"], old_row.current), "current13_original_replay")
    need(np.array_equal(features["pooled"], old_row.pooled), "pooled13_original_replay")
    return {"key": old_row.key, "split": old_row.split,
            "context9": old_row.current[:9].copy(), "current4": old_row.current[9:].copy(),
            "pooled4": old_row.pooled[9:].copy(), "originalnorm4": features["norm"][9:].copy(),
            **values, "effect": old_row.effect.copy(), "zero_actual_E": old_row.zero_actual_E,
            "provenance": old_row.provenance}


def load_rows(stage, cache, deadline_epoch, *, synthetic=False):
    """Return {rows, frozen, original_qa, input_receipt}; no cloud calls.

    stage contains files named exactly INPUTS.json input_objects[].path. cache
    must be an empty private directory. Real neural arrays require Linux.
    """
    need(synthetic or sys.platform.startswith("linux"), "real_neural_array_processing_cloud_Linux_only")
    stage, cache = Path(stage).resolve(), Path(cache).resolve()
    need(stage.is_dir() and not stage.is_symlink(), "existing_exact_stage_directory")
    cache.mkdir(parents=True, exist_ok=True)
    need(not any(cache.iterdir()), "empty_private_adapter_cache")
    manifest_raw = (HERE / "INPUTS.json").read_bytes()
    manifest = json.loads(manifest_raw)
    refs = {r["uri"]: r for r in manifest["input_objects"]}
    need(len(refs) == manifest["input_object_count"] == 1500, "exact1500_unique_input_objects")
    need(sum(r["bytes"] for r in refs.values()) == manifest["input_body_bytes"] == 769677897,
         "exact_unique_input_body_bytes")
    checked = {}

    def check():
        need(time.time() < deadline_epoch, "signed_transfer_absolute_science_deadline")

    def restore(ref, *, destination, deadline_epoch):
        check()
        bound = refs.get(ref["uri"])
        need(bound is not None and all(bound[k] == ref[k] for k in ("bytes", "sha256")), "exact_staged_input_ref")
        source = stage / bound["path"]
        need(source.is_file() and not source.is_symlink() and source.parent == stage, "regular_flat_staged_input")
        # The retained reader independently validates the copied whole object
        # and all TAR member hashes, not just the staging transport receipt.
        shutil.copyfile(source, destination)
        checked[ref["uri"]] = {k: bound[k] for k in ("path", "bytes", "sha256")}

    class Budget:
        def download_bound(self, _):
            return nullcontext()

    package = HERE / "legacy"
    with legacy_modules(package) as modules:
        C, A, S, science = (modules[n] for n in ("contract", "ablation", "sources", "science"))
        _, _, _, planned = modules["prepare_runtime"].verify(package)
        need(set(planned["objects"]) == set(refs), "same_original_unique_URI_set")
        readers = {}
        def get(name):
            if name not in readers:
                readers[name] = modules["resolver"].Reads(package, cache / name, planned["phases"][name],
                                                         restore, Budget(), deadline_epoch)
            return readers[name]
        original_row = S.row
        expanded = {}
        def capture(D, model, query, effect, **kwargs):
            row = original_row(D, model, query, effect, **kwargs)
            need(row.key not in expanded, "unique_expanded_input_identity")
            expanded[row.key] = expand_row(D, model, query, row)
            return row
        S.row = capture
        try:
            D = S.dependencies({n: package / "worker/core" / n for n in S.PINS})
            phase0 = planned["phases"][C.SHARDS[0]]["science_plan"]
            r0 = get(C.SHARDS[0])
            artifact = r0(phase0["fit_manifest"], lambda p: json.loads(p.read_bytes()))
            arrays = r0(phase0["fit_arrays"], science.read_npz)
            need(synthetic or artifact["synthetic"] is False, "actual_original_fit_artifact")
            old = D.P.load_frozen(artifact["model"], arrays, expected_identity=artifact["model_identity"])
            scales = A.FrozenScales(old, synthetic=synthetic)
            devphase = planned["phases"]["DEV"]
            rd = get("DEV")
            def devread(ref, fn):
                return rd(devphase["by_original_uri"][ref["uri"]], fn)
            dev, devproof = S.dev_rows(D, old, devphase["science_plan"], devread,
                                      check=check, synthetic=synthetic)
            rd.cleanup()
            rq = get("QUAL")
            qualified = S.qualification(rq, planned["qualification"], synthetic=synthetic)
            rq.cleanup()
            conf, missing, errors = [], {}, []
            for sid in C.SHARDS:
                check()
                phase = planned["phases"][sid]
                reader = get(sid)
                if sid != C.SHARDS[0]:
                    plan = phase["science_plan"]
                    af = reader(plan["fit_manifest"], lambda p: json.loads(p.read_bytes()))
                    aa = reader(plan["fit_arrays"], science.read_npz)
                    same = D.P.load_frozen(af["model"], aa, expected_identity=af["model_identity"])
                    need(same.identity == old.identity, "same_frozen_fit_across_tasks")
                values, lost, issues = science.conf_task(D, old, artifact, arrays, phase, reader,
                                                        qualified, check, synthetic=synthetic)
                conf.extend(values)
                missing.update(lost)
                errors.extend({"shard": sid, **x} for x in issues)
                reader.cleanup()
            need(not missing and not errors, "complete_original_qualified_population")
            need(len(dev) == 48 and len(conf) == 120 and set(expanded) == set(A.DEV_KEYS + A.CONF_KEYS),
                 "exact_shared48_KT96_KC24")
            need(sum(len(r.receipts) for r in readers.values()) == planned["physical_read_count"],
                 "all_original_phase_reads_consumed")
            need(set(checked) == set(refs), "all1500_unique_staged_inputs_consumed")
            fitted = A.fit(scales, dev)
            original_qa = A.evaluate(fitted, conf)
            frozen = {n: np.array(getattr(scales, n), copy=True)
                      for n in ("target_mean", "target_scale", "feature_mean", "feature_scale")}
            receipt = {"input_manifest": identity(manifest_raw), "unique_objects": len(checked),
                       "unique_body_bytes": sum(r["bytes"] for r in checked.values()),
                       "phase_read_count": sum(len(r.receipts) for r in readers.values()),
                       "phase_read_bytes": sum(x["bytes"] for r in readers.values() for x in r.receipts),
                       "verified_every_tar_member_proofs": sum(x["every_tar_member_proof"] is not None
                                                               for r in readers.values() for x in r.receipts),
                       "phase_failures": {n: r.failures for n, r in readers.items()},
                       "original_DEV": devproof, "original_fit_identity": old.identity,
                       "old_scores_replaced": False, "posthoc_not_new_blind_test": True,
                       "model_calls": 0, "env_steps": 0, "GPU": 0, "original_arrays_to_Mac": False}
            return {"rows": [expanded[k] for k in (*A.DEV_KEYS, *A.CONF_KEYS)], "frozen": frozen,
                    "original_qa": original_qa, "input_receipt": receipt}
        finally:
            S.row = original_row
            for reader in readers.values():
                reader.cleanup()
