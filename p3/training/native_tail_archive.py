"""Owned 30k metadata adapter; native member PUT/GET publication body unchanged."""
import importlib.util
import json
import os
from pathlib import Path
from protocol import ROOT, HERE, ORIGINAL, require, digest, identity, write_new


def writer_class(data_prefix, checkpoint_prefix):
    require(data_prefix.startswith("redacted-24fc043a6cae") and data_prefix.endswith("/"), "exact data run prefix required")
    require(checkpoint_prefix.startswith("redacted-53ad6e2d54fe") and checkpoint_prefix.endswith("/"), "exact checkpoint run prefix required")
    path = ROOT / "docs/experiments/response-training-feasibility-20261006-v1/multi-anchor-update-candidate-v1/execution-candidate-v2/bos_transport.py"
    pins = json.loads((HERE / "DEPENDENCIES-v1.json").read_bytes())["files"]
    row = next(r for r in pins if r["path"] == str(path.relative_to(ROOT)))
    require(identity(path) == {k: row[k] for k in ("bytes", "sha256")}, "actual-qualified BOS transport differs")
    spec = importlib.util.spec_from_file_location("cul_private_checkpoint_transport", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Only this fresh private module namespace gets the two root-reviewed new
    # exact scopes. The frozen source/SDK/config and all old scopes are intact.
    module.ALLOWED_SCOPES = {data_prefix, checkpoint_prefix}
    return module.BOSWriter


def archive(step_directory, sidecar, local_record, contract, *, checkpoint_writer, data_writer, out, check_deadline):
    """Called only after the native stock manager has waited for closed save."""
    step_directory, sidecar, out = Path(step_directory), Path(sidecar), Path(out)
    count = local_record["step"]
    require(contract["schema"] == "T20-optimization-continuation-30k-contract-v1" and 20000 < count <= 30000, "tail_checkpoint_step")
    prefix = "step-" + str(count).zfill(6) + "/"
    objects = []
    progress = out / ("CHECKPOINT_VERIFIED_OBJECTS-step" + str(count) + ".jsonl")
    with progress.open("x"): pass
    def remember(row):
        objects.append(row)
        # A later upload/manifest failure retains every already-proven object.
        with progress.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush(); os.fsync(stream.fileno())
    def snapshot(folder):
        found = []
        for path in sorted(Path(folder).rglob("*")):
            if path.is_dir(): continue
            require(path.is_file() and not path.is_symlink(), "closed checkpoint support contains nonregular member")
            s = path.lstat()
            found.append((str(path.relative_to(folder)), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
        return found
    before = snapshot(step_directory)
    require([r[0] for r in before] == [r["relative"] for r in local_record["stock_files"]], "closed checkpoint inventory differs before archive")
    for row in local_record["stock_files"]:
        check_deadline()
        relative = prefix + "checkpoint/" + row["relative"]
        receipt = checkpoint_writer.put(step_directory / row["relative"], relative)
        require({k: receipt[k] for k in ("bytes", "sha256")} == {k: row[k] for k in ("bytes", "sha256")}, "stock source/BOS whole identity differs")
        remember({**receipt, "cache_relative": "checkpoint/" + str(count) + "/" + row["relative"], "stock_relative": row["relative"], "role": "stock_native_checkpoint"})
    for name in ("CURSOR-v1.json", "STATE_IDENTITY-v1.json", "PREFIX_IDENTITY-v1.json", "LOCAL_COMMIT-v1.json"):
        check_deadline()
        receipt = checkpoint_writer.put(sidecar / name, prefix + "sidecar/" + name)
        require({k: receipt[k] for k in ("bytes", "sha256")} == identity(sidecar / name), "sidecar/BOS identity differs")
        remember({**receipt, "cache_relative": "sidecar/" + str(count) + "/" + name, "role": "matching_state_cursor_sidecar"})
    require(snapshot(step_directory) == before, "closed checkpoint changed before manifest commit")
    manifest = {"schema": "T30-full-checkpoint-manifest-v1", "logical_run_id": contract["logical_run_id"],
        "arm": contract["arm"], "training_seed": contract["seed"], "initialization_seed": 0,
        "effective_updates": count, "scientific_contract_sha256": digest(contract),
        "input_index_sha256": contract["input_index_identity"]["sha256"], "files": objects,
        "whole_member_independent_GET_SHA_verified": True, "params_online_opt_EMA_cursor_complete": True,
        "native_full_state_tree_sha256": local_record["full_state_tree_sha256"],
        "original_checkpoint_artifact_id": ORIGINAL, "automatic_retry": False}
    target = out / ("CHECKPOINT_MANIFEST-step" + str(count) + ".json")
    write_new(target, manifest)
    manifest_object = checkpoint_writer.put(target, prefix + "CHECKPOINT_MANIFEST-v1.json")
    check_deadline()
    role = "new_endpoint30k" if count == 30000 else "recovery_process25k" if count == 25000 else "resume_partial"
    find = lambda name: next({k: r[k] for k in ("uri", "bytes", "sha256")} for r in objects if r.get("cache_relative", "").endswith("/" + name))
    trigger = {"schema": "T30-native-checkpoint-commit-v1", "logical_run_id": contract["logical_run_id"],
        "arm": contract["arm"], "training_seed": contract["seed"], "initialization_seed": 0,
        "effective_updates": count, "actual_stock_step": count, "logical_target_updates": 30000,
        "checkpoint_role": role, "eval_eligible": count == 30000, "main_evaluation_parameters": "EMA",
        "scientific_contract_sha256": digest(contract), "input_index_sha256": contract["input_index_identity"]["sha256"],
        "original_checkpoint_artifact_id": ORIGINAL, "checkpoint_manifest": manifest_object,
        "full_state_identity": find("STATE_IDENTITY-v1.json"), "prefix_identity": find("PREFIX_IDENTITY-v1.json"),
        "local_commit": find("LOCAL_COMMIT-v1.json"),
        "EMA_inference_members": [r for r in objects if r.get("stock_relative", "").startswith(("params/", "assets/"))],
        "full_checkpoint_restore_permitted": True, "stock_params_item_contains_EMA": True,
        "whole_member_independent_GET_SHA_verified": True, "success_or_learning_claimed": False}
    trigger_path = out / ("CHECKPOINT_TRIGGER-step" + str(count) + ".json")
    write_new(trigger_path, trigger)
    receipt = data_writer.put(trigger_path, "checkpoint-triggers/" + prefix + "CHECKPOINT_COMMIT-v1.json")
    require({k: receipt[k] for k in ("bytes", "sha256")} == identity(trigger_path), "trigger did not independently close")
    return {**trigger, "trigger_object": receipt}
