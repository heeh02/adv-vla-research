#!/usr/bin/env python3
"""Exact P08 selected-NPZ reader; no original cache/model access or statistics."""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import tarfile
import time
import zipfile
import numpy as np

SCHEMA = "P08-late-joint-native-selected-v1"
BUCKET = "redacted-15cbc6ba826f"
PREFIX = "s3://" + BUCKET + "/one-vla/artifacts/users/jinwen/adv-vla/libero-light-v1/analysis-system-20260929/late-three-layer-selected-20261001-v1/P08-late-joint-ed1c90766bf48f05/"
FIELD_NAMES = {
    "Hclean": "full/free/hidden_clean", "Hdelta": "full/free/hidden_delta",
    "Aclean": "derived16/clean", "Aactual": "derived16/actual",
    "flow_clean": "full/free/velocity_clean", "flow_delta": "full/free/velocity_delta",
    "times": "compact/times",
}
VALID_NAMES = {k: "valid/free/" + k for k in
               ("clean_source_mass", "actual_source_mass", "source_mass_delta")}
NATIVE_AXES = {
    **{f"full/free/hidden_{b}": ["free_tau", "layer", "action_token", "hidden_dim"] for b in ("clean", "delta")},
    **{f"full/free/velocity_{b}": ["free_tau", "action_token", "velocity_dim"] for b in ("clean", "delta")},
    **{f"derived16/{b}": ["free_tau", "layer", "head_k_times_g", "query_group", "camera", "map_y", "map_x"] for b in ("clean", "actual")},
    **{f"valid/free/{b}": ["free_tau", "layer", "head", "query_group", "source"] for b in VALID_NAMES},
    "compact/times": ["free_tau"],
}
INT_FIELDS = ("slot", "control_step", "query_index", "task", "init", "noise_repeat", "task_fold")
FLOAT_FIELDS = ("query_time", "sensor_time", "physical_prefix_translation_rms",
                "physical_prefix_rotation_rms", "physical_prefix_gripper_rms")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checked_file(path, expected, limit):
    p = Path(path)
    need(not p.is_symlink() and p.is_file() and p.stat().st_size <= limit, "bounded_regular_file")
    raw = p.read_bytes()
    need(len(raw) == expected["bytes"] and sha(raw) == expected["sha256"], "frozen_file_bytes_SHA")
    return raw


def load_inputs(path):
    """Read the small prepared manifest/ledger; no scientific arrays or network."""
    p = Path(path)
    config = json.loads(p.read_text())
    need(config["schema"] == "P08-dynamic-readset-v1", "readset_schema")
    ledger = config["owner_ledger"]
    raw = checked_file(p.parent / ledger["bundle_name"], ledger, 4_000_000)
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode())))
    need(len(rows) == 6720 and len({r["anchor_id"] for r in rows}) == 6720, "planned_owner_domain")
    present = [r for r in rows if r["original_present"] == "True"]
    need(len(present) == 6554 and len({r["file_id"] for r in present}) == 6554, "selected_file_domain")
    need(sum(int(r["selected_file_bytes"]) for r in present) == 1024812888, "selected_NPZ_byte_domain")
    need(set(config["parts"]) == {"analysis-and-source"} | {f"s{i:02d}" for i in range(40)}, "archive_part_domain")
    need(config.get("parts_to_read") == [f"s{i:02d}" for i in range(40)], "frozen_forty_selected_parts")
    for row in rows:
        metadata(row)  # validates typed support without filling missing values
        if row["original_present"] == "True":
            need(row["extraction_status"] == "complete", "original_extraction_status")
            name = PurePosixPath(row["selected_file_path"])
            need(len(name.parts) == 4 and name.parts[0] == "shards" and name.parts[1] in config["parts"]
                 and name.parts[2] == "selected" and name.name == row["file_id"] + ".npz", "exact_selected_locator")
        else:
            need(row["extraction_status"] == "missing_original_episode"
                 and not row["selected_file_path"] and not row["original_success"], "missing_owner_not_imputed")
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["episode_id"], {})
        need(row["slot"] not in pair, "duplicate_episode_slot")
        pair[row["slot"]] = row
    need(len(pairs) == 3360 and all(set(p) == {"1", "2"} for p in pairs.values()), "two_slot_pairs")
    for pair in pairs.values():
        need(all(pair["1"][k] == pair["2"][k] for k in
                 ("original_present", "original_success", "suite", "task", "init", "noise_repeat",
                  "history_condition", "original_source_id", "original_result_sha256")), "episode_tuple_join")
    return config, rows


def metadata(row):
    """Typed convenience fields plus the complete untouched original string row."""
    owner = {k: row[k] for k in ("anchor_id", "episode_id", "file_id", "root", "suite",
                                "history_condition", "declared_current_condition")}
    for key in INT_FIELDS:
        owner[key] = int(row[key]) if row[key] != "" else None
    for key in FLOAT_FIELDS:
        owner[key] = float(row[key]) if row[key] != "" else None
        need(owner[key] is None or math.isfinite(owner[key]), "finite_owner_scalar")
    need(row["original_present"] in ("True", "False") and row["reached"] == row["original_present"], "original_support")
    owner["original_present"] = row["original_present"] == "True"
    need(row["original_success"] in ("", "0", "1"), "outcome_domain")
    owner["original_success"] = int(row["original_success"]) if row["original_success"] != "" else None
    owner["task_id"] = f'{row["suite"]}:t{int(row["task"]):02d}'
    owner["generation_call_index"] = 0
    owner["generation_time"] = 1.0
    owner["planned_action_positions"] = [0, 1, 2, 3, 4]
    owner["actual_control_step"] = owner["control_step"]
    need(owner["slot"] in (1, 2) and owner["control_step"] == 5 * owner["query_index"], "physical_query_clock")
    owner["issued_action_status"] = "not_in_selected_P08_readset"
    owner["physical_motion_status"] = "not_in_selected_P08_readset"
    return {"owner": owner, "ledger": dict(row)}


def _json_array(archive, name):
    info = archive.getinfo(name + ".npy")
    need(info.file_size <= 250_000, "bounded_NPZ_metadata")
    value = np.load(io.BytesIO(archive.read(info)), allow_pickle=False)
    need(value.shape == () and value.dtype.kind in ("U", "S"), "scalar_string_metadata")
    return json.loads(value.item())


def decode_selected(raw, row, config):
    """Validate current NPZ and original BF16 payload hashes, return one owner.

    Numerical BF16 values are losslessly expanded to float32 via their bits.
    No arithmetic on H/A/flow, averaging, normalization or differencing occurs.
    """
    need(len(raw) == int(row["selected_file_bytes"]) and sha(raw) == row["selected_file_sha256"],
         "selected_NPZ_bytes_SHA")
    fields = config["fields"]
    need(set(fields) == set(FIELD_NAMES.values()) | set(VALID_NAMES.values()), "exact_ten_fields")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        expected = {k + ".npy" for k in fields} | {"__schema__.npy", "__selected_meta__.npy"}
        need(len(infos) == len(expected) and {i.filename for i in infos} == expected, "NPZ_unique_complete_inventory")
        need(all(i.compress_type == zipfile.ZIP_STORED and not i.flag_bits & 1 for i in infos)
             and sum(i.file_size for i in infos) <= 500_000, "NPZ_stored_bounded")
        schema = _json_array(archive, "__schema__")
        meta = _json_array(archive, "__selected_meta__")
        need(set(schema) == set(fields) | {"__selected_meta__"}, "selected_schema_inventory")
        need(meta["schema"] == SCHEMA and meta["index_sha256"] == config["producer_identity"]["index_sha256"]
             and meta["contract_sha256"] == config["producer_identity"]["contract_sha256"]
             and meta["program_sha256"] == config["producer_identity"]["program_sha256"], "producer_identity")
        need(meta["selection"] == {"tau": 0, "layers": [1, 9, 18], "queries": [0, 1, 2, 3, 4]}
             and meta["source_whole_SHA_recomputed"] is True and meta["persisted_logical_bytes"] == 111560,
             "original_selection_qualification")
        need(meta["source_cache_sha256"] == row["cache_sha256"]
             and meta["source_trace_sha256"] == row["trace_sha256"], "trace_cache_identity")
        need(set(meta["owner"]) == set(config["embedded_owner_keys"])
             and all(str(meta["owner"][k]) == row[k] for k in config["embedded_owner_keys"]), "embedded_owner_full_join")
        need(set(meta["selected_members"]) == set(fields), "per_field_receipt_inventory")
        values, proofs = {}, {}
        for name, spec in fields.items():
            descriptor = schema[name]
            need(descriptor["shape"] == spec["shape"] and descriptor["dtype"] == spec["logical_dtype"]
                 and descriptor["stored_dtype"] == spec["stored_dtype"], "logical_field_schema")
            a = np.load(io.BytesIO(archive.read(name + ".npy")), allow_pickle=False)
            need(list(a.shape) == spec["shape"] and a.dtype.str == spec["stored_dtype"]
                 and not a.dtype.hasobject and a.flags.c_contiguous, "actual_field_shape_dtype")
            bits = a.tobytes(order="C")
            original = meta["selected_members"][name]
            need(len(bits) == spec["payload_bytes"] == original["slice_bytes"]
                 and sha(bits) == original["slice_SHA256"] and original["slice_shape"] == spec["shape"]
                 and original["native_dtype"] == spec["logical_dtype"]
                 and original["native_axes"] == NATIVE_AXES[name]
                 and original["selection"] == "tau0/first5; times full10", "original_slice_bits_SHA")
            if spec["logical_dtype"] == "bfloat16":
                need(descriptor["dtype_str"] == "<V2" and a.dtype.str == "<u2", "BF16_uint16_encoding")
                value = (a.astype(np.uint32) << np.uint32(16)).view(np.float32)
            else:
                need(descriptor["dtype_str"] == spec["stored_dtype"], "native_dtype_encoding")
                value = a.copy()
            need(np.isfinite(value).all(), "finite_selected_values")
            value.flags.writeable = False
            values[name] = value
            proofs[name] = {"slice_SHA256": sha(bits), "stored_dtype": a.dtype.str,
                            "payload_bytes": len(bits), "original_full_NPY_SHA256": original["member_SHA256"]}
    need(values["compact/times"].shape == (10,) and values["compact/times"][0] == np.float32(1), "recorded_j0_time")
    result = metadata(row)
    result.update({k: values[name] for k, name in FIELD_NAMES.items()})
    result["validity"] = {k: values[name] for k, name in VALID_NAMES.items()}
    result["status"] = "selected_owner_verified"
    result["source"] = {"selected_file_path": row["selected_file_path"],
                        "selected_file_sha256": row["selected_file_sha256"],
                        "selected_file_bytes": int(row["selected_file_bytes"]),
                        "field_receipts": proofs}
    return result


class VerifiedPart:
    """One complete 18–28 MB selected archive, never the original cache."""
    def __init__(self, directory, part):
        self.directory, self.part = Path(directory), part
        manifest_path = self.directory / "MEMBER_MANIFEST.json"
        need(manifest_path.is_file() and not manifest_path.is_symlink()
             and manifest_path.stat().st_size <= 2_000_000
             and file_sha(manifest_path) == part["manifest_sha256"], "member_manifest_SHA")
        self.manifest = json.loads(manifest_path.read_text())
        need(self.manifest["part"] == part["part"] and self.manifest["tar_sha256"] == part["tar_sha256"]
             and self.manifest["tar_bytes"] == part["tar_bytes"], "part_manifest_identity")
        tarpath = self.directory / "members.tar"
        need(tarpath.is_file() and not tarpath.is_symlink() and tarpath.stat().st_size == part["tar_bytes"]
             and file_sha(tarpath) == part["tar_sha256"], "whole_tar_bytes_SHA")
        records = self.manifest["members"]
        self.records = {r["path"]: r for r in records}
        need(len(records) == len(self.records) == part["member_count"]
             and sum(r["bytes"] for r in records) == part["source_member_bytes"], "tar_manifest_domain")
        self.tar = tarfile.open(tarpath, "r:")
        self.members = {}
        try:
            for member in self.tar:
                name = PurePosixPath(member.name)
                need(member.isfile() and not name.is_absolute() and ".." not in name.parts
                     and str(name) == member.name and member.name in self.records
                     and member.name not in self.members
                     and member.size == self.records[member.name]["bytes"], "actual_tar_header_name_type_size")
                self.members[member.name] = member
            need(set(self.members) == set(self.records), "complete_tar_member_domain")
        except BaseException:
            self.tar.close()
            raise

    def read(self, name, limit=4_000_000):
        need(name in self.members and self.members[name].size <= limit, "exact_bounded_member")
        with self.tar.extractfile(self.members[name]) as f:
            raw = f.read(limit + 1)
        rec = self.records[name]
        need(len(raw) == rec["bytes"] and sha(raw) == rec["sha256"], "current_member_payload_SHA")
        return raw

    def close(self):
        self.tar.close()


def cloud_root(path):
    need(platform.system() == "Linux", "cloud_Linux_only_never_Mac")
    p = Path(path)
    need(p.is_absolute() and str(p).startswith("/tmp/") and p.resolve() == p
         and not any(x.is_symlink() for x in (p, *p.parents)), "private_tmp_no_links")
    p.mkdir(parents=True, exist_ok=True, mode=0o700)
    need(p.stat().st_dev == Path("/tmp").stat().st_dev, "tmp_device_only")
    for mounted in (Path("/mnt/workspace"), Path("/mnt/jpfs")):
        if mounted.exists() and os.path.ismount(mounted):
            need(p.stat().st_dev != mounted.stat().st_dev, "no_JPFS_device")
    need(os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", "CPU_only")
    return p


def restore_part(download, part, destination, deadline):
    """Exact-object callback transport; no SDK/profile/credential dependency.

    download(uri, expected, destination) receives bytes (when known), max_bytes,
    and sha256. Parent supplies authenticated or presigned transport in memory.
    This function independently rechecks every returned file; no URL is logged.
    """
    need(time.time() < deadline, "read_deadline")
    uri = part["artifact_uri"]
    need(uri.startswith(PREFIX + part["part"] + "/sha256-") and uri.endswith("/")
         and part["artifact_id"] == "sha256:" + uri.rstrip("/").split("sha256-")[-1]
         and 0 < part["tar_bytes"] <= 32_000_000, "frozen_P08_archive_scope")
    dest = Path(destination)
    need(not dest.is_symlink(), "part_directory_no_link")
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, expected in (
        ("MEMBER_MANIFEST.json", {"sha256": part["manifest_sha256"], "max_bytes": 2_000_000}),
        ("members.tar", {"sha256": part["tar_sha256"], "bytes": part["tar_bytes"], "max_bytes": part["tar_bytes"]}),
    ):
        need(time.time() < deadline, "download_deadline")
        target = dest / name
        if not target.exists():
            need(not target.is_symlink(), "object_destination_no_link")
            download(uri + "payload/" + name, dict(expected), target)
        need(target.is_file() and not target.is_symlink() and target.stat().st_size <= expected["max_bytes"]
             and ("bytes" not in expected or target.stat().st_size == expected["bytes"])
             and file_sha(target) == expected["sha256"], "downloaded_object_current_SHA")
    need(time.time() < deadline, "verified_part_deadline")
    return VerifiedPart(dest, part)


def iter_owners(inputs_path, download, workdir, deadline):
    """Stream all 6,720 planned owners, preserving 166 missing rows.

    Only the frozen forty selected-data shards are read. The bundled original
    owner ledger is independently pinned, so analysis-and-source is unnecessary.
    Parent launcher owns callback credentials, network timeout, deadline and BOS.
    """
    config, rows = load_inputs(inputs_path)
    work = cloud_root(workdir)
    for row in rows:
        if row["original_present"] == "False":
            value = metadata(row)
            value.update({k: None for k in FIELD_NAMES})
            value.update(status="missing_original_episode", validity=None, source=None)
            yield value
    for shard in config["parts_to_read"]:
        part = restore_part(download, config["parts"][shard], work / shard, deadline)
        try:
            selected = [r for r in rows if r["selected_file_path"].startswith(f"shards/{shard}/selected/")]
            need({n for n in part.members if n.endswith(".npz")} ==
                 {"results/" + r["selected_file_path"] for r in selected}, "shard_selected_file_set")
            for row in selected:
                need(time.time() < deadline, "owner_read_deadline")
                member = "results/" + row["selected_file_path"]
                rec = part.records[member]
                need(rec["bytes"] == int(row["selected_file_bytes"])
                     and rec["sha256"] == row["selected_file_sha256"], "owner_manifest_member_identity")
                value = decode_selected(part.read(member, limit=500_000), row, config)
                value["source"].update(artifact_uri=config["parts"][shard]["artifact_uri"],
                                       tar_sha256=config["parts"][shard]["tar_sha256"], tar_member=member)
                yield value
        finally:
            part.close()
