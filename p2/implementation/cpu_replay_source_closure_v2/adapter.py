"""Verify archived runtime bytes and export textual source; never execute it."""
import argparse
import ast
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import time
import zipfile

HERE = Path(__file__).resolve().parent
RESULT = "cpu-replay-source-closure-result.json"
OUTPUTS = (RESULT, "SOURCE-BUNDLE.zip", "SOURCE-INDEX.json", "QA.json", "PROTOCOL.json")
SOURCE_FILES = ("adapter.py", "INPUTS.json", "PROTOCOL.json", "test_extract.py",
                "freeze.py", "README.md", "IMAGE-SOURCE-PROVENANCE.json")
COUNTS = {"archives": 2, "input_objects": 15}
MIB = 1024 * 1024
PACKAGE_NAMES = {"libero", "robosuite", "mujoco", "mujoco_py"}
TEXT_SUFFIXES = {".py", ".pyx", ".pxd", ".json", ".yaml", ".yml", ".toml", ".cfg"}
REQUIRED = {
    "libero_env_wrapper": ("libero/envs/env_wrapper.py",),
    "robosuite_base": ("robosuite/environments/base.py",),
    "robosuite_observables": ("robosuite/utils/observables.py",),
    "robosuite_binding_utils": ("robosuite/utils/binding_utils.py",),
    "robosuite_control_utils": ("robosuite/utils/control_utils.py",),
}


class ExtractionError(RuntimeError):
    pass


def need(condition, code):
    if not condition:
        raise ExtractionError(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fid(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def json_read(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, "duplicate_JSON_key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ExtractionError("nonfinite_JSON")))


def relative(name):
    need(isinstance(name, str) and name and "\x00" not in name and "\\" not in name,
         "unsafe_relative_path")
    path = PurePosixPath(name)
    need(not path.is_absolute() and str(path) == name and
         all(p not in ("", ".", "..") for p in name.split("/")), "unsafe_relative_path")
    return path


def valid_ref(ref):
    need(type(ref["bytes"]) is int and 0 <= ref["bytes"] <= 1024 * MIB and
         isinstance(ref["sha256"], str) and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
         "invalid_byte_SHA_reference")


def regular(path):
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "regular_input_required")
    return info


def signature(info):
    return tuple(getattr(info, k) for k in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink"))


def hash_file(path, ref, check):
    before = regular(path)
    need(before.st_size == ref["bytes"], "input_size")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as src:
        while block := src.read(MIB):
            check()
            total += len(block)
            digest.update(block)
    need({"bytes": total, "sha256": digest.hexdigest()} ==
         {k: ref[k] for k in ("bytes", "sha256")}, "input_SHA")
    need(signature(before) == signature(regular(path)), "input_changed")


class Parts(io.RawIOBase):
    """One read pass over parts, verifying part and concatenated archive digests."""
    def __init__(self, stage, parts, archive, check):
        self.stage, self.parts, self.archive, self.check = stage, parts, archive, check
        self.index = 0
        self.stream = None
        self.part_digest = None
        self.part_count = 0
        self.total = 0
        self.digest = hashlib.sha256()
        self.before = None
        self.verified_parts = 0

    def readable(self):
        return True

    def _open(self):
        if self.index == len(self.parts):
            return False
        part = self.parts[self.index]
        need(part["offset"] == self.total, "part_offset")
        path = self.stage / part["stage_path"]
        self.before = signature(regular(path))
        need(path.stat().st_size == part["bytes"], "part_size")
        self.stream = path.open("rb")
        self.part_digest, self.part_count = hashlib.sha256(), 0
        return True

    def _finish_part(self):
        part = self.parts[self.index]
        need({"bytes": self.part_count, "sha256": self.part_digest.hexdigest()} ==
             {k: part[k] for k in ("bytes", "sha256")}, "part_SHA")
        path = self.stage / part["stage_path"]
        need(self.before == signature(regular(path)), "part_changed")
        self.stream.close()
        self.stream = None
        self.index += 1
        self.verified_parts += 1

    def read(self, size=-1):
        need(type(size) is int and 0 <= size <= 16 * MIB, "bounded_stream_read")
        self.check()
        if size == 0:
            return b""
        chunks = []
        remaining = size
        while remaining:
            if self.stream is None and not self._open():
                break
            block = self.stream.read(remaining)
            if not block:
                self._finish_part()
                continue
            self.check()
            self.part_count += len(block)
            self.total += len(block)
            need(self.total <= self.archive["bytes"], "archive_overflow")
            self.part_digest.update(block)
            self.digest.update(block)
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def finish(self):
        # tarfile may stop after two zero blocks; drain padding before archive SHA.
        while self.read(MIB):
            pass
        need({"bytes": self.total, "sha256": self.digest.hexdigest()} == self.archive,
             "whole_archive_SHA")
        need(self.verified_parts == len(self.parts), "all_parts_verified")

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        super().close()


def selected(path):
    parts = relative(path).parts
    lower = [p.lower() for p in parts]
    name = parts[-1]
    if any(p in PACKAGE_NAMES for p in lower) and PurePosixPath(path).suffix.lower() in TEXT_SUFFIXES:
        # Package assets may contain JSON/YAML configuration but binary arrays are never decoded.
        return "package_text_source_or_config"
    if name.startswith("ADV_VLA_ENVIRONMENT") and name.endswith(".json"):
        return "pinned_environment_plan"
    if path in {"environment/python/_adv_vla_plus_environment_v1.py",
                "environment/ENVIRONMENT_PREPARATION-v1.json",
                "reports/ACTUAL_OPERATOR_IMPORTS-v1.json",
                "reports/RUNTIME_FRAMEWORK_ASSETS-v1.json"}:
        return "pinned_runtime_provenance"
    if name in {"METADATA", "PKG-INFO", "WHEEL", "top_level.txt"} and any(
        p.lower().startswith(("robosuite", "libero", "mujoco")) and
        p.endswith((".dist-info", ".egg-info")) for p in parts[:-1]
    ):
        return "package_version_metadata"
    return None


def parse_manifest(stage, archive, refs, check):
    rows, names = [], set()
    for name in archive["manifest_paths"]:
        ref = refs[name]
        need(ref["bytes"] <= 32 * MIB, "bounded_file_manifest")
        hash_file(stage / name, ref, check)
        with (stage / name).open("rb") as stream:
            for line in stream:
                check()
                need(0 < len(line) <= 65536, "bounded_manifest_row")
                row = json_read(line)
                relative(row["path"])
                valid_ref(row)
                need(row["path"] not in names, "duplicate_manifest_member")
                need(type(row.get("mode")) is int and 0 <= row["mode"] <= 0o7777,
                     "manifest_mode")
                names.add(row["path"])
                rows.append(row)
    need(len(rows) == archive["members"] and
         sum(r["bytes"] for r in rows) == archive["logical_file_bytes"], "manifest_population")
    return rows


def execute(stage, inputs, check, *, synthetic=False):
    need(inputs["schema"] == "paper2-cpu-replay-source-closure-inputs-v1", "input_schema")
    refs = {r["path"]: r for r in inputs["input_objects"]}
    need(len(refs) == len(inputs["input_objects"]) == inputs["input_object_count"] and
         len({r["uri"] for r in refs.values()}) == len(refs), "unique_input_roster")
    need(sum(r["bytes"] for r in refs.values()) == inputs["input_body_bytes"] <= 1024 * MIB,
         "bounded_input_total")
    need(stage.is_dir() and not stage.is_symlink(), "regular_stage_directory")
    need({p.name for p in stage.iterdir()} == set(refs), "exact_stage_roster")
    for name, ref in refs.items():
        need(len(relative(name).parts) == 1, "flat_stage_input")
        valid_ref(ref)
        need(regular(stage / name).st_size == ref["bytes"], "stage_input_size")
    archives = inputs["archives"]
    need(len({a["name"] for a in archives}) == len(archives), "unique_archive_names")
    used = []
    for archive in archives:
        need(len(relative(archive["name"]).parts) == 1, "flat_archive_label")
        used += archive["manifest_paths"] + [p["stage_path"] for p in archive["parts"]]
        offset = 0
        for p in archive["parts"]:
            need(p["offset"] == offset and all(p[k] == refs[p["stage_path"]][k]
                 for k in ("bytes", "sha256")), "part_roster_identity")
            offset += p["bytes"]
        need(offset == archive["archive"]["bytes"], "archive_size_from_parts")
    need(len(used) == len(refs) and set(used) == set(refs), "all_inputs_used_once")
    if not synthetic:
        need(len(refs) == 15 and inputs["input_body_bytes"] == 786530837 and
             len(archives) == 2 and sum(a["members"] for a in archives) == 30498,
             "fixed_real_runtime_population")
    bundle = io.BytesIO()
    index, proofs = [], []
    selected_bytes = 0
    all_names = set()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for archive in archives:
            check()
            rows = parse_manifest(stage, archive, refs, check)
            stream = Parts(stage, archive["parts"], archive["archive"], check)
            count = 0
            try:
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    for member in tar:
                        check()
                        need(count < len(rows), "extra_tar_member")
                        row = rows[count]
                        relative(member.name)
                        need(member.isfile() and not member.issparse() and
                             member.name == row["path"] and member.size == row["bytes"],
                             "actual_tar_member_header")
                        need(member.mode == row["mode"], "actual_tar_member_mode")
                        reason = selected(member.name)
                        if reason:
                            need(member.size <= 2 * MIB, "selected_member_size_cap")
                        digest, length = hashlib.sha256(), 0
                        captured = io.BytesIO() if reason else None
                        src = tar.extractfile(member)
                        need(src is not None, "regular_member_reader")
                        with src:
                            while block := src.read(MIB):
                                check()
                                digest.update(block)
                                length += len(block)
                                if captured is not None:
                                    captured.write(block)
                        need({"bytes": length, "sha256": digest.hexdigest()} ==
                             {k: row[k] for k in ("bytes", "sha256")}, "member_SHA")
                        if captured is not None:
                            raw = captured.getvalue()
                            need(b"\x00" not in raw, "selected_binary_NUL")
                            try:
                                raw.decode("utf-8")
                            except UnicodeDecodeError:
                                raise ExtractionError("selected_not_UTF8") from None
                            selected_bytes += len(raw)
                            need(selected_bytes <= 16 * MIB, "selected_total_text_cap")
                            name = archive["name"] + "/" + member.name
                            need(name not in all_names, "duplicate_output_member")
                            all_names.add(name)
                            info = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
                            info.compress_type = zipfile.ZIP_DEFLATED
                            info.external_attr = 0o100644 << 16
                            output.writestr(info, raw)
                            need(bundle.tell() <= 4 * MIB, "source_bundle_cap")
                            index.append({"archive": archive["name"], "original_path": member.name,
                                          "bundle_path": name, "selection": reason,
                                          "source_domain": "capsule_member_manifest",
                                          **fid(raw), "original_mode": member.mode})
                        count += 1
                need(count == len(rows), "missing_tar_member")
                stream.finish()
                proofs.append({"name": archive["name"], "archive": archive["archive"],
                               "parts_verified": stream.verified_parts, "members_verified": count,
                               "logical_file_bytes_verified": archive["logical_file_bytes"],
                               "all_part_archive_member_SHA_verified": True,
                               "files_materialized_outside_source_bundle": 0})
            finally:
                stream.close()
    raw = bundle.getvalue()
    need(len(raw) <= 4 * MIB and index, "bounded_nonempty_source_bundle")
    required = {key: [r["bundle_path"] for r in index if any(
        r["original_path"].endswith(suffix) for suffix in suffixes)]
        for key, suffixes in REQUIRED.items()}
    missing = [key for key, matches in required.items() if not matches]
    # A missing pinned interface must stop production; never silently replace it.
    # Keep verified archive sources even when interfaces belong to the base image.
    counts = {"archives": len(archives), "input_objects": len(refs)}
    summary = {"members_verified": sum(p["members_verified"] for p in proofs),
               "selected_source_files": len(index), "selected_source_bytes": selected_bytes,
               "source_bundle": fid(raw), "required_source_candidates": required,
               "multiple_source_candidates": [k for k, v in required.items() if len(v) > 1],
               "missing_source_interfaces": missing,
               "archive_proofs": proofs}
    return raw, index, summary, counts


def no_link_ancestors(path):
    need(path.is_absolute() and path == Path(os.path.normpath(str(path))), "canonical_image_path")
    for ancestor in reversed((path, *path.parents)):
        info = ancestor.lstat()
        need(not stat.S_ISLNK(info.st_mode), "image_symlink_ancestor")
    need(path.is_dir(), "image_directory_required")


def append_image_sources(source, index, summary, provenance, check, *, synthetic=False):
    """Read an explicit historical image path; do not import or discover modules."""
    expected_site = "/opt/capsule/.venv/lib/python3.12/site-packages"
    need(provenance["package"] == "robosuite" and provenance["package_version"] == "1.4.1",
         "fixed_image_package_version")
    need(provenance["historical_module"] ==
         {"imported": True, "origin": expected_site + "/robosuite/__init__.py"},
         "historical_image_module_origin")
    if not synthetic:
        need(provenance["fixed_site"] == expected_site and
             provenance["image_sha256"] ==
             "8a0b1fb7132dcef3b799acb883cc1b6e6e08487eb220a580b7051cb5db4b2fb4",
             "fixed_immutable_image_location")
    site = Path(provenance["fixed_site"])
    no_link_ancestors(site)
    package = site / "robosuite"
    no_link_ancestors(package)
    # A single direct-level enumeration is bounded to the recorded site.
    distributions = []
    for number, entry in enumerate(os.scandir(site), 1):
        check()
        need(number <= 10000, "image_site_entry_cap")
        if entry.name.lower().startswith("robosuite-") and entry.name.endswith(".dist-info"):
            need(not entry.is_symlink() and entry.is_dir(follow_symlinks=False),
                 "image_metadata_regular_directory")
            distributions.append(Path(entry.path))
    need(len(distributions) == 1, "exact_image_distribution")
    distribution = distributions[0]
    no_link_ancestors(distribution)
    metadata_path = distribution / "METADATA"
    need(regular(metadata_path).st_size <= 2 * MIB, "image_metadata_size")
    metadata = metadata_path.read_bytes()
    text = metadata.decode("utf-8")
    versions = [line[9:] for line in text.splitlines() if line.startswith("Version: ")]
    names = [line[6:] for line in text.splitlines() if line.startswith("Name: ")]
    need(versions == ["1.4.1"] and len(names) == 1 and names[0].lower() == "robosuite",
         "image_metadata_version")
    init_path = package / "__init__.py"
    init_info = regular(init_path)
    need(init_info.st_size <= 2 * MIB, "image_init_size")
    init_raw = init_path.read_bytes()
    need(signature(init_info) == signature(regular(init_path)), "image_init_changed")
    tree = ast.parse(init_raw.decode("utf-8"), filename="image/robosuite/__init__.py")
    init_versions = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            need(isinstance(node.value, ast.Constant) and isinstance(node.value.value, str),
                 "image_init_version_not_literal")
            init_versions.append(node.value.value)
    need(init_versions == versions, "image_init_metadata_version_mismatch")
    image_paths, visited = [], 0
    pending = [package]
    while pending:
        folder = pending.pop()
        no_link_ancestors(folder)
        for entry in sorted(os.scandir(folder), key=lambda e: e.name):
            check()
            visited += 1
            need(visited <= 50000, "image_package_entry_cap")
            need(not entry.is_symlink(), "image_package_symlink")
            if entry.is_dir(follow_symlinks=False):
                pending.append(Path(entry.path))
            else:
                need(entry.is_file(follow_symlinks=False), "image_special_file")
                path = Path(entry.path)
                if path.suffix.lower() in TEXT_SUFFIXES:
                    image_paths.append((path, "image_package_text_source_or_config"))
    for name in provenance["allowed_metadata_basenames"]:
        relative(name)
        need("/" not in name and name in {"METADATA", "WHEEL", "RECORD", "top_level.txt"},
             "image_metadata_allowlist")
        path = distribution / name
        if path.exists():
            image_paths.append((path, "image_package_version_metadata"))
    need(image_paths and package / "__init__.py" in [p for p, _ in image_paths],
         "image_package_init_required")
    result_index = list(index)
    result_summary = dict(summary)
    raw_output = io.BytesIO(source)
    selected_bytes = summary["selected_source_bytes"]
    seen = {r["bundle_path"] for r in index}
    with zipfile.ZipFile(raw_output, "a", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path, reason in sorted(image_paths):
            check()
            before = regular(path)
            need(before.st_size <= 2 * MIB, "image_source_member_size")
            raw = path.read_bytes()
            need(signature(before) == signature(regular(path)), "image_source_changed")
            need(b"\x00" not in raw, "image_source_binary_NUL")
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                raise ExtractionError("image_source_not_UTF8") from None
            selected_bytes += len(raw)
            need(selected_bytes <= 16 * MIB, "selected_total_text_cap")
            name = "image/" + str(path.relative_to(site))
            need(name not in seen, "image_source_name_collision")
            seen.add(name)
            info = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, raw)
            need(raw_output.tell() <= 4 * MIB, "source_bundle_cap")
            result_index.append({
                "source_domain": "immutable_image_digest",
                "image_sha256": provenance["image_sha256"], "site": str(site),
                "original_path": str(path), "bundle_path": name, "selection": reason,
                "package_version": "1.4.1", **fid(raw),
                "not_in_capsule_member_manifest": True})
    blob = raw_output.getvalue()
    need(len(blob) <= 4 * MIB, "source_bundle_cap")
    required = {key: [r["bundle_path"] for r in result_index if any(
        r["original_path"].endswith(suffix) for suffix in suffixes)]
        for key, suffixes in REQUIRED.items()}
    result_summary.update(
        selected_source_files=len(result_index), selected_source_bytes=selected_bytes,
        source_bundle=fid(blob), required_source_candidates=required,
        missing_source_interfaces=[k for k, v in required.items() if not v],
        multiple_source_candidates=[k for k, v in required.items() if len(v) > 1],
        image_source_proof={
            "source_domain": "immutable_image_digest",
            "image_sha256": provenance["image_sha256"],
            "image_identity_proof_owner": "root_transport_provider_actual_image_binding",
            "self_observed_image_digest": False,
            "fixed_site": str(site), "package": "robosuite", "version": versions[0],
            "metadata": fid(metadata), "package_entries_inspected": visited,
            "initializer": fid(init_raw), "initializer_literal_version": init_versions[0],
            "selected_image_files": len(image_paths),
            "historical_module_origin": provenance["historical_module"],
            "package_modules_imported": 0, "find_spec_calls": 0,
            "old_per_file_manifest_equivalence_claimed": False,
            "runtime_import_precedence_or_behavior_qualified": False,
        })
    return blob, result_index, result_summary


def source_check(expected_sha):
    raw = (HERE / "SOURCE-MANIFEST.json").read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "source_manifest_SHA")
    manifest = json_read(raw)
    need(manifest["schema"] == "paper2-cpu-replay-source-closure-source-v2", "source_schema")
    rows = manifest["files"]
    need(len(rows) == len(SOURCE_FILES) and {r["path"] for r in rows} == set(SOURCE_FILES),
         "exact_source_roster")
    for row in rows:
        p = HERE / row["path"]
        need(regular(p).st_size == row["bytes"] and fid(p.read_bytes()) ==
             {k: row[k] for k in ("bytes", "sha256")}, "source_file_SHA")
    return fid(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    need(sys.platform.startswith("linux"), "real_source_archives_cloud_Linux_only")
    need(not args.out.exists() and not args.out.is_symlink(), "new_output_directory")
    args.out.mkdir()
    start = time.monotonic()
    state = {}
    def check():
        need(time.time() < args.deadline_epoch, "science_absolute_deadline")
    def persist(status, error=None):
        archive_complete = bool(state.get("archive_complete"))
        desired_complete = archive_complete and not state["summary"].get("missing_source_interfaces") and error is None
        result = {
            "schema": "paper2-cpu-replay-source-closure-result-v2", "status": status,
            "input_qualified": archive_complete,
            "analysis_complete": status == "CPU_REPLAY_SOURCE_CLOSURE_COMPLETE",
            "desired_source_interfaces_complete": bool(desired_complete),
            "source_only_image_text": True, "synthetic": False,
            "source_manifest": state.get("manifest_ref"),
            "input_manifest": fid(state["input_raw"]) if "input_raw" in state else None,
            "input_objects_verified": len(state["inputs"]["input_objects"]) if archive_complete else 0,
            "input_body_bytes": state["inputs"]["input_body_bytes"] if archive_complete else 0,
            "counts": state.get("counts", {}), "summary": state.get("summary"),
            "elapsed_seconds": time.monotonic() - start,
            "new_model_calls": 0, "new_env_steps": 0, "environment_constructions": 0,
            "runtime_imports": 0, "new_fits": 0, "training_updates": 0,
            "GPU_count": 0, "JPFS_new_bytes": 0, "new_scientific_behavior_result": False,
            "scope": "Exact archive and separately image-bound source extraction; no environment or replay qualification.",
        }
        if error is not None:
            result["error_type"] = type(error).__name__
            result["code"] = str(error) if isinstance(error, ExtractionError) else "source_extraction_error"
        values = {RESULT: encoded(result)}
        if archive_complete:
            qa = {"schema": "paper2-cpu-replay-source-closure-QA-v2",
                  "status": "SOURCE_INTERFACES_PRESENT" if desired_complete else "VERIFIED_ARCHIVE_SOURCE_RETAINED_INTERFACES_INCOMPLETE",
                  **state["summary"], "source_modules_executed": 0,
                  "environment_qualified": False, "native_library_binaries_exported": False,
                  "source_presence_does_not_resolve_runtime_import_precedence": True,
                  "full_runtime_call_graph_verified": False,
                  "desired_source_interfaces_complete": bool(desired_complete),
                  "old_full_replay_failure_unchanged": True}
            values.update({"SOURCE-BUNDLE.zip": state["source"],
                           "SOURCE-INDEX.json": encoded({"schema": "paper2-cpu-replay-source-index-v2",
                                                        "summary": state["summary"], "files": state["index"]}),
                           "QA.json": encoded(qa), "PROTOCOL.json": state["protocol_raw"]})
        need(sum(len(v) for v in values.values()) <= 8 * MIB, "output_total_cap")
        for name, data in values.items():
            need(len(data) <= (4 if name.endswith(".zip") else 2) * MIB, "output_file_cap")
            with (args.out / name).open("xb") as stream:
                stream.write(data)
        return result
    try:
        check()
        state["manifest_ref"] = source_check(args.source_manifest_sha256)
        state["input_raw"] = (HERE / "INPUTS.json").read_bytes()
        state["inputs"] = json_read(state["input_raw"])
        state["protocol_raw"] = (HERE / "PROTOCOL.json").read_bytes()
        source, index, summary, counts = execute(args.stage, state["inputs"], check)
        state.update(source=source, index=index, summary=summary, counts=counts, archive_complete=True)
        provenance = json_read((HERE / "IMAGE-SOURCE-PROVENANCE.json").read_bytes())
        source, index, summary = append_image_sources(source, index, summary, provenance, check)
        state.update(source=source, index=index, summary=summary)
        complete = not summary["missing_source_interfaces"]
        status = "CPU_REPLAY_SOURCE_CLOSURE_COMPLETE" if complete else "CPU_REPLAY_SOURCE_CLOSURE_INCOMPLETE"
        result = persist(status)
        print(encoded({"status": result["status"], "counts": counts,
                       "selected_source_files": summary["selected_source_files"]}).decode())
        if not complete:
            raise SystemExit(2)
    except Exception as exc:
        # A verified archive-only source bundle survives an image-source failure.
        persist("CPU_REPLAY_SOURCE_CLOSURE_INCOMPLETE" if state.get("archive_complete") else "FAILED", exc)
        raise


if __name__ == "__main__":
    main()
