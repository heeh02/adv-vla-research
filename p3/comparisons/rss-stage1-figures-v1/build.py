"""Deterministic, locally verified CPU0 source/request package; no cloud calls."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import tarfile

HERE = Path(__file__).resolve().parent
CLOSEOUT = HERE.parents[1]
BOS = "redacted-24fc043a6cae"
QUEUE = "redacted-68f1c47a0c15"
PARENT_QUEUE = "redacted-6e77221ca60e"
POOL = "aihc-serverless"
RESOURCES = {"CPU": 2, "memory_GiB": 8, "GPU": 0}
PHASES = {"science_runtime_seconds": 3000, "archive_target_seconds": 3480, "max_runtime_seconds": 3600}
LIMITS = {
    "input_limit_bytes": 400 * 1024**2,
    "transfer_limit_bytes": 512 * 1024**2,
    "output_limit_bytes": 32 * 1024**2,
    "private_tmp_limit_bytes": 1024**3,
    "mac_readback_limit_bytes": 8 * 1024**2,
}
IMAGE = "onevla-model-cn-north-1.jcr.service.jdcloud.com/one-vla/adv-vla-light@sha256:8a0b1fb7132dcef3b799acb883cc1b6e6e08487eb220a580b7051cb5db4b2fb4"
OLD_CPU0 = CLOSEOUT / "comparison-v1/incremental-floor0-u1136-n474-v1/cpu-execution-v1"
OLD_BOOTSTRAP_SHA = "ed72c33929556ecc3d8411108a03239a530212bfb12d2f1094ac32d77f614cae"
OLD_BOS_SHA = "c7a4b40421081a4a7a6e1ed9e74f74b6db05e571fb10c12f35d003d25dd75d42"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def exact(path, ref):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "regular_exact_source")
    data = path.read_bytes()
    need(identity(data) == {k: ref[k] for k in ("bytes", "sha256")}, "exact_source_SHA")
    return data


def safe_name(name):
    path = PurePosixPath(name)
    need(
        not path.is_absolute()
        and str(path) == name
        and "\\" not in name
        and all(part not in ("", ".", "..") for part in name.split("/")),
        "safe_unique_source_member",
    )
    return name


def input_scope(refs):
    need(len(refs)==2 and all(r['uri'].startswith(BOS+'paper3-comparison-increment/jinwen-paper3-rss-panel-stage1-joint-20261021-v1/')for r in refs),'two_published_analysis_inputs_only')
    return sorted(r['uri'].removeprefix('bos://')for r in refs)


def request(prepared):
    prepared = Path(prepared)
    spec_path = prepared / "INPUT-SPEC-v1.json"
    task = json.loads(spec_path.read_bytes())
    need(
        task["schema"] == "paper3-data-driven-endpoint-increment-v1"
        and task["GPU"] == task["model_calls"] == 0
        and task["parent_T30_total"] == 3626,
        "exact_added_endpoint_science_scope",
    )
    for name in ("DEPENDENCIES-v1.json", "NEW-ENDPOINTS-v1.json.gz", "NEW-SUMMARIES-v1.json.gz"):
        exact(prepared / name, task[name])
    for name in ("worker.py", "cloud_adapter.py"):
        need(
            identity((HERE / name).read_bytes()) == task[name.removesuffix(".py") + "_source"],
            "prepared_science_source_sha",
        )
    refs = task["readset"]["objects"]
    need(
        len(refs) == task["readset"]["object_count"] == task["new_count"] + 2
        and sum(ref["bytes"] for ref in refs) == task["readset"]["exact_total_bytes"] <= LIMITS["input_limit_bytes"]
        and task["readset"]["old_OUTCOME_GETs"] == task["readset"]["summary_cloud_GETs"] == 0,
        "exact_bounded_runtime_readset",
    )
    sources = input_scope(refs)
    digest = hashlib.sha256(encoded(task["new_endpoint_identity"])).hexdigest()[:16]
    run = "jinwen-paper3-inc-" + digest
    need(
        task["output_prefix"].startswith(BOS + "paper3-comparison-increment/") and task["output_prefix"].endswith("/"),
        "exact_Jinwen_output_prefix",
    )
    return {
        "schema": "paper3-data-driven-CPU0-request-v1",
        "run_id": run,
        "scientific_run_id": task["run_id"],
        "model_id": task["model_id"],
        "new_count": task["new_count"],
        "parent_T30_total": 3626,
        "queue_id": QUEUE,
        "physical_queue_id": PARENT_QUEUE,
        "pool": POOL,
        "resources": RESOURCES,
        "phases": PHASES,
        "limits": LIMITS,
        **PHASES,
        **LIMITS,
        "image": IMAGE,
        "output_prefix": task["output_prefix"],
        "input_spec": identity(spec_path.read_bytes()),
        "increment_input_spec": identity(spec_path.read_bytes()),
        "frozen_dependencies": task["frozen_dependencies"],
        "borrowed_worker": {k: task["borrowed_worker"][k] for k in ("bytes", "sha256")},
        "driver_source": identity((HERE / "driver.py").read_bytes()),
        "runtime_exact_URI_count": len(refs),
        "runtime_exact_input_bytes": task["readset"]["exact_total_bytes"],
        "input_objects": len(refs),
        "input_bytes": task["readset"]["exact_total_bytes"],
        "input_credential_resources": sources,
        "input_permission": ["READ"],
        "credential_scope_wider_than_runtime_readset": True,
        "output_credential_resource": task["output_prefix"].removeprefix("bos://") + "*",
        "JPFS_new_bytes": 0,
        "new_model_calls": 0,
        "new_episodes": 0,
        "automatic_Create_retry": False,
        "full_population_evaluation_complete": False,
        "root_thread": "01a0d223-c34b-7290-83da-aeee6a8e3fdd",
    }


def members(prepared):
    prepared = Path(prepared)
    q = request(prepared)
    task = json.loads((prepared / "INPUT-SPEC-v1.json").read_bytes())
    files = {}
    for name in ("worker.py", "cloud_adapter.py", "driver.py"):
        files["worker/science/" + name] = (HERE / name).read_bytes()
    files["worker/driver.py"] = (HERE / "driver.py").read_bytes()
    for path in sorted(prepared.iterdir()):
        need(path.is_file() and not path.is_symlink(), "prepared_regular_file_only")
        files["worker/science/prepared-v1/" + path.name] = path.read_bytes()
    borrowed = Path(task["borrowed_worker"]["path"])
    files["worker/borrowed/worker.py"] = exact(borrowed, task["borrowed_worker"])
    deps = json.loads(exact(prepared / "DEPENDENCIES-v1.json", task["frozen_dependencies"]))
    for ref in deps["files"]:
        name = safe_name(ref["package_path"])
        files["worker/deps/" + name] = exact(ref["source_path"], ref)
    files["metadata/REQUEST-v1.json"] = encoded(q)
    manifest = {
        "schema": "paper3-data-driven-CPU0-source-manifest-v1",
        "run_id": q["run_id"],
        "input_spec": q["input_spec"],
        "files": [{"path": name, **identity(data)} for name, data in sorted(files.items())],
    }
    files["metadata/SOURCE_MANIFEST-v1.json"] = encoded(manifest)
    need(
        len(files) == len(set(files)) and sum(len(raw) for raw in files.values()) <= 12 * 1024**2,
        "small_unique_source_package",
    )
    return files, q


def bootstrap_source(q):
    source = (OLD_CPU0 / "bootstrap_runtime.py").read_bytes()
    need(hashlib.sha256(source).hexdigest() == OLD_BOOTSTRAP_SHA, "accepted_CPU0_bootstrap_source_SHA")
    text = source.decode()
    replacements = {
        "RUN='jinwen-paper3-floor0-plus-u1136-added474-20261020-v1'": "RUN=" + repr(q["run_id"]),
        "OUTPUT='redacted-9dff0b6ceaff'+RUN+'/'": "OUTPUT="
        + repr(q["output_prefix"]),
        "paper3-Plus474-increment-CPU-package-v1": "paper3-data-driven-CPU0-package-v1",
        "paper3-Plus474-increment-CPU-source-manifest-v1": "paper3-data-driven-CPU0-source-manifest-v1",
        "paper3-Plus474-increment-CPU-request-v1": "paper3-data-driven-CPU0-request-v1",
        "paper3-Plus474-increment-CPU-runtime-admission-v1": "paper3-data-driven-CPU0-runtime-admission-v1",
        "start+3300": "start+3000",
        "science_runtime_seconds']==3300": "science_runtime_seconds']==3000",
    }
    for before, after in replacements.items():
        need(text.count(before) == 1, "one_exact_bootstrap_site:" + before[:50])
        text = text.replace(before, after)
    need(
        "RUN=" + repr(q["run_id"]) in text and "OUTPUT=" + repr(q["output_prefix"]) in text,
        "bound_dynamic_CPU0_bootstrap",
    )
    return text.encode()


def build(prepared, output):
    output = Path(output)
    need(not output.exists(), "new_immutable_package_directory")
    files, q = members(prepared)
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "SOURCE_PACKAGE-v1.tar.gz"
    with (
        archive.open("xb") as target,
        gzip.GzipFile(fileobj=target, mode="wb", mtime=0, filename="") as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tf,
    ):
        for name, raw in sorted(files.items()):
            safe_name(name)
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o644
            info.mtime = 0
            tf.addfile(info, io.BytesIO(raw))
    descriptor = {
        "schema": "paper3-data-driven-CPU0-package-v1",
        "run_id": q["run_id"],
        "output_prefix": q["output_prefix"],
        "source_package": identity(archive.read_bytes()),
        "source_manifest": identity(files["metadata/SOURCE_MANIFEST-v1.json"]),
        "request": identity(files["metadata/REQUEST-v1.json"]),
        "input_spec": q["input_spec"],
        "unpacked_max_bytes": sum(map(len, files.values())),
        "prepared_directory": str(Path(prepared).resolve()),
    }
    need(descriptor["source_package"]["bytes"] <= 8 * 1024**2, "source_package_cap8MiB")
    (output / "REQUEST-v1.json").write_bytes(files["metadata/REQUEST-v1.json"])
    (output / "DESCRIPTOR-v1.json").write_bytes(encoded(descriptor))
    (output / "bootstrap_runtime.py").write_bytes(bootstrap_source(q))
    bos = (OLD_CPU0 / "worker/bos_stdlib.py").read_bytes()
    need(hashlib.sha256(bos).hexdigest() == OLD_BOS_SHA, "accepted_CPU0_BOSWriter_source_SHA")
    (output / "worker").mkdir()
    (output / "worker/bos_stdlib.py").write_bytes(bos)
    staged = output / "prepared-v1"
    staged.mkdir()
    for path in sorted(Path(prepared).iterdir()):
        (staged / path.name).write_bytes(path.read_bytes())
    return descriptor


def verify(prepared, output):
    output = Path(output)
    descriptor = json.loads((output / "DESCRIPTOR-v1.json").read_bytes())
    expected, q = members(prepared)
    need(
        exact(output / "SOURCE_PACKAGE-v1.tar.gz", descriptor["source_package"])
        and exact(output / "REQUEST-v1.json", descriptor["request"]) == encoded(q),
        "same_current_source_and_request",
    )
    actual = {}
    with tarfile.open(output / "SOURCE_PACKAGE-v1.tar.gz", "r:gz") as archive:
        for member in archive:
            safe_name(member.name)
            need(
                member.isfile() and not member.issparse() and member.name not in actual and member.size <= 8 * 1024**2,
                "safe_unique_archive_member",
            )
            actual[member.name] = archive.extractfile(member).read()
            need(sum(map(len, actual.values())) <= 12 * 1024**2, "bounded_archive_decode")
    need(
        actual == expected
        and descriptor["source_manifest"] == identity(actual["metadata/SOURCE_MANIFEST-v1.json"])
        and descriptor["input_spec"] == q["input_spec"]
        and descriptor["run_id"] == q["run_id"],
        "byte_exact_frozen_and_new_source_package",
    )
    need(
        (output / "bootstrap_runtime.py").read_bytes() == bootstrap_source(q)
        and hashlib.sha256((output / "worker/bos_stdlib.py").read_bytes()).hexdigest() == OLD_BOS_SHA,
        "exact_accepted_CPU0_controller_assets",
    )
    need(
        {p.name for p in (output / "prepared-v1").iterdir()} == {p.name for p in Path(prepared).iterdir()}
        and all((output / "prepared-v1" / p.name).read_bytes() == p.read_bytes() for p in Path(prepared).iterdir()),
        "staged_prepared_controls_identical",
    )
    return descriptor, q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    value, _q = verify(args.prepared, args.out) if args.verify else (build(args.prepared, args.out), None)
    print(
        json.dumps(
            {"status": "verified" if args.verify else "built_not_submitted", "descriptor": value, "cloud_calls": 0}
        )
    )


if __name__ == "__main__":
    main()
