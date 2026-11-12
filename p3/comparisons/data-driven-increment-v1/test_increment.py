"""CPU-only metadata and tiny frozen-bridge fixtures; never access cloud."""

import ast
import base64
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import build
import cloud_adapter
import launch
import prepare
import pytest
import review
import root_adapter
import worker


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_real_partial270_preparation_metadata_only(tmp_path):
    closeout = prepare.CLOSEOUT
    adoption = closeout / "root/results/reheat1-v12-partial270-v1/ADOPTION-v1.json"
    result = prepare.prepare(closeout, [adoption], tmp_path / "prepared")
    assert result["status"] == "prepared_not_submitted"
    assert result["new_count"] == 270
    assert result["input_objects"] == 272
    assert result["input_bytes"] == 286_615_046
    task = json.loads((tmp_path / "prepared/INPUT-SPEC-v1.json").read_bytes())
    assert task["model_id"] == "T-reheat-s1-u30000-EMA"
    assert task["run_id"] == "jinwen-t30-eval-reheat-s1-plus-u1008-20261019-v12"
    assert (
        task["model_source"]["checkpoint"]["sha256"]
        == "5f14691cea8b362ee36949b0f2187ce18a1f8bc2baa37d300d85c298e5a06749"
    )
    adopted = json.loads(adoption.read_text())["adopted_ordinals"]
    assert sorted(x["ordinal"] for x in task["new_endpoint_identity"]) == sorted(adopted)
    assert 1277 not in adopted  # A gap is not an inferred failure or unstarted case.
    assert {x["id"] for x in task["contrasts"]} == {"reheat_s1_T30_minus_T20", "reheat_s1_T30_minus_Original"}
    assert task["readset"]["new_outcome_GETs"] == 270
    assert task["readset"]["old_OUTCOME_GETs"] == 0
    assert task["readset"]["summary_cloud_GETs"] == 0
    assert not task["old_baselines_or3626_tables_reemitted"]
    assert not any("PAIRED-GROUPS" in x["uri"] for x in task["readset"]["objects"])
    assert result["control_bytes"] < 2 * 1024**2
    prepared = json.loads(gzip.decompress((tmp_path / "prepared/NEW-ENDPOINTS-v1.json.gz").read_bytes()))
    assert sum(row["success"] for row in prepared) == 253
    assert sum(not row["success"] and row["status"] == "complete" for row in prepared) == 17


def test_only_durable_processed_set_is_subtracted_from_later_adoption(tmp_path):
    closeout = prepare.CLOSEOUT
    adoption = closeout / "root/results/reheat1-v12-partial270-v1/ADOPTION-v1.json"
    first = tmp_path / "first"
    prepare.prepare(closeout, [adoption], first)
    task = json.loads((first / "INPUT-SPEC-v1.json").read_bytes())
    result = {
        "schema": "paper3-added-endpoint-pairs-v1",
        "model_id": task["model_id"],
        "run_id": task["run_id"],
        "parent_T30_total": 3626,
        "new_endpoint_identity": task["new_endpoint_identity"][:10],
        "new_trace": {"bytes": 10, "sha256": "a" * 64},
        "new_normalized_delta": {"bytes": 20, "sha256": "b" * 64},
    }
    source = tmp_path / "synthetic-prior-result.json"
    source.write_bytes(prepare.canonical(result) + b"\n")
    ref = {"path": str(source), **prepare.identity(source.read_bytes())}
    receipt = {
        "schema": "paper3-added-endpoints-durable-receipt-v1",
        "root_reviewed": True,
        "whole_GET_SHA_verified": True,
        "source_result": ref,
        "result_publication": {
            "uri": prepare.BOS_PREFIX + "synthetic-reviewed-prior/RESULT-v1.json",
            **prepare.identity(source.read_bytes()),
            "whole_GET_SHA_verified": True,
        },
        "trace_publication": {
            "uri": prepare.BOS_PREFIX + "synthetic-reviewed-prior/NEW-PAIR-TRACE-v1.jsonl.gz",
            "bytes": 10,
            "sha256": "a" * 64,
            "whole_GET_SHA_verified": True,
        },
        "normalized_delta_publication": {
            "uri": prepare.BOS_PREFIX + "synthetic-reviewed-prior/NORMALIZED-NEW-v1.jsonl.gz",
            "bytes": 20,
            "sha256": "b" * 64,
            "whole_GET_SHA_verified": True,
        },
        "parent_T30_total": 3626,
        "scope": "SYNTHETIC ONLY",
    }
    proof = tmp_path / "synthetic-prior-durable.json"
    proof.write_text(json.dumps(receipt))
    after = prepare.prepare(closeout, [adoption], tmp_path / "after", processed=[proof])
    assert after["already_processed"] == 10
    assert after["new_count"] == 260
    assert after["input_objects"] == 262
    tampered = json.loads(proof.read_text())
    tampered["whole_GET_SHA_verified"] = False
    proof.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="prior_delta_must_be_root_reviewed"):
        prepare.prepare(closeout, [adoption], tmp_path / "rejected", processed=[proof])


def test_worker_calls_frozen_gate_and_reads_only_new_outcome_with_small_fixture(tmp_path):
    closeout = prepare.CLOSEOUT
    repair = closeout / "comparison-cpu-execution-v1/repair-v1"
    old = closeout / "comparison-v1/incremental-floor0-u1136-n474-v1"
    sys.path.insert(0, str(repair))
    recover = load("recover", repair / "recover.py")
    future = load("future_trace_overlay", repair / "future_trace_overlay.py")
    fixture_module = load("_paper3_repair_fixture", repair / "test_repair.py")
    borrowed = load("_paper3_borrowed_fixture", old / "worker.py")
    bridge = recover.bridge()
    spec, docs, store = fixture_module.fixture(bridge, tmp_path / "controls")
    model = "T-reheat-s1-u30000-EMA"
    new = next(row for row in docs["t30_endpoints"] if row["model_id"] == model and row["ordinal"] == 1)
    deps = {"files": []}
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "DEPENDENCIES-v1.json").write_bytes(borrowed.canonical(deps))
    (inputs / "NEW-ENDPOINTS-v1.json.gz").write_bytes(gzip.compress(borrowed.canonical([new]), mtime=0))
    summary_raw = store.objects[new["summary"]["uri"]]
    (inputs / "NEW-SUMMARIES-v1.json.gz").write_bytes(
        gzip.compress(borrowed.canonical({new["summary"]["uri"]: base64.b64encode(summary_raw).decode()}), mtime=0)
    )
    refs = [spec["caches"][key] for key in ("old_cache", "merged_delta")] + [new["outcome"]]
    refs = [{k: r[k] for k in ("uri", "bytes", "sha256")} for r in refs]
    task = {
        "schema": "paper3-data-driven-endpoint-increment-v1",
        "parent_T30_total": 3626,
        "GPU": 0,
        "model_calls": 0,
        "model_id": model,
        "model_source": spec["models"][model],
        "run_id": new["run_id"],
        "case_list_identity": spec["case_list_identity"],
        "new_count": 1,
        "already_processed": 0,
        "new_endpoint_identity": [{"ordinal": 1, "outcome_canonical_sha256": new["outcome_canonical_sha256"]}],
        "readset": {"objects": refs, "object_count": 3, "exact_total_bytes": sum(r["bytes"] for r in refs)},
        "contrasts": [
            r
            for r in spec["comparison"]["contrasts"]
            if r["treatment"] == model and r["control"] in ("original", "T-s1-u20000-EMA")
        ],
        "parent_snapshot": {
            "T30_total": 3626,
            "scope": "SYNTHETIC ONLY",
            "result": {"sha256": worker.PARENT_RESULT_SHA},
            "tables": [{"sha256": sha} for sha in sorted(worker.PARENT_TABLE_SHAS)],
        },
        "borrowed_worker": {"path": str(old / "worker.py"), **borrowed.file_identity(old / "worker.py")},
        "worker_source": borrowed.file_identity(worker.__file__),
        "cloud_adapter_source": borrowed.file_identity(cloud_adapter.__file__),
    }
    for key, name in (
        ("DEPENDENCIES-v1.json", "DEPENDENCIES-v1.json"),
        ("NEW-ENDPOINTS-v1.json.gz", "NEW-ENDPOINTS-v1.json.gz"),
        ("NEW-SUMMARIES-v1.json.gz", "NEW-SUMMARIES-v1.json.gz"),
    ):
        task[key] = borrowed.file_identity(inputs / name)
    (inputs / "INPUT-SPEC-v1.json").write_bytes(borrowed.canonical(task))
    original_stream = bridge.stream_cache

    def tiny_stream(path, count):
        return original_stream(path, 6 if Path(path).name == "old_cache.jsonl.gz" else 0)

    with (
        patch.object(worker.platform, "system", return_value="Linux"),
        patch.object(worker, "load_borrowed", return_value=borrowed),
        patch.object(
            borrowed,
            "dependencies",
            return_value=(
                bridge,
                recover,
                future,
                {"frozen-bridge/inputs-2648-v1/INPUT-SPEC-v1.json": tmp_path / "controls/INPUT-SPEC-v1.json"},
            ),
        ),
        patch.object(bridge, "load_controls", return_value=(spec, docs)),
        patch.object(bridge, "stream_cache", side_effect=tiny_stream),
    ):
        value = worker.run(
            inputs, tmp_path / "deps", old / "worker.py", store, tmp_path / "work", tmp_path / "out", time.time() + 30
        )
    assert value["status"] == "computed_unpublished"
    assert value["new_count"] == 1
    assert value["new_trace_rows"] == 2
    assert value["input_objects"] == 3
    assert value["new_OUTCOME_GETs"] == 1
    assert value["old_OUTCOME_GETs"] == 0
    assert set(store.calls) == {r["uri"] for r in refs}
    assert len(store.calls) == 3
    trace = [
        json.loads(x) for x in gzip.decompress((tmp_path / "out/NEW-PAIR-TRACE-v1.jsonl.gz").read_bytes()).splitlines()
    ]
    assert len(trace) == 2
    assert {x["contrast"] for x in trace} == {r["id"] for r in task["contrasts"]}
    assert all(x["diagnostic"]["reason"] != "qualified_recorded_pair" for x in trace)
    assert all(x["delta_SR"] is None for x in value["new_cohort_pairs"])
    assert all(x["qualified_pairs"] == 0 for x in value["new_cohort_pairs"])
    assert not value["old3626_tables_reemitted"]
    assert not value["old_baseline_cache_reemitted"]
    assert value["publication_status"] == "not_invoked_by_worker"


def test_admitted_adapter_publishes_only_new_science_and_delivery_with_fake_store(tmp_path, monkeypatch):
    closeout = prepare.CLOSEOUT
    adoption = closeout / "root/results/reheat1-v12-partial270-v1/ADOPTION-v1.json"
    inputs = tmp_path / "inputs"
    prepare.prepare(closeout, [adoption], inputs)
    task = json.loads((inputs / "INPUT-SPEC-v1.json").read_text())
    borrowed = load("_paper3_adapter_borrowed", closeout / "comparison-v1/incremental-floor0-u1136-n474-v1/worker.py")
    calls = []

    class FakeBudget:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def archive_phase(self):
            return self

    class FakeWriter:
        def __init__(self, prefix, deadline):
            assert prefix == task["output_prefix"]
            self.prefix = prefix

        def put(self, path, name):
            assert Path(path).name == name
            calls.append(name)
            return {"uri": self.prefix + name, **borrowed.file_identity(path)}

    def fake_module(name, _path):
        if name.endswith("exact_bos"):
            return SimpleNamespace(BOSWriter=FakeWriter)
        if name.endswith("byte_budget"):
            return SimpleNamespace(bounded_http=lambda **_: FakeBudget())
        if name.endswith("exact_store"):
            return SimpleNamespace(ExactStore=lambda *_: object())
        if name == "contract":
            return SimpleNamespace()
        raise AssertionError(name)

    def fake_worker_run(_inputs, _deps, _borrowed, _store, _work, output, _deadline):
        Path(output).mkdir()
        for name in (
            "NEW-PAIR-TRACE-v1.jsonl.gz",
            "NORMALIZED-NEW-v1.jsonl.gz",
            "INPUT-RECEIPT-v1.json",
            "RESULT-v1.json",
        ):
            (Path(output) / name).write_bytes(name.encode())
        return {"new_OUTCOME_GETs": 270, "input_bytes": 286615046}

    borrowed.dependencies = lambda *_: (
        None,
        None,
        None,
        {
            "transport/bos_stdlib.py": tmp_path / "bos.py",
            "transport/byte_budget.py": tmp_path / "budget.py",
            "transport/contract.py": tmp_path / "contract.py",
            "transport/exact_store.py": tmp_path / "store.py",
        },
    )
    borrowed.load_module = fake_module
    admission = tmp_path / "admission.json"
    now = time.time()
    admission.write_text(
        json.dumps(
            {
                "resources": {"CPU": 2, "memory_GiB": 8, "GPU": 0},
                "input_spec": borrowed.file_identity(inputs / "INPUT-SPEC-v1.json"),
                "source_package_bytes": 1000,
                "science_deadline_epoch": now + 3000,
                "archive_target_epoch": now + 3300,
                "deadline_epoch": now + 3500,
            }
        )
    )
    monkeypatch.setattr(cloud_adapter.platform, "system", lambda: "Linux")
    monkeypatch.setattr(worker, "load_borrowed", lambda *_: borrowed)
    monkeypatch.setattr(worker, "run", fake_worker_run)
    result = cloud_adapter.execute(
        inputs, tmp_path / "deps", tmp_path / "borrowed.py", tmp_path / "work", tmp_path / "out", admission
    )
    assert result["status"] == "science_and_delta_whole_GET_SHA_published"
    assert calls == [
        "NEW-PAIR-TRACE-v1.jsonl.gz",
        "NORMALIZED-NEW-v1.jsonl.gz",
        "INPUT-RECEIPT-v1.json",
        "RESULT-v1.json",
        "DELIVERY-v1.json",
    ]
    assert result["new_count"] == 270
    assert not result["old_baseline_or3626_republished"]
    delivery = json.loads((tmp_path / "out/DELIVERY-v1.json").read_text())
    assert delivery["new_count"] == 270
    assert len(delivery["snapshot_publication"]) == 4
    assert all(row["whole_GET_SHA_verified"] for row in delivery["snapshot_publication"])


def test_exact_cpu0_package_review_dry_and_no_unreviewed_create(tmp_path, capsys):
    closeout = prepare.CLOSEOUT
    adoption = closeout / "root/results/reheat1-v12-partial270-v1/ADOPTION-v1.json"
    inputs = tmp_path / "inputs"
    prepare.prepare(closeout, [adoption], inputs)
    package = tmp_path / "package"
    descriptor = build.build(inputs, package)
    checked, q = build.verify(package / "prepared-v1", package)
    assert checked == descriptor
    assert q["resources"] == {"CPU": 2, "memory_GiB": 8, "GPU": 0}
    assert q["phases"] == {"science_runtime_seconds": 3000, "archive_target_seconds": 3480, "max_runtime_seconds": 3600}
    assert q["runtime_exact_URI_count"] == 272
    assert q["runtime_exact_input_bytes"] == 286615046
    assert len(q["input_credential_resources"]) == 3
    assert q["credential_scope_wider_than_runtime_readset"]
    archive = package / "SOURCE_PACKAGE-v1.tar.gz"
    assert archive.stat().st_size < 8 * 1024**2
    ast.parse((package / "bootstrap_runtime.py").read_text())
    bootstrap = (package / "bootstrap_runtime.py").read_text()
    assert "science==start+3000" in bootstrap
    assert q["run_id"] in bootstrap
    assert q["output_prefix"] in bootstrap
    assert "paper3-Plus474-increment-CPU-request-v1" not in bootstrap
    candidate = review.review(package, package / "ROOT-REVIEW-CANDIDATE-v1.json")
    assert not candidate["root_reviewed"]
    assert candidate["input_credential_resources"] == q["input_credential_resources"]
    assert launch.launch(package, "dry") == 0
    assert "CANDIDATE_VERIFIED_NO_CLOUD" in capsys.readouterr().out
    with pytest.raises(ValueError, match="root_exclusive_science_CPU_route"):
        launch.launch(package, "execute", review=package / "ROOT-REVIEW-CANDIDATE-v1.json")
    with pytest.raises(ValueError, match="root_reviewed_exact_CPU_run"):
        launch.launch(
            package,
            "execute",
            review=package / "ROOT-REVIEW-CANDIDATE-v1.json",
            root_exclusive=True,
            no_other_route=True,
        )


def test_offline_wire_uses_exact_two_cpu_eight_gib_and_rejects_downsize(tmp_path):
    closeout = prepare.CLOSEOUT
    adoption = closeout / "root/results/reheat1-v12-partial270-v1/ADOPTION-v1.json"
    inputs = tmp_path / "inputs"
    prepare.prepare(closeout, [adoption], inputs)
    package = tmp_path / "package"
    build.build(inputs, package)
    host, descriptor, q = root_adapter.prepare(package)
    assert host.data_increment_constant_changes["create_once"] == {"4": 1, "1200": 1, "600": 1, "900": 1, "3600": 2}
    admission = sys.modules["_paper3_accepted_CPU0_admission"]
    public = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "void",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu",
    }
    env = {**public, **dict.fromkeys(host.AUTH, "synthetic-private-not-saved")}
    body = {
        "name": q["run_id"],
        "queue": q["queue_id"],
        "command": "synthetic immutable bootstrap command",
        "jobType": "PyTorchJob",
        "priority": "normal",
        "enableBccl": False,
        "faultTolerance": False,
        "retentionPeriod": "1d",
        "datasources": [],
        "labels": [
            {"key": "one_vla_run_id", "value": q["run_id"]},
            {"key": "source_sha", "value": descriptor["source_package"]["sha256"][:32]},
        ],
        "jobSpec": {
            "image": q["image"],
            "replicas": 1,
            "resources": [{"name": "cpu", "quantity": 2}, {"name": "memory", "quantity": 8}],
            "envs": [{"name": key, "value": value} for key, value in env.items()],
            "imageConfig": {"synthetic": True},
            "enableRDMA": False,
        },
    }
    target = {"run_id": q["run_id"], "resources": q["resources"], "source_package": descriptor["source_package"]}
    proof = admission.wire(body, host, target)
    assert proof["GPU_resource_entries"] == 0
    assert proof["resources"] == {"CPU": 2, "memory_GiB": 8, "GPU": 0}
    assert proof["queue_id"] == "redacted-68f1c47a0c15"
    body["jobSpec"]["resources"][1]["quantity"] = 4
    with pytest.raises(ValueError, match="no_GPU_or_extra_resource_wire"):
        admission.wire(body, host, target)


def test_reviewed_launch_routes_exact_package_to_old_host_without_create_in_test(tmp_path, monkeypatch):
    closeout = prepare.CLOSEOUT
    adoption = closeout / "root/results/reheat1-v12-partial270-v1/ADOPTION-v1.json"
    inputs = tmp_path / "inputs"
    prepare.prepare(closeout, [adoption], inputs)
    package = tmp_path / "package"
    build.build(inputs, package)
    approved = package / "ROOT-REVIEW-v1.json"
    review.review(package, approved, root_reviewed=True)
    host, descriptor, q = root_adapter.prepare(package)
    captured = []

    def fake_main():
        captured.append(sys.argv.copy())
        return 0

    host.main = fake_main
    monkeypatch.setattr(root_adapter, "prepare", lambda _: (host, descriptor, q))
    fake_registry = tmp_path / "ACCOUNT.json"
    fake_registry.write_text("SYNTHETIC ONLY")
    original_module = root_adapter.module

    def fake_module(name, path):
        if name == "_paper3_thread_scope":
            return SimpleNamespace(authority=lambda: None, ACCOUNT=fake_registry)
        return original_module(name, path)

    monkeypatch.setattr(root_adapter, "module", fake_module)
    assert launch.launch(package, "execute", review=approved, root_exclusive=True, no_other_route=True) == 0
    assert len(captured) == 1
    argv = captured[0]
    assert argv[1] == "execute"
    assert argv[argv.index("--prepared") + 1] == str(package / "prepared-v1")
    assert argv[argv.index("--review") + 1] == str(approved)
    assert argv[argv.index("--registry") + 1] == str(fake_registry)
    assert not (package / "run-v1").exists()  # fake host never mutates or Create


def test_readback_checks_new_only_whole_sha_and_never_fetches_large_trace(tmp_path):
    closeout = prepare.CLOSEOUT
    adoption = closeout / "root/results/reheat1-v12-partial270-v1/ADOPTION-v1.json"
    inputs = tmp_path / "inputs"
    prepare.prepare(closeout, [adoption], inputs)
    package = tmp_path / "package"
    build.build(inputs, package)
    host, _descriptor, q = root_adapter.prepare(package)
    result = {
        "schema": "paper3-added-endpoint-pairs-v1",
        "new_count": 270,
        "new_trace_rows": 540,
        "new_OUTCOME_GETs": 270,
        "old_OUTCOME_GETs": 0,
        "full_population_evaluation_complete": False,
        "CI": None,
        "GPU": 0,
        "model_calls": 0,
        "input_spec": q["input_spec"],
    }
    receipt = {
        "input_spec": q["input_spec"],
        "verified_objects": 272,
        "verified_bytes": 286615046,
        "new_OUTCOME_GETs": 270,
        "old_OUTCOME_GETs": 0,
    }
    raw = {"RESULT-v1.json": build.encoded(result), "INPUT-RECEIPT-v1.json": build.encoded(receipt)}
    refs = []
    for name in ("NEW-PAIR-TRACE-v1.jsonl.gz", "NORMALIZED-NEW-v1.jsonl.gz", "INPUT-RECEIPT-v1.json", "RESULT-v1.json"):
        payload = raw.get(name, b"large-BOS-object-not-read-on-Mac")
        refs.append(
            {
                "relative": name,
                "uri": q["output_prefix"] + name,
                **build.identity(payload),
                "whole_GET_SHA_verified": True,
            }
        )
    delivery = {
        "schema": "paper3-added-endpoint-delivery-v1",
        "status": "science_and_delta_whole_GET_SHA_published",
        "run_id": q["scientific_run_id"],
        "model_id": q["model_id"],
        "new_count": 270,
        "old_baseline_or3626_republished": False,
        "new_OUTCOME_GETs": 270,
        "snapshot_publication": refs,
    }
    raw["DELIVERY-v1.json"] = build.encoded(delivery)
    reads = []

    def fake_get(_client, _bucket, key, *, meter, cap, expect):
        name = Path(key).name
        assert name in raw
        reads.append(name)
        payload = raw[name]
        meter["consumed"] += len(payload)
        assert meter["consumed"] < cap
        if expect is not None:
            assert build.identity(payload) == {k: expect[k] for k in ("bytes", "sha256")}
        return payload

    host.bounded_get_function = lambda: fake_get
    backend = SimpleNamespace(sts=lambda *_args: {}, old=SimpleNamespace(_bos_client=lambda _: object()))
    state = tmp_path / "state"
    state.mkdir()
    proof = root_adapter.small_readback(host, backend, state, {}, q)
    assert proof["snapshot_cloud_durable"]
    assert reads == ["DELIVERY-v1.json", "RESULT-v1.json", "INPUT-RECEIPT-v1.json"]
    assert not any("TRACE" in name or "NORMALIZED" in name for name in reads)
    bad = dict(delivery, model_id="wrong-model")
    raw["DELIVERY-v1.json"] = build.encoded(bad)
    another = tmp_path / "other-state"
    another.mkdir()
    with pytest.raises(ValueError, match="actual_new_endpoint_delivery_scope"):
        root_adapter.small_readback(host, backend, another, {}, q)
