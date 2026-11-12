"""Synthetic codec/identity tests only. No real arrays, network, or cloud."""
from pathlib import Path
import json
import importlib.util
import shutil
import tempfile
import time
import unittest
import zipfile
from unittest import mock
import numpy as np
import audit as A


def writer():
    path = Path(__file__).resolve().parent / 'writer_reference/ledger_original.py'
    raw = path.read_bytes()
    if len(raw) != 7943 or A.sha(raw) != '4a99a3edfeee6c6e7069422d5e39215b8c6069bef0dc827df4bf70baa0a521ea':
        raise AssertionError('original_parent_writer_source_SHA')
    spec = importlib.util.spec_from_file_location('_original_parent_ledger', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit_initial(folder, name, selection, value):
    """Exercise the byte-pinned historical writer, never hand-build valid schema."""
    module = writer()
    local = folder / ('writer-' + name)
    ledger = module.Ledger(local, limits={'full': 1, 'suffix': 10, 'prefill': 1, 'bytes': 1024**2},
                           deadline_epoch=time.time() + 60, plan={'synthetic': True},
                           temporary_limit_bytes=1024**2, inflight_limit_bytes=256*1024)
    ledger.emit('ACTUAL_ROOT_INITIAL_VECTOR', selection, {'initial_vector': value})
    row = ledger.members[-1]
    target = folder / name
    shutil.copyfile(local / row['path'], target)
    return target


def mutate_npz(path, transform):
    """Fault injection into a writer-generated archive; not a valid-fixture codec."""
    with np.load(path, allow_pickle=False) as archive:
        stored = {key: archive[key].copy() for key in archive.files}
    tree = json.loads(stored['__schema__'].tobytes())
    transform(tree, stored)
    stored['__schema__'] = np.frombuffer(A.canonical(tree), np.uint8)
    np.savez_compressed(path, **stored)


def rebind(ref, path):
    raw = path.read_bytes()
    return {**ref, 'bytes': len(raw), 'sha256': A.sha(raw)}


def fixture(folder, duplicate=False):
    refs = []
    for number, root in enumerate(A.ROOTS):
        task, rank = divmod(number, 2)
        a = np.arange(123 if task == 0 else 84, dtype=np.float64) + (0 if duplicate else rank) * .125
        library = {"path": f"synthetic-task-{task}.pruned_init", "bytes": 1000, "sha256": str(task + 1) * 64}
        selection = {"root_trajectory_id": root, "unique_vector_rank": rank, "original_library_index": rank,
                     "native_initial_vector": A.typed_identity(a), "source_library": library,
                     "original_init0_proof": {"synthetic": True},
                     "rule": "first_two_typed_byte_distinct_vectors_in_original_library_order",
                     "old_optics_case_is_not_current_state": True}
        expected = {"source_run_id": "synthetic-" + root, "task_component_id": f"task-{task}",
                    "parent_task_id": f"parent-{task}", **{k: selection[k] for k in
                    ("unique_vector_rank", "original_library_index", "native_initial_vector", "source_library")}}
        summary = {"root_id": root, "run_id": expected["source_run_id"], "root_complete": True,
                   "worker_result": {"actual_initial_selection": selection}}
        for kind, suffix in [("parent_summary", "json"), ("initial_vector", "npz")]:
            name = root + "." + suffix
            path = folder / name
            if suffix == "json":
                path.write_bytes(A.canonical(summary))
            else:
                emit_initial(folder, name, selection, a)
            raw = path.read_bytes()
            refs.append({"kind": kind, "root_id": root, "path": name, "uri": "synthetic:" + name,
                         "bytes": len(raw), "sha256": A.sha(raw), "expected": expected})
    return refs


class InitialAuditTests(unittest.TestCase):
    def test_original_writer_exact_roster_and_values_descriptor(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); path = p / refs[1]['path']
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(set(archive.namelist()), {'__schema__.npy', 'root/values/initial_vector.npy'})
            with np.load(path, allow_pickle=False) as archive:
                tree = json.loads(archive['__schema__'].tobytes())
            self.assertEqual(set(tree), {'metadata', 'values'})
            self.assertEqual(set(tree['values']['initial_vector']), {'array', 'shape', 'logical_dtype', 'sha256'})
            self.assertEqual(tree['values']['initial_vector']['array'], 'root/values/initial_vector')
            self.assertEqual(A.validate(p, refs, synthetic=True)['distinct_initialization_clusters'], 4)

    def test_full_support_distinct_payloads_and_no_IID_claim(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); calls = []
            result = A.validate(p, refs, lambda: calls.append(1), synthetic=True)
            self.assertEqual(result["distinct_initialization_clusters"], 4)
            self.assertFalse(result["statistical_independence_claim"])
            self.assertFalse(result["array_payloads_returned"])
            self.assertGreaterEqual(len(calls), 16)

    def test_whole_file_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); path = p / refs[1]["path"]
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(ValueError, "root_input_size"):
                A.validate(p, refs, synthetic=True)

    def test_identical_vectors_do_not_create_new_clusters(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p, duplicate=True)
            with self.assertRaisesRegex(ValueError, "two_actual_distinct"):
                A.validate(p, refs, synthetic=True)

    def test_missing_root_and_cross_root_expected_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p)
            with self.assertRaisesRegex(ValueError, "exact_four_root"):
                A.validate(p, refs[:-1], synthetic=True)
            refs[1] = {**refs[1], "expected": refs[3]["expected"]}
            with self.assertRaisesRegex(ValueError, "same_expected_root_source"):
                A.validate(p, refs, synthetic=True)

    def test_descriptor_and_typed_identity_both_required(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); ref = refs[1]; path = p / ref["path"]
            mutate_npz(path, lambda tree, stored: tree['values']['initial_vector'].update(sha256='0'*64))
            refs[1] = rebind(ref, path)
            with self.assertRaisesRegex(ValueError, "initial_descriptor_payload_SHA"):
                A.validate(p, refs, synthetic=True)

    def test_rebound_whole_file_and_descriptor_cannot_bypass_typed_raw_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); path = p / refs[1]['path']
            def modify(tree, stored):
                stored['root/values/initial_vector'][0] += 1
                tree['values']['initial_vector']['sha256'] = A.sha(stored['root/values/initial_vector'].tobytes())
            mutate_npz(path, modify); refs[1] = rebind(refs[1], path)
            with self.assertRaisesRegex(ValueError, 'initial_typed_and_raw_SHA'):
                A.validate(p, refs, synthetic=True)

    def test_later_arrays_codec_name_is_rejected_without_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); path = p / refs[1]['path']
            def modify(tree, stored):
                stored['root/arrays/initial_vector'] = stored.pop('root/values/initial_vector')
                tree['arrays'] = tree.pop('values')
            mutate_npz(path, modify); refs[1] = rebind(refs[1], path)
            with self.assertRaisesRegex(ValueError, 'exact_initial_NPZ_roster'):
                A.validate(p, refs, synthetic=True)

    def test_extra_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); path = p / refs[1]['path']
            mutate_npz(path, lambda tree, stored: stored.update(unexpected=np.zeros(1)))
            refs[1] = rebind(refs[1], path)
            with self.assertRaisesRegex(ValueError, 'exact_initial_NPZ_roster'):
                A.validate(p, refs, synthetic=True)

    def test_real_payload_guard_on_nonlinux(self):
        with mock.patch.object(A.sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "cloud_Linux_only"):
                A.validate("/does-not-exist", [])

    def test_driver_seam_filters_other_inputs_and_exposes_groups(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p)
            result = A.audit(p, {"input_objects": refs + [{"kind": "DEV695", "path": "never-read"}]}, synthetic=True)
            self.assertEqual(result["schema"], "rss-CPU-inventory-audit-v1")
            self.assertEqual(len(result["groups"]), 4)
            self.assertEqual({g["task_id"] for g in result["groups"]}, {"T00", "T01"})
            self.assertEqual(len({g["cluster_id"] for g in result["groups"]}), 4)
            self.assertEqual(result["DEV695_objects_redecoded_here"], 0)
            self.assertFalse(result["pulse_inventory"]["CONF_metadata_checked"])

    def test_CONF_metadata_is_not_DEV_or_random_pulse(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); refs = fixture(p); coverage = []
            for t in (6, 7, 8, 9):
                shard = f"CCONF-T{t:02d}"
                keys = [[f"SDRU-{shard}-V-synthetic-P{parent}", q] for parent in ("C", "T") for q in (11, 16, 22)]
                value = {"keys": keys, "roles": ["baseline_none_1", "shared_pulse", "shared_persistent"],
                         "adapter_axes": "role/action_slot/7command_dim", "first5_proposed_not_issued": True}
                name = shard + ".json"; raw = A.canonical(value); (p / name).write_bytes(raw)
                refs.append({"kind": "conf_pvp_axes", "expected_shard": shard, "path": name,
                             "bytes": len(raw), "sha256": A.sha(raw), "uri": "synthetic:" + name})
                coverage.extend({"root_id": r, "query_index": q, "status": "COMPLETE",
                                 "same_j5_and_x6": True, "same_j6_L1_L9_QP": True} for r, q in keys)
            value = {"schema": "PVP-existing-Gstar-CPU-readout-v1", "complete": True, "synthetic": False,
                     "planned_queries": 24, "completed_queries": 24, "errors": [], "new_fit": False,
                     "new_model_calls": 0, "new_environment_steps": 0, "coverage": coverage}
            name = "pvp-result.json"; raw = A.canonical(value); (p / name).write_bytes(raw)
            refs.append({"kind": "conf_pvp_result", "path": name, "bytes": len(raw),
                         "sha256": A.sha(raw), "uri": "synthetic:" + name})
            result = A.audit(p, {"input_objects": refs}, synthetic=True)
            self.assertTrue(result["pulse_inventory"]["CONF_metadata_checked"])
            self.assertEqual(result["pulse_inventory"]["CONF_shared_pulse_anchors"], 24)
            self.assertFalse(result["pulse_inventory"]["CONF_relabel_as_DEV_allowed"])
            self.assertEqual(result["inventory_objects_verified"], 13)


if __name__ == "__main__":
    unittest.main()
