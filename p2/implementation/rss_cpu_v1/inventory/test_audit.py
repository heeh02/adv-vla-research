"""Synthetic codec/identity tests only. No real arrays, network, or cloud."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock
import numpy as np
import audit as A


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
        desc = {"__array__": "root/arrays/initial_vector", "shape": list(a.shape), "logical_dtype": "float64",
                "storage_dtype": "float64", "bytes": a.nbytes, "payload_sha256": A.sha(a.tobytes())}
        tree = {"metadata": selection, "arrays": {"initial_vector": desc}}
        for kind, suffix in [("parent_summary", "json"), ("initial_vector", "npz")]:
            name = root + "." + suffix
            path = folder / name
            if suffix == "json":
                path.write_bytes(A.canonical(summary))
            else:
                np.savez_compressed(path, __schema__=np.frombuffer(A.canonical(tree), np.uint8),
                                    **{"root/arrays/initial_vector": a})
            raw = path.read_bytes()
            refs.append({"kind": kind, "root_id": root, "path": name, "uri": "synthetic:" + name,
                         "bytes": len(raw), "sha256": A.sha(raw), "expected": expected})
    return refs


class InitialAuditTests(unittest.TestCase):
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
            with np.load(path, allow_pickle=False) as z:
                tree = json.loads(z["__schema__"].tobytes()); a = z["root/arrays/initial_vector"].copy()
            tree["arrays"]["initial_vector"]["payload_sha256"] = "0" * 64
            np.savez_compressed(path, __schema__=np.frombuffer(A.canonical(tree), np.uint8),
                                **{"root/arrays/initial_vector": a})
            raw = path.read_bytes(); refs[1] = {**ref, "bytes": len(raw), "sha256": A.sha(raw)}
            with self.assertRaisesRegex(ValueError, "initial_descriptor_payload_SHA"):
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
