"""Offline two-row smoke for the five-arm lifecycle.

The official LIBERO-Plus runtime is Linux-only and is intentionally not
started on the Mac.  This smoke injects a tiny environment/store seam and
checks the worker's ordering, fresh-wrapper lifecycle, and receipt fields.
It is software-contract evidence, not a behavior result.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


HERE = Path(__file__).resolve().parent
EVENTS: list[tuple] = []


class _Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def json(self, name, value):
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")

    def arrays(self, name, values, role):
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{key: np.asarray(value) for key, value in values.items()})

    def close_case(self, value):
        self.json("ANALYSIS-CASE-v1.json", value)


class _Env:
    def __init__(self, kind):
        self.kind = kind
        self.prompt = "synthetic official task"
        self.source = {"kind": kind, "official": True}
        self.closed = False
        self.env = SimpleNamespace(get_sim_state=lambda: np.array([1.0, 2.0]))
        EVENTS.append(("construct", kind))

    def close(self):
        if not self.closed:
            self.closed = True
            EVENTS.append(("close", self.kind))


def _observation_identity(obs):
    return {key: {"shape": list(np.asarray(value).shape), "dtype": np.asarray(value).dtype.str}
            for key, value in sorted(obs.items())}


def _array_identity(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "bytes": value.nbytes,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def _install_seams():
    envs = ModuleType("environments")
    envs.OfficialPlusEnvironment = lambda classification, **kwargs: _Env("variant")
    envs.BaseReferenceEnvironment = lambda classification, **kwargs: _Env("clean")

    def initialize(env, case, store, prefix, check, **kwargs):
        check()
        EVENTS.append(("initialize", env.kind, prefix))
        obs = {"robot0_eef_pos": np.zeros(3, np.float32),
               "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
               "robot0_gripper_qpos": np.zeros(2, np.float32)}
        store.arrays(prefix + "/q0-native-input.npz", obs, "smoke_q0")
        return {"env": env, "observation": obs, "physical": np.array([1.0, 2.0])}

    envs.initialize = initialize
    envs.snapshot = lambda state, store, prefix, **kwargs: {"snapshot": prefix}
    sys.modules["environments"] = envs

    prod = ModuleType("production")
    prod.Store = _Store
    sys.modules["production"] = prod

    adapter = ModuleType("adapter")
    adapter.observation_identity = _observation_identity
    adapter.array_identity = _array_identity
    sys.modules["adapter"] = adapter

    collector = ModuleType("collect")

    def run_episode(state, record, backend, store, prefix, check, **kwargs):
        check()
        EVENTS.append(("episode", prefix))
        store.arrays(prefix + "/query-0000-actions.npz",
                     {"raw_actions": np.zeros((1, 10, 32), np.float32),
                      "actions": np.zeros((1, 10, 7), np.float64)}, "smoke_command")
        store.arrays(prefix + "/terminal.npz",
                     {"robot0_eef_pos": np.zeros(3, np.float32),
                      "robot0_eef_quat": np.array([0, 0, 0, 1], np.float32),
                      "robot0_gripper_qpos": np.zeros(2, np.float32)}, "smoke_terminal")
        result = {"status": "accepted_failure", "success": False,
                  "policy_steps": 1, "queries": 1,
                  "terminal_reason": "policy_budget_exhausted"}
        store.json(prefix + "/Y-v1.json", result)
        return result

    collector.run_episode = run_episode
    sys.modules["collect"] = collector


def _load_worker():
    _install_seams()
    path = HERE / "five_arm_collect.py"
    spec = importlib.util.spec_from_file_location("five_arm_collect_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    worker = _load_worker()
    manifest = json.loads((HERE / "VARIANT-MANIFEST-v1.json").read_text())
    assert manifest["row_count"] == 268
    assert len(manifest["rows"]) == 268
    records = []
    for row in manifest["rows"][:2]:
        records.append({
            "identity": {"case_id": row["row_id"], "task": row["canonical_parent_task_candidate"]["family_id"],
                         "family": row["official_category"]},
            "official_case": {"max_policy_steps": 1},
            "reference_contract": {"base_bddl": row["clean_BDDL"]},
        })
    with tempfile.TemporaryDirectory(prefix="five-arm-smoke-") as raw:
        outputs = []
        for index, record in enumerate(records):
            result = worker.collect_case_five(
                record, classification={}, init_reader=None, plus_root=Path(raw),
                runtime_manifest={}, backend=object(), out=Path(raw) / f"case-{index:02d}",
                check=lambda: None, binding={"source": "smoke"}, run_id="smoke-v1",
                failure_location=lambda error, phase: {"phase": phase, "type": type(error).__name__},
            )
            assert result["analysis_status"] == "complete"
            assert result["arm_order"] == ["R0", "T0", "R1", "T1", "R0prime"]
            assert list(result["Y"]) == result["arm_order"]
            assert all(result["Y"][arm]["success"] is False for arm in result["arm_order"])
            outputs.append(result)
    arm_initializations = [event for event in EVENTS if event[0] == "initialize" and event[2].startswith("arms/")]
    assert len(arm_initializations) == 10
    assert [event[2].split("/")[1] for event in arm_initializations[:5]] == list(worker.ARMS)
    receipt = {"schema": "plus-five-arm-two-row-smoke-v1", "status": "PASS",
               "rows": 2, "arms_per_row": 5, "arm_order": list(worker.ARMS),
               "fresh_arm_initializations": len(arm_initializations),
               "official_runtime_executed": False,
               "blocker": "official LIBERO-Plus wrapper requires Linux and restored runtime/checkpoint; Mac smoke uses injected contract seams",
               "cloud_CreateJob_called": False}
    (HERE / "SMOKE-v1.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
