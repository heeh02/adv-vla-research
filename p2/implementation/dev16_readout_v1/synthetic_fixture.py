"""Artificial-only in-memory inputs. Never reads or copies real label arrays."""
import copy
import hashlib
import json

import analysis as A


def ident(value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def counter(shape, count):
    return {"shape": shape, "coordinates": count, "logical_dtype": "float64",
            "numeric_nonzero_coordinates": 0, "bitwise_changed_coordinates": 0,
            "squared_norm": 0., "maximum_absolute_difference": 0.,
            "physical_meaningful_threshold": None,
            "numeric_nonzero_is_not_physical_motion": True}


def make():
    observations = []
    for index, (root, q) in enumerate(A.KEYS):
        s = [[0.0 for _ in range(7)] for _ in range(10)]
        r = copy.deepcopy(s)
        p = copy.deepcopy(s)
        for slot in range(10):
            # All zero, source zero, target zero, equal, S bigger, R bigger,
            # and a signed position pattern deliberately occur in each fixture.
            sv = [0., 0., 2., 1., 3., 1., (-1.)**slot * (index + 1)]
            rv = [0., 2., 0., 1., 1., 3., 0.5]
            for d in range(7):
                s[slot][d] = sv[d]
                r[slot][d] = rv[d]
                p[slot][d] = sv[d] - rv[d] / 4.
        arrays = {"S": s, "R": r, "P": p}
        arms = {}
        for arm in ("S", "R", "P"):
            arms[arm] = {"adapter7_effect": [arrays[arm]],
                         "actual_E_numeric_zero": False, "Vmod_bitwise_same_as_VT": False,
                         "actual_E_squared_norm": 4., "native_compute_precision_changed": False,
                         "stage_changes": {
                             "adapter7_first5": counter([1, 5, 7], 35),
                             "adapter7_full10": counter([1, 10, 7], 70)}}
        item = {"root_id": root, "query_index": q, "arms": arms,
                "qualification_status": "ALL72_CHECKS_PASSED",
                "eligible_for_scientific_summary": True,
                "physical_motion_and_task_success": None,
                "P_minus_S_identifies_duration_or_feedback_alone": False,
                "R_delayed_repeat_performed": False,
                "actual_dose": {"shared_global_norm": 2., "random_global_norm": 2.000002,
                                "global_random_minus_shared_norm": .000002,
                                "exact_actual_dose_match": False, "redraw_or_rescale": False,
                                "valid_tokens": 5, "nonzero_valid_token_residuals": 5,
                                "maximum_absolute_valid_token_residual": .01,
                                "full_report_sha256": "a" * 64,
                                "original_edit_record": {"path": "SYNTHETIC", "bytes": 1,
                                                         "sha256": "b" * 64},
                                "full_token_arrays": "SYNTHETIC only"}}
        for name, x, y in (("S_minus_R", s, r), ("P_minus_S", p, s)):
            item[name + "_adapter7"] = [[[x[j][d] - y[j][d] for d in range(7)]
                                        for j in range(10)]]
            item[name] = counter([1, 10, 7], 70)
        observations.append(item)
    worker = {"schema": "DEV-pulse-worker-result-v1", "synthetic": True,
              "status": "DEV_PULSE_NATIVE_COMPLETE", "complete": True,
              "qualification_complete": True, "initial_qualification_complete": True,
              "source_roster_native_qualified": True,
              "planned_queries": [list(k) for k in A.KEYS],
              "observability": observations, "failures": [],
              "environment_steps": 0, "training_updates": 0, "new_statistical_fits": 0,
              "full_generations_charged": 72, "prefix_prefills_charged": 144,
              "suffix_evaluations_charged": 720, "R_delayed_repeat": False,
              "physical_meaningful_threshold": None,
              "old_G_endpoint_bitwise_parity_claim": False,
              "old_persistent_results_relabelled_as_pulse": False,
              "query_status": [{"root_id": root, "query_index": q, "planned_calls": 9,
                                "charged_calls": 9, "passed_calls": 9, "status": "COMPLETE",
                                "unexecuted_call_ids": []} for root, q in A.KEYS]}
    for key in ("charged_calls", "returned_native_calls", "completed_calls", "passed_calls"):
        worker[key] = list(A.CALL_IDS)
    terminal = {"schema": "rss-pulse-compact-terminal-v1", "result": worker,
                "charged_calls": list(A.CALL_IDS), "persisted_calls": sorted(A.CALL_IDS),
                "stopped_on_storage_failure": False, "native_graph_modified": False,
                "unexecuted_call_ids": [], "GPU_authorization_granted": False,
                "release_proven_by_storage": False}
    refs = {"TERMINAL.json": {"path": "TERMINAL.json", "uri": "synthetic://terminal",
                             **ident(terminal)}}
    launcher = {"schema": "DEV-pulse-native-launcher-result-v1", "run_id": A.RUN,
                "synthetic": True, "status": "DEV_PULSE_NATIVE_COMPLETE",
                "source_manifest_sha256": A.PRODUCER_SOURCE_SHA,
                "worker_complete": True, "worker_returned": True, "worker_started": True,
                "qualification_complete": True, "worker_status": worker["status"],
                "model_binding": {"model_bound": True, "adapter_bound": True},
                "model_loads_started": 1, "model_loads_completed": 1,
                "environment_steps": 0, "training_updates": 0, "automatic_retry": False,
                "new_scientific_grid": False, "GPU_release_confirmed": False,
                "worker_terminal": {"path": "ledger/TERMINAL.json",
                                    **{k: refs["TERMINAL.json"][k] for k in ("bytes", "sha256")}}}
    for key in ("charged_calls", "returned_native_calls", "completed_calls", "passed_calls"):
        launcher[key] = list(A.CALL_IDS)
    refs["RESULT.json"] = {"path": "RESULT.json", "uri": "synthetic://result", **ident(launcher)}
    archive = []
    for name, path in (("TERMINAL.json", "ledger/TERMINAL.json"), ("RESULT.json", "RESULT.json")):
        archive.append({"path": path, "state": "VERIFIED",
                        **{k: refs[name][k] for k in ("bytes", "sha256")},
                        "receipt": {"relative": "science/" + path,
                                    **{k: refs[name][k] for k in ("bytes", "sha256", "uri")}}})
    runtime = {"schema": "DEV-pulse-native-parent-result-v1", "run_id": A.RUN,
               "failure": None, "child": {"returncode": 0}, "automatic_retry": False,
               "science_archive": {"complete_for_observed_files": True, "errors": [],
                                   "objects": archive},
               "storage": {"JPFS_new_bytes": 0}, "provider_release_proven": False}
    refs["RUNTIME-RESULT.json"] = {"path": "RUNTIME-RESULT.json", "uri": "synthetic://runtime",
                                 **ident(runtime)}
    return terminal, launcher, runtime, refs
