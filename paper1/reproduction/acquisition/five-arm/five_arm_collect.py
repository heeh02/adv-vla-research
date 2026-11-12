"""Black-box five-arm collector for the approved Plus paired protocol.

This module deliberately reuses the already qualified Plus environment and
policy wrapper. It adds no model or cloud code. The caller supplies an
immutable case record containing both selected and clean BDDL identities.
"""
import copy
from pathlib import Path
import sys

try:
    import environments as E
    import production as P
    from adapter import observation_identity, array_identity
    from collect import run_episode
except ModuleNotFoundError:
    # The cloud entrypoint puts the qualified runtime directory on PYTHONPATH.
    # Keep direct source inspection/import usable from this isolated package.
    _RUNTIME = Path(__file__).resolve().parent / "runtime-v6"
    sys.path.insert(0, str(_RUNTIME))
    import environments as E
    import production as P
    from adapter import observation_identity, array_identity
    from collect import run_episode

ARMS = ("R0", "T0", "R1", "T1", "R0prime")


def _case_record(record, noise_label):
    value = copy.deepcopy(record)
    value["noise"] = {
        "parent_family_id": record["identity"]["task"],
        "noise_label": int(noise_label),
        "protocol": "plus-shared-policy-noise-v1",
    }
    return value


def _make_base(record, classification, init_reader, plus_root, runtime_manifest,
               variant_snapshot=None):
    return E.BaseReferenceEnvironment(
        classification,
        plus_root=plus_root,
        reference=record["reference_contract"],
        init_reader=init_reader,
        runtime_manifest=runtime_manifest,
        variant_preinit_snapshot=variant_snapshot,
    )


def _make_variant(classification, init_reader):
    return E.OfficialPlusEnvironment(
        classification,
        protocol_mode="released_task_language",
        init_reader=init_reader,
    )


def collect_case_five(record, *, classification, init_reader, plus_root,
                      runtime_manifest, backend, out, check, binding, run_id,
                      failure_location):
    """Run one case in exactly the frozen five-arm order.

    The return value follows the existing case archive schema so the parent
    controller can archive it without treating this as a second experiment.
    """
    from production import Store

    store = Store(Path(out))
    result = {
        "schema": "plus-five-arm-case-result-v1",
        "identity": copy.deepcopy(record["identity"]),
        "binding": {**binding, "run_id": run_id,
                    "attempt_id": run_id + ":" + record["identity"]["case_id"]},
        "arm_order": list(ARMS),
        "Y": {},
        "analysis_status": "pending",
        "fatal": None,
    }

    phase = "case_start"
    # Keep every selected row in the denominator.  An arm-level technical
    # error is recorded and the next arm still receives a fresh wrapper.
    # Only a failure to build the common compiled snapshot is case-fatal.
    try:
        # The clean wrapper's compiled-physics gate needs a variant snapshot.
        # Build this technical snapshot before the fixed arm order; it is not
        # an inference call or an episode and is recorded separately.
        phase = "variant_compiled_snapshot"
        snapshot_env = _make_variant(classification, init_reader)
        try:
            snapshot_state = E.initialize(
                snapshot_env, record["official_case"], store,
                "technical/variant-snapshot", check,
                sensor_tap=record["identity"].get("family") == "Sensor Noise",
            )
            variant_snapshot = E.snapshot(
                snapshot_state, store, "technical/variant-snapshot/model",
                plus_root=plus_root, runtime_manifest=runtime_manifest,
            )
        finally:
            snapshot_env.close()
        for arm in ARMS:
            env = None
            phase = "initialize_" + arm
            noise = 0 if arm in ("R0", "T0", "R0prime") else 1
            is_treatment = arm in ("T0", "T1")
            arm_prefix = "arms/" + arm
            try:
                check()
                env = _make_variant(classification, init_reader) if is_treatment else _make_base(
                    record, classification, init_reader, plus_root, runtime_manifest,
                    variant_snapshot=variant_snapshot)
                state = E.initialize(
                    env, record["official_case"], store, arm_prefix + "/environment", check,
                    sensor_tap=record["identity"].get("family") == "Sensor Noise",
                )
                result.setdefault("q0", {})[arm] = {
                    "observation": observation_identity(state["observation"]),
                    "physical": array_identity(state["physical"]),
                    "noise_label": noise,
                }
                y = run_episode(state, _case_record(record, noise), backend, store,
                                arm_prefix + "/Y", check)
                # run_episode writes native action chunks and query/terminal
                # NPZs.  The receipt names those immutable artifacts and the
                # exact EEF fields needed by downstream motion readout.
                result["Y"][arm] = {
                    "status": y["status"], "success": y["success"],
                    "policy_steps": y["policy_steps"], "queries": y["queries"],
                    "terminal_reason": y["terminal_reason"],
                    "artifacts": {
                        "commands": [arm_prefix + "/Y/query-%04d-actions.npz" % q
                                     for q in range(y["queries"])],
                        "eef_observation_fields": ["robot0_eef_pos", "robot0_eef_quat",
                                                    "robot0_gripper_qpos"],
                        "terminal": arm_prefix + "/Y/terminal.npz",
                        "success_source": arm_prefix + "/Y/Y-v1.json",
                    },
                }
            except Exception as error:
                # Preserve the concrete phase and exception class, while
                # avoiding arbitrary exception text/locals in durable output.
                result["Y"][arm] = {
                    "status": "technical_missing", "success": None,
                    "policy_steps": 0, "queries": 0, "terminal_reason": None,
                    "error": {"phase": phase, "type": type(error).__name__},
                    "artifacts": {
                        "commands": [], "eef_observation_fields": [],
                        "terminal": None, "success_source": None,
                    },
                }
                result.setdefault("arm_failures", []).append({
                    "arm": arm, "phase": phase, "type": type(error).__name__})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as error:
                        result.setdefault("arm_close_failures", []).append({
                            "arm": arm, "type": type(error).__name__})
        result["analysis_status"] = "complete"
    except Exception as error:
        result["analysis_status"] = "failed"
        result["fatal"] = failure_location(error, phase)
        for arm in ARMS:
            result["Y"].setdefault(arm, {"status": "technical_missing", "success": None})
    store.json("FIVE-ARM-CASE-v1.json", result)
    store.close_case(result)
    return result


# Keep the parent entry-point contract unchanged while binding the new
# five-arm implementation explicitly.
collect_case = collect_case_five

