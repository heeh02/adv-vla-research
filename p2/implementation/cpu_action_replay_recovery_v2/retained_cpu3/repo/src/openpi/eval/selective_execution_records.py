"""Local D35B admission, attempt accounting and three-level execution records.

No model, environment, network or provider is constructed here. One owner must
serialize calls to each ledger. ``append`` must durably append the supplied JSON
line before returning; an ambiguous append failure poisons the ledger. The caller
owns clock identity, actual release evidence and the account-wide resource gate.
Hash bindings use canonical JSON, not arbitrary file formatting or ETags.
Real GPU mode is the default and requires a per-attempt live allocation. Pure
CPU fixtures must opt into synthetic_cpu; every receipt retains that mode.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


NS = 1_000_000_000
# Direct human update 2026-10-12: B concurrency raised to 20; total GPU-hours
# and the approved 20-case/80-short/12-full manifest remain unchanged.
HARD_LIMITS = {"cases": 24, "arms_per_case": 4, "short": 96, "full": 60, "gpu_hours": 16, "concurrent_gpus": 20}
KINDS = frozenset(("single_call", "generation", "short", "full"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical_sha256(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CaseIdentity:
    case_id: str
    task_id: str
    variant_id: str
    init_id: str
    noise_id: str

    def __post_init__(self):
        _require(all(_text(v) for v in asdict(self).values()), "Explicit case/task/variant/init/noise identities required")


@dataclass(frozen=True)
class ExecutionIdentity:
    case: CaseIdentity
    run_id: str
    worker_id: str
    host_id: str
    device_id: str
    checkpoint_sha256: str
    input_sha256: str
    noise_sha256: str
    arm: str

    def __post_init__(self):
        _require(isinstance(self.case, CaseIdentity), "Immutable CaseIdentity required")
        _require(all(_text(getattr(self, k)) for k in ("run_id", "worker_id", "host_id", "device_id", "arm")),
                 "Explicit run/worker/host/device/arm identity required")
        _require(all(_sha(getattr(self, k)) for k in ("checkpoint_sha256", "input_sha256", "noise_sha256")),
                 "Actual checkpoint/input/noise SHA-256 required")


@dataclass(frozen=True)
class CasePlan:
    identity: CaseIdentity
    arms: tuple[str, ...]
    full_episode_selected: bool
    startup_steps: int
    full_max_policy_steps: int


@dataclass(frozen=True)
class Admission:
    protocol_sha256: str
    manifest_sha256: str
    approval_sha256: str
    approval_evidence: str
    cases: tuple[CasePlan, ...]
    limits: tuple[tuple[str, float | int], ...]


def admit_b(approval: Mapping, protocol: Mapping, manifest: Mapping) -> Admission:
    """Fail closed on proposed budgets, unilateral freezes or non-DEV cases.

    The authority record must come from a verified direct user confirmation;
    this pure validator checks its explicit scope/bindings, not human identity.
    Frozen rows require resolved variant_id/init_id/noise_id, not guessed labels
    from a proposal. Every row schedules one short attempt per ordered arm; a
    selected full-episode row schedules one full attempt per arm as well.
    Each row also freezes startup_steps=10 and full_max_policy_steps in 1..520.
    """
    _require(approval.get("kind") == "explicit_user_budget_approval" and approval.get("scope") == "D35B"
             and approval.get("approved") is True and _text(approval.get("evidence_id")),
             "B requires explicit user budget approval; proposals and research Goals are not approval")
    _require(protocol.get("status") == "frozen" and protocol.get("scope") == "D35B", "B protocol must be frozen")
    reviews = protocol.get("reviews", {})
    for owner in ("paper", "execution"):
        review = reviews.get(owner, {})
        _require(review.get("approved") is True and _text(review.get("reviewer_id")) and _text(review.get("evidence_id")),
                 f"Frozen protocol requires {owner} review")
    _require(reviews["paper"]["reviewer_id"] != reviews["execution"]["reviewer_id"], "Two independent reviewers required")
    _require(manifest.get("status") == "frozen" and manifest.get("split") == "development", "Exact DEV manifest must be frozen")
    protocol_sha, manifest_sha = canonical_sha256(protocol), canonical_sha256(manifest)
    _require(approval.get("protocol_sha256") == protocol_sha and approval.get("manifest_sha256") == manifest_sha,
             "User approval must bind this exact frozen protocol and DEV manifest")
    limits = dict(approval.get("limits", {}))
    _require(set(limits) == set(HARD_LIMITS), "All six independent B budget limits must be explicit")
    for name, maximum in HARD_LIMITS.items():
        value = limits[name]
        valid = (type(value) in (int, float) and math.isfinite(value) and value > 0) if name == "gpu_hours" else _integer(value)
        _require(valid and value <= maximum, f"Invalid or excessive independent {name} budget")
    _require(limits["cases"] > 0 and limits["arms_per_case"] > 0 and limits["concurrent_gpus"] > 0,
             "Nonzero case/arm/concurrency limits required")
    dev, evaluation = protocol.get("development_task_ids"), protocol.get("evaluation_task_ids")
    _require(isinstance(dev, list) and dev and all(_text(x) for x in dev) and len(set(dev)) == len(dev), "Explicit DEV task set required")
    _require(isinstance(evaluation, list) and all(_text(x) for x in evaluation), "Explicit evaluation task set required")
    _require(not set(dev).intersection(evaluation), "Development and evaluation tasks overlap")
    rows = manifest.get("rows")
    _require(isinstance(rows, list) and 0 < len(rows) <= limits["cases"], "Case budget exceeded or empty manifest")
    plans, identities, keys = [], set(), set()
    for row in rows:
        case = CaseIdentity(**{k: row.get(k) for k in CaseIdentity.__dataclass_fields__})
        key = (case.task_id, case.variant_id, case.init_id, case.noise_id)
        _require(case.case_id not in identities and key not in keys, "Duplicate case ID or task x variant x init x noise")
        _require(row.get("split") == "development" and case.task_id in dev, "Case is outside frozen DEV tasks")
        arms = row.get("planned_arm_order")
        _require(isinstance(arms, list) and 0 < len(arms) <= limits["arms_per_case"]
                 and all(_text(a) for a in arms) and len(set(arms)) == len(arms), "Per-case arm budget/order invalid")
        _require(row.get("short_control_steps") == 5 and type(row["short_control_steps"]) is int,
                 "B short execution is exactly the first query's five policy steps")
        _require(type(row.get("full_episode_selected")) is bool, "Explicit full-episode selection required")
        _require(type(row.get("startup_steps")) is int and row["startup_steps"] == 10,
                 "Each case must freeze exactly ten startup steps")
        _require(_integer(row.get("full_max_policy_steps"), minimum=1) and row["full_max_policy_steps"] <= 520,
                 "Each case must freeze full_max_policy_steps in 1..520")
        plans.append(CasePlan(case, tuple(arms), row["full_episode_selected"], row["startup_steps"], row["full_max_policy_steps"]))
        identities.add(case.case_id)
        keys.add(key)
    short = sum(len(p.arms) for p in plans)
    full = sum(len(p.arms) for p in plans if p.full_episode_selected)
    _require(type(manifest.get("planned_short_executions")) is int and manifest["planned_short_executions"] == short
             and short <= limits["short"], "Short-execution count/budget disagrees with exact rows")
    _require(type(manifest.get("planned_full_episodes")) is int and manifest["planned_full_episodes"] == full
             and full <= limits["full"], "Full-episode count/budget disagrees with exact rows")
    return Admission(protocol_sha, manifest_sha, canonical_sha256(approval), approval["evidence_id"], tuple(plans), tuple(limits.items()))


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    identity: ExecutionIdentity
    kind: str
    allocation_id: str | None
    execution_mode: str


class BudgetLedger:
    """Single-owner, append-before-permission ledger; no automatic retries.

    Allocations reserve their entire declared interval before admission. Their
    actual GPU time continues until verified release, including failed/unknown
    attempts and overruns. Expiry never implies provider release. A caller must
    enforce deadlines externally; this object cannot stop a remote allocation.
    """

    def __init__(self, admission: Admission, append: Callable[[str], None], *, execution_mode="real_gpu"):
        _require(isinstance(admission, Admission) and callable(append), "Validated admission and durable append callback required")
        _require(execution_mode in ("real_gpu", "synthetic_cpu"), "Explicit real_gpu or synthetic_cpu execution mode required")
        self._execution_mode = execution_mode
        self._admission, self._append = admission, append
        self._limits = dict(admission.limits)
        self._events, self._attempts, self._outcomes, self._allocations = [], {}, {}, {}
        self._workers, self._used_slots = {}, set()
        self._recorders = {}
        self._now, self._gpu_ns, self._poisoned = 0, 0, False
        self._emit({"event": "admitted", "admission": asdict(admission), "execution_mode": execution_mode,
                    "synthetic_cpu": execution_mode == "synthetic_cpu"})

    @property
    def admission(self):
        return self._admission

    @property
    def events(self):
        return tuple(self._events)

    def _emit(self, value):
        _require(not self._poisoned, "Ledger append outcome unknown; no further execution permitted")
        event = {"sequence": len(self._events), "previous_sha256": hashlib.sha256(self._events[-1].encode()).hexdigest()
                 if self._events else None, "execution_mode": self._execution_mode, **value}
        line = _json(event)
        self._events.append(line)
        try:
            self._append(line + "\n")
        except BaseException:
            self._poisoned = True
            raise

    def _advance(self, at_ns):
        _require(not self._poisoned, "Ledger append outcome unknown; no further execution permitted")
        _require(_integer(at_ns) and at_ns >= self._now, "Monotonic integer nanosecond clock required")
        self._gpu_ns += (at_ns - self._now) * sum(a["gpus"] for a in self._allocations.values() if a["released"] is None)
        self._now = at_ns

    def _resource_gate(self):
        _require(all(a["released"] is not None or self._now < a["deadline"] for a in self._allocations.values()),
                 "Allocation expired without verified release; stop admission")
        _require(self._gpu_ns < self._limits["gpu_hours"] * 3600 * NS, "GPU time budget exhausted")

    def _allocation_gate(self, allocation_id):
        self._resource_gate()
        if self._execution_mode == "real_gpu":
            allocation = self._allocations.get(allocation_id)
            _require(allocation is not None and allocation["released"] is None,
                     "Real GPU execution requires this attempt's active allocation_id")
        elif allocation_id is not None:
            allocation = self._allocations.get(allocation_id)
            _require(allocation is not None and allocation["released"] is None, "Bound synthetic allocation has been released")

    def start_allocation(self, allocation_id, gpus, *, at_ns, deadline_ns):
        self._advance(at_ns)
        self._resource_gate()
        _require(_text(allocation_id) and allocation_id not in self._allocations, "Unique allocation ID required")
        _require(_integer(gpus, minimum=1) and _integer(deadline_ns) and deadline_ns > at_ns, "Positive GPU allocation interval required")
        active = [a for a in self._allocations.values() if a["released"] is None]
        _require(sum(a["gpus"] for a in active) + gpus <= self._limits["concurrent_gpus"], "Concurrent GPU budget exceeded")
        reserved = self._gpu_ns + sum((a["deadline"] - at_ns) * a["gpus"] for a in active) + (deadline_ns - at_ns) * gpus
        _require(reserved <= self._limits["gpu_hours"] * 3600 * NS, "Allocated intervals exceed GPU time budget")
        self._allocations[allocation_id] = {"gpus": gpus, "start": at_ns, "deadline": deadline_ns, "released": None}
        self._emit({"event": "allocation_started", "allocation_id": allocation_id, **self._allocations[allocation_id]})

    def release_allocation(self, allocation_id, *, at_ns, release_evidence):
        self._advance(at_ns)  # Retain overruns; never reject recording real late release.
        allocation = self._allocations.get(allocation_id)
        _require(allocation is not None and allocation["released"] is None and _text(release_evidence), "Verified, unique release required")
        allocation["released"] = at_ns
        self._emit({"event": "allocation_released", "allocation_id": allocation_id, "at_ns": at_ns,
                    "release_evidence": release_evidence, "gpu_nanoseconds": self._gpu_ns})

    def start_attempt(self, attempt_id, identity: ExecutionIdentity, kind, *, at_ns, allocation_id=None) -> Attempt:
        self._advance(at_ns)
        self._allocation_gate(allocation_id)
        _require(_text(attempt_id) and attempt_id not in self._attempts and isinstance(identity, ExecutionIdentity), "Unique immutable attempt identity required")
        _require(kind in KINDS, "Unknown B execution layer")
        plan = next((p for p in self.admission.cases if p.identity == identity.case), None)
        _require(plan is not None and identity.arm in plan.arms, "Attempt is not in the exact frozen case/arm manifest")
        _require(kind != "full" or plan.full_episode_selected, "Full episode was not preselected")
        slot = (identity.case.case_id, identity.arm, kind)
        _require(slot not in self._used_slots, "No automatic retry or repeated slot; a new approved manifest is required")
        previous = sum(a.identity.case == identity.case and a.kind == kind for a in self._attempts.values())
        _require(identity.arm == plan.arms[previous], "Attempt arm differs from the frozen order")
        worker = (identity.run_id, identity.worker_id, identity.host_id, identity.device_id, identity.checkpoint_sha256, identity.noise_sha256)
        _require(identity.case.case_id not in self._workers or self._workers[identity.case.case_id] == worker,
                 "Paired arms must retain the same run/worker/host/device/checkpoint/noise")
        if kind in ("short", "full"):
            _require(sum(a.kind == kind for a in self._attempts.values()) < self._limits[kind], f"{kind} attempt budget exhausted")
        attempt = Attempt(attempt_id, identity, kind, allocation_id, self._execution_mode)
        self._attempts[attempt_id] = attempt
        self._workers[identity.case.case_id] = worker
        self._used_slots.add(slot)
        self._emit({"event": "attempt_charged", "at_ns": at_ns, "attempt": asdict(attempt)})
        return attempt  # Permission is returned only after durable append succeeds.

    def finish_attempt(self, attempt: Attempt, *, at_ns, status, success=None, endpoint_evidence=None):
        self._advance(at_ns)
        _require(self._attempts.get(attempt.attempt_id) == attempt and attempt.attempt_id not in self._outcomes, "Unknown or already closed attempt")
        _require(status in ("completed", "early_terminated", "technical_failure", "unknown"), "Explicit completion status required")
        recorder = self._recorders.get(attempt.attempt_id)
        if status == "completed":
            _require(recorder is not None and not recorder._unconfirmed, "Completed execution requires recorded, confirmed results")
            _require(not recorder._late_model_result, "Late model result is retained but cannot complete the attempt")
            _require(recorder._counts == recorder._returned, "Unconfirmed controls cannot complete")
            if attempt.kind == "single_call":
                _require(recorder._queries == {(0, 0)}, "Completed single call requires the fixed q0/j0 velocity32 record")
            elif attempt.kind == "generation":
                _require(set(recorder._generations) == {0} and recorder._flow_counts.get(0) == 10,
                         "Completed generation requires all ten flows and raw32/adapter7")
            elif attempt.kind == "short":
                _require(recorder._returned["startup"] == 10 and recorder._returned["policy"] == 5
                         and set(recorder._generations) == {0} and recorder._flow_counts.get(0) == 10,
                         "Completed short execution requires startup, all flows and five returned policy steps")
            else:
                _require(recorder._returned["startup"] == 10 and recorder._returned["policy"] > 0
                         and bool(recorder._generations)
                         and all(recorder._flow_counts[q] == 10 and q in recorder._generations for q in recorder._flow_counts),
                         "Incomplete generation data cannot complete a full episode")
                _require(recorder._last_return is not None and
                         (recorder._last_return.terminal or recorder._returned["policy"] == recorder._plan.full_max_policy_steps),
                         "Full episode must reach its frozen policy horizon or actual terminal return")
        if status == "early_terminated":
            _require(recorder is not None and not recorder._unconfirmed and recorder._last_return is not None
                     and recorder._last_return.terminal, "Early termination requires an actual terminal return")
        if status != "completed" or attempt.kind != "full":
            _require(success is None, "Technical failures, unknowns and short executions have success=null")
        else:
            _require(type(success) is bool and _text(endpoint_evidence), "Full development episode requires an observed endpoint")
            _require(success == recorder._last_return.observed_success and endpoint_evidence == recorder._last_return.receipt_id,
                     "Full success and endpoint must match the last actual return")
        outcome = {"event": "attempt_closed", "attempt_id": attempt.attempt_id, "at_ns": at_ns, "status": status,
                   "success": success, "endpoint_evidence": endpoint_evidence, "formal_sr_eligible": False,
                   "control_attempted": dict(recorder._counts) if recorder else {"startup": 0, "policy": 0},
                   "control_returned": dict(recorder._returned) if recorder else {"startup": 0, "policy": 0}}
        self._outcomes[attempt.attempt_id] = outcome
        self._emit(outcome)
        return dict(outcome)

    def snapshot(self, *, at_ns):
        self._advance(at_ns)
        return {"attempts": len(self._attempts), "short_attempts": sum(a.kind == "short" for a in self._attempts.values()),
                "full_attempts": sum(a.kind == "full" for a in self._attempts.values()),
                "unknown_or_open_attempts": sum(a.attempt_id not in self._outcomes or self._outcomes[a.attempt_id]["status"] == "unknown"
                                                for a in self._attempts.values()),
                "gpu_nanoseconds": self._gpu_ns, "active_gpus": sum(a["gpus"] for a in self._allocations.values() if a["released"] is None),
                "poisoned": self._poisoned, "formal_sr_eligible": False}


def tensor_record(value, shape):
    """Preserve native bytes, including BF16 and signed zero, without casting."""
    array = np.asarray(value)
    _require(array.shape == shape and not array.dtype.hasobject and np.isfinite(array).all(), "Finite tensor with exact declared axes required")
    _require(array.dtype.kind == "f" or array.dtype.name == "bfloat16", "Floating model/command tensor required")
    raw = np.ascontiguousarray(array).tobytes()
    return {"shape": list(shape), "dtype": array.dtype.name, "dtype_str": array.dtype.str,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes_hex": raw.hex()}


@dataclass(frozen=True)
class StepReturn:
    receipt_id: str
    observed_success: bool
    terminal: bool

    def __post_init__(self):
        _require(_text(self.receipt_id) and type(self.observed_success) is bool and type(self.terminal) is bool,
                 "An actual control return receipt and boolean observations are required")


class ExecutionRecords:
    """One admitted attempt. Only ``dispatch`` can create issue events.

    An issue means invocation at the actual transport/env.step boundary, not a
    confirmed physical transition. A thrown call is unconfirmed and halts this
    recorder. Startup and policy counters are separate. The callback is supplied
    by the real execution owner; this module never creates an environment.
    """

    def __init__(self, budget: BudgetLedger, attempt: Attempt, clock_ns: Callable[[], int]):
        _require(budget._attempts.get(attempt.attempt_id) == attempt and callable(clock_ns), "Charged attempt and clock required")
        _require(attempt.attempt_id not in budget._recorders, "An attempt cannot recreate its execution recorder")
        self._budget, self._attempt, self._clock = budget, attempt, clock_ns
        self._generations, self._queries, self._issued = {}, set(), set()
        self._flow_counts, self._last_times, self._forward_ids = {}, {}, set()
        self._model_started, self._last_return = False, None
        self._late_model_result = False
        self._plan = next(p for p in budget.admission.cases if p.identity == attempt.identity.case)
        self._counts, self._halted = {"startup": 0, "policy": 0}, False
        self._returned, self._unconfirmed = {"startup": 0, "policy": 0}, False
        budget._recorders[attempt.attempt_id] = self

    @property
    def attempt(self):
        return self._attempt

    def _write(self, event):
        _require(not self._halted and self.attempt.attempt_id not in self._budget._outcomes, "Closed or unconfirmed attempt cannot continue")
        now = self._clock()
        self._budget._advance(now)
        if event["event"] in ("issue", "forward_attempt_charged", "forward_authorized"):
            self._budget._allocation_gate(self.attempt.allocation_id)
        if event["event"] in ("single_call", "generation"):
            allocation = self._budget._allocations.get(self.attempt.allocation_id)
            late = (self.attempt.execution_mode == "real_gpu" and (
                allocation is None or allocation["released"] is not None or now >= allocation["deadline"]
                or self._budget._gpu_ns >= self._budget._limits["gpu_hours"] * 3600 * NS
            ))
            self._late_model_result |= late
            event = {**event, "recorded_after_execution_window": late,
                     "permits_new_execution": False}
        self._budget._emit({"attempt_id": self.attempt.attempt_id, "allocation_id": self.attempt.allocation_id,
                            "identity": asdict(self.attempt.identity), "at_ns": now, **event})

    def _model_gate(self, query_index):
        _require(_integer(query_index), "Integer query index required")
        kind = self.attempt.kind
        if kind in ("single_call", "generation", "short"):
            _require(query_index == 0, "This execution layer is restricted to query0")
        if kind in ("short", "full"):
            _require(self._returned["startup"] == self._plan.startup_steps,
                     "Model records require ten returned startup controls")
            horizon = 5 if kind == "short" else self._plan.full_max_policy_steps
            _require(self._returned["policy"] < horizon, "Frozen policy horizon already reached")
            _require(self._returned["policy"] % 5 == 0 and query_index == self._returned["policy"] // 5,
                     "Model query must follow the previous five returned policy slots")

    def authorize_forward(self, call_id, *, query_index, flow_index, purpose, charge_forward):
        """Optional pre-call seam for the root's separately frozen B1 counter.

        The external callback must persistently charge its approved call limit.
        This module deliberately invents no forward budget. A callback failure
        leaves the attempted call charged and halts this recorder; never retry.
        No actual model function is called here.
        """
        self._model_gate(query_index)
        _require(_text(call_id) and call_id not in self._forward_ids and _text(purpose) and callable(charge_forward),
                 "Unique forward ID, purpose and external budget charger required")
        expected = self._flow_counts.get(query_index, 0)
        _require(_integer(flow_index) and flow_index == expected and flow_index < 10,
                 "Forward admission must name the next unrecorded flow")
        _require(self.attempt.kind != "single_call" or flow_index == 0, "Single-call probe is fixed at q0/j0")
        self._write({"event": "forward_attempt_charged", "call_id": call_id, "query_index": query_index,
                     "flow_index": flow_index, "purpose": purpose, "automatic_retry": False,
                     "external_budget_result": "pending", "model_execution_started": False})
        self._forward_ids.add(call_id)
        self._model_started = True
        try:
            charge_forward(self.attempt, call_id, purpose)
            self._write({"event": "forward_authorized", "call_id": call_id, "model_execution_started": False})
        except BaseException:
            self._halted = True
            raise

    def single_call(self, *, query_index, flow_index, x_t32, time, velocity32):
        self._model_gate(query_index)
        key = (query_index, flow_index)
        _require(key not in self._queries, "Single-call record is immutable")
        _require(_integer(flow_index) and flow_index == self._flow_counts.get(query_index, 0) and flow_index < 10,
                 "Record each query's native flows in order 0..9")
        _require(self.attempt.kind != "single_call" or flow_index == 0, "Single-call probe is fixed at q0/j0")
        _require(query_index not in self._generations, "A completed generation cannot accept more flow records")
        native_time = np.asarray(time)
        _require(native_time.shape == () and native_time.dtype == np.dtype(np.float32) and np.isfinite(native_time),
                 "Native time must be a finite FP32 scalar; do not coerce it to Python float")
        expected_time = np.float32(1) if flow_index == 0 else np.float32(self._last_times[query_index] + np.float32(-0.1))
        _require(native_time.tobytes() == expected_time.tobytes(), "Supplied time differs from the native ten-step FP32 recurrence")
        time_record = tensor_record(time, ())
        time_record.update(weak_type=bool(getattr(time, "weak_type", False)),
                           weak_type_source="array_attribute" if hasattr(time, "weak_type") else "numpy_explicit_dtype")
        record = {"event": "single_call", "role": "model_velocity_not_issued_command", "query_index": query_index,
                  "flow_index": flow_index, "time": time_record, "x_t32": tensor_record(x_t32, (10, 32)),
                  "velocity32": tensor_record(velocity32, (10, 32))}
        self._write(record)
        self._queries.add(key)
        self._flow_counts[query_index] = flow_index + 1
        self._last_times[query_index] = native_time[()]
        self._model_started = True

    def generation(self, *, query_index, raw32, adapter7):
        self._model_gate(query_index)
        _require(self.attempt.kind != "single_call", "Single-call probe cannot record a complete generation")
        _require(_integer(query_index) and query_index not in self._generations, "Unique generation query index required")
        _require(self._flow_counts.get(query_index) == 10, "Generation requires ten ordered xt32/time/velocity32 records")
        raw, adapter = tensor_record(raw32, (10, 32)), tensor_record(adapter7, (10, 7))
        self._write({"event": "generation", "role": "planned_model_and_adapter_output_not_issued",
                     "query_index": query_index, "raw32": raw, "adapter7": adapter})
        self._generations[query_index] = np.array(adapter7, copy=True)
        self._model_started = True

    def dispatch(self, command7, send: Callable[[np.ndarray], StepReturn], *, phase, query_index=None, action_position=None):
        _require(self.attempt.kind in ("short", "full"), "Single-call and generation attempts cannot issue control")
        _require(phase in ("startup", "policy") and callable(send), "Explicit startup/policy dispatch callback required")
        command = np.array(command7, copy=True)
        packed = tensor_record(command, (7,))
        if phase == "policy":
            _require(self._returned["startup"] == self._plan.startup_steps, "Policy controls require ten returned startup controls")
            horizon = 5 if self.attempt.kind == "short" else self._plan.full_max_policy_steps
            _require(self._counts["policy"] < horizon, "Frozen policy horizon already reached")
            _require(_integer(query_index) and _integer(action_position) and query_index == self._counts["policy"] // 5
                     and action_position == self._counts["policy"] % 5, "Policy controls require sequential queries and slots 0..4")
            _require(query_index in self._generations and self._flow_counts.get(query_index) == 10,
                     "Policy issue must reference an existing adapter7 generation position")
            expected = self._generations[query_index][action_position]
            _require(command.dtype == expected.dtype and command.tobytes() == expected.tobytes(), "Issued command must match actual adapter7 bytes")
            _require((query_index, action_position) not in self._issued, "No repeated policy control position")
            _require(self.attempt.kind != "short" or query_index == 0, "Short execution permits only query0")
        else:
            _require(query_index is None and action_position is None and self._counts["policy"] == 0
                     and not self._model_started and self._counts["startup"] < self._plan.startup_steps,
                     "Startup controls must precede all model/policy calls and cannot exceed ten")
        index = self._counts[phase]
        self._write({"event": "issue", "meaning": "transport_invocation_not_confirmed_motion", "phase": phase,
                     "phase_step": index, "query_index": query_index, "action_position": action_position, "command7": packed})
        self._counts[phase] += 1
        if phase == "policy":
            self._issued.add((query_index, action_position))
        try:
            result = send(command.copy())
            _require(isinstance(result, StepReturn), "Actual callback did not return a StepReturn receipt")
        except BaseException as error:
            self._unconfirmed = True
            try:
                self._write({"event": "unconfirmed", "phase": phase, "phase_step": index, "success": None,
                             "error_type": type(error).__name__, "automatic_retry": False})
            finally:
                self._halted = True
            raise
        try:
            self._write({"event": "returned", "phase": phase, "phase_step": index, **asdict(result)})
        except BaseException:
            self._unconfirmed = True
            self._halted = True
            raise
        self._returned[phase] += 1
        self._last_return = result
        if result.terminal:
            self._halted = True
        return result
