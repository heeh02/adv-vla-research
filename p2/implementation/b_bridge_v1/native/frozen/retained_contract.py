"""Pure intervention contract. No model, network, arrays or global hook state.

The reference evaluator in cpu_reference.py is SYNTHETIC SOFTWARE ONLY.
Production candidate arithmetic lives in jax_candidate.py and requires a
separate actual-checkpoint BF16 qualification before any action is issued.
"""
from dataclasses import dataclass
from typing import Any, Callable


class ContractError(ValueError):
    pass


def require(ok, code):
    if not ok:
        raise ContractError(code)


@dataclass(frozen=True)
class Intervention:
    mode: str
    layer: int = 17
    call_index: int = 0
    continuation: str = "base_prefix_only"

    def __post_init__(self):
        require(self.mode in ("noop", "observe", "replace_k", "replace_v", "replace_kv"), "unknown_mode")
        require(type(self.layer) is int and self.layer in (0, 8, 17), "three_representative_layers")
        require(type(self.call_index) is int and 0 <= self.call_index < 10, "call_out_of_range")
        require(self.continuation == "base_prefix_only", "one_shot_requires_base_continuation")

    @property
    def edits_k(self):
        return self.mode in ("replace_k", "replace_kv")

    @property
    def edits_v(self):
        return self.mode in ("replace_v", "replace_kv")


@dataclass(frozen=True)
class ResumeState:
    """The actual PRE-update receiver state, not an interpolated trajectory.

    time is opaque: do not cast or substitute an ideal decimal. The caller
    binds original typed identity/weak_type, state source, noise and query.
    """
    start_call: int
    x_t: Any
    time: Any

    def __post_init__(self):
        require(type(self.start_call) is int and 0 <= self.start_call < 10, "resume_call_out_of_range")


class OneShotTransaction:
    """CPU/host contract fixture: request-local, exception-safe, not a JIT hook.

    It never assigns into a caller's cache. The supplied select function must
    be functional. Subsequent calls ALWAYS receive base; no commit API exists.
    Use one object per arm/query. JAX implements the same rule with pure values.
    """
    def __init__(self, plan, base, donor, select: Callable):
        require(isinstance(plan, Intervention), "explicit_intervention_required")
        self.plan, self.base, self.donor, self.select = plan, base, donor, select
        self._entered = self._closed = self._used = False
        self._last_call = None

    def __enter__(self):
        require(not self._entered and not self._closed, "transaction_not_reusable")
        self._entered = True
        return self

    def cache_for(self, call_index):
        require(self._entered and not self._closed, "transaction_not_active")
        require(type(call_index) is int and 0 <= call_index < 10, "call_out_of_range")
        require(self._last_call is None or call_index == self._last_call + 1, "calls_must_be_sequential_once")
        require(self.plan.mode == "noop" or self._last_call is not None or call_index <= self.plan.call_index, "target_call_already_passed")
        self._last_call = call_index
        if call_index != self.plan.call_index or self.plan.mode in ("noop", "observe"):
            return self.base
        require(not self._used, "intervention_already_used")
        self._used = True
        return self.select(self.base, self.donor, self.plan)

    def __exit__(self, kind, value, tb):
        self._closed = True
        self.donor = None  # no donor retained in any shared/session state
        return False


def dispatch_query(plan, *, stock_sample, candidate_sample, capture=False):
    """No-op MUST call the original fast path, without candidate preparation.

    Identity-donor/capture checks belong to observe/replace modes. They do not
    qualify by returning an unchanged result through this bypass.
    """
    require(isinstance(plan, Intervention), "explicit_intervention_required")
    if plan.mode == "noop":
        require(not capture, "noop_cannot_claim_operand_capture")
        return stock_sample()
    return candidate_sample()


def require_real_qualification(receipt):
    """Future controller gate, never synthesizes a passing receipt."""
    require(receipt.get("schema") == "sru-native-qualification-v1", "missing_native_qualification")
    for field in ("actual_checkpoint", "source_bound", "native_BF16", "stock_noop_bytes_exact",
                  "observe_control_bytes_exact", "identity_KV_bytes_exact", "receiver_fixed_bytes_exact",
                  "capture_vs_control_bytes_exact", "base_cache_unchanged", "remaining_sampling_checked",
                  "raw32_and_adapter7_checked"):
        require(receipt.get(field) is True, "native_gate_" + field)
    require(receipt.get("synthetic_cpu") is False, "synthetic_is_not_native_qualification")
    return True
