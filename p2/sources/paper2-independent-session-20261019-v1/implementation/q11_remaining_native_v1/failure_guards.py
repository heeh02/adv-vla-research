"""Candidate failure-path guards only; importing this file runs no environment.

Not an execution entrypoint. The future root-owned runner must bind source,
inputs, authorization, resources and physical qualification separately.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path


CONTROL_SOURCE = {
    "bytes": 18963,
    "sha256": "e91d7b7b35cc3ad2b073606c99860def6b3ac43b66eba00ef0ddb6bfd8f39c3c",
}


class CandidateGuardError(RuntimeError):
    pass


class InteractivePlacementFailure(CandidateGuardError):
    pass


def bind_control_source(path):
    """Read the exact source; never import it or use upstream replacement code."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CandidateGuardError("regular_bound_source_required")
    raw = path.read_bytes()
    if {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != CONTROL_SOURCE:
        raise CandidateGuardError("original_ControlEnv_source_mismatch")
    return dict(CONTROL_SOURCE)


@dataclass
class ResetBudget:
    maximum: int = 10
    attempted: int = 0
    successful: int = 0

    def charge(self):
        if type(self.maximum) is not int or not 1 <= self.maximum <= 10:
            raise CandidateGuardError("bounded_reset_budget_required")
        if self.attempted >= self.maximum:
            raise CandidateGuardError("reset_attempt_cap")
        self.attempted += 1


def reset_once_zero_noise(wrapper, budget, *, role, emit, check):
    """Known noise=0 success path, one lower reset attempt, no retry.

    The original ControlEnv.reset repeatedly called wrapper.env.reset and
    skipped its image-corruption branch for integer noise=0. Its finally:
    continue could suppress exceptions. This candidate performs the same
    single successful call and returns the same object. Failure is retained
    and propagated, including KeyboardInterrupt/SystemExit. It neither sets
    RNG/state nor performs a second reset, observation refresh or rendering.

    The caller must not install this by wrapping the old retry loop: it must
    replace that entry in a separately reviewed overlay. This helper alone
    does not prove native equivalence or qualify an environment.
    """
    check()
    if type(wrapper.noise) is not int or wrapper.noise != 0:
        raise CandidateGuardError("only_bound_Camera_zero_noise_path")
    budget.charge()
    emit("RESET_ATTEMPT", {"role": role, "attempt": budget.attempted})
    try:
        result = wrapper.env.reset()
    except BaseException as error:
        emit("RESET_FAILED", {"role": role, "error_type": type(error).__name__,
                              "attempt": budget.attempted})
        raise
    budget.successful += 1
    emit("RESET_SUCCEEDED", {"role": role, "attempt": budget.attempted})
    check()
    return result


def reject_interactive_placement(*_args, **_kwargs):
    """Replace only the pinned placement failure's debugger callback later."""
    raise InteractivePlacementFailure("placement_failed_no_debugger_no_retry")
