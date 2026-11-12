"""CPU-only transaction guards for a separately qualified clean-reference path.

No renderer, simulator, policy, or checkpoint is imported here. Callers supply
passive snapshot/restore functions and a monotonic physical-step counter which
must live outside restorable simulator state. This guard cannot discover hidden
state or an uninstrumented step. A successful receipt never qualifies a renderer.
"""

from collections.abc import Callable, Mapping, Sequence
import copy
import dataclasses
import hashlib
import json
import random
import struct
import threading
from typing import Any

import numpy as np

_LOCK = threading.Lock()
_POISONED = False


def _identity(value: Any) -> str:
    """Exact typed identity for explicit data trees; reject arbitrary objects."""
    digest = hashlib.sha256()

    def item(tag, payload=b""):
        digest.update(tag + str(len(payload)).encode() + b":" + payload)

    def visit(node):
        if isinstance(node, np.ndarray | np.generic):
            array = np.asarray(node)
            if array.dtype.hasobject or array.dtype.fields is not None:
                raise TypeError("Snapshot arrays must have a non-object, unstructured dtype")
            item(b"array" if isinstance(node, np.ndarray) else b"scalar")
            item(b"dtype", (array.dtype.str + "/" + array.dtype.name).encode())
            item(b"shape", json.dumps(array.shape).encode())
            item(b"value", array.tobytes(order="C"))
        elif node is None:
            item(b"none")
        elif type(node) is bool:
            item(b"bool", bytes([node]))
        elif type(node) is int:
            item(b"int", str(node).encode())
        elif type(node) is float:
            item(b"float", struct.pack(">d", node))
        elif type(node) is str:
            item(b"str", node.encode())
        elif type(node) is bytes:
            item(b"bytes", node)
        elif type(node) in (tuple, list):
            item(b"tuple" if type(node) is tuple else b"list", str(len(node)).encode())
            for child in node:
                visit(child)
        elif type(node) is dict:
            if any(type(key) is not str for key in node):
                raise TypeError("Snapshot dictionaries require string keys")
            item(b"dict", str(len(node)).encode())
            for key in sorted(node):
                visit(key)
                visit(node[key])
        else:
            raise TypeError("Snapshot must be an explicit typed data tree")

    visit(value)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class StateBinding:
    """One caller-owned state; snapshots must be passive and independently copyable.

    Default: any observed change is forbidden even if subsequently restored.
    Set allow_temporary_change only for fields intentionally transacted, such
    as camera parameters. Do not set it for physical progression or clocks.
    """

    name: str
    snapshot: Callable[[], Any]
    restore: Callable[[Any], None]
    allow_temporary_change: bool = False


class ReferenceTransactionError(RuntimeError):
    def __init__(self, report):
        super().__init__("Clean-reference transaction failed; inspect the structured report")
        self.report = copy.deepcopy(report)


class ReferenceRestoreError(ReferenceTransactionError):
    """Restoration is uncertain; all further transactions in this process fail."""


class ReferenceOverlapError(RuntimeError):
    """Global RNG ownership forbids overlapping transactions, including nesting."""


class ReferenceProcessPoisonedError(RuntimeError):
    """Restart the owning worker after an uncertain restore or forbidden mutation."""


class ReferenceTransaction:
    """Single-use guard with exception-safe rollback and an inspectable receipt.

    ``required_states`` must declare the application's observation coverage.
    Missing bindings are reported, never inferred as unchanged. They produce
    ``restored_with_unobserved_state``, not a qualification PASS. Callers must
    gate real execution on complete required coverage themselves.

    ``physical_step_counter`` is mandatory and monotonic across the entire
    transaction, including restoration; it must count *attempted* physical
    steps and must not be reset by any binding. The counter cannot be replaced
    by sim.time, qpos, or a restorable environment step counter.

    Restoration errors or forbidden state/step changes poison this process.
    Catching the exception cannot permit another transaction. The caller must
    stop the affected branch/worker; this utility does not terminate processes.
    """

    def __init__(
        self,
        bindings: Sequence[StateBinding],
        *,
        required_states: Sequence[str],
        physical_step_counter: Callable[[], int],
        generators: Mapping[str, np.random.Generator] | None = None,
    ):
        bindings = tuple(bindings)
        if any(not isinstance(binding, StateBinding) for binding in bindings):
            raise TypeError("Explicit StateBinding objects required")
        names = [binding.name for binding in bindings]
        if any(not name or not isinstance(name, str) or name.startswith("rng.") for name in names):
            raise ValueError("Unique nonempty caller names outside reserved rng namespace required")
        if len(names) != len(set(names)) or any(
            not callable(binding.snapshot) or not callable(binding.restore) for binding in bindings
        ):
            raise ValueError("Unique callable state bindings required")
        if any(type(binding.allow_temporary_change) is not bool for binding in bindings):
            raise TypeError("Temporary change permission must be an explicit boolean")
        if isinstance(required_states, str) or not required_states or any(
            not isinstance(name, str) or not name for name in required_states
        ):
            raise ValueError("Explicit nonempty required state names required")
        if not callable(physical_step_counter):
            raise TypeError("An external monotonic attempted-step counter is required")
        generators = dict(generators or {})
        if any(not isinstance(name, str) or not name for name in generators) or any(
            not isinstance(generator, np.random.Generator) for generator in generators.values()
        ):
            raise TypeError("Register named numpy Generator instances explicitly")
        if len({id(generator) for generator in generators.values()}) != len(generators):
            raise ValueError("The same Generator cannot be registered twice")

        # Restore in reverse order, so global streams are restored last after
        # caller callbacks. Registered Generator states include their algorithms.
        rng_bindings = [
            StateBinding("rng.python_global", random.getstate, random.setstate, True),
            StateBinding("rng.numpy_global", np.random.get_state, np.random.set_state, True),
        ]
        for name, generator in sorted(generators.items()):
            rng_bindings.append(StateBinding(
                "rng.generator." + name,
                lambda generator=generator: generator.bit_generator.state,
                lambda state, generator=generator: setattr(generator.bit_generator, "state", state),
                True,
            ))
        self._bindings = (*rng_bindings, *bindings)
        self._step_counter = physical_step_counter
        self._saved = []
        self._used = False
        observed = {binding.name for binding in self._bindings}
        self.report = {
            "schema": "selective-reference-transaction-v1",
            "status": "not_entered",
            "native_renderer_qualified": False,
            "required_states": sorted(set(required_states)),
            "unobserved_states": sorted(set(required_states) - observed),
            "before": {}, "after_call": {}, "after_restore": {},
            "changed_during_call": [], "forbidden_changes": [],
            "observation_errors": [], "restore_errors": [],
            "step_before": None, "step_after_call": None, "step_after_restore": None,
            "body_exception_type": None,
        }

    def _step(self):
        value = self._step_counter()
        if type(value) is not int or value < 0:
            raise TypeError("Step witness must return a nonnegative Python int")
        return value

    def __enter__(self):
        if self._used:
            raise RuntimeError("Reference transactions are single use")
        self._used = True
        if not _LOCK.acquire(blocking=False):
            raise ReferenceOverlapError("A reference transaction already owns process-global state")
        try:
            if _POISONED:
                raise ReferenceProcessPoisonedError("Reference worker state is uncertain; restart the worker")
            self.report["step_before"] = self._step()
            for binding in self._bindings:
                saved = copy.deepcopy(binding.snapshot())
                # Keep the state even if identity validation rejects it, so the
                # corresponding restore still runs during failed entry.
                self._saved.append((binding, saved))
                self.report["before"][binding.name] = _identity(saved)
            if self._step() != self.report["step_before"]:
                self.report["forbidden_changes"].append("physical_step_during_snapshot")
                raise RuntimeError("Snapshot callbacks must not advance physics")
            self.report["status"] = "active"
            return self
        except BaseException as error:
            try:
                if self._saved:
                    self._finish(error, entry_failed=True)
            finally:
                _LOCK.release()
            raise

    def __exit__(self, _kind, error, _traceback):
        try:
            self._finish(error)
        finally:
            _LOCK.release()
        return False

    def _finish(self, body_error, *, entry_failed=False):
        global _POISONED  # One process-wide owner is required for global RNG.
        report = self.report
        report["body_exception_type"] = type(body_error).__name__ if body_error is not None else None
        if entry_failed:
            report["observation_errors"].append({"state": "entry", "error_type": type(body_error).__name__})

        def observe(phase):
            for binding, _saved in reversed(self._saved):
                try:
                    value = _identity(binding.snapshot())
                    report[phase][binding.name] = value
                    if phase == "after_call" and value != report["before"].get(binding.name):
                        report["changed_during_call"].append(binding.name)
                        if not binding.allow_temporary_change:
                            report["forbidden_changes"].append(binding.name)
                    if phase == "after_restore" and value != report["before"].get(binding.name):
                        report["restore_errors"].append({"state": binding.name, "error_type": "IdentityMismatch"})
                except BaseException as error:
                    report["observation_errors"].append({
                        "state": binding.name, "phase": phase, "error_type": type(error).__name__,
                    })
            try:
                report["step_" + phase] = self._step()
                if report["step_" + phase] != report["step_before"]:
                    report["forbidden_changes"].append("physical_step_" + phase)
            except BaseException as error:
                report["observation_errors"].append({
                    "state": "physical_step_witness", "phase": phase, "error_type": type(error).__name__,
                })

        observe("after_call")
        for binding, saved in reversed(self._saved):
            try:
                binding.restore(copy.deepcopy(saved))
            except BaseException as error:
                report["restore_errors"].append({"state": binding.name, "error_type": type(error).__name__})
        observe("after_restore")
        # Do not keep full snapshots alive past exit; receipts contain typed
        # identities only. Recording source arrays remains the caller's job.
        self._saved.clear()
        if report["restore_errors"] or report["observation_errors"]:
            report["status"] = "restoration_uncertain_stop_worker"
            _POISONED = True
            raise ReferenceRestoreError(report) from body_error
        if report["forbidden_changes"]:
            report["status"] = "forbidden_change_restored_stop_worker"
            _POISONED = True
            raise ReferenceTransactionError(report) from body_error
        report["status"] = (
            "restored_with_unobserved_state" if report["unobserved_states"] else "declared_state_restored"
        )


def reference_cache_key(
    *, physical_state, query_identity, clean_input, prompt, checkpoint_identity,
    model_config_identity, render_config_identity, prefix_identity, x_t, t,
) -> str:
    """Key only, no cache: bind actual current-branch inputs and generation state.

    query_identity must identify run/case/arm/query; prefix_identity must bind
    preprocessing, masks, token layout and native dtype. Physics/render inputs
    must also bind the actual sensor-producing state/time. The utility requires
    these fields explicitly but cannot verify the caller's provenance claims.
    """
    fields = {
        "physical_state": physical_state, "query_identity": query_identity,
        "clean_input": clean_input, "prompt": prompt, "checkpoint_identity": checkpoint_identity,
        "model_config_identity": model_config_identity, "render_config_identity": render_config_identity,
        "prefix_identity": prefix_identity, "x_t": x_t, "t": t,
    }
    if any(value is None for value in fields.values()):
        raise ValueError("All clean-reference cache identity fields are mandatory")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("The actual prompt is required")
    return _identity({"schema": "selective-reference-cache-key-v1", **fields})
