"""Pre-call native budget hooks. Importing this module performs no native work."""
from contextlib import ExitStack
from dataclasses import dataclass, field
import math
import time
from failure_guards import ResetBudget, reset_once_zero_noise, reject_interactive_placement


class NativeBudgetError(RuntimeError):
    pass


LIMITS = {
    "construct": 10, "reset": 10, "set_init_state": 10,
    "mj_step": 8750, "mj_step1": 20, "mj_step2": 0,
    "mj_forward": 30000,
}


@dataclass
class Budget:
    deadline_epoch: float
    counts: dict = field(default_factory=lambda: {k: 0 for k in LIMITS})
    attempted: dict = field(default_factory=lambda: {k: 0 for k in LIMITS})
    aborted: str | None = None
    resets: ResetBudget = field(default_factory=ResetBudget)

    def check(self):
        if self.aborted:
            raise NativeBudgetError(self.aborted)
        if not math.isfinite(self.deadline_epoch) or time.time() >= self.deadline_epoch:
            self.aborted = "native_absolute_deadline"
            raise NativeBudgetError(self.aborted)

    def charge(self, kind, amount=1):
        self.check()
        if kind not in LIMITS or type(amount) is not int or amount < 1:
            self.aborted = "invalid_native_call_charge"
            raise NativeBudgetError(self.aborted)
        self.attempted[kind] += amount
        if self.counts[kind] + amount > LIMITS[kind]:
            self.aborted = "native_cap_" + kind
            raise NativeBudgetError(self.aborted)
        self.counts[kind] += amount

    def snapshot(self):
        return {"limits": dict(LIMITS), "entered_calls": dict(self.counts),
                "attempted_calls": dict(self.attempted), "aborted": self.aborted,
                "reset_attempts": self.resets.attempted,
                "reset_successes": self.resets.successful,
                "counts_are_entered_calls_not_successful_native_returns": True}


def replace(stack, owner, name, value):
    old = getattr(owner, name)
    setattr(owner, name, value)
    stack.callback(setattr, owner, name, old)
    return old


def verify_clock(wrapper):
    """Require the fixed source cadence before dummy or policy stepping."""
    env = wrapper.env
    values = {"control_freq": float(env.control_freq),
              "control_timestep": float(env.control_timestep),
              "model_timestep": float(env.model_timestep),
              "compiled_timestep": float(env.sim.model._model.opt.timestep)}
    if values != {"control_freq": 20.0, "control_timestep": .05,
                  "model_timestep": .002, "compiled_timestep": .002}:
        raise NativeBudgetError("actual_native_clock_differs")
    if int(values["control_timestep"] / values["model_timestep"]) != 25:
        raise NativeBudgetError("actual_substep_ratio_differs")
    return values


def install(control_class, mujoco, pdb, budget, *, role, emit):
    """Install before constructor invocation; every hook restores on close."""
    stack = ExitStack()
    original_construct = control_class.__init__
    original_init_state = control_class.set_init_state
    def clock(self):
        try:
            return verify_clock(self)
        except BaseException:
            budget.aborted = "actual_native_clock_differs"
            raise
    def construct(self, *args, **kwargs):
        budget.charge("construct")
        original_construct(self, *args, **kwargs)
        emit("ACTUAL_NATIVE_CLOCK", {"role": role, **clock(self)})
    def reset(self):
        budget.charge("reset")
        out = reset_once_zero_noise(self, budget.resets, role=role, emit=emit, check=budget.check)
        clock(self)
        return out
    def set_init_state(self, *args, **kwargs):
        budget.charge("set_init_state")
        result = original_init_state(self, *args, **kwargs)
        clock(self)
        return result
    def no_debugger(*args, **kwargs):
        budget.aborted = "placement_debugger_requested"
        emit("PLACEMENT_FAILED_NO_INTERACTION", {"role": role})
        reject_interactive_placement(*args, **kwargs)
    try:
        replace(stack, control_class, "__init__", construct)
        replace(stack, control_class, "reset", reset)
        replace(stack, control_class, "set_init_state", set_init_state)
        replace(stack, pdb, "set_trace", no_debugger)
        for kind in ("mj_step", "mj_step1", "mj_step2", "mj_forward"):
            original = getattr(mujoco, kind)
            def counted(*args, _name=kind, _original=original, **kwargs):
                # Pinned MjSim.step passes exactly model,data. Reject an alternate batch ABI.
                if _name == "mj_step" and (len(args) != 2 or kwargs):
                    budget.aborted = "unexpected_mj_step_batch_ABI"
                    raise NativeBudgetError(budget.aborted)
                budget.charge(_name)
                return _original(*args, **kwargs)
            replace(stack, mujoco, kind, counted)
        return stack
    except BaseException:
        stack.close()
        raise
