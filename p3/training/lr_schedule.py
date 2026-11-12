"""Only the future callable changes; the restored Optax counter stays global."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class ContinuationSchedule:
    branch: str
    start_step: int = 20000
    target_step: int = 30000
    ramp_steps: int = 500
    floor_lr: float = 2.5e-6
    peak_lr: float = 2.5e-5

    def __post_init__(self):
        if dataclasses.asdict(self) != dict(branch=self.branch, start_step=20000,
                target_step=30000, ramp_steps=500, floor_lr=2.5e-6, peak_lr=2.5e-5):
            raise ValueError('only_root_frozen_schedule_supported')
        if self.branch not in ('floor', 'reheat'):
            raise ValueError('unknown_schedule_branch')

    def create(self):
        import jax.numpy as jnp
        import optax
        # Exact original implementation, including native FP32 tail rounding.
        original = optax.warmup_cosine_decay_schedule(
            init_value=2.5e-5 / 1001, peak_value=2.5e-5,
            warmup_steps=1000, decay_steps=10000, end_value=2.5e-6)
        if self.branch == 'floor':
            return original
        future = optax.warmup_cosine_decay_schedule(
            init_value=self.floor_lr, peak_value=self.peak_lr,
            warmup_steps=self.ramp_steps,
            decay_steps=self.target_step-self.start_step, end_value=self.floor_lr)
        def schedule(count):
            return jnp.where(count < self.start_step, original(count),
                             future(jnp.maximum(count-self.start_step, 0)))
        return schedule
