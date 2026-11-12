"""Persistent, explicitly batched policy for the native-light campaign.

Use one instance per GPU process. Simulator actors exchange arrays with this
process; they never load separate model replicas. No sampling RNG is implicit.
"""

# The pinned Policy has no public transform/model accessors; reusing its
# exact pipeline is preferable to maintaining a second preprocessing copy.
# ruff: noqa: SLF001

import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.model import Observation
from openpi.shared import nnx_utils


def stack_inputs(inputs):
    if not inputs:
        raise ValueError("Empty inference batch")
    return jax.tree.map(lambda *values: jnp.stack([jnp.asarray(x) for x in values]), *inputs)


def _argument_layout(leaves):
    """Record the array ABI without casting values, including typed RNG keys."""
    return tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves)


class NativeTrajectoryParityError(RuntimeError):
    """Retain the small genuine mismatch arrays for qualification evidence."""

    def __init__(self, condition, reference, candidate):
        self.condition = condition
        self.reference, self.candidate = np.asarray(reference), np.asarray(candidate)
        super().__init__(f"Native trajectory recorder changed canonical {condition} actions; exact parity required")


def require_native_actions(condition, reference, candidate):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if (
        reference.dtype != candidate.dtype
        or reference.shape != candidate.shape
        or reference.tobytes() != candidate.tobytes()
    ):
        raise NativeTrajectoryParityError(condition, reference, candidate)


def attach_native_sources(
    diagnostic, clean_reference, actual_reference, clean_candidate, actual_candidate, clean_path, actual_path
):
    """Merge independently executed graphs outside JIT; never integrate shadow v."""
    if any(key in diagnostic for key in ("actions", "actual_actions", "clean_actions")):
        raise ValueError("Shadow diagnostics must never provide sampling actions")
    result = dict(diagnostic)
    result.update(
        clean_actions=clean_reference,
        actual_actions=actual_reference,
        native_clean_candidate_actions=clean_candidate,
        native_actual_candidate_actions=actual_candidate,
        native_clean_layer_rms=jnp.moveaxis(clean_path["layer_rms"], 2, 0),
        native_actual_layer_rms=jnp.moveaxis(actual_path["layer_rms"], 2, 0),
        native_times=actual_path["time"],
        native={
            side: {key: path[key] for key in ("x_t", "time", "velocity", "layer_rms")}
            for side, path in (("clean", clean_path), ("actual", actual_path))
        },
        schema_name="native_shadow_v2",
        provenance={
            "actions": "canonical_native_sampler",
            "native_states": "independent_native_recorder",
            "dense_scalars": "native_recorder",
            "free_hidden_readout_velocity": "native_recording",
            "free_attention_visual_branch_output": "shadow_on_condition_native_states",
            "shared_xt": "shadow_on_shared_clean_native_states_same_three_tau_abi",
            "shadow_error": "source_dtype_shadow_minus_native; never an action path",
        },
    )
    return result


@dataclasses.dataclass
class CompiledCampaignBatch:
    """An explicitly lowered executable, for allocated-GPU timing only.

    The frozen graph/state and method call match ``module_jit``. Executing
    this object calls the compiled executable directly, so a timed warm call
    cannot include another implicit compilation. Model tensors are not cast.
    """

    mode: str
    executable: object
    arguments: tuple
    model_state: object
    argument_treedef: object
    argument_layout: tuple
    actual_states: object
    clean_states: object
    timings: dict
    memory_analysis: dict
    canonical_reference: object = None

    def execute(self):
        start = time.monotonic()
        reference = None
        if self.canonical_reference is not None:
            reference, _ = self.canonical_reference.execute()
        result = self.executable(*self.arguments)
        jax.block_until_ready(result)
        if reference is not None:
            candidate, scalars = result
            require_native_actions("scalar", reference, candidate)
            result = (reference, scalars)
        return result, time.monotonic() - start


@dataclasses.dataclass
class CompiledPairedCampaignBatch:
    """Five separate executable boundaries, matching the production sequence."""

    policy: object
    reference_actual: CompiledCampaignBatch
    reference_clean: CompiledCampaignBatch
    trajectory_actual: CompiledCampaignBatch
    trajectory_clean: CompiledCampaignBatch
    shadow: CompiledCampaignBatch
    actual_observation: object
    clean_observation: object
    mode: str
    arguments: tuple
    model_state: object
    argument_treedef: object
    argument_layout: tuple
    actual_states: object
    clean_states: object
    timings: dict
    memory_analysis: dict

    def execute(self):
        started = time.monotonic()
        actual_reference, _ = self.reference_actual.execute()
        clean_reference, _ = self.reference_clean.execute()
        (actual_candidate, actual_path), _ = self.trajectory_actual.execute()
        (clean_candidate, clean_path), _ = self.trajectory_clean.execute()
        require_native_actions("actual", actual_reference, actual_candidate)
        require_native_actions("clean", clean_reference, clean_candidate)
        flat, treedef = jax.tree.flatten(
            (self.shadow.model_state, self.clean_observation, self.actual_observation, clean_path, actual_path)
        )
        if treedef != self.shadow.argument_treedef or _argument_layout(flat) != self.shadow.argument_layout:
            raise ValueError("Shadow input ABI changed after native recording")
        diagnostic = self.shadow.executable(*flat)
        jax.block_until_ready(diagnostic)
        result = attach_native_sources(
            diagnostic, clean_reference, actual_reference, clean_candidate, actual_candidate, clean_path, actual_path
        )
        return result, time.monotonic() - started


class CampaignPolicy:
    def __init__(self, checkpoint):
        # Imports are delayed so pure batching/metrics tests need no LIBERO.
        from openpi.policies.policy_config import create_trained_policy  # noqa: PLC0415
        from openpi.training.config import get_config  # noqa: PLC0415

        if not jax.devices("gpu"):
            raise RuntimeError("Campaign model must run on an allocated company GPU")
        config = get_config("pi05_libero")
        self.policy = create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": 10})
        self.policy._model.eval()  # Same model/transform pipeline as standard LIBERO policy.
        self.sample = nnx_utils.module_jit(self.policy._model.sample_actions)
        self.sample_scalars = nnx_utils.module_jit(self.policy._model.sample_actions_with_scalars)
        self.sample_trajectory = nnx_utils.module_jit(self.policy._model.sample_actions_with_trajectory)
        self.shadow_pair = nnx_utils.module_jit(
            self.policy._model.trace_pair_on_native_states, static_argnames=("save_visual", "save_attention")
        )
        self.trace_pair = nnx_utils.module_jit(
            self.policy._model.trace_pair, static_argnames=("save_visual", "save_attention")
        )

    def encode(self, observations, prompts):
        from openpi.eval.libero_inputs import _preprocess_obs_images  # noqa: PLC0415
        from openpi.eval.libero_inputs import _state_from_obs  # noqa: PLC0415

        inputs = []
        for observation, prompt in zip(observations, prompts, strict=True):
            main, wrist = _preprocess_obs_images(observation, 224)
            data = {
                "observation/image": main,
                "observation/wrist_image": wrist,
                "observation/state": _state_from_obs(observation),
                "prompt": prompt,
            }
            inputs.append(self.policy._input_transform(data))
        batch = stack_inputs(inputs)
        return Observation.from_dict(batch), batch

    def decode(self, raw_actions, states):
        raw_actions, states = np.asarray(raw_actions), np.asarray(states)
        return np.stack(
            [
                self.policy._output_transform({"actions": action, "state": state})["actions"]
                for action, state in zip(raw_actions, states, strict=True)
            ]
        )

    def compile_batch(
        self,
        observations,
        prompts,
        noise,
        *,
        mode="scalars",
        clean_observations=None,
        save_visual=False,
        save_attention=False,
    ):
        """Measure explicit lower/compile separately from synchronized inference.

        Compilation can read an existing JAX disk cache; the receipt therefore
        calls this interval compile-or-cache-load, not an assumed cold compile.
        This is a measurement entry point, not a new production sampler.
        """
        import flax.nnx as nnx  # noqa: PLC0415

        if mode == "paired":
            return self._compile_paired(
                observations, prompts, noise, clean_observations, save_visual=save_visual, save_attention=save_attention
            )
        if mode not in ("baseline", "scalars", "trajectory"):
            raise ValueError("Unknown compiled campaign mode")
        start = time.monotonic()
        actual, actual_batch = self.encode(observations, prompts)
        noise = jnp.asarray(noise)
        if noise.shape != (len(observations), 10, 32):
            raise ValueError("Each request must carry its own 10x32 noise")
        clean_states = None
        if mode == "baseline":
            method = self.policy._model.sample_actions
            args = (jax.random.key(0), actual, noise)
            kwargs = {}
        else:
            method = (
                self.policy._model.sample_actions_with_trajectory
                if mode == "trajectory"
                else self.policy._model.sample_actions_with_scalars
            )
            args = (actual, noise)
            kwargs = {}
        jax.block_until_ready((args, kwargs, actual_batch, clean_states))
        encode_seconds = time.monotonic() - start
        graphdef, state = nnx.split(self.policy._model)
        flat_arguments, argument_treedef = jax.tree.flatten((state, *args))
        argument_layout = _argument_layout(flat_arguments)

        def frozen(*flat_leaves):
            # .lower() constructs ArgInfo input trees outside tracing. Passing
            # Observation there invokes its checked dataclass constructor with
            # non-Array leaves. Rebuild only inside JIT, with actual tracers.
            model_state, *arguments = argument_treedef.unflatten(flat_leaves)
            module = nnx.merge(graphdef, model_state)
            if mode == "baseline":
                return method.__func__(module, arguments[0], arguments[1], noise=arguments[2], num_steps=10)
            return method.__func__(module, *arguments, **kwargs)

        start = time.monotonic()
        lowered = jax.jit(frozen).lower(*flat_arguments)
        lower_seconds = time.monotonic() - start
        start = time.monotonic()
        executable = lowered.compile()
        compile_seconds = time.monotonic() - start
        analysis = executable.memory_analysis()
        memory = {}
        if analysis is not None:
            for name in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes"):
                value = getattr(analysis, name, None)
                if value is not None:
                    memory[name] = int(value)
        compiled = CompiledCampaignBatch(
            mode=mode,
            executable=executable,
            arguments=tuple(flat_arguments),
            model_state=state,
            argument_treedef=argument_treedef,
            argument_layout=argument_layout,
            actual_states=actual_batch["state"],
            clean_states=clean_states,
            timings={
                "encode_and_transfer_seconds": encode_seconds,
                "lower_seconds": lower_seconds,
                "compile_or_cache_load_seconds": compile_seconds,
            },
            memory_analysis=memory,
        )
        if mode == "scalars":
            reference = self.compile_batch(observations, prompts, noise, mode="baseline")
            compiled.canonical_reference = reference
            compiled.timings = {key: compiled.timings[key] + reference.timings[key] for key in compiled.timings}
            compiled.memory_analysis = {
                "scalar_stage": memory,
                "canonical_reference": reference.memory_analysis,
                "scope": "separate executables; actual peak measured separately",
            }
        return compiled

    def _compile_paired(self, observations, prompts, noise, clean_observations, *, save_visual, save_attention):
        if clean_observations is None or len(clean_observations) != len(observations):
            raise ValueError("Compiled paired inference requires matched clean observations")
        stages = {
            "reference_actual": self.compile_batch(observations, prompts, noise, mode="baseline"),
            "reference_clean": self.compile_batch(clean_observations, prompts, noise, mode="baseline"),
            "trajectory_actual": self.compile_batch(observations, prompts, noise, mode="trajectory"),
            "trajectory_clean": self.compile_batch(clean_observations, prompts, noise, mode="trajectory"),
        }
        started = time.monotonic()
        actual_reference, _ = stages["reference_actual"].execute()
        clean_reference, _ = stages["reference_clean"].execute()
        (actual_candidate, actual_path), _ = stages["trajectory_actual"].execute()
        (clean_candidate, clean_path), _ = stages["trajectory_clean"].execute()
        require_native_actions("actual", actual_reference, actual_candidate)
        require_native_actions("clean", clean_reference, clean_candidate)
        example_seconds = time.monotonic() - started
        started = time.monotonic()
        actual, _ = self.encode(observations, prompts)
        clean, _ = self.encode(clean_observations, prompts)
        shadow_encode_seconds = time.monotonic() - started
        import flax.nnx as nnx  # noqa: PLC0415

        graphdef, state = nnx.split(self.policy._model)
        arguments, tree = jax.tree.flatten((state, clean, actual, clean_path, actual_path))
        method = self.policy._model.trace_pair_on_native_states

        def frozen_shadow(*leaves):
            model_state, *items = tree.unflatten(leaves)
            module = nnx.merge(graphdef, model_state)
            return method.__func__(module, *items, save_visual=save_visual, save_attention=save_attention)

        started = time.monotonic()
        lowered = jax.jit(frozen_shadow).lower(*arguments)
        lower_seconds = time.monotonic() - started
        started = time.monotonic()
        executable = lowered.compile()
        compile_seconds = time.monotonic() - started
        memory = executable.memory_analysis()
        shadow_memory = {
            name: int(getattr(memory, name))
            for name in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
            if memory is not None and getattr(memory, name, None) is not None
        }
        stages["shadow"] = CompiledCampaignBatch(
            mode="shadow",
            executable=executable,
            arguments=tuple(arguments),
            model_state=state,
            argument_treedef=tree,
            argument_layout=_argument_layout(arguments),
            actual_states=None,
            clean_states=None,
            timings={
                "encode_and_transfer_seconds": shadow_encode_seconds,
                "lower_seconds": lower_seconds,
                "compile_or_cache_load_seconds": compile_seconds,
            },
            memory_analysis=shadow_memory,
        )
        combined, combined_tree = jax.tree.flatten((state, actual, clean, jnp.asarray(noise)))
        timings = {
            name: sum(stage.timings[name] for stage in stages.values())
            for name in ("encode_and_transfer_seconds", "lower_seconds", "compile_or_cache_load_seconds")
        }
        timings.update(
            native_example_execution_for_shadow_abi_seconds=example_seconds,
            stages={name: stage.timings for name, stage in stages.items()},
            execution_scope="four independent native executable calls, then one independent shadow executable",
        )
        return CompiledPairedCampaignBatch(
            policy=self,
            **stages,
            actual_observation=actual,
            clean_observation=clean,
            mode="paired",
            arguments=tuple(combined),
            model_state=state,
            argument_treedef=combined_tree,
            argument_layout=_argument_layout(combined),
            actual_states=stages["reference_actual"].actual_states,
            clean_states=stages["reference_clean"].actual_states,
            timings=timings,
            memory_analysis={
                "separate_stages": {name: stage.memory_analysis for name, stage in stages.items()},
                "scope": "per executable estimates, not a summed peak; actual GPU peak measured separately",
            },
        )

    def materialize_compiled(self, compiled, device_result):
        """Use the same native-dtype host decoding as the ordinary policy."""
        start = time.monotonic()
        result = jax.device_get(device_result)
        transfer_seconds = time.monotonic() - start
        start = time.monotonic()
        if compiled.mode == "paired":
            if not bool(result["free"]["prefix_mask_equal"]):
                raise RuntimeError("Paired inputs have different valid prefix keys")
            result["clean_physical_actions"] = self.decode(result["clean_actions"], compiled.clean_states)
            result["actual_physical_actions"] = self.decode(result["actual_actions"], compiled.actual_states)
        else:
            raw, scalars = result if compiled.mode in ("scalars", "trajectory") else (result, None)
            result = {"raw_actions": np.asarray(raw), "actions": self.decode(raw, compiled.actual_states)}
            if scalars is not None:
                result["layer_rms"] = np.moveaxis(np.asarray(scalars["layer_rms"]), 2, 0)
                result["denoise_times"] = np.asarray(scalars["time"])
                if compiled.mode == "trajectory":
                    result["native_path"] = scalars
        return result, {
            "device_to_host_seconds": transfer_seconds,
            "decode_seconds": time.monotonic() - start,
        }

    def rebind_compiled(self, compiled, observations, prompts, noise, *, clean_observations=None):
        """Re-encode requests while reusing the exact executable and shape."""
        actual, actual_batch = self.encode(observations, prompts)
        noise = jnp.asarray(noise)
        if noise.shape != (len(observations), 10, 32):
            raise ValueError("Each request must carry its own 10x32 noise")
        state = compiled.model_state
        clean_states = None
        if isinstance(compiled, CompiledPairedCampaignBatch):
            if clean_observations is None or len(clean_observations) != len(observations):
                raise ValueError("Compiled paired inference requires matched clean observations")
            clean, clean_batch = self.encode(clean_observations, prompts)
            arguments = (state, actual, clean, noise)
            flat, tree = jax.tree.flatten(arguments)
            if tree != compiled.argument_treedef:
                raise ValueError("Compiled request pytree structure changed")
            if _argument_layout(flat) != compiled.argument_layout:
                raise ValueError("Compiled request leaf shape/dtype changed")
            stages = {
                name: self.rebind_compiled(
                    getattr(compiled, name),
                    clean_observations if name.endswith("clean") else observations,
                    prompts,
                    noise,
                )
                for name in ("reference_actual", "reference_clean", "trajectory_actual", "trajectory_clean")
            }
            return dataclasses.replace(
                compiled,
                **stages,
                actual_observation=actual,
                clean_observation=clean,
                arguments=tuple(flat),
                actual_states=actual_batch["state"],
                clean_states=clean_batch["state"],
            )
        if compiled.mode == "baseline":
            arguments = (state, jax.random.key(0), actual, noise)
        elif compiled.mode in ("scalars", "trajectory"):
            arguments = (state, actual, noise)
        else:
            if clean_observations is None or len(clean_observations) != len(observations):
                raise ValueError("Compiled paired inference requires matched clean observations")
            clean, clean_batch = self.encode(clean_observations, prompts)
            arguments = (state, clean, actual, noise)
            clean_states = clean_batch["state"]
        jax.block_until_ready((arguments, actual_batch, clean_states))
        flat_arguments, treedef = jax.tree.flatten(arguments)
        if treedef != compiled.argument_treedef:
            raise ValueError("Compiled request pytree structure changed; a fixed executable cannot rebind it")
        if _argument_layout(flat_arguments) != compiled.argument_layout:
            raise ValueError("Compiled request leaf shape/dtype changed; a fixed executable cannot rebind it")
        reference = (
            None
            if compiled.canonical_reference is None
            else self.rebind_compiled(compiled.canonical_reference, observations, prompts, noise)
        )
        return dataclasses.replace(
            compiled,
            arguments=tuple(flat_arguments),
            actual_states=actual_batch["state"],
            clean_states=clean_states,
            canonical_reference=reference,
        )

    def infer_batch(self, observations, prompts, noise, *, capture_scalars=True):
        obs, batch = self.encode(observations, prompts)
        noise = jnp.asarray(noise)
        if noise.shape != (len(observations), 10, 32):
            raise ValueError("Each request must carry its own 10x32 noise")
        start = time.monotonic()
        raw = self.sample(jax.random.key(0), obs, noise=noise, num_steps=10)
        raw.block_until_ready()
        if capture_scalars:
            candidate, scalars = self.sample_scalars(obs, noise)
            jax.block_until_ready((candidate, scalars))
            require_native_actions("scalar", raw, candidate)
        seconds = time.monotonic() - start
        result = {"actions": self.decode(raw, batch["state"]), "raw_actions": np.asarray(raw), "seconds": seconds}
        if capture_scalars:
            result["layer_rms"] = np.moveaxis(np.asarray(scalars["layer_rms"]), 2, 0)
            result["denoise_times"] = np.asarray(scalars["time"])
        return result

    def native_trajectory_batch(self, observations, prompts, noise):
        """Expose genuine recorder candidates for independent exact qualification.

        Callers must retain mismatch actions/scalars, not dump full hidden paths
        for every padding case. No canonical action is substituted here.
        """
        obs, batch = self.encode(observations, prompts)
        noise = jnp.asarray(noise)
        if noise.shape != (len(observations), 10, 32):
            raise ValueError("Each request must carry its own 10x32 noise")
        raw, path = self.sample_trajectory(obs, noise)
        jax.block_until_ready((raw, path))
        raw, path = jax.device_get((raw, path))
        return {
            "raw_actions": raw,
            "actions": self.decode(raw, batch["state"]),
            "layer_rms": np.moveaxis(np.asarray(path["layer_rms"]), 2, 0),
            "denoise_times": path["time"],
            "native_path": path,
        }

    def paired_batch(
        self, clean_observations, actual_observations, prompts, noise, *, save_visual=False, save_attention=False
    ):
        if len(clean_observations) != len(actual_observations):
            raise ValueError("Unpaired observations")
        clean, clean_batch = self.encode(clean_observations, prompts)
        actual, actual_batch = self.encode(actual_observations, prompts)
        noise = jnp.asarray(noise)
        if noise.shape != (len(actual_observations), 10, 32):
            raise ValueError("Each pair must carry shared 10x32 noise")
        start = time.monotonic()
        # Independent host/JIT boundaries are a scientific contract. Retaining
        # diagnostic outputs must never let CSE/fusion change the control path.
        actual_reference = self.sample(jax.random.key(0), actual, noise=noise, num_steps=10)
        actual_reference.block_until_ready()
        clean_reference = self.sample(jax.random.key(0), clean, noise=noise, num_steps=10)
        clean_reference.block_until_ready()
        actual_candidate, actual_path = self.sample_trajectory(actual, noise)
        jax.block_until_ready((actual_candidate, actual_path))
        clean_candidate, clean_path = self.sample_trajectory(clean, noise)
        jax.block_until_ready((clean_candidate, clean_path))
        require_native_actions("actual", actual_reference, actual_candidate)
        require_native_actions("clean", clean_reference, clean_candidate)
        diagnostic = self.shadow_pair(
            clean, actual, clean_path, actual_path, save_visual=save_visual, save_attention=save_attention
        )
        jax.block_until_ready(diagnostic)
        result = attach_native_sources(
            diagnostic, clean_reference, actual_reference, clean_candidate, actual_candidate, clean_path, actual_path
        )
        seconds = time.monotonic() - start
        result = jax.device_get(result)
        if not bool(result["free"]["prefix_mask_equal"]):
            raise RuntimeError("Paired inputs have different valid prefix keys")
        result["clean_physical_actions"] = self.decode(result["clean_actions"], clean_batch["state"])
        result["actual_physical_actions"] = self.decode(result["actual_actions"], actual_batch["state"])
        result["seconds"] = seconds
        return result
