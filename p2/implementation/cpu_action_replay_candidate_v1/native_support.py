"""Native adapter additions; target modules are imported only after direct approval."""
from contextlib import ExitStack
import copy
import importlib
from pathlib import Path
import sys
import numpy as np
from campaign_core import ROOT, ROLES, digest, need
from failure_guards import bind_control_source
from native_budget import Budget, install, verify_clock


class NativeSupport:
    synthetic = False

    def _init_support(self, context):
        self.role = context.get("role")
        self.budget = context.get("native_budget")
        self.support_stack = ExitStack()
        self.compiled_parts = {}
        self.last_terminal = False
        self.live_clock = None

    def _require_native_authority(self):
        authority = self.context.get("authority", {})
        need(authority.get("direct_human_approved") is True and
             authority.get("scope") == "q11_cached_command_physical_pilot" and
             authority.get("root_id") == ROOT and authority.get("query_index") == 11 and
             authority.get("maximum_explicit_env_steps") == 350 and
             authority.get("new_model_calls") == authority.get("GPU") == 0,
             "new_direct_environment_authority_required")
        need(self.role in ROLES and isinstance(self.budget, Budget), "bound_role_and_native_budget")
        need(self.context["resources"] == {"CPU": 8, "GPU": 0, "memory_GiB": 32},
             "exact_CPU_resources")
        self.budget.check()

    def _install_support(self, native, check):
        check()
        cls = native["wrapper_class"]
        module = sys.modules[cls.__module__]
        bind_control_source(Path(module.__file__))
        mujoco = importlib.import_module("mujoco")
        pdb = importlib.import_module("pdb")
        hooks = install(cls, mujoco, pdb, self.budget, role=self.role, emit=self.emit)
        self.support_stack.callback(hooks.close)
        self.emit("BOUNDED_NATIVE_HOOKS_INSTALLED",
                  {"role": self.role, "failure_path_only_reset_change": True,
                   "native_budget": self.budget.snapshot(), "no_target_state_or_RNG_injection": True})

    def _capture_compiled(self, event, metadata, values):
        if event == "SETUP_COMPILED_ARRAYS" and metadata["role"] == "current_clean_parent_after_exact_selected_init":
            name = metadata["name"]
            need(name not in self.compiled_parts, "unique_compiled_snapshot_part")
            self.compiled_parts[name] = digest(values)

    def _validate_live_parent(self):
        self.live_clock = verify_clock(self.parent.env)
        physical = self.parent.env.env
        need(len(physical.robots) == 1, "one_actual_robot")
        robot = physical.robots[0]
        controller = robot.controller
        need(type(robot).__name__ == "SingleArm" and
             type(robot.robot_model).__name__ == "OnTheGroundPanda" and
             type(robot.gripper).__name__ == "PandaGripper" and
             type(controller).__name__ == "OperationalSpaceController",
             "actual_robot_controller_gripper_classes")
        need(controller.control_dim == 6 and controller.impedance_mode == "fixed" and
             controller.use_delta is True and controller.use_ori is True and
             controller.interpolator_pos is None and controller.interpolator_ori is None and
             controller.position_limits is None and controller.orientation_limits is None,
             "actual_fixed_delta_OSC_configuration")
        scale = np.array([.05, .05, .05, .5, .5, .5])
        need(np.array_equal(controller.input_min, np.full(6, -1.)) and
             np.array_equal(controller.input_max, np.ones(6)) and
             np.array_equal(controller.output_max, scale) and
             np.array_equal(controller.output_min, -scale), "actual_OSC_clip_scale")
        need(self.compiled_parts, "actual_compiled_model_snapshot_captured")

    def critical_state(self):
        self.budget.check()
        self._validate_live_parent()
        physical = self.parent.env.env
        robot = physical.robots[0]
        current = np.array(robot.gripper.current_action, copy=True)
        need(current.shape == (2,) and current.dtype == np.float64 and np.isfinite(current).all(),
             "actual_gripper_memory_after_dummy_prefix")
        env = {k: copy.deepcopy(getattr(physical, k)) for k in
               ("timestep", "cur_time", "done", "model_timestep", "control_timestep", "deterministic_reset")}
        actuator_ids = list(robot._ref_joint_gripper_actuator_indexes)
        return {"gripper_current_action": current, "environment": env,
                "controller": {k: v for k, v in self.group_ids().items()
                               if k.startswith(("controller.", "robot."))},
                "compiled_model": dict(self.compiled_parts),
                "gripper_actuator_indices": actuator_ids,
                "gripper_actuator_ranges": np.array(self.sim.model.actuator_ctrlrange[actuator_ids], copy=True),
                "clock_binding": dict(self.live_clock)}

    def bind_cached_commands(self, actions, raw32, q, key):
        self.budget.check()
        need(q == 11 and key in ("N", "S", "R") and self.role != "source_qualification",
             "cached_commands_only_target_q11")
        need(actions.shape == (1, 10, 7) and actions.dtype == np.float64 and
             raw32.shape == (1, 10, 32) and raw32.dtype == np.float32 and
             np.isfinite(actions).all() and np.isfinite(raw32).all(), "cached_native_array_ABI")
        self.collector.cursor.bind_generated({"actions": actions, "raw_actions": raw32}, query_index=q)
        self.collector.pending_query = None
        self.emit("CACHED_COMMANDS_BOUND", {"query_index": q, "role": self.role, "cached_arm": key,
                  "origin": "qualified_saved_native_command", "model_calls": self.model_calls,
                  "issued_slots": [0, 1, 2, 3, 4]})

    def physical_outputs(self):
        self.budget.check()
        obs = self.state["observation"]
        images = {"agentview_image", "robot0_eye_in_hand_image", "image-state"}
        observed = {}
        for name, value in obs.items():
            if name in images:
                continue
            array = np.asarray(value)
            need(array.dtype.kind in "biuf" and array.size <= 4096 and
                 np.isfinite(array).all(), "bounded_nonimage_physical_observation")
            observed[name] = np.array(array, copy=True)
        need({"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"} <= set(observed),
             "required_physical_observations")
        ncon = int(self.sim.data.ncon)
        need(0 <= ncon <= 4096, "bounded_native_contacts")
        contacts = []
        for i in range(ncon):
            contact = self.sim.data.contact[i]
            contacts.append({"geom1": int(contact.geom1), "geom2": int(contact.geom2),
                             "geom1_name": self.sim.model.geom_id2name(contact.geom1),
                             "geom2_name": self.sim.model.geom_id2name(contact.geom2),
                             "distance": float(contact.dist)})
        robot = self.parent.env.env.robots[0]
        return {"source": "actual_returned_observations_and_native_state_no_forced_refresh",
                "observation_units": "original_simulator_observation_units_not_controller_input",
                "returned_nonimage_observations": observed,
                "actuator_ctrl": np.array(self.sim.data.ctrl, copy=True),
                "gripper_actuator_ctrl": np.array(
                    self.sim.data.ctrl[robot._ref_joint_gripper_actuator_indexes], copy=True),
                "gripper_current_action": np.array(robot.gripper.current_action, copy=True),
                "contacts": contacts, "contact_count": ncon,
                "contact_forces_measured": False}

    def _close_support(self):
        self.support_stack.close()
