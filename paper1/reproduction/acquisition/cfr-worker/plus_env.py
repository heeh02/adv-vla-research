"""LIBERO-Plus treated (E_T) and base (E_C) environments for one unit (TECH-CONTRACT-v1 2-3).

E_T is the unchanged adapter ``OfficialPlusEnvironment`` (A/official_environment.py:12-110,
packaged in the scientific source). E_C (``BaseEnvironment``) adapts its ``reset`` the way
P/parallel-collection-runtime-v6/environments.py:144-194 (``BaseReferenceEnvironment``) does:
the base BDDL is the variant name before ``_view_`` plus ``.bddl`` (the wrapper's own rule,
U/libero/libero/envs/env_wrapper.py:207-231), same env seed, same trusted official init0,
noise id 0, and the variant's prompt. The physics pair gate adapts environments.py:223-241
(``physics_xml``) and :259-290 (``qualify_pair``) with an explicit compiled-field list.
Cloud Linux only; local tests inject fakes.
"""
# ruff: noqa: PLC0415

from __future__ import annotations

import hashlib
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np

from .common import TechnicalFailure, array_id, need

DUMMY = [0.0] * 6 + [-1.0]
WARMUP_STEPS = 10
PAIR_FIELDS = ("body_pos", "body_quat", "body_mass", "body_inertia", "body_parentid", "jnt_type", "jnt_qposadr",
               "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss", "geom_type", "geom_size", "geom_pos",
               "geom_quat", "geom_friction", "geom_bodyid", "geom_contype", "geom_conaffinity", "actuator_gear",
               "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm", "eq_data", "opt_timestep")


def physics_xml(text):
    """Copy of environments.py:223-241: strip render-only tags/attributes, canonical attributes."""
    root = ET.fromstring(text)
    root.attrib.pop("model", None)
    visual_tags = {"visual", "camera", "light", "texture", "material"}

    def strip(node):
        for child in list(node):
            if child.tag in visual_tags:
                node.remove(child)
            else:
                strip(child)
        for key in ("rgba", "material"):
            node.attrib.pop(key, None)
        if node.tag == "compiler":
            node.attrib.pop("texturedir", None)
        node.attrib = dict(sorted(node.attrib.items()))
        node.text = node.text.strip() if node.text and node.text.strip() else None
        node.tail = None

    strip(root)
    return ET.tostring(root, encoding="unicode")


def variant_case(unit, classification):
    """Official case for a UNIT-MAP row: the variant is resolved by its exact BDDL name."""
    import plus_contract as C
    suite = unit["suite"]
    names = [entry["name"] for entry in classification[suite]]
    need(names.count(unit["variant_name"]) == 1, "unit_variant_name_not_unique_in_official_catalog")
    task_id = names.index(unit["variant_name"])
    case = C.make_case(suite, task_id, classification)
    if unit.get("classification_id") is not None:
        need(int(unit["classification_id"]) == case["classification_id"], "unit_classification_id_differs")
    need(case["category"] == unit["family"], "unit_family_differs_from_official_category")
    return case


def base_bddl_name(variant_bddl_name):
    need("_view_" in variant_bddl_name and "_initstate_" in variant_bddl_name, "plus_variant_bddl_suffix_required")
    return variant_bddl_name.split("_view_")[0] + ".bddl"


def make_base_environment_class():
    from official_environment import OfficialPlusEnvironment
    import plus_contract as C

    class BaseEnvironment(OfficialPlusEnvironment):
        """Clean base wrapper for the N_C arm and the dual-render checks."""

        def reset(self, case):
            C.validate_case(case, self.classification)
            self.close()
            np.random.seed(case["env_seed"])
            random.seed(case["env_seed"])
            bench = self.registry[case["suite"]](task_order_index=0)
            task = bench.get_task(case["task_id"])
            need(task.name == case["task_name"], "base_reference_task_identity")
            requested = Path(self.get_path("bddl_files")) / task.problem_folder / task.bddl_file
            base = requested.with_name(base_bddl_name(requested.name))
            need(base.is_file() and not base.is_symlink(), "base_BDDL_missing")
            self.env = self.env_factory(bddl_file_name=str(base), camera_heights=256, camera_widths=256)
            self.env.seed(case["env_seed"])
            self.env.reset()
            states, proof = self.init_reader.select(self.benchmark_module, bench, case["task_id"], case["suite"])
            initial = np.array(states[0], copy=True)
            need(self.env.noise == 0, "base_wrapper_must_use_noise0")
            obs = self.env.set_init_state(initial)
            # Same released task language as the variant arm (OfficialPlusEnvironment protocol).
            self.prompt = task.language
            self.source = {"role": "cf_repair_base_reference", "variant_case": dict(case),
                           "base_BDDL": {"path": str(base), **C.identity(base)},
                           "native_init0": array_id(initial), "explicit_init_contract": proof,
                           "noise_id": 0, "prompt_source": "official_Benchmark.get_task.language (variant task)",
                           "parsed_problem_sha256": C.digest(self.env.env.parsed_problem)}
            return obs

    return BaseEnvironment


def image_convention():
    import robosuite.macros as macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    return IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]


def warmup(env, *, tapped_step=None):
    """Official 10 dummy steps after reset; ``tapped_step(action)`` runs the 10th (index 9)."""
    obs = None
    for index in range(WARMUP_STEPS):
        if index == WARMUP_STEPS - 1 and tapped_step is not None:
            obs, done = tapped_step(DUMMY)
        else:
            obs, done = env.step(DUMMY)
        need(type(done) in (bool, np.bool_), "warmup_terminal_must_be_boolean")
    return obs


def compiled_physics_digest(model):
    rows = {}
    for name in PAIR_FIELDS:
        if hasattr(model, name):
            rows[name] = array_id(np.asarray(getattr(model, name)))["sha256"]
    return rows


def pair_gate(variant_env, base_env):
    """Physics equality between E_T and E_C after both official warm-ups (no optics)."""
    vt, bt = variant_env.env, base_env.env
    v_flat = np.asarray(vt.get_sim_state())
    b_flat = np.asarray(bt.get_sim_state())
    v_xml = physics_xml(vt.env.model.get_xml())
    b_xml = physics_xml(bt.env.model.get_xml())
    v_fields = compiled_physics_digest(vt.env.sim.model)
    b_fields = compiled_physics_digest(bt.env.sim.model)
    checks = {"same_prompt": variant_env.prompt == base_env.prompt,
              "same_q0_physical_bytes": v_flat.dtype == b_flat.dtype and v_flat.tobytes() == b_flat.tobytes(),
              "physics_xml_projection_equal": v_xml == b_xml,
              "compiled_fields_equal": v_fields == b_fields,
              "same_parsed_problem": variant_env.source.get("parsed_problem_sha256") == base_env.source.get("parsed_problem_sha256")}
    return {"schema": "cf-repair-pair-gate-v1", "checks": checks, "qualified": all(checks.values()),
            "compared_compiled_fields": sorted(v_fields),
            "differing_compiled_fields": sorted(k for k in set(v_fields) | set(b_fields) if v_fields.get(k) != b_fields.get(k)),
            "physics_xml_sha256": {"variant": hashlib.sha256(v_xml.encode()).hexdigest(),
                                   "base": hashlib.sha256(b_xml.encode()).hexdigest()},
            "late_queries_same_state_claim": False}


def render_base_frame(base_env, physics_flat, *, convention, width=256, height=256):
    """T3(d): set_state -> forward -> render agentview in the separate E_C instance."""
    sim = base_env.env.env.sim
    sim.set_state_from_flattened(np.asarray(physics_flat))
    sim.forward()
    image = sim.render(camera_name="agentview", width=width, height=height, depth=False)
    return np.array(np.asarray(image)[::convention], copy=True)


def render_base_frame_exact_source(base_env, render_data, mujoco_module, *, convention, width=256, height=256):
    """T3(d') diagnostic: copy the render-time MjData arrays, mj_camlight for the base camera, render."""
    sim = base_env.env.env.sim
    raw_model = getattr(sim.model, "_model", sim.model)
    raw_data = getattr(sim.data, "_data", sim.data)
    for name, value in render_data.items():
        target = getattr(raw_data, name)
        if np.asarray(target).shape != np.asarray(value).shape:
            raise TechnicalFailure("render_data_shape_differs:" + name)
        target[...] = value
    mujoco_module.mj_camlight(raw_model, raw_data)
    image = sim.render(camera_name="agentview", width=width, height=height, depth=False)
    return np.array(np.asarray(image)[::convention], copy=True)


def image_stats(reference, candidate):
    a, b = np.asarray(reference).astype(np.int16), np.asarray(candidate).astype(np.int16)
    if a.shape != b.shape:
        return {"comparable": False}
    delta = np.abs(a - b)
    differing = np.count_nonzero(delta)
    changed_pixels = np.any(delta != 0, axis=-1) if delta.ndim == 3 else (delta != 0)
    fraction_pixels = float(np.mean(changed_pixels)) if changed_pixels.size else 0.0
    return {"comparable": True, "exact": differing == 0, "max_abs_u8": int(delta.max(initial=0)),
            "differing_values": int(differing), "fraction_values": float(differing / delta.size),
            "differing_pixels": int(np.count_nonzero(changed_pixels)), "fraction_pixels": fraction_pixels,
            "within_frozen_1lsb_0p1pct": bool(delta.max(initial=0) <= 1 and fraction_pixels <= 0.001)}
