"""Passive source / compiled-model diagnostics; never a counterfactual proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


FIELDS = {
    "body": "body_parentid body_pos body_quat body_mass body_inertia body_ipos body_iquat",
    "joint": "jnt_type jnt_bodyid jnt_qposadr jnt_dofadr jnt_pos jnt_axis jnt_limited jnt_range jnt_stiffness",
    "dof": "dof_bodyid dof_jntid dof_damping dof_frictionloss dof_armature",
    "geom": "geom_type geom_bodyid geom_pos geom_quat geom_size geom_contype geom_conaffinity geom_friction geom_solimp geom_solref geom_matid",
    "actuator": "actuator_trnid actuator_trntype actuator_gear actuator_ctrlrange actuator_ctrllimited actuator_gainprm actuator_biasprm actuator_dynprm",
    "camera": "cam_bodyid cam_pos cam_quat cam_fovy",
    "light": "light_bodyid light_pos light_dir light_diffuse light_ambient light_specular light_active light_directional light_castshadow",
    "material": "mat_texid mat_texrepeat mat_texuniform mat_rgba mat_shininess mat_specular mat_reflectance",
    "texture": "tex_type tex_height tex_width tex_adr",
}
COUNTS = {"body": "nbody", "joint": "njnt", "dof": "nv", "geom": "ngeom",
          "actuator": "nu", "camera": "ncam", "light": "nlight", "material": "nmat", "texture": "ntex"}
ALIASES = {"material": ("mat_id2name", "material_id2name"), "texture": ("tex_id2name", "texture_id2name")}


def file_identity(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("regular source file required")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def texture_references(xml, xml_path=None):
    """A relative final-XML ref without a known origin remains unresolved."""
    root = ET.fromstring(xml)
    rows = []
    compiler = root.find("compiler")
    texture_dir = compiler.get("texturedir") if compiler is not None else None
    for node in root.findall(".//texture"):
        row = {"attributes": dict(node.attrib), "texture_dir": texture_dir,
               "resolved_path": None, "identity": None, "resolution": "no_file_attribute"}
        reference = node.get("file")
        if reference:
            rel = Path(reference)
            if rel.is_absolute():
                target, basis = rel, "absolute_final_reference"
            elif xml_path is not None:
                base = Path(xml_path).resolve().parent
                target, basis = base / (texture_dir or "") / rel, "source_XML_directory_and_compiler_texturedir"
            else:
                target, basis = None, "relative_final_XML_origin_unknown"
            row["resolution"] = basis
            if target is not None:
                target = target.resolve()
                row["resolved_path"] = str(target)
                if target.is_file() and not target.is_symlink():
                    row["identity"] = file_identity(target)
                else:
                    row["resolution"] += ":missing_or_nonregular"
        rows.append(row)
    return rows


def snapshot(adapter, save_arrays, save_text, *, np):
    physical = adapter.env.env
    model = physical.sim.model
    arrays, missing, names, dimensions = {}, [], {}, {}
    for entity, fields in FIELDS.items():
        count = getattr(model, COUNTS[entity], None)
        dimensions[entity] = int(count) if count is not None else None
        methods = ALIASES.get(entity, (entity + "_id2name",))
        name_method = next((getattr(model, n) for n in methods if callable(getattr(model, n, None))), None)
        names[entity] = [{"id": i, "name": name_method(i) if name_method else None}
                         for i in range(int(count or 0))]
        for key in fields.split():
            if hasattr(model, key):
                a = np.asarray(getattr(model, key))
                if a.dtype.hasobject:
                    raise TypeError("compiled field object dtype")
                arrays[key] = np.array(a, copy=True)
            else:
                missing.append(key)
    option = getattr(model, "opt", None)
    for key in ("timestep", "gravity", "integrator", "solver", "iterations", "tolerance", "cone", "jacobian"):
        if option is not None and hasattr(option, key):
            arrays["opt_" + key] = np.array(getattr(option, key), copy=True)
        else:
            missing.append("opt_" + key)
    compiled = save_arrays("compiled-model-fields", arrays)
    problem = json.loads(canonical(physical.parsed_problem))
    # The original parsed tree/entities are retained; this is not a new semantic parser.
    source = {}
    for role, key in (("BDDL", "resolved_BDDL"), ("scene", "scene")):
        path = Path(adapter.source[key]["path"])
        source[role] = {"path": str(path), **file_identity(path)}
        if role == "scene":
            text = path.read_text()
            source[role]["texture_references"] = texture_references(text, path)
    final_xml = None
    for role, owner in (("physical.model.get_xml", getattr(physical, "model", None)),
                        ("physical.sim.model.get_xml", model)):
        if owner is not None and callable(getattr(owner, "get_xml", None)):
            text = owner.get_xml()
            if not isinstance(text, str):
                raise TypeError("model XML must be text")
            final_xml = {"origin": role, "saved": save_text("final-model.xml", text),
                         "texture_references": texture_references(text)}
            break
    return {"schema": "plus-passive-runtime-model-snapshot-v1", "source": source,
            "parsed_problem": problem, "parsed_problem_sha256": hashlib.sha256(canonical(problem).encode()).hexdigest(),
            "named_compiled_entities": names, "dimensions": dimensions,
            "compiled_fields": compiled, "compiled_missing_fields": missing,
            "final_model_XML": final_xml, "texture_bytes_in_compiled_model": "not_copied",
            "controller_and_observable_replay_complete": False,
            "clean_variant_goal_or_physics_equivalence": "not_tested"}


def compare_snapshots(left, right):
    """Untapped/tapped only; ignore distinct output file locations."""
    def content(value):
        if isinstance(value, dict):
            return {k: content(v) for k, v in value.items() if k not in ("saved_path",)}
        if isinstance(value, list):
            return [content(v) for v in value]
        return value
    result = {}
    for key in ("source", "parsed_problem", "named_compiled_entities", "dimensions",
                "compiled_fields", "compiled_missing_fields", "final_model_XML"):
        # Compare native array identities, not zip container timestamps.
        if key == "compiled_fields":
            result[key] = left[key]["arrays"] == right[key]["arrays"]
        else:
            result[key] = content(left[key]) == content(right[key])
    return {"comparison": "same_official_case_untapped_vs_tapped_only", "checks": result,
            "exact_snapshots": all(result.values()), "clean_variant_physics_equivalence": "not_tested"}
