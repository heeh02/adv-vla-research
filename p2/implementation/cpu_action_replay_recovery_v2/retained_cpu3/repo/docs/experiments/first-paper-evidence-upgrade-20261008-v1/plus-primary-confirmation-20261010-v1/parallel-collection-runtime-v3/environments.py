"""Sequential real wrappers: save variant q0 evidence before creating base."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import numpy as np
import orchestrator as O
import plus_contract as C
from official_environment import OfficialPlusEnvironment
from canonical_step_capture import StepCapture, rng_copy, rng_arrays
import runtime_snapshot as S


def exact_source(root, ref):
    relative = ref['relative_to_plus_root']
    C.need(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'frozen reference relative path')
    path = (Path(root) / relative).resolve(strict=True)
    C.need(path.is_relative_to(Path(root).resolve()), 'reference source escaped Plus root')
    C.exact(path, ref)
    return path


def matched_base_prompt(record, plus_root, runtime_manifest, *, parser_module=None):
    """Use the exact official producer consumed by ControlEnv, not lexical text."""
    reference = record.get('reference_contract')
    C.need(reference is not None, 'explicit_base_BDDL_prompt_source_required')
    root = Path(plus_root).resolve(strict=True)
    bddl = exact_source(root, reference['base_bddl'])
    parser = parser_module or importlib.import_module('libero.libero.envs.bddl_utils')
    parser_path = Path(parser.__file__).resolve(strict=True)
    C.need(parser_path.is_relative_to(root), 'official_BDDL_language_parser_origin')
    member = 'official/' + str(parser_path.relative_to(root))
    C.need(member in runtime_manifest, 'official_BDDL_language_parser_manifest_identity')
    C.exact(parser_path, runtime_manifest[member])
    value = parser.get_problem_info(str(bddl))
    prompt = value['language_instruction']
    C.need(type(prompt) is str and bool(prompt), 'actual_base_parser_language_required')
    lexical = record.get('matched_prompt')
    proof = {'schema': 'plus-matched-base-prompt-v2', 'actual_prompt': prompt,
             'actual_prompt_utf8_sha256': hashlib.sha256(prompt.encode('utf8')).hexdigest(),
             'source': 'official_bddl_utils.get_problem_info(base_BDDL).language_instruction',
             'base_BDDL': {'path': str(bddl), **C.identity(bddl)},
             'official_parser': {'path': str(parser_path), **C.identity(parser_path)},
             'frame_lexical_prompt_metadata': lexical, 'frame_lexical_exact_equal': lexical == prompt,
             'normalization_implemented_by_this_worker': False,
             'model_token_IDs_source': 'q0 recorder actual encoded and consumed model-input NPZ; never inferred from text'}
    return prompt, proof


def verify_preinit_compiled(reference_model, reference_xml, saved):
    """Compare the live base BEFORE init with the closed variant's q0 receipt."""
    C.need(saved['schema'] == 'plus-passive-runtime-model-snapshot-v1', 'saved_variant_compiled_snapshot_required')
    for entity in ('body', 'joint', 'dof', 'geom', 'actuator'):
        count_name = S.COUNTS[entity]
        count = getattr(reference_model, count_name)
        C.need(count == saved['dimensions'][entity], 'reference compiled count before init')
        method = entity + '_id2name'
        if callable(getattr(reference_model, method, None)):
            names = [{'id': i, 'name': getattr(reference_model, method)(i)} for i in range(count)]
            C.need(names == saved['named_compiled_entities'][entity], 'reference named address before init')
        for field in S.FIELDS[entity].split():
            if field == 'geom_matid': continue
            C.need(hasattr(reference_model, field) and field in saved['compiled_fields']['fields'], 'reference compiled field missing before init')
            actual = O.array_id(np.asarray(getattr(reference_model, field)))
            frozen = saved['compiled_fields']['fields'][field]
            C.need(all(frozen.get(key) == value for key, value in actual.items()),
                   'reference compiled physics before init')
    C.need(physics_xml(reference_xml) == saved['physics_XML_projection'],
           'reference physics XML before init')


class BaseReferenceEnvironment(OfficialPlusEnvironment):
    def __init__(self, classification, *, plus_root, reference, init_reader, runtime_manifest, variant_preinit_snapshot):
        super().__init__(classification, protocol_mode='released_task_language', init_reader=init_reader)
        self.plus_root, self.reference = Path(plus_root), reference
        self.runtime_manifest = runtime_manifest
        self.variant_preinit_snapshot = copy.deepcopy(variant_preinit_snapshot)

    def reset(self, case):
        C.validate_case(case, self.classification)
        self.close()
        np.random.seed(case['env_seed']); random.seed(case['env_seed'])
        bench = self.registry[case['suite']](task_order_index=0)
        task = bench.get_task(case['task_id'])
        C.need(task.name == case['task_name'], 'reference init task identity')
        bddl = exact_source(self.plus_root, self.reference['base_bddl'])
        self.env = self.env_factory(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        self.env.seed(case['env_seed']); self.env.reset()
        scene = Path(self.env.env._arena_xml).resolve()
        C.need(scene.is_relative_to(self.plus_root.resolve()), 'actual base scene escaped Plus root')
        scene_member = 'official/' + str(scene.relative_to(self.plus_root.resolve()))
        C.need(scene_member in self.runtime_manifest, 'actual base scene absent from original runtime manifest')
        C.exact(scene, self.runtime_manifest[scene_member])
        if 'base_scene' in self.reference:
            C.need(scene == exact_source(self.plus_root, self.reference['base_scene']), 'explicit actual base scene differs')
        # Preserve the v2 BEFORE-init compiled gate without a live variant.
        # P.Store recorded these typed field identities and the XML projection
        # while the variant was active, before its Y rollout and close.
        verify_preinit_compiled(self.env.env.sim.model, self.env.env.model.get_xml(),
                                self.variant_preinit_snapshot)
        states, proof = self.init_reader.select(self.benchmark_module, bench, case['task_id'], case['suite'])
        initial = np.array(states[0], copy=True)
        C.need(initial.dtype.str == proof['init0_dtype'] and list(initial.shape) == proof['init0_shape']
               and hashlib.sha256(initial.tobytes(order='C')).hexdigest() == proof['init0_raw_sha256'], 'reference actual native init')
        physical = self.env.env
        C.need(Path(physical.bddl_file_name).resolve() == bddl and Path(physical._arena_xml).resolve() == scene, 'actual reference BDDL scene differs')
        C.need(self.env.noise == 0, 'base reference wrapper must use noise0')
        obs = self.env.set_init_state(initial)
        # The paired protocol holds actual prompt fixed; this is not original F0.
        self.prompt = self.env.language_instruction
        self.source = {'role': 'explicit_base_reference_new_paired_protocol', 'official_commit': C.COMMIT,
             'classification_sha256': C.CLASS_SHA, 'source_variant_case': copy.deepcopy(case),
             'reference_contract_id': self.reference['id'], 'policy_prompt': self.prompt,
             'released_task_language': task.language, 'resolved_BDDL_language': self.env.language_instruction,
             'resolved_BDDL': {'path': str(bddl), **C.identity(bddl)}, 'scene': {'path': str(scene), **C.identity(scene)},
             'explicit_init_contract': proof, 'native_init0': {'dtype': initial.dtype.str, 'shape': list(initial.shape),
                'sha256_raw': hashlib.sha256(initial.tobytes(order='C')).hexdigest()},
             'noise_id': 0, 'parsed_problem_sha256': C.digest(physical.parsed_problem),
             'goal_predicate': 'official_ControlEnv.check_success', 'not_original_F0': True}
        return obs


def restore_rng(state):
    np.random.set_state(copy.deepcopy(state['numpy']))
    random.setstate(copy.deepcopy(state['python']))


def initialize(env, case, store, prefix, check, *, sensor_tap=False):
    check(); env.reset(case)
    capture = None
    for index in range(10):
        check()
        if sensor_tap and index == 9:
            capture = StepCapture(env.env, lambda: env.env.get_sim_state())
            obs, done = capture.step([0.0]*6+[-1.0], invoke=env.step)
        else:
            obs, done = env.step([0.0]*6+[-1.0])
    state = {'env': env, 'observation': obs, 'rng': rng_copy(),
             'physical': np.array(env.env.get_sim_state(), copy=True)}
    store.arrays(prefix + '/q0-native-input.npz', {**obs, 'physical_sim_state': state['physical'], **rng_arrays(state['rng'])}, 'real_wrapper_q0_after10dummy')
    store.json(prefix + '/SOURCE-v1.json', env.source)
    if capture is not None:
        store.arrays(prefix + '/q0-preSensor.npz', {**capture.record['pre_sensor']['observation'],
             'physical_sim_state': capture.record['pre_sensor']['physics'], **rng_arrays(capture.record['pre_sensor']['rng'])}, 'actual_underlying_step_before_original_sensor')
        store.json(prefix + '/SENSOR-TAP-v1.json', capture.receipt())
    return state


def physics_xml(text):
    root = ET.fromstring(text)
    root.attrib.pop('model', None)
    visual_tags = {'visual', 'camera', 'light', 'texture', 'material'}
    def strip(node):
        for child in list(node):
            if child.tag in visual_tags:
                node.remove(child)
            else:
                strip(child)
        for key in ('rgba', 'material'):
            node.attrib.pop(key, None)
        if node.tag == 'compiler':
            node.attrib.pop('texturedir', None)
        node.attrib = dict(sorted(node.attrib.items()))
        node.text = node.text.strip() if node.text and node.text.strip() else None
        node.tail = None
    strip(root)
    return ET.tostring(root, encoding='unicode')


def snapshot(state, store, prefix):
    def text(name, value):
        path = store.path / prefix / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream: stream.write(value)
        return {'saved_path': str(path.relative_to(store.path)), **C.identity(path)}
    result = S.snapshot(state['env'], lambda name, arrays: store.arrays(prefix+'/'+name+'.npz', arrays, 'compiled_model_diagnostic'), text, np=np)
    store.json(prefix + '/MODEL-SNAPSHOT-v1.json', result)
    model = state['env'].env.env.model
    result['physics_XML_projection'] = physics_xml(model.get_xml())
    return result


def qualify_pair(ref, var, ref_snapshot, var_snapshot, record):
    # Compiled indexed fields plus exact named correspondence prevent flat-state
    # equality from being treated as model equality. XML adds constraints/assets.
    checks = {'same_actual_prompt': ref['env'].prompt == var['prompt'],
              'same_q0_physical_bytes': ref['physical'].dtype == var['physical'].dtype
                  and ref['physical'].shape == var['physical'].shape
                  and ref['physical'].tobytes() == var['physical'].tobytes(),
              'physics_XML_except_declared_render_fields': ref_snapshot['physics_XML_projection'] == var_snapshot['physics_XML_projection']}
    for entity in ('body', 'joint', 'dof', 'geom', 'actuator'):
        checks['named_' + entity] = ref_snapshot['named_compiled_entities'][entity] == var_snapshot['named_compiled_entities'][entity]
        for field in S.FIELDS[entity].split():
            if field == 'geom_matid': continue
            a = ref_snapshot['compiled_fields']['fields'].get(field)
            b = var_snapshot['compiled_fields']['fields'].get(field)
            checks['compiled_' + field] = a is not None and a == b
    for key in ('opt_timestep', 'opt_gravity', 'opt_integrator'):
        a = ref_snapshot['compiled_fields']['fields'].get(key); b = var_snapshot['compiled_fields']['fields'].get(key)
        checks[key] = a is not None and a == b
    # Preserve the actual parsed task tree/entity/init/region values; ignore only
    # parser's class/language metadata, never infer a base task by suffix.
    rp, vp = ref_snapshot['parsed_problem'], var_snapshot['parsed_problem']
    keys = sorted((set(rp) | set(vp)) - {'problem_name', 'language', 'scene_properties'})
    checks['actual_goal_and_entity_tree'] = bool(keys) and all(rp.get(k) == vp.get(k) for k in keys)
    checks['explicit_goal_binding'] = bool(record.get('goal_binding_id')) and record['reference_contract']['goal_binding_id'] == record['goal_binding_id']
    return {'schema': 'plus-runtime-base-variant-q0-pair-v1', 'checks': checks, 'qualified': all(checks.values()),
            'pair_id': record['identity']['pair_id'], 'goal_binding_id': record.get('goal_binding_id'),
            'source': 'two_independent_real_wrappers_same_native_init_selector',
            'late_queries_same_state_claim': False, 'full_controller_replay_claim': False}
