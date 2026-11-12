"""Use the real Plus registry/wrapper; no stock scene or task-range override."""
import random
from pathlib import Path
import sys

import numpy as np

import plus_contract as C
from adapter import array_identity, observation_identity, strict_bool


class OfficialPlusEnvironment:
    def __init__(self, classification, *, protocol_mode='released_task_language', registry=None, env_factory=None, get_path=None, init_reader=None, benchmark_module=None):
        # Dependency injection exists for explicit synthetic tests only. Normal
        # runtime imports happen after runtime.prepare_official checked origins.
        self.real_runtime=registry is None
        self.init_reader=init_reader;self.benchmark_module=benchmark_module
        if registry is None:
            C.need(init_reader is not None,'real Plus requires qualified explicit init contract')
            C.need(sys.platform.startswith('linux'), 'real Plus arrays remain on cloud Linux')
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
            registry, env_factory, get_path = benchmark.get_benchmark_dict(), OffScreenRenderEnv, get_libero_path
            self.benchmark_module=benchmark
        self.registry, self.env_factory, self.get_path = registry, env_factory, get_path
        C.need(protocol_mode in C.PROMPT_MODES, 'named prompt protocol required')
        self.protocol_mode = protocol_mode
        self.classification = classification
        self.env = None
        self.source = None

    def reset(self, case):
        C.validate_case(case, self.classification)
        self.close()
        np.random.seed(case['env_seed'])
        random.seed(case['env_seed'])
        bench = self.registry[case['suite']](task_order_index=0)
        C.need(bench.get_num_tasks() == C.COUNTS[case['suite']], 'actual Plus suite count differs')
        C.need(bench.get_task_names() == [e['name'] for e in self.classification[case['suite']]], 'actual official variant ordering differs')
        task = bench.get_task(case['task_id'])
        C.need(task.name == case['task_name'] and task.problem_folder == case['suite'], 'resolved official variant identity differs')
        requested_bddl = str(Path(self.get_path('bddl_files')) / task.problem_folder / task.bddl_file)
        # Keep every official suffix. The wrapper resolves view/initstate/noise;
        # Path itself is not accepted by its string-membership/parser interface.
        self.env = self.env_factory(bddl_file_name=requested_bddl, camera_heights=256, camera_widths=256)
        self.env.seed(case['env_seed'])
        self.env.reset()
        # Crucial: do not concatenate stock init paths. Official method maps
        # language/view/light/texture variants and layout's libero_newobj path.
        init_proof=None
        if self.init_reader is None:
            C.need(not self.real_runtime,'unbridged real init loading forbidden')
            states=bench.get_task_init_states(case['task_id'])
        else:
            states,init_proof=self.init_reader.select(self.benchmark_module,bench,case['task_id'],case['suite'])
        C.need(len(states) > 0, 'official init0 missing')
        initial = np.array(states[0], copy=True)
        C.need(initial.ndim == 1 and initial.dtype.kind == 'f' and np.isfinite(initial).all(), 'official native init0 invalid')
        if init_proof is not None:
            import hashlib
            C.need(initial.dtype.str==init_proof['init0_dtype'] and list(initial.shape)==init_proof['init0_shape'] and hashlib.sha256(initial.tobytes(order='C')).hexdigest()==init_proof['init0_raw_sha256'],'independent native init0 copy differs')
        obs = self.env.set_init_state(initial)
        released, resolved = task.language, self.env.language_instruction
        C.need(type(released) is str and bool(released) and type(resolved) is str and bool(resolved), 'both actual official language fields required')
        # The second arm is a FULL BDDL correction, not suffix-only stripping:
        # base wording can differ too. Never reconstruct a base prompt by name.
        self.prompt = released if self.protocol_mode == 'released_task_language' else resolved
        if case['category'] == 'Language Instructions':
            C.need(released == resolved, 'official rewritten Language task and BDDL disagree; do not substitute base language')
        C.need(type(self.prompt) is str and bool(self.prompt), 'official task language missing')
        noise = self.env.noise
        C.need(type(noise) is int and 0 <= noise <= 50, 'official wrapper noise id invalid')
        C.need((noise > 0) == (case['category'] == 'Sensor Noise'), 'classification and actual official noise wrapper disagree')
        physical = self.env.env
        resolved_bddl = Path(physical.bddl_file_name)
        scene = Path(physical._arena_xml)
        self.source = {'official_commit': C.COMMIT, 'classification_sha256': C.CLASS_SHA,
                       'task_name': task.name, 'task_language': task.language,
                       'resolved_BDDL_language': resolved,
                       'policy_prompt': self.prompt, 'protocol_mode': self.protocol_mode,
                       'prompt_source': 'official_Benchmark.get_task.language' if self.protocol_mode == 'released_task_language' else 'official_ControlEnv.language_instruction',
                       'prompt_texts_equal': released == resolved,
                       'released_prompt_sha256': C.digest(released), 'resolved_prompt_sha256': C.digest(resolved),
                       'correction_interpretation': 'full_BDDL_correction_not_suffix_only;may_change_base_wording',
                       'requested_variant_BDDL': requested_bddl,
                       'resolved_BDDL': {'path': str(resolved_bddl), **C.identity(resolved_bddl)},
                       'scene': {'path': str(scene), **C.identity(scene)},
                       'official_init_selector': 'Benchmark.get_task_init_states(task_id)[0]',
                       'declared_variant_init_file': task.init_states_file, 'native_init0': array_identity(initial),'explicit_init_contract':init_proof,'native_init0_independent_copy':True,
                       'noise_id': noise, 'sensor_noise_target': 'official_wrapper_agentview_only',
                       'goal_predicate': 'official_ControlEnv.check_success',
                       'parsed_problem_sha256': C.digest(physical.parsed_problem),
                       'robot_classes': [type(robot).__name__ for robot in self.env.robots],
                       'set_init_observation_used_for_policy': False}
        return obs

    def step(self, action):
        a = np.asarray(action)
        C.need(a.shape == (7,) and a.dtype.kind == 'f' and np.isfinite(a).all(), 'finite seven-dimensional physical action required')
        obs, _reward, done, _info = self.env.step(action)
        observation_identity(obs)
        return obs, strict_bool(done, 'official step done')

    def check_success(self):
        return strict_bool(self.env.check_success(), 'official check_success')

    def close(self):
        if self.env is not None:
            env, self.env = self.env, None
            env.close()
