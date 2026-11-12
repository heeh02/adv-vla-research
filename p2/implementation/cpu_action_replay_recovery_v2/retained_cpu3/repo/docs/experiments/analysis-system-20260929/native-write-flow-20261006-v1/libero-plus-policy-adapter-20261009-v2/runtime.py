"""Cloud-only lazy runtime and exact reuse of already-run C/U/L policy gates."""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import plus_contract as C
from adapter import PolicyAdapter
import official_namespace
import trusted_init
import q0_consumer


def verify_sources():
    bindings = C.read(C.HERE / 'SOURCE_BINDINGS-v1.json')
    for row in bindings['repository_files']:
        C.exact(C.ROOT / row['path'], row)
    for row in bindings['adapter_files']:
        C.exact(C.HERE / row['path'], row)
    return bindings


def loaded_origins(prefix, root):
    result = {}
    for name, module in list(sys.modules.items()):
        if name == prefix or name.startswith(prefix + '.'):
            for package_path in getattr(module, '__path__', []):
                C.need(Path(package_path).resolve().is_relative_to(root), 'namespace package includes non-frozen source: '+name)
            path = getattr(module, '__file__', None)
            if path:
                p = Path(path).resolve()
                C.need(p.is_relative_to(root), 'same-name module imported outside frozen source: ' + name)
                result[name] = {'path': str(p), **C.identity(p)}
    return result


def prepare_official(plus_root, config_dir, bindings):
    C.need(sys.platform.startswith('linux'), 'Plus runtime is cloud Linux only')
    root, config = Path(plus_root).resolve(strict=True), Path(config_dir).resolve(strict=True)
    C.need((config / 'config.yaml').is_file() and str(config).startswith('/tmp/Jinwen/adv-vla/'), 'dedicated prebuilt Plus config in cloud tmp required')
    C.need(os.environ.get('MUJOCO_GL') == 'egl' and os.environ.get('PYOPENGL_PLATFORM') == 'egl', 'actual EGL runtime required')
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True, timeout=10).strip()
    C.need(commit == C.COMMIT, 'official checkout commit differs')
    # Assets can be untracked; tracked source changes are rejected.
    C.need(not subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', 'HEAD'], text=True, timeout=10).strip(), 'official tracked source has changed')
    for row in bindings['official_files']:
        C.exact(root / row['path'], row)
    loaded_origins('libero', root)  # Reject earlier stock imports before modifying search path.
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    official_namespace.install(root)
    importlib.invalidate_caches()
    importlib.import_module('libero.libero.benchmark')
    importlib.import_module('libero.libero.envs.env_wrapper')
    modules = loaded_origins('libero', root)
    from libero.libero import get_libero_path
    # Require the active configuration, not only a correct sys.path entry.
    for key in ('bddl_files', 'init_states', 'assets'):
        p = Path(get_libero_path(key)).resolve(strict=True)
        C.need(p.is_relative_to(root), 'Plus config points outside frozen Plus tree: ' + key)
    return {'official_commit': commit, 'modules': modules, 'config': {'path': str(config / 'config.yaml'), **C.identity(config / 'config.yaml')}}


def validate_q0_gate(path, expected):
    return q0_consumer.validate_gate(path,expected)


def prepare_init_reader(plus_root,reference_path,q0):
    import numpy
    import torch
    root=Path(plus_root).resolve();reference=Path(reference_path).resolve()
    C.need(reference.is_relative_to(root.parent),'init reference must come from same restored base capsule')
    rows=trusted_init.read_reference(reference,q0['variant_reference'])
    return trusted_init.TrustedInitReader(root,rows,torch,numpy)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy(model, checkpoint_root, checkpoint_metadata, checkpoint_manifest=None):
    C.need(sys.platform.startswith('linux'), 'model and arrays must remain on cloud Linux')
    C.model_identity(model)
    root = Path(checkpoint_root).resolve(strict=True)
    C.need(str(root).startswith('/tmp/Jinwen/adv-vla/'), 'preverified dedicated cloud-tmp checkpoint required')
    C.need(os.environ.get('OPENPI_DATA_HOME', '').startswith('/tmp/Jinwen/adv-vla/'), 'private preverified tokenizer cache required')
    sys.path[:0] = [str(C.ROOT / 'src'), str(C.ROOT / 'packages/openpi-client/src')]
    old = C.N / 'cul-closedloop-evaluation-v1'
    old_contract = _load('_plus_frozen_CUL_contract', old / 'contract.py')
    # Private imports retain the original function globals without overwriting
    # the frozen backend, contract, model loader or process-global torch.load.
    saved = {k: sys.modules.get(k) for k in ('contract', 'init_inventory')}
    try:
        sys.modules['contract'] = old_contract
        sys.modules['init_inventory'] = _load('_plus_frozen_CUL_inventory', old / 'init_inventory.py')
        runner = _load('_plus_frozen_CUL_runner', old / 'run.py')
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    runner.source_gate('3239396228a46c1adf4297f54bf0f1893876e78c776f15354aab3941a9707861')
    role = next(r for r in old_contract.model_roles() if r['id'] == model['id'])
    C.need(role['parameter_role'] == model['parameter_role'], 'canonical parameter role differs')
    proof = runner.checkpoint_gate({'model': role, 'checkpoint_commit_or_original_manifest': model['checkpoint']},
                                   checkpoint_metadata, checkpoint_manifest, root)
    from openpi.training.config import get_config
    from openpi.eval.campaign_policy import CampaignPolicy
    cfg = get_config('pi05_libero')
    C.need(cfg.model.pi05 is True and cfg.model.discrete_state_input is False
           and cfg.model.action_horizon == 10 and cfg.model.action_dim == 32 and cfg.model.dtype == 'bfloat16'
           and cfg.data.extra_delta_transform is False, 'canonical pi05_libero config differs')
    C.need(Path(sys.modules[CampaignPolicy.__module__].__file__).resolve() == C.ROOT / 'src/openpi/eval/campaign_policy.py', 'wrong canonical policy module origin')
    loaded = CampaignPolicy(root)  # Exactly once; run_case only receives this instance.
    return PolicyAdapter(loaded), {'checkpoint': proof, 'model_loads': 1, 'config': 'pi05_libero',
                                  'precision': 'stock_BF16_and_native_FP32_no_extra_cast',
                                  'horizon': 10, 'execute_prefix': 5, 'denoise_steps': 10,
                                  'extra_delta_transform': False, 'discrete_state_input': False,
                                  'policy_module': {'path': str(C.ROOT / 'src/openpi/eval/campaign_policy.py'), **C.identity(C.ROOT / 'src/openpi/eval/campaign_policy.py')}}
