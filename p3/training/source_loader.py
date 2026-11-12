"""Use an already frozen T20 repository; never repair or bypass its hashes."""
import contextlib
import functools
import importlib.util
import os
from pathlib import Path
import sys
import types
import continuation_contract as C

N_REL = Path('docs/experiments/analysis-system-20260929/native-write-flow-20261006-v1')
LEGACY_REL = N_REL/'training-extension-20k-20261010-v1/continuation-candidate-v1/revision-v2'
FORMAL_REL = N_REL/'temporal-support-clean-candidate-v1/formal-training-integration-candidate-v1'

@contextlib.contextmanager
def aliases(mapping):
    previous = {name:sys.modules.get(name) for name in mapping}
    sys.modules.update(mapping)
    try:
        yield
    finally:
        for name,value in previous.items():
            if value is None:
                sys.modules.pop(name,None)
            else:
                sys.modules[name] = value

def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    C.need(Path(module.__file__).resolve() == Path(path).resolve(), 'exact_module_origin')
    return module

def configure(root):
    root = Path(root).resolve(strict=True)
    C.need((root/'AGENTS.md').is_file(), 'complete_frozen_repository_root')
    previous = os.environ.get('T30_FROZEN_REPO')
    C.need(previous is None or Path(previous).resolve() == root, 'one_immutable_source_root_per_process')
    os.environ['T30_FROZEN_REPO'] = str(root)
    return load_legacy()

def verify_sources():
    manifest = C.read(C.HERE/'SOURCE_MANIFEST-v1.json')
    for row in manifest['files']:
        C.need(C.identity(C.HERE/row['path']) == {key:row[key] for key in ('bytes','sha256')}, 'overlay_source_drift:'+row['path'])
    root = Path(os.environ['T30_FROZEN_REPO'])
    pins = C.read(C.HERE/'DEPENDENCIES-v1.json')
    for row in pins['files']:
        C.need(C.identity(root/row['path']) == {key:row[key] for key in ('bytes','sha256')}, 'frozen_T20_source_drift:'+row['path'])

@functools.lru_cache(maxsize=1)
def load_legacy():
    verify_sources()
    root = Path(os.environ['T30_FROZEN_REPO'])
    contract = load('_T30_parent20_contract',root/LEGACY_REL/'tail_contract.py')
    with aliases({'tail_contract':contract}):
        loader = load('_T30_parent20_loader',root/LEGACY_REL/'source_loader.py')
        checkpoint = load('_T30_parent20_checkpoint',root/LEGACY_REL/'tail_checkpoint.py')
        temporal = load('_T30_parent20_temporal',root/LEGACY_REL/'temporal_tail.py')
        old,common,epoch,original_checkpoint = loader.load_original()
    return types.SimpleNamespace(root=root,contract=contract,checkpoint=checkpoint,temporal=temporal,
        old=old,common=common,epoch=epoch,original_checkpoint=original_checkpoint)

def load_original():
    b = load_legacy()
    return b.old,b.common,b.epoch,b.original_checkpoint

def load_native_archive(old,common):
    verify_sources()
    with common.aliases({'protocol':old.protocol}):
        return load('_T30_native_archive',C.HERE/'native_tail_archive.py')
