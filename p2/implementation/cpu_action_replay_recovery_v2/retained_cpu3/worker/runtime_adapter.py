"""Local stream adapter to unchanged composite_restore and capsule_runtime.

All cloud IO is the parent's already metered ExactRestore. This module cannot
download, authenticate, load a policy, install dependencies, or call env.step.
"""
import contextlib,glob,importlib.util,json,os,sys,time
from pathlib import Path
import contract as C
ADAPTER='docs/experiments/analysis-system-20260929/native-write-flow-20261006-v1/libero-plus-policy-adapter-20261009-v2'
ASSEMBLY='docs/experiments/first-paper-evidence-upgrade-20261008-v1/plus-primary-confirmation-20261010-v1/parallel-collection-execution-v5/assembly'
def cpu_science_visibility(base_env):
    """RoboSuite 1.4.1 treats empty CUDA as int(''); unset only in GPU0 child."""
    env=dict(base_env);nodes=sorted(glob.glob('/dev/nvidia*'))
    C.need(C.RESOURCES['GPU']==0 and not nodes and env.get('NVIDIA_VISIBLE_DEVICES')=='void'
           and env.get('CUDA_VISIBLE_DEVICES')=='','CPU_only_empty_CUDA_environment_required')
    env.pop('CUDA_VISIBLE_DEVICES')
    return env,{'scope':'credential_free_science_child_only','GPU_requested':0,'visible_NVIDIA_device_count':0,
        'CUDA_VISIBLE_DEVICES_before':'','CUDA_VISIBLE_DEVICES_present_after':False,'NVIDIA_VISIBLE_DEVICES':'void',
        'backend':'egl','new_GPU_device_selection':False,'actual_child_GL_software_check_required':True}
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
@contextlib.contextmanager
def alias(name,value):
    old=sys.modules.get(name);sys.modules[name]=value
    try:yield
    finally:
        if old is None:sys.modules.pop(name,None)
        else:sys.modules[name]=old
class LocalExactStore:
    def __init__(self,index,paths):
        self.rows={r['relative']:r for r in index['parts']+index['file_manifests']};self.paths=paths
    def stream(self,relative,size,consume):
        r=self.rows.get(relative);C.need(r is not None and r['bytes']==size,'only_original_runtime_part_or_manifest')
        path=Path(self.paths[r['uri']]);C.need(C.ident(path)=={k:r[k]for k in('bytes','sha256')},'runtime_cache_exact_SHA')
        with path.open('rb')as f:
            for raw in iter(lambda:f.read(1024**2),b''):consume(raw)
        return {k:r[k]for k in('bytes','sha256')}
def restore(package,root,paths,*,check):
    package,root=Path(package),Path(root);folder=package/'repo'/ASSEMBLY
    assembly=module('_replay_exact_composite_restore',folder/'composite_restore.py')
    indexes=[json.loads((folder/name).read_bytes())for name in('BASE_INDEX-v1.json','OVERLAY_INDEX-v1.json')]
    base,overlay,reports,base_proof,overlay_proof,repair=assembly.restore_pair(root,
        LocalExactStore(indexes[0],paths),LocalExactStore(indexes[1],paths),*indexes,check)
    gatepath=folder/'Q0_GATE-v1.json';gate=json.loads(gatepath.read_bytes())
    reference=base/gate['variant_reference']['member']
    C.need(C.ident(reference)=={k:gate['variant_reference'][k]for k in('bytes','sha256')},'same_base_original_init_reference')
    manifest={}
    for path in (root/'compatibility-work/base-cache').glob('files-*.jsonl'):
        for line in path.read_text().splitlines():r=json.loads(line);manifest[r['path']]=r
    C.need(len(manifest)==indexes[0]['members'],'all_actual_base_manifest_members')
    config=root/'libero-config';config.mkdir();(config/'config.yaml').write_bytes(C.encoded({
        'benchmark_root':str(base/'official/libero/libero'),'bddl_files':str(base/'official/libero/libero/bddl_files'),
        'init_states':str(base/'official/libero/libero/init_files'),'assets':str(base/'official/libero/libero/assets'),'datasets':str(base/'unused-datasets')}))
    trusted=module('_replay_environment_trusted',package/'repo'/ADAPTER/'trusted_init.py')
    with alias('trusted_init',trusted):environment=module('_replay_exact_environment',package/'repo'/ADAPTER/'capsule_runtime.py')
    child_env=environment.local_child_environment(dict(os.environ),root/'science-tmp')
    child_env,dependency=environment.bind(base/'environment',gate['environment_plan'],C.IMAGE.rsplit('sha256:',1)[1],child_env)
    child_env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void',
        LIBERO_CONFIG_PATH=str(config),XDG_CACHE_HOME=str(root/'cache'),MESA_SHADER_CACHE_DIR=str(root/'mesa-cache'),
        PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    child_env,visibility=cpu_science_visibility(child_env)
    C.need(not any(k.startswith(('ONE_VLA_','AWS_','BCE_','BAIDU_','ADV_VLA_SOURCE'))for k in child_env),'credential_free_native_child')
    restored={'repo_root':str(package/'repo'),'plus_root':str(base/'official'),'config_dir':str(config),
        'q0_gate_path':str(gatepath),'q0_gate_identity':C.ident(gatepath),'q0_gate':gate,
        'init_reference':str(reference),'runtime_manifest':manifest}
    proof={'schema':'recorded-replay-runtime-restore-v1','base':base_proof,'overlay':overlay_proof,'Git_repair':repair,
        'dependencies':dependency,'expanded_runtime_bytes':sum(i['logical_file_bytes']for i in indexes),
        'model_payloads':0,'dependency_installations':0,'restore_helper':'unchanged_composite_restore.restore_pair',
        'CPU_renderer_environment':visibility}
    return restored,child_env,proof
