"""Bind the already qualified v6 environment without installing dependencies."""
import importlib.util
import json
from pathlib import Path
import trusted_init as T

PLAN='python/ADV_VLA_ENVIRONMENT-v1.json'
HELPER='python/_adv_vla_plus_environment_v1.py'

def local_child_environment(base_env,private_tmp):
    """Only local runtime variables; no cloud auth, signed URL or proxy values."""
    allowed={'PATH','HOME','LANG','LANGUAGE','TZ','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','JAX_PLATFORMS',
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MUJOCO_GL','PYOPENGL_PLATFORM','PYTHONUNBUFFERED','PYTHONDONTWRITEBYTECODE',
        'OPENPI_DATA_HOME','HF_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TRANSFORMERS_CACHE'}
    env={k:v for k,v in base_env.items() if k in allowed or k.startswith('LC_')}
    tmp=Path(private_tmp).resolve();tmp.mkdir(exist_ok=True);env['TMPDIR']=str(tmp)
    return env

def bind(root,expected_plan,image_sha256,base_env=None):
    root=Path(root).resolve();plan_path=root/PLAN
    T.need(T.identity(plan_path)=={k:expected_plan[k] for k in ('bytes','sha256')},'qualified v6 environment plan identity differs')
    plan=json.loads(plan_path.read_bytes())
    T.need(plan['module_relative']==HELPER and plan['qualified_actual_import'] is True and plan['image_sha256']==image_sha256,'qualified v6 environment/image required')
    helper=root/HELPER;T.need(helper.resolve().is_relative_to(root) and T.identity(helper)==plan['module_identity'],'archived environment helper identity differs')
    spec=importlib.util.spec_from_file_location('_plus_verified_v6_environment',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=module.bind(root,image_sha256,base_env)
    return env,{'plan':dict(expected_plan),'helper':plan['module_identity'],'qualified_actual_import':True,
                'restored_private_environment_rebased':True,'dependency_installations':0}
