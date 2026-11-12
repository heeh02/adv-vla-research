"""One in-job CPU EGL attempt before runtime assets. Never switch backends."""
import glob,importlib,json,os,subprocess,sys,time
from pathlib import Path

def current_software_renderer(*,allow_unset_cuda=False):
    if glob.glob('/dev/nvidia*'):raise ValueError('NVIDIA_device_visible_on_GPU0_job_exit')
    cuda=os.environ.get('CUDA_VISIBLE_DEVICES');nvidia=os.environ.get('NVIDIA_VISIBLE_DEVICES')
    allowed_cuda=cuda in('','-1')or(allow_unset_cuda is True and cuda is None and nvidia=='void')
    if not allowed_cuda or nvidia not in('','none','void'):
        raise ValueError('explicit_GPU0_visibility_required')
    gl=importlib.import_module('OpenGL.GL')
    def label(enum):
        value=gl.glGetString(enum)
        if not isinstance(value,bytes):raise ValueError('actual_GL_string_unavailable')
        return value.decode('utf8','strict')
    result={'GL_vendor':label(gl.GL_VENDOR),'GL_renderer':label(gl.GL_RENDERER),'GL_version':label(gl.GL_VERSION)}
    renderer=result['GL_renderer'].lower();vendor=result['GL_vendor'].lower()
    if 'nvidia'in renderer+' '+vendor:raise ValueError('GPU_renderer_on_GPU0_job_exit')
    if not any(token in renderer for token in('llvmpipe','softpipe','swrast','lavapipe','software rasterizer')):
        raise ValueError('unproven_software_GL_renderer_on_GPU0_exit')
    result.update(visible_NVIDIA_device_count=0,device_visibility={'CUDA_VISIBLE_DEVICES':cuda,'NVIDIA_VISIBLE_DEVICES':nvidia},
                  CUDA_VISIBLE_DEVICES_present='CUDA_VISIBLE_DEVICES'in os.environ,science_child_unset_cuda_allowed=allow_unset_cuda)
    return result

def probe_cpu_egl(*,deadline_epoch):
    report={'schema':'recorded-replay-CPU-EGL-probe-v1','status':'UNAVAILABLE','GPU_requested':0,
            'model_calls':0,'env_steps':0,'runtime_assets_restored':False,'backend_fallbacks':0,'renderer_qualified':False}
    context=None
    try:
        if not sys.platform.startswith('linux')or time.time()>=deadline_epoch:raise ValueError('allocated_Linux_before_deadline')
        devices=sorted(glob.glob('/dev/nvidia*'))
        report['visible_NVIDIA_device_count']=len(devices)
        report['visible_NVIDIA_devices']=devices
        report['device_visibility']={k:os.environ.get(k)for k in('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES')}
        if devices:raise ValueError('NVIDIA_device_visible_on_GPU0_job_exit')
        if os.environ.get('CUDA_VISIBLE_DEVICES')not in('','-1')or os.environ.get('NVIDIA_VISIBLE_DEVICES')not in('','none','void'):
            raise ValueError('explicit_GPU0_visibility_required')
        report['requested_backend']={k:os.environ.get(k)for k in('MUJOCO_GL','PYOPENGL_PLATFORM')}
        if report['requested_backend']!={'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'}:raise ValueError('only_original_EGL_requested')
        egl=importlib.import_module('mujoco.egl')
        context=egl.GLContext(8,8);context.make_current()
        report.update(current_software_renderer())
        if time.time()>=deadline_epoch:raise ValueError('EGL_probe_deadline')
        report.update(status='SOFTWARE_EGL_AVAILABLE',software_renderer=True,renderer_qualified=False,
                      meaning='8x8_GLContext_only_not_environment_or_pixel_replay')
    except BaseException as error:
        report['failure']={'type':type(error).__name__,'code':str(error)[:160],'automatic_retry':False}
    finally:
        if context is not None:
            try:context.free()
            except BaseException as error:
                report['status']='UNAVAILABLE';report['cleanup_failure_type']=type(error).__name__
    return report


def probe_isolated(*,deadline_epoch):
    """Same CPU job, fresh process; do not contaminate restored module origins."""
    try:
        timeout=min(60.,deadline_epoch-time.time())
        if timeout<=0:raise ValueError('probe_deadline')
        keep={'PATH','LD_LIBRARY_PATH','HOME','TMPDIR','LANG','LC_ALL','TZ','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES',
              'MUJOCO_GL','PYOPENGL_PLATFORM','XDG_CACHE_HOME','MESA_SHADER_CACHE_DIR',
              'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE'}
        child_env={k:v for k,v in os.environ.items()if k in keep}
        probe_cwd=Path(child_env['TMPDIR'])
        if not probe_cwd.is_absolute()or probe_cwd.is_symlink()or probe_cwd.resolve()!=probe_cwd or not probe_cwd.is_dir():
            raise ValueError('probe_requires_existing_canonical_private_TMPDIR')
        result=subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve()),'--probe-child',str(deadline_epoch)],
                              capture_output=True,text=True,timeout=timeout,env=child_env,cwd=probe_cwd)
        if len(result.stdout.encode())+len(result.stderr.encode())>1024**2:raise ValueError('bounded_probe_log')
        value=json.loads(result.stdout.splitlines()[-1]);value['isolated_same_job_probe']=True
        return value
    except BaseException as error:
        return {'status':'UNAVAILABLE','failure':{'type':type(error).__name__,'code':str(error)[:160]},'runtime_assets_restored':False,'backend_fallbacks':0}

def before_restore(*,restore_runtime,emit,deadline_epoch,probe=probe_isolated):
    """Parent hook: no runtime/input GET unless the same-job probe passes."""
    receipt=probe(deadline_epoch=deadline_epoch);emit('CPU_EGL_PRE_RESTORE',receipt)
    if receipt['status']!='SOFTWARE_EGL_AVAILABLE':return {'status':'CPU_EGL_UNAVAILABLE_NO_ASSET_RESTORE','probe':receipt,'restored':None}
    restored=restore_runtime()
    return {'status':'RUNTIME_RESTORED_AFTER_SOFTWARE_EGL_PROBE','probe':receipt,'restored':restored}

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--probe-child':print(json.dumps(probe_cpu_egl(deadline_epoch=float(sys.argv[2])),sort_keys=True))
    else:print(json.dumps({'status':'dry_only_no_GL_context_created'}))
