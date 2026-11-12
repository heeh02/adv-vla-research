"""One small deterministic source package; frozen analysis files unchanged."""
import argparse,gzip,io,json,tarfile
from pathlib import Path
import contract as C
HERE=Path(__file__).resolve().parent;SCIENCE=HERE.parent;CLOSEOUT=HERE.parents[2]
BRIDGE=CLOSEOUT/'comparison-cloud-bridge-v1';REPAIR=CLOSEOUT/'comparison-cpu-execution-v1/repair-v1'

def members():
    files={};freeze=C.read(SCIENCE/'SOURCE-MANIFEST-v1.json')
    for r in freeze['files']:
        p=SCIENCE/r['path'];C.exact(p,r);files['worker/science/'+r['path']]=p.read_bytes()
    files['worker/science/SOURCE-MANIFEST-v1.json']=(SCIENCE/'SOURCE-MANIFEST-v1.json').read_bytes()
    deps=C.read(SCIENCE/'prepared-v2/DEPENDENCIES-v1.json')
    for r in deps['files']:
        C.relative(r['package_path']);C.exact(r['source_path'],r)
        files['worker/deps/'+r['package_path']]=Path(r['source_path']).read_bytes()
    refs=C.read(SCIENCE/'prepared-v2/EXACT-READSET-v1.json')['objects']
    q={'schema':'paper3-Plus-increment-CPU-request-v1','run_id':C.RUN,'output_prefix':C.OUTPUT,
       'queue_id':C.QUEUE,'physical_queue_id':C.PARENT,'resources':C.RESOURCES,'phases':C.PHASES,'limits':C.LIMITS,**C.PHASES,**C.LIMITS,
       'input_objects':len(refs),'input_bytes':sum(r['bytes'] for r in refs),'old2648_OUTCOME_GETs':0,
       'bridge_ready':C.ident(BRIDGE/'READY-v1.json'),'repair_ready':C.ident(REPAIR/'READY-v1.json'),
       'increment_source_manifest':C.ident(SCIENCE/'SOURCE-MANIFEST-v1.json'),
       'increment_input_spec':C.ident(SCIENCE/'prepared-v2/INPUT-SPEC-v1.json'),
       'input_credential_resources':C.read_acl(refs),'input_permission':['READ'],
       'credential_scope_wider_than_runtime_readset':True,'runtime_exact_URI_count':1013,
       'new_model_calls':0,'new_episodes':0,'JPFS_new_bytes':0,'snapshot_T30_count':3152,
       'full_population_evaluation_complete':False,'root_thread':'01a0d223-c34b-7290-83da-aeee6a8e3fdd'}
    C.request_gate(q);files['metadata/REQUEST-v1.json']=C.encoded(q)
    for p in sorted((HERE/'worker').glob('*.py')):files['worker/'+p.name]=p.read_bytes()
    files['worker/contract.py']=(HERE/'contract.py').read_bytes()
    mf={'schema':'paper3-Plus-increment-CPU-source-manifest-v1','run_id':C.RUN,
        'files':[{'path':name,**C.identity(raw)} for name,raw in sorted(files.items())]}
    files['metadata/SOURCE_MANIFEST-v1.json']=C.encoded(mf)
    return files,q

def prepare(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);files,q=members()
    archive=out/'SOURCE_PACKAGE-v1.tar.gz'
    with archive.open('xb') as target,gzip.GzipFile(fileobj=target,mode='wb',mtime=0,filename='') as gz,tarfile.open(fileobj=gz,mode='w',format=tarfile.USTAR_FORMAT) as tf:
        for name,raw in sorted(files.items()):
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.mtime=0;tf.addfile(info,io.BytesIO(raw))
    d={'schema':'paper3-Plus-increment-CPU-package-v1','run_id':C.RUN,'output_prefix':C.OUTPUT,
       'source_package':C.ident(archive),'source_manifest':C.identity(files['metadata/SOURCE_MANIFEST-v1.json']),
       'request':C.identity(files['metadata/REQUEST-v1.json']),'unpacked_max_bytes':sum(map(len,files.values())),
       'bridge_ready':q['bridge_ready'],'repair_ready':q['repair_ready'],
       'increment_source_manifest':q['increment_source_manifest'],'increment_input_spec':q['increment_input_spec']}
    C.need(d['source_package']['bytes']<=8*1024**2 and d['unpacked_max_bytes']<=12*1024**2,'small_source_package')
    (out/'DESCRIPTOR-v1.json').write_bytes(C.encoded(d));(out/'REQUEST-v1.json').write_bytes(files['metadata/REQUEST-v1.json']);return d

def verify(output):
    out=Path(output);d=C.read(out/'DESCRIPTOR-v1.json');C.exact(out/'SOURCE_PACKAGE-v1.tar.gz',d['source_package']);C.exact(out/'REQUEST-v1.json',d['request'])
    q=C.request_gate(C.read(out/'REQUEST-v1.json'));expected,current_q=members();C.need(q==current_q,'same_current_frozen_request')
    raw={}
    with tarfile.open(out/'SOURCE_PACKAGE-v1.tar.gz','r:gz') as tf:
        for m in tf:
            C.relative(m.name);C.need(m.isfile() and not m.issparse() and m.name not in raw and m.size<=8*1024**2,'safe_unique_source_member')
            raw[m.name]=tf.extractfile(m).read()
    C.need(raw==expected,'every_actual_member_equals_exact_frozen_or_new_source')
    C.need(d['run_id']==C.RUN and d['output_prefix']==C.OUTPUT and d['source_manifest']==C.identity(raw['metadata/SOURCE_MANIFEST-v1.json']),'descriptor_binding')
    return d,q

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.out)[0] if a.verify else prepare(a.out)))
