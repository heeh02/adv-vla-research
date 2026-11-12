"""Root-only narrow composition of the actually successful repair CPU host.

Only this new CPU run's memory/deadlines/source review/readback change. The
original source file and original scientific package are never modified.
"""
import ast,copy,importlib.util,json,sys,time
from pathlib import Path
from urllib.parse import urlsplit
import contract as C
import build
HERE=Path(__file__).resolve().parent;CLOSEOUT=HERE.parents[2]
ORIGINAL=CLOSEOUT/'comparison-cpu-execution-v1/repair-v1/cpu-execution-v1'

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def transform_function(module,name,replacements):
    node=next(n for n in ast.parse((ORIGINAL/'root.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    changed={repr(k):0 for k in replacements}
    class ExactConstants(ast.NodeTransformer):
        def visit_Constant(self,n):
            for old,new in replacements.items():
                if type(n.value)is type(old) and n.value==old:
                    changed[repr(old)]+=1;return ast.copy_location(ast.Constant(value=new),n)
            return n
    node=ExactConstants().visit(node)
    C.need(all(v>0 for v in changed.values()),'expected_transport_constant_sites:'+name)
    ns=module.__dict__;exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(__file__)+':'+name,'exec'),ns)
    return changed

def small_readback(R,backend,state,record):
    prior=R.logged(state,'READBACK-v1.json')
    if prior:return prior
    get=R.bounded_get_function();cap=C.LIMITS['mac_readback_limit_bytes'];meter={'consumed':max([C.read(p)['consumed'] for p in Path(state).glob('READBACK-BYTES-*.json')],default=0)};files={};env={}
    try:
        env=backend.sts([{'service':'bce:bos','region':'bj','effect':'Allow','resource':[C.OUTPUT.removeprefix('bos://')+x for x in C.SMALL],'permission':['READ']}],900)
        client=backend.old._bos_client(env)
        def read(name,expected=None):
            C.need(name in C.SMALL,'small_metadata_only');u=urlsplit(C.OUTPUT+name)
            raw=get(client,u.netloc,u.path.lstrip('/'),meter=meter,cap=cap,expect=expected)
            C.need(len(raw)<=4*1024**2,'individual_readback4MiB')
            p=Path(state)/'readback'/name;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():C.need(p.read_bytes()==raw,'immutable_readback')
            else:p.write_bytes(raw)
            files[name]={'uri':C.OUTPUT+name,**C.identity(raw)};return json.loads(raw)
        delivery=read('DELIVERY-v1.json')
        C.need(delivery['snapshot_comparison_complete'] is True and delivery['snapshot_cloud_durable'] is True
               and delivery['full_population_evaluation_complete'] is False and delivery['T30_total']==3626
               and delivery['new_T30']==474 and delivery['old3152_OUTCOME_GETs']==0
               and delivery['input_bytes']==322946637,'exact_increment_snapshot_delivery')
        refs={r['relative']:r for r in delivery['snapshot_publication']}
        expected={'SINGLE-GROUPS-v1.csv.gz','PAIRED-GROUPS-v1.csv.gz','PAIRED-CASE-QUALIFICATION-DELTA-v1.jsonl.gz',
                  'QUALIFIED-COVERAGE-CHANGE-v1.json','RESULT-v1.json','INPUT-RECEIPT-v1.json','SCIENTIFIC-MANIFEST-v1.json'}
        C.need(set(refs)==expected and len(delivery['snapshot_publication'])==len(expected),'complete_exact_snapshot_archive')
        C.need(all(r['uri']==C.OUTPUT+n and r['whole_GET_SHA_verified'] is True for n,r in refs.items()),'owned_verified_snapshot_objects')
        C.need({'RESULT-v1.json','INPUT-RECEIPT-v1.json','SCIENTIFIC-MANIFEST-v1.json'}<=set(refs),'complete_snapshot_small_receipts')
        result=read('RESULT-v1.json',refs['RESULT-v1.json']);receipt=read('INPUT-RECEIPT-v1.json',refs['INPUT-RECEIPT-v1.json'])
        _,q=build.verify(HERE/'prepared-v1')
        C.need(C.identity(C.encoded(receipt['controls']))==q['increment_input_spec'],'exact_frozen_increment_input_spec')
        C.need(receipt['verified_objects']==478 and receipt['verified_bytes']==322946637 and receipt['old3152_OUTCOME_GETs']==0,'exact_readset_execution')
        C.need(result['snapshot_comparison_complete'] is True and result['full_population_evaluation_complete'] is False and result['T30_total']==3626 and result['unique_endpoints']==32405 and result['delta_trace_rows']==1422,'exact_snapshot_and_delta_trace')
        change=read('QUALIFIED-COVERAGE-CHANGE-v1.json',refs['QUALIFIED-COVERAGE-CHANGE-v1.json'])
        C.need(change['old_T30']==3152 and change['added_T30']==474 and change['cumulative_T30']==3626
               and change['population_effect_identified'] is False and change['plots_created']==0,'coverage_question_not_population_or_plot_claim')
        science=read('SCIENTIFIC-MANIFEST-v1.json',refs['SCIENTIFIC-MANIFEST-v1.json'])
        for r in science['files']:
            n=r['file'];C.need(n in refs and all(r[k]==refs[n][k] for k in ('bytes','sha256')),'snapshot_manifest_receipts_agree')
        cache=None
        C.need(delivery['cache_status'] in ('COMPLETE_VERIFIED_NEW474_DELTA','NEW474_DELTA_INCOMPLETE_STATISTICS_PRESERVED'),'separate_explicit_new_delta_state')
        C.need(delivery['full_baseline_or_historical_cache_reemitted'] is False,'no_repeated_full_cache_output')
        if delivery['cache_status']=='COMPLETE_VERIFIED_NEW474_DELTA':
            published={r['relative']:r for r in delivery['cache_publication']}
            C.need(set(published)=={'NORMALIZED-NEW474-v1.jsonl.gz','CACHE-MANIFEST-v1.json'}
                   and all(r['uri']==C.OUTPUT+n and r['whole_GET_SHA_verified']is True for n,r in published.items()),'complete_exact_new_delta_publication')
            cache=read('CACHE-MANIFEST-v1.json',published['CACHE-MANIFEST-v1.json'])
            C.need(cache['input_spec']==q['increment_input_spec'] and cache['rows']==474 and cache['baseline_rows']==cache['old3152_rows']==0
                   and cache['model_id']=='T-floor-s0-u30000-EMA' and cache['ordinals']==[1136,1609]
                   and cache['cache']['bytes']<=24*1024**2,'exact474_only_cache_metadata')
        return R.journal(state,'READBACK-v1.json',{'schema':'paper3-increment-CPU-root-readback-v1','run_id':C.RUN,
            'snapshot_comparison_complete':True,'full_population_evaluation_complete':False,'cache_status':delivery['cache_status'],
            'files':files,'large_tables_and_cache_remain_cloud':True,'provider_release_separately_required':True})
    finally:
        R.event(state,'READBACK-BYTES',consumed=meter['consumed'],cap=cap,failed_reads_not_refunded=True);env.clear()

def prepare():
    source=C.read(HERE/'BORROWED-SOURCES-v1.json')
    pinned=next(r for r in source['files'] if r['source']==str(ORIGINAL/'root.py'));C.exact(pinned['source'],pinned)
    R=load('_increment_original_repair_host',ORIGINAL/'root.py');R.C=C;R.build=build;R.HERE=HERE
    changes={}
    for name,mapping in [('explicit_create',{4:8}),('cpu_capacity',{4:8}),
        ('exact_job',{4:8,'actual_GPU0_CPU2_memory4':'actual_GPU0_CPU2_memory8'}),
        ('create_once',{4:8,1200:3600,600:3300,900:3480,3600:7200}),
        ('observe',{'hard20min_deadline':'hard1h_deadline'}),
        ('main',{'statistics_recovery_complete':'snapshot_comparison_complete'})]:
        changes[name]=transform_function(R,name,mapping)
    old_review=R.review_gate
    # The actual full comparison CPU already used exact-run filename READ
    # patterns. Keep this wider credential envelope explicit; runtime exactness
    # remains independently enforced by the frozen 478-object Store.
    node=next(n for n in ast.parse((ORIGINAL/'root.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='review_gate')
    found=0
    for n in ast.walk(node):
        if isinstance(n,ast.Compare) and isinstance(n.left,ast.Subscript) and isinstance(n.left.slice,ast.Constant) and n.left.slice.value=='credential_scope_wider_than_runtime_readset':
            C.need(len(n.comparators)==1 and n.comparators[0].value is False,'original_honesty_gate');n.comparators[0]=ast.Constant(value=True);found+=1
    C.need(found==1,'one_compact_scope_honesty_gate');exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(__file__)+':review_gate','exec'),R.__dict__)
    original_review=R.review_gate
    def review(review,d,q):
        original_review(review,d,q)
        C.need(review['increment_source_manifest']==q['increment_source_manifest'] and review['increment_input_spec']==q['increment_input_spec'],'reviewed_frozen_science_increment')
        C.need(review['snapshot_T30_count']==3626 and review['full_population_evaluation_complete'] is False,'reviewed_snapshot_scope')
    R.review_gate=review
    original_verify=R.verify_sources
    def verify():
        original_verify()
        rows=C.read(HERE/'BORROWED-SOURCES-v1.json')['actual_repair_receipts'];docs=[]
        for r in rows:C.exact(r['path'],r);docs.append(C.read(r['path']))
        registered,terminal,result=docs
        C.need(registered['job_id']==terminal['job_id']=='redacted-d4f0b30ed5b6' and terminal['CPU_instance_release_proven'] is True and terminal['provider_status']=='Succeeded' and result['statistics_recovery_complete'] is True,'actual_successful_repair_transport')
    R.verify_sources=verify;R.readback=lambda backend,state,record:small_readback(R,backend,state,record)
    R.transport_constant_changes=changes
    return R

if __name__=='__main__':raise SystemExit(prepare().main())
