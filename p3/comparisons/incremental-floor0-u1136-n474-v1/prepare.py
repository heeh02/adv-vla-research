"""Prepare exact474 metadata/readset from root gen7 and recovered summaries only."""
import argparse,base64,gzip,json
from pathlib import Path
import worker as W

def reference(path):
    path=Path(path).resolve();return dict(path=str(path),**W.file_identity(path))

def prepare(closeout,out):
    c,out=Path(closeout).resolve(),Path(out).resolve()
    here=Path(__file__).resolve().parent
    old=c/'comparison-v1/incremental-floor1-u504-execution-v1'
    deps=json.loads((old/'prepared-v2/DEPENDENCIES-v1.json').read_bytes())
    b,r,f,paths=W.dependencies(deps)
    spec,docs=b.load_controls(c/'comparison-cloud-bridge-v1/inputs-2648-v1')
    actual=here/'summary-recovery-actual-v1/READBACK-001.json'
    W.need(W.file_identity(actual)=={'bytes':674909,'sha256':'7fe01a9f50e4bb071cdbf6c3fc763dea70b27f2f2032e9225eb0faf5f6f6b9db'},'actual403_recovery_readback')
    read=json.loads(actual.read_bytes())
    W.need(read['status']=='ALL474_SUMMARIES_VERIFIED403_GET71_REUSED'
           and read['exact474_OUTCOME_GET_bytes']==74297967 and read['exact474_OUTCOME_GET_count']==474
           and read['GET_bytes_verified']==1270348 and read['OUTCOME_GETs']==read['array_GETs']==0,'root_summary_only_actual_read')
    ledger_path=c/'plus-cpu4-successor-v1/floor0/CASE-LEDGER-v1.json'
    applied_path=c/'control-recovery-v1/PLUS-ADOPTION-APPLIED-v1.json'
    preservation_path=c/'control-recovery-v1/PLUS-PRESERVATION-ACTUAL-v1/PRESERVATION-v1.json'
    ledger=json.loads(ledger_path.read_bytes());applied=json.loads(applied_path.read_bytes());preservation=json.loads(preservation_path.read_bytes())
    W.need(ledger['generation']==7 and W.file_identity(ledger_path)==applied['after_identity']
           and applied['applied'] is True and applied['after_closed']==474
           and preservation['closed_case_indexes_verified']==474
           and preservation['status']=='ALL474_AND_UNRESOLVED_SNAPSHOT_METADATA_PRESERVED','gen7_root_adopted_and_preserved474')
    old_extra=json.loads(gzip.decompress((here/'OLD-FLOOR1-504-ENDPOINTS-v1.json.gz').read_bytes()))
    old_keys={(x['model_id'],x['ordinal'])for x in docs['t30_endpoints']+old_extra}
    W.need(len(old_keys)==3152,'accepted3152_old_metadata_only')
    partners={c['control'] if c['treatment']==W.MODEL else c['treatment']
              for c in spec['comparison']['contrasts'] if W.MODEL in (c['control'],c['treatment'])}
    W.need(not any((m,o)in old_keys for m in partners if m not in b.BASELINES for o in W.NEW),
           'no_old_T30_partner_input_required_for_this_exact_increment')
    new=[];summaries={}
    for metadata,row in zip(read['summaries'],ledger['rows'][:474],strict=True):
        raw=Path(metadata['local']['path']).read_bytes();W.exact(raw,metadata['summary']);s=json.loads(raw)
        ordinal=row['ordinal']
        W.need(ordinal==metadata['ordinal'] and row['state']=='closed' and row['scientific_attempt']==0
               and row['original_evidence_authenticated'] is True and row['success'] is s['success']
               and row['id']==s['id'] and row['case_definition_sha256']==b.digest(s['case']),'same_exact_adopted_endpoint')
        W.need(s['run_id']==W.RUN and s['attempt']==0 and s['population_ordinal']==ordinal
               and s['model_identity_sha256']==b.digest(spec['models'][W.MODEL])
               and s['outcome']==metadata['outcome'],'same_model_outcome_source')
        W.need((W.MODEL,ordinal)not in old_keys,'all474_new_relative_to3152')
        e=dict(panel='Plus',model_id=W.MODEL,checkpoint_sha256=spec['models'][W.MODEL]['checkpoint']['sha256'],
            ordinal=ordinal,official_case=s['case'],case_definition_sha256=b.digest(s['case']),
            population_sha256=s['population_sha256'],id=s['id'],run_id=s['run_id'],attempt=0,adopted=True,
            success=s['success'],status=s['status'],summary=metadata['summary'],outcome=s['outcome'],
            outcome_canonical_sha256=s['outcome_canonical_sha256'],
            input={'status':'not_materialized_for_pairing','signature':None},
            source_package_sha256=s['source_package_sha256'],local_adoption=str(applied_path))
        b.validate_endpoint(e,spec['models'],docs['historical_plan']['population_cases'])
        W.need(e['official_case']==docs['historical_plan']['population_cases'][ordinal],'same_original10030_case')
        new.append(e);summaries[e['summary']['uri']]=base64.b64encode(raw).decode()
    W.need(len(new)==474 and {e['ordinal']for e in new}==W.NEW and sum(e['success']for e in new)==421,'exact474_421_success53_normal_failure')
    delivery=json.loads((old/'cpu-execution-v1/run-v1/readback/DELIVERY-v1.json').read_bytes())
    publication={x['relative']:x for x in delivery['snapshot_publication']}
    W.need(delivery['snapshot_cloud_durable'] is True and delivery['T30_total']==3152,'actual3152_tables_persisted')
    tables=[]
    for name,limit in [('SINGLE-GROUPS-v1.csv.gz',16*1024**2),('PAIRED-GROUPS-v1.csv.gz',64*1024**2)]:
        ref=publication[name];W.need(ref['whole_GET_SHA_verified'] is True,'whole3152_table_SHA')
        tables.append(dict(name=name,decoded_limit_bytes=limit,**{k:ref[k]for k in ('uri','bytes','sha256')}))
    trace=publication['PAIRED-CASE-QUALIFICATION-v2.jsonl.gz']
    refs=[{k:x[k]for k in ('uri','bytes','sha256')}for x in list(spec['caches'].values())+tables+[e['outcome']for e in new]]
    W.need(len(refs)==len({x['uri']for x in refs})==478 and sum(x['bytes']for x in refs)==322946637,'exact478_322946637B')
    out.mkdir(parents=True,exist_ok=False)
    payloads={'DEPENDENCIES-v1.json':W.canonical(deps)+b'\n',
              'NEW-ENDPOINTS-v1.json.gz':gzip.compress(W.canonical(new),mtime=0),
              'NEW-SUMMARIES-v1.json.gz':gzip.compress(W.canonical(summaries),mtime=0),
              'EXACT-READSET-v1.json':W.canonical(dict(schema='added474-exact-readset-v1',objects=sorted(refs,key=lambda x:x['uri']),
                object_count=478,exact_total_bytes=322946637,baseline_normalized_cache_bytes=247558566,
                existing3152_table_bytes=1090104,new474_OUTCOME_bytes=74297967,
                old3152_OUTCOME_GETs=0,baseline_original_array_GETs=0))+b'\n',
              'OLD-FLOOR1-504-ENDPOINTS-v1.json.gz':(here/'OLD-FLOOR1-504-ENDPOINTS-v1.json.gz').read_bytes(),
              'OLD-3152-RESULT-v1.json':(here/'OLD-3152-RESULT-v1.json').read_bytes()}
    for name,raw in payloads.items():(out/name).write_bytes(raw)
    task=dict(schema='paper3-floor0-u1136-added474-input-v1',worker_source=W.file_identity(here/'worker.py'),
        cloud_adapter_source=W.file_identity(here/'cloud_adapter.py'),case_list_identity=spec['case_list_identity'],
        new_endpoints=W.file_identity(out/'NEW-ENDPOINTS-v1.json.gz'),new_summaries=W.file_identity(out/'NEW-SUMMARIES-v1.json.gz'),
        dependencies=W.file_identity(out/'DEPENDENCIES-v1.json'),readset=W.file_identity(out/'EXACT-READSET-v1.json'),
        prior_floor1_endpoints=W.file_identity(out/'OLD-FLOOR1-504-ENDPOINTS-v1.json.gz'),
        prior_result=W.file_identity(out/'OLD-3152-RESULT-v1.json'),prior_tables=tables,
        prior_trace={k:trace[k]for k in ('uri','bytes','sha256')},
        source_authority={k:reference(p)for k,p in [('gen7_ledger',ledger_path),('root_applied',applied_path),
            ('preserved474_and_partial',preservation_path),('summary_recovery',actual),
            ('prior3152_delivery',old/'cpu-execution-v1/run-v1/readback/DELIVERY-v1.json')]},
        output_prefix='redacted-0bf4976d7ed4',
        scientific_population=10030,old_T30=3152,new_T30=474,cumulative_T30=3626,delta_trace_rows=1422,
        resources=dict(CPU=2,memory_GiB=8,GPU=0,max_wall_seconds=3600,GET_limit_bytes=W.MAX_INPUT,
                       output_limit_bytes=W.MAX_OUTPUT,JPFS_growth=0,cloud_private_disk_limit_bytes=1024**3),
        optional_full_baseline_cache_output=False,optional_new474_delta_cache_max_bytes=24*1024**2,
        statistics_persist_before_delta_cache=True,L10_late_cases_included=False)
    (out/'INPUT-SPEC-v1.json').write_bytes(W.canonical(task)+b'\n')
    return dict(input_objects=478,input_bytes=322946637,new474_OUTCOME_bytes=74297967,
        summary_cloud_GETs_in_CPU_job=0,prepared=str(out),control_bytes=sum(p.stat().st_size for p in out.iterdir()),
        cloud_calls=0,model_calls=0,full_cache_decodes_on_Mac=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--closeout',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(prepare(a.closeout,a.out)))
