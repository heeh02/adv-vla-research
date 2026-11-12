"""Small local control preparation only. No remote cache/OUTCOME reads."""
import argparse
import gzip
import json
from pathlib import Path
import worker as W


def prepare(closeout, out):
    c, out = Path(closeout).resolve(), Path(out).resolve()
    b = c/'comparison-cloud-bridge-v1'
    repair = c/'comparison-cpu-execution-v1/repair-v1'
    rows = []
    ready = json.loads((b/'READY-v1.json').read_bytes())
    for r in ready['files'] + [{'path':'READY-v1.json',**W.file_identity(b/'READY-v1.json')}]:
        p = b/r['path']
        W.need(W.file_identity(p)=={k:r[k] for k in ('bytes','sha256')}, 'frozen_bridge_unchanged')
        rows.append({'source_path':str(p),'package_path':'frozen-bridge/'+r['path'],**W.file_identity(p)})
    repair_pins={Path(r['path']).name:r for r in json.loads((repair/'READY-v1.json').read_bytes())['files']}
    for name in ('recover.py','future_trace_overlay.py','INPUT-SPEC-v1.json'):
        p=repair/name
        W.need(W.file_identity(p)=={k:repair_pins[name][k] for k in ('bytes','sha256')},'frozen_repair_unchanged')
        rows.append({'source_path':str(p),'package_path':'frozen-repair/'+name,**W.file_identity(p)})
    transport_pins={r['path']:r for r in json.loads((repair/'cpu-execution-v1/CONTROLLER-MANIFEST-v1.json').read_bytes())['files']}
    for name in ('bos_stdlib.py','byte_budget.py','exact_store.py','contract.py'):
        relative='contract.py' if name=='contract.py' else 'worker/'+name
        p=repair/'cpu-execution-v1'/relative
        W.need(W.file_identity(p)=={k:transport_pins[relative][k] for k in ('bytes','sha256')},'executed_transport_unchanged')
        rows.append({'source_path':str(p),'package_path':'transport/'+name,**W.file_identity(p)})
    dependencies={'schema':'paper3-increment-frozen-dependencies-v1','files':rows,
                  'copy_by_root_to_dependency_root_only':True,'large_cloud_inputs_included':False}
    B,R,F,paths=W.dependencies(dependencies)
    spec,docs=B.load_controls(b/'inputs-2648-v1')
    reader=W.load_module('_increment_adoption_reader',c/'comparison-v1/compare_endpoints.py')
    new=[];authority=[]
    for suffix in ('partial01','partial02','complete'):
        p=c/f'root/results/floor-s1-u504-v7-{suffix}/ADOPTION-v1.json'
        ad=json.loads(p.read_bytes())
        proof=reader.read_ref(ad['proof']);request=reader.read_ref(proof['request'])
        W.need(request['model']==spec['models'][W.MODEL], 'exact_existing_floor1_model')
        authority.append({'adoption':{'path':str(p),**W.file_identity(p)},'proof':ad['proof'],'request':proof['request']})
        new.extend(reader.collect_adoption(p))
    W.need(len(new)==504 and {r['ordinal'] for r in new}==W.NEW,'exact_new504_unique')
    old={(r['model_id'],r['ordinal']) for r in docs['t30_endpoints']}
    for r in new:
        W.need(r['model_id']==W.MODEL and r['run_id']==W.RUN and r['attempt']==0,'exact_new_scope')
        W.need((r['model_id'],r['ordinal']) not in old,'no_old_endpoint_overwrite')
        B.validate_endpoint(r,spec['models'],docs['historical_plan']['population_cases'])
        W.need(r['official_case']==docs['historical_plan']['population_cases'][r['ordinal']],'same_full_population_case')
    repair_spec=json.loads((repair/'INPUT-SPEC-v1.json').read_bytes())
    refs=list(spec['caches'].values())+repair_spec['tables']+[r[k] for r in new for k in ('summary','outcome')]
    refs=[{k:r[k] for k in ('uri','bytes','sha256')} for r in refs]
    W.need(len({r['uri'] for r in refs})==len(refs),'unique_readset')
    W.need(sum(r['bytes'] for r in refs)<=W.MAX_INPUT,'input_cap400MiB')
    # Only the two baseline cache objects, the three exact old tables, and the
    # 504 new summaries/OUTCOMEs are authorized. No old2648 OUTCOME URI is added.
    new_body=gzip.compress(W.canonical(sorted(new,key=lambda r:r['ordinal'])),mtime=0)
    out.mkdir(parents=True,exist_ok=False)
    (out/'NEW-ENDPOINTS-v1.json.gz').write_bytes(new_body)
    (out/'DEPENDENCIES-v1.json').write_bytes(W.canonical(dependencies)+b'\n')
    readset={'schema':'paper3-increment-exact-GET-readset-v1','objects':sorted(refs,key=lambda r:r['uri']),
             'exact_total_bytes':sum(r['bytes'] for r in refs),'object_count':len(refs),
             'old2648_OUTCOME_reads':0,'model_or_array_reads':0,'GPU':0,'GET_only':True}
    (out/'EXACT-READSET-v1.json').write_bytes(W.canonical(readset)+b'\n')
    task={'schema':'paper3-floor1-u504-increment-input-v1','worker_source':W.file_identity(Path(__file__).with_name('worker.py')),
          'cloud_adapter_source':W.file_identity(Path(__file__).with_name('cloud_adapter.py')),
          'case_list_identity':spec['case_list_identity'],'new_endpoints':W.file_identity(out/'NEW-ENDPOINTS-v1.json.gz'),
          'dependencies':W.file_identity(out/'DEPENDENCIES-v1.json'),'readset':W.file_identity(out/'EXACT-READSET-v1.json'),
          'new_case_authority':authority,'adoption_reader':{'path':str(c/'comparison-v1/compare_endpoints.py'),**W.file_identity(c/'comparison-v1/compare_endpoints.py')},
          'output_prefix':'redacted-4f8a66f31a70',
          'resources':{'CPU':2,'memory_GiB':8,'GPU':0,'max_wall_seconds':3600,'GET_limit_bytes':W.MAX_INPUT,
                       'output_limit_bytes':W.MAX_OUTPUT,'JPFS_growth':0,'cloud_private_disk_limit_bytes':1024**3},
          'scientific_population':10030,'old_T30':2648,'new_T30':504,
          'runtime_callback':'Root supplies publish_phase(phase, paths) returning exact Jinwen URI/bytes/SHA/whole_GET_SHA_verified receipts; science must be durable before cache production.'}
    (out/'INPUT-SPEC-v1.json').write_bytes(W.canonical(task)+b'\n')
    return {'prepared':str(out),'GET_bytes':readset['exact_total_bytes'],'GET_objects':len(refs),
            'control_bytes':sum(p.stat().st_size for p in out.iterdir()),
            'borrowed_dependencies_bytes':sum(r['bytes'] for r in rows),'cloud_calls':0,'large_data_reads':0}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--closeout',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.closeout,a.out)))
