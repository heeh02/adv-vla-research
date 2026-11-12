"""Tiny synthetic metadata only, using the actual frozen source functions."""
import copy
import gzip
import json
import tempfile
import time
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch
import worker as W
import cloud_adapter as A

HERE=Path(__file__).resolve().parent
C=HERE.parents[1]
R=W.load_module('recover',C/'comparison-cpu-execution-v1/repair-v1/recover.py')
F=W.load_module('future_trace_overlay',C/'comparison-cpu-execution-v1/repair-v1/future_trace_overlay.py')
T=W.load_module('_prior_repair_fixture',C/'comparison-cpu-execution-v1/repair-v1/test_repair.py')
B=R.bridge()


def fixture(root):
    spec,docs,store=T.fixture(B,root/'controls')
    removed=[r for r in docs['t30_endpoints'] if r['ordinal']==1 and r['model_id'] in
             (W.MODEL,'T-reheat-s1-u30000-EMA')]
    docs['t30_endpoints']=[r for r in docs['t30_endpoints'] if r not in removed]
    spec['expected_T30_unique']=len(docs['t30_endpoints'])
    normalize,restore,gate=B.source_functions()
    raw=gzip.decompress(store.objects[spec['caches']['old_cache']['uri']])
    old_raw=[json.loads(line) for line in raw.splitlines()]
    baseline=list(B.cache_rows(docs['historical_plan'],'old_cache',
        F.tagged_cache_rows(old_raw,spec['caches']['old_cache']),restore,spec['case_list_identity']))
    bucket={}
    for r in baseline:
        B.add_endpoint(bucket,r)
    old=[]
    for r in docs['t30_endpoints']:
        e=B.materialize_t30(r,spec['models'][r['model_id']],store,normalize)
        e['population_identity']=spec['case_list_identity'];old.append(e);B.add_endpoint(bucket,e)
    out=root/'prior';out.mkdir()
    try:B.write_tables(spec,docs['historical_plan']['population_cases'],bucket,gate,out)
    except KeyError as e:
        assert e.args==('summary',)
    tables=[]
    for name in ('SINGLE-GROUPS-v1.csv','PAIRED-GROUPS-v1.csv','PAIRED-CASE-QUALIFICATION-v1.jsonl.gz'):
        raw=(out/name).read_bytes();packed=raw if name.endswith('.gz') else gzip.compress(raw,mtime=0)
        uri='redacted-01b22164a52b'+name+'.packed'
        store.objects[uri]=packed
        tables.append({'uri':uri,**W.identity(packed),'original':W.identity(raw),
                       'original_name':name,'decoded_limit_bytes':1024**2})
    rs=root/'repair.json';rs.write_bytes(W.canonical({'tables':tables}))
    new=copy.deepcopy(next(r for r in removed if r['model_id']==W.MODEL))
    new['run_id']=W.RUN
    summary=json.loads(store.objects[new['summary']['uri']]);summary['run_id']=W.RUN
    new['summary']=store.json('redacted-2e975ea9919d',summary)
    new['population_identity']=spec['case_list_identity']
    return spec,docs,store,tables,rs,baseline,old,new


def prepared(root,spec,docs,store,tables,new):
    inp=root/'inputs';inp.mkdir()
    refs=list(spec['caches'].values())+tables+[new['summary'],new['outcome']]
    refs=[{k:r[k] for k in ('uri','bytes','sha256')} for r in refs]
    for name,raw in [('NEW-ENDPOINTS-v1.json.gz',gzip.compress(W.canonical([new]),mtime=0)),
                     ('DEPENDENCIES-v1.json',W.canonical({'files':[]})),
                     ('EXACT-READSET-v1.json',W.canonical({'objects':refs}))]:
        (inp/name).write_bytes(raw)
    task={'schema':'paper3-floor1-u504-increment-input-v1','worker_source':W.file_identity(W.__file__),
          'case_list_identity':spec['case_list_identity'],
          'output_prefix':'redacted-723a4212007a'}
    for k,n in [('new_endpoints','NEW-ENDPOINTS-v1.json.gz'),('dependencies','DEPENDENCIES-v1.json'),('readset','EXACT-READSET-v1.json')]:
        task[k]=W.file_identity(inp/n)
    (inp/'INPUT-SPEC-v1.json').write_bytes(W.canonical(task));return inp,task,refs


class Tests(unittest.TestCase):
    def full_fixture_run(self,fail_cache=False):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,docs,store,tables,rs,baseline,old,new=fixture(root)
            inp,task,refs=prepared(root,spec,docs,store,tables,new)
            original_stream=B.stream_cache;store.calls=[];phases=[];snapshots={}
            def stream(p,count):return original_stream(p,6 if Path(p).name=='old_cache.jsonl.gz' else 0)
            def publisher(phase,paths):
                phases.append(phase)
                snapshots[phase]={p.name:p.read_bytes() for p in paths}
                self.assertFalse(any('NORMALIZED' in n for n in snapshots['science']))
                return [{'relative':p.name,'uri':task['output_prefix']+p.name,
                         **W.file_identity(p),'whole_GET_SHA_verified':True} for p in paths]
            original_gzip=W.OutputBudget.gzip
            def gz(self,name):
                if fail_cache and name.startswith('NORMALIZED-'):
                    raise ValueError('output_cap32MiB')
                return original_gzip(self,name)
            with ExitStack() as s:
                s.enter_context(patch.multiple(W,NEW={1},NEW_COUNT=1,OLD_T30_COUNT=6,POPULATION_COUNT=2))
                s.enter_context(patch.object(W.platform,'system',return_value='Linux'))
                s.enter_context(patch.object(B,'load_controls',return_value=(spec,docs)))
                s.enter_context(patch.object(B,'stream_cache',side_effect=stream))
                s.enter_context(patch.object(W,'dependencies',return_value=(B,R,F,{
                    'frozen-bridge/inputs-2648-v1/INPUT-SPEC-v1.json':root/'controls/INPUT-SPEC-v1.json',
                    'frozen-repair/INPUT-SPEC-v1.json':rs})))
                s.enter_context(patch.object(W.OutputBudget,'gzip',gz))
                value=W.run(inp,root/'deps',store,root/'work',root/'out',time.time()+30,publisher)
            self.assertTrue(value['snapshot_comparison_complete'])
            self.assertFalse(value['full_population_evaluation_complete'])
            self.assertEqual(set(store.calls),{r['uri'] for r in refs})
            self.assertEqual(len(store.calls),len(refs))
            self.assertEqual(value['T30_total'],7)
            result=json.loads(snapshots['science']['RESULT-v1.json'])
            self.assertEqual(result['unique_endpoints'],13)
            trace=[json.loads(x) for x in gzip.decompress(snapshots['science']['PAIRED-CASE-QUALIFICATION-v2.jsonl.gz']).splitlines()]
            self.assertEqual(len(trace),24)
            self.assertEqual(sum(x['qualification_evidence'].startswith('direct') for x in trace),3)
            original=[x['control_source'] for x in trace if x['control_source'] and
                      x['control_source']['source_kind']=='verified_normalized_cache_row_no_summary_in_legacy_ABI']
            self.assertTrue(original);self.assertIsNone(original[0]['summary'])
            self.assertEqual(phases[0],'science');self.assertEqual(phases[-1],'delivery')
            if fail_cache:
                self.assertEqual(value['cache_status'],'CACHE_INCOMPLETE_SCIENTIFIC_RESULTS_PRESERVED')
                self.assertEqual(phases,['science','delivery'])
                self.assertEqual(json.loads((root/'out/RESULT-v1.json').read_bytes()),result)
            else:
                self.assertEqual(value['cache_status'],'COMPLETE_VERIFIED_CACHE')
                cached=[json.loads(x) for x in gzip.decompress(snapshots['cache']['NORMALIZED-BASELINE-AND-DELTA-v1.jsonl.gz']).splitlines()]
                self.assertEqual(len(cached),7)
                self.assertEqual(sum(r['model_id'] not in B.BASELINES for r in cached),1)
                self.assertTrue(all('cache_provenance' in r for r in cached if r['model_id'] in B.BASELINES))
            # Compare both complete CSVs with a fresh evaluation of the same
            # tiny inputs using the original writer and trace-only fix.
            raw_new=B.materialize_t30(new,spec['models'][W.MODEL],store,B.source_functions()[0])
            raw_new['population_identity']=spec['case_list_identity']
            fresh=B.__dict__.copy()
            clone=type(B)('fixture_bridge');clone.__dict__.update(fresh)
            F.install(clone,spec['caches'])
            bucket={('Plus',r['model_id'],r['ordinal']):r for r in baseline+old+[raw_new]}
            expected=root/'fresh';expected.mkdir()
            clone.write_tables(spec,docs['historical_plan']['population_cases'],bucket,B.source_functions()[2],expected)
            for name in ('SINGLE-GROUPS-v1.csv','PAIRED-GROUPS-v1.csv'):
                self.assertEqual(gzip.decompress(snapshots['science'][name+'.gz']),(expected/name).read_bytes())

    def test_full_frozen_fixture_increment_and_reusable_cache(self):
        self.full_fixture_run()

    def test_cache_failure_preserves_published_snapshot_and_never_rereads(self):
        self.full_fixture_run(fail_cache=True)

    def test_new_direct_gate_uses_late_noise_and_old_labels_are_not_transitive(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);spec,docs,store,tables,rs,baseline,old,new=fixture(root)
            single=(root/'prior/SINGLE-GROUPS-v1.csv').read_bytes();pair=(root/'prior/PAIRED-GROUPS-v1.csv').read_bytes()
            _,_,ends,labels,_=R.validate_tables(B,spec,docs,single,pair)
            inp=copy.deepcopy(baseline[0]['input']);inp['recorded_noise_by_query']={0:{'typed':'same'},1:{'typed':'same-later'}}
            for row in baseline:row['input']=copy.deepcopy(inp)
            new['input']=copy.deepcopy(inp);new['input']['recorded_noise_by_query'][1]={'typed':'changed-later'}
            cells=W.MergedCells(B,spec,docs['historical_plan']['population_cases'],ends,labels,docs['t30_endpoints'],baseline,[new])
            self.assertEqual(cells.direct['floor_s1_T30_minus_T20',1]['reason'],'observed_shared_query_noise_difference')
            self.assertEqual(cells.label(spec['comparison']['contrasts'][0],0),labels[spec['comparison']['contrasts'][0]['id'],0])
            new['input']['signature']='new-q0'
            cells=W.MergedCells(B,spec,docs['historical_plan']['population_cases'],ends,labels,docs['t30_endpoints'],baseline,[new])
            self.assertEqual(cells.direct['floor_s1_T30_minus_T20',1]['reason'],'q0_input_difference')
            with self.assertRaisesRegex(ValueError,'new_case_already_adopted|duplicate_new_case'):
                W.MergedCells(B,spec,docs['historical_plan']['population_cases'],ends,labels,docs['t30_endpoints'],baseline,[new,new])

    def test_compact_projection_keeps_each_gate_result_and_unknown(self):
        with tempfile.TemporaryDirectory() as td:
            _,_,_,_,_,baseline,_,_=fixture(Path(td));a=baseline[0];b=copy.deepcopy(a)
            restore,gate=B.source_functions()[1:]
            cases=[b]
            for field,value in [('signature',None),('signature','different'),('runtime_signature',None),('recorded_noise_by_query',None)]:
                x=copy.deepcopy(b);x['input'][field]=value;cases.append(x)
            for row in cases:
                ca=W.compact_endpoint(a);cr=W.compact_endpoint(row)
                self.assertEqual(gate(a,row),gate(restore(ca),restore(cr)))
                self.assertEqual(cr['input_projection']['normalized_input_before_projection_sha256'],B.digest(row['input']))

    def test_readset_budget_unknown_repeat_and_SHA_refusal(self):
        raw=b'{}';uri='redacted-086ffe4b5966'
        ref={'uri':uri,**W.identity(raw)}
        class Store:
            def get(self,uri,limit):return raw
        s=W.ExactReads(Store(),[ref],time.time()+30)
        self.assertEqual(s.get(uri,2),raw)
        for wrong in (uri,uri+'unknown'):
            with self.assertRaisesRegex(ValueError,'unlisted_or_repeated'):s.get(wrong,2)
        with self.assertRaisesRegex(ValueError,'input_cap'):
            W.ExactReads(Store(),[dict(ref,bytes=W.MAX_INPUT+1)],time.time()+30)
        with self.assertRaisesRegex(ValueError,'exact_input_SHA'):
            W.ExactReads(Store(),[dict(ref,sha256='0'*64)],time.time()+30).get(uri,2)

    def test_existing_publisher_has_no_unknown_PUT_retry(self):
        class Budget:
            @contextmanager
            def archive_phase(self):yield
        class BOSWriter:
            bucket='redacted-417e8f2cb1dd';prefix='research-storage/adv-vla/fixture/'
            def __init__(self):self.calls=0
            def put(self,p,relative):self.calls+=1;raise OSError('unknown PUT')
        writer=BOSWriter();pub=A.phase_publisher(writer,Budget(),'bos://'+writer.bucket+'/'+writer.prefix,time.time()+30,W.file_identity(__file__))
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'result.json';p.write_bytes(b'{}')
            with self.assertRaises(OSError):pub('science',[p])
            with self.assertRaisesRegex(ValueError,'no_repeat'):pub('science',[p])
            self.assertEqual(writer.calls,1)

    def test_dependency_source_mismatch_and_Darwin_block_before_reads(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad.py';p.write_text('pass')
            with self.assertRaisesRegex(ValueError,'frozen_dependency_SHA'):
                W.dependencies({'files':[{'source_path':str(p),'package_path':'bad.py','bytes':4,'sha256':'0'*64}]})
        with patch.object(W.platform,'system',return_value='Darwin'):
            with self.assertRaisesRegex(ValueError,'cloud_CPU_only'):
                W.run('absent','absent',None,'absent','absent',time.time()+10)

    def test_actual_output_cap_keeps_room_for_explicit_failure_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            budget=W.OutputBudget(Path(td)/'out',maximum=W.SUMMARY_RESERVE+128)
            with self.assertRaisesRegex(ValueError,'output_cap32MiB'):
                with budget.gzip('cache.partial.gz') as f:
                    f.write(bytes(range(256))*200)
            budget.json('FAILURE-v1.json',{'cache_incomplete':True},reserve=False)
            self.assertTrue((budget.directory/'FAILURE-v1.json').is_file())


if __name__=='__main__':unittest.main()
