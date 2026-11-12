"""Tiny fixtures only; original gate/statistic code and byte-identical CSV reference."""
import base64,copy,gzip,inspect,json,tempfile,time,unittest
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import worker as W
HERE=Path(__file__).resolve().parent;C=HERE.parents[1]
OLD=W.load_module('_prior_increment_tests_for_fixture_only',C/'comparison-v1/incremental-floor1-u504-execution-v1/test_increment.py')
B,R,F=OLD.B,OLD.R,OLD.F
# Match the actual floor0 extension: the corresponding reheat-s0 endpoint is
# absent at new ordinals. Never relax the gate to infer an old T30 input.
fixture_source=inspect.getsource(OLD.fixture).replace("'T-reheat-s1-u30000-EMA'","'T-reheat-s0-u30000-EMA'")
exec(compile(fixture_source,__file__+'::same_scope_tiny_fixture','exec'),OLD.__dict__)

def inputs(root,spec,docs,store,tables,new):
    inp=root/'inputs';inp.mkdir()
    decoded={r['original_name']:gzip.decompress(store.objects[r['uri']])for r in tables[:2]}
    singles,pairs,_,_,_=R.validate_tables(B,spec,docs,decoded['SINGLE-GROUPS-v1.csv'],decoded['PAIRED-GROUPS-v1.csv'])
    expected=dict(T30_total=len(docs['t30_endpoints']),snapshot_comparison_complete=True,
                  single_endpoints=[singles[m,'all','ALL']for m in spec['models']],
                  paired_contrasts=[pairs[c['id'],'all','ALL']for c in spec['comparison']['contrasts']])
    prior=[dict(name=r['original_name']+'.gz',**{k:r[k]for k in ('uri','bytes','sha256','decoded_limit_bytes')})for r in tables[:2]]
    refs=[{k:r[k]for k in ('uri','bytes','sha256')}for r in list(spec['caches'].values())+prior+[new['outcome']]]
    blobs={'NEW-ENDPOINTS-v1.json.gz':gzip.compress(W.canonical([new]),mtime=0),
           'NEW-SUMMARIES-v1.json.gz':gzip.compress(W.canonical({new['summary']['uri']:base64.b64encode(store.objects[new['summary']['uri']]).decode()}),mtime=0),
           'DEPENDENCIES-v1.json':W.canonical({'files':[]}),
           'EXACT-READSET-v1.json':W.canonical({'objects':refs}),
           'OLD-3152-RESULT-v1.json':W.canonical(expected)}
    for name,raw in blobs.items():(inp/name).write_bytes(raw)
    task=dict(schema='paper3-floor0-u1136-added474-input-v1',worker_source=W.file_identity(W.__file__),
              case_list_identity=spec['case_list_identity'],prior_tables=prior,
              prior_trace={'uri':'redacted-9b0b1d941cc2','bytes':1,'sha256':'a'*64},
              output_prefix='redacted-f1b824871b4b')
    for key,name in [('new_endpoints','NEW-ENDPOINTS-v1.json.gz'),('new_summaries','NEW-SUMMARIES-v1.json.gz'),
                     ('dependencies','DEPENDENCIES-v1.json'),('readset','EXACT-READSET-v1.json'),('prior_result','OLD-3152-RESULT-v1.json')]:
        task[key]=W.file_identity(inp/name)
    (inp/'INPUT-SPEC-v1.json').write_bytes(W.canonical(task))
    return inp,task,refs

class Tests(unittest.TestCase):
    def run_fixture(self,fail_delta=False):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);spec,docs,store,tables,rs,baseline,old,new=OLD.fixture(root)
            inp,task,refs=inputs(root,spec,docs,store,tables,new)
            stream=B.stream_cache;publications={};phases=[]
            def counted(path,count):return stream(path,6 if Path(path).name=='old_cache.jsonl.gz'else 0)
            def publish(phase,paths):
                phases.append(phase);publications[phase]={p.name:p.read_bytes()for p in paths}
                return [dict(relative=p.name,uri=task['output_prefix']+p.name,**W.file_identity(p),whole_GET_SHA_verified=True)for p in paths]
            gzip_method=W.OutputBudget.gzip
            def gz(self,name,**kwargs):
                if fail_delta and name.startswith('NORMALIZED-NEW'):raise ValueError('new474_delta_compressed_cap24MiB')
                return gzip_method(self,name,**kwargs)
            with ExitStack()as stack:
                stack.enter_context(patch.multiple(W,NEW={1},NEW_COUNT=1,OLD_T30_COUNT=len(docs['t30_endpoints']),POPULATION_COUNT=2))
                stack.enter_context(patch.object(W.platform,'system',return_value='Linux'))
                stack.enter_context(patch.object(W,'dependencies',return_value=(B,R,F,{'frozen-bridge/inputs-2648-v1/INPUT-SPEC-v1.json':root/'controls/INPUT-SPEC-v1.json'})))
                stack.enter_context(patch.object(B,'load_controls',return_value=(spec,docs)))
                stack.enter_context(patch.object(W,'extend_prior_snapshot',side_effect=lambda b,s,d,i,t:(s,d)))
                stack.enter_context(patch.object(B,'stream_cache',side_effect=counted))
                stack.enter_context(patch.object(W.OutputBudget,'gzip',gz))
                result=W.run(inp,None,store,root/'work',root/'output',time.time()+600,publish_phase=publish)
            normalize,restore,gate=B.source_functions();fresh=B.materialize_t30(new,spec['models'][W.MODEL],store,normalize);fresh['population_identity']=spec['case_list_identity']
            full={('Plus',x['model_id'],x['ordinal']):x for x in baseline+old+[restore(fresh)]}
            reference=root/'full';reference.mkdir()
            try:B.write_tables(spec,docs['historical_plan']['population_cases'],full,gate,reference)
            except KeyError as error:self.assertEqual(error.args,('summary',))
            for name in ('SINGLE-GROUPS-v1.csv','PAIRED-GROUPS-v1.csv'):
                self.assertEqual(gzip.decompress(publications['science'][name+'.gz']),(reference/name).read_bytes())
            self.assertEqual(result['delta_trace_rows'],3)
            delta=[json.loads(x)for x in gzip.decompress(publications['science']['PAIRED-CASE-QUALIFICATION-DELTA-v1.jsonl.gz']).splitlines()]
            self.assertEqual(len(delta),3);self.assertTrue(all(x['ordinal']==1 and x['qualification_evidence']=='direct_original_gate_new_endpoint'for x in delta))
            report=json.loads(publications['science']['QUALIFIED-COVERAGE-CHANGE-v1.json'])
            self.assertTrue(all(r['added_closed_endpoint_count']in (0,1)for r in report['rows']))
            self.assertFalse(report['population_effect_identified'])
            self.assertNotIn('NORMALIZED-BASELINE-AND-DELTA-v1.jsonl.gz',{n for docs in publications.values()for n in docs})
            self.assertEqual(phases[0],'science');self.assertEqual(phases[-1],'delivery')
            if fail_delta:
                self.assertEqual(result['cache_status'],'NEW474_DELTA_INCOMPLETE_STATISTICS_PRESERVED')
                self.assertTrue(result['snapshot_cloud_durable']);self.assertIsNone(result['cache_publication'])
            else:
                cache=[json.loads(x)for x in gzip.decompress(publications['cache']['NORMALIZED-NEW474-v1.jsonl.gz']).splitlines()]
                self.assertEqual([(x['model_id'],x['ordinal'])for x in cache],[(W.MODEL,1)])
                self.assertEqual(result['cache_status'],'COMPLETE_VERIFIED_NEW474_DELTA')
            return result

    def test_increment_is_full_frozen_statistic_equivalent_and_new_only_cache(self):
        self.run_fixture()

    def test_delta_overflow_preserves_published_statistics_and_is_not_complete(self):
        self.run_fixture(True)

    def test_new_input_gate_does_not_inherit_old_qualified_label(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);spec,docs,store,tables,rs,baseline,old,new=OLD.fixture(root)
            raw=[gzip.decompress(store.objects[x['uri']])for x in tables[:2]]
            singles,pairs,endpoints,labels,groups=R.validate_tables(B,spec,docs,*raw)
            normalize,restore,gate=B.source_functions();row=B.materialize_t30(new,spec['models'][W.MODEL],store,normalize)
            row['population_identity']=spec['case_list_identity'];row=restore(row)
            row['input']['signature']='changed_new_q0_signature'
            merged=W.MergedCells(B,spec,docs['historical_plan']['population_cases'],endpoints,labels,docs['t30_endpoints'],baseline,[row])
            self.assertEqual(merged.direct['floor_s0_T30_minus_T20',1]['reason'],'q0_input_difference')

    def test_actual_prepared_readset_contains_only_two_caches_two_tables474_outcomes(self):
        inp=HERE/'prepared-v2'
        if not inp.exists():inp=HERE/'prepared-v1'
        d=json.loads((inp/'EXACT-READSET-v1.json').read_bytes())
        self.assertEqual(d['object_count'],478);self.assertEqual(d['exact_total_bytes'],322946637)
        self.assertEqual(sum(r['uri'].endswith('/OUTCOME-v1.json')for r in d['objects']),474)
        self.assertFalse(any('/CASE_SUMMARY-v1.json' in r['uri']for r in d['objects']))
        self.assertEqual(d['new474_OUTCOME_bytes'],74297967)

if __name__=='__main__':unittest.main()
