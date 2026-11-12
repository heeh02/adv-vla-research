"""Read only exact new OUTCOMEs and frozen baseline caches on cloud CPU.

No provider API, model/array IO, adoption, registry write or automatic publish.
Original frozen normalizer and pair gate are called for every newly affected
pair. Prior 3626 tables and old outcomes remain immutable references.
"""

import argparse
import base64
from collections import Counter
import gzip
import importlib.util
import io
import json
from pathlib import Path
import platform
import time


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


PARENT_RESULT_SHA = "5938d81bbf9fa746c2944e84aca6d150dcb7dade9c5e1b4d3576f770f18ef6a7"
PARENT_TABLE_SHAS = {
    "870059244895a2ed6f716cd48b11340de8945caf2ad1cb2d052b719e039909bd",
    "aa37c54aeec3dfc3264922afeeabd772c96e8d972d07a7a4f678c21eb97cf8a7",
}


def bounded_gzip(raw, limit):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        value = stream.read(limit + 1)
    need(len(value) <= limit, "bounded_control_gzip")
    return value


def load_borrowed(path, ref):
    p = Path(path)
    spec = importlib.util.spec_from_file_location("_paper3_borrowed_increment", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(module.file_identity(p) == {k: ref[k] for k in ("bytes", "sha256")}, "borrowed_worker_SHA")
    return module


class SummaryOverlay:
    """Locally bundled exact summaries; only original OUTCOME uses GET."""

    def __init__(self, body, rows, transport, borrowed):
        raw = json.loads(bounded_gzip(body, 8 * 1024**2))
        self.refs = {r["summary"]["uri"]: r["summary"] for r in rows}
        need(
            len(self.refs) == len(rows) == len(raw) and set(raw) == set(self.refs), "complete_unique_packaged_summaries"
        )
        self.summaries = {}
        for uri, encoded in raw.items():
            data = base64.b64decode(encoded, validate=True)
            self.summaries[uri] = borrowed.exact(data, self.refs[uri])
        self.transport, self.seen = transport, set()

    def get(self, uri, limit):
        if uri not in self.summaries:
            return self.transport.get(uri, limit)
        need(uri not in self.seen and limit == self.refs[uri]["bytes"], "one_exact_local_summary_read")
        self.seen.add(uri)
        return self.summaries[uri]


def run(inputs, dependency_root, borrowed_worker_path, store, work, output, deadline_epoch):
    from fractions import Fraction
    need(platform.system()=='Linux' and 0<deadline_epoch-time.time()<=3600,'bounded_cloud_CPU_only')
    inputs,work=Path(inputs),Path(work);task=json.loads((inputs/'INPUT-SPEC-v1.json').read_bytes())
    need(task['analysis_kind']=='fixed_RSS_panel_partial_joint' and task['GPU']==task['model_calls']==0,'explicit_four_endpoint_science_scope')
    borrowed=load_borrowed(borrowed_worker_path,task['borrowed_worker']);need(borrowed.file_identity(__file__)==task['worker_source'],'actual_worker_SHA')
    deps=json.loads(borrowed.exact((inputs/'DEPENDENCIES-v1.json').read_bytes(),task['DEPENDENCIES-v1.json']))
    b,_repair,f,paths=borrowed.dependencies(deps,dependency_root);spec,docs=b.load_controls(paths['frozen-bridge/inputs-2648-v1/INPUT-SPEC-v1.json'].parent)
    endpoints=json.loads(bounded_gzip(borrowed.exact((inputs/'NEW-ENDPOINTS-v1.json.gz').read_bytes(),task['NEW-ENDPOINTS-v1.json.gz']),8*1024**2))
    panel=task['panel'];ords=set(panel['ordinals']);need(len(ords)==1053 and b.digest(panel['case_definitions'])==task['selected_case_definition_SHA'],'fixed1053_case_support')
    need(all(docs['historical_plan']['population_cases'][o]==c for o,c in zip(panel['ordinals'],panel['case_definitions'])),'original_cases_not_redefined')
    expected={(x['model_id'],x['ordinal'],x['outcome_canonical_sha256'])for x in task['new_endpoint_identity']}
    need(len(endpoints)==423==len(expected) and expected=={(x['model_id'],x['ordinal'],x['outcome_canonical_sha256'])for x in endpoints},'exact423_new_endpoints')
    for row in endpoints:
        b.validate_endpoint(row,spec['models'],docs['historical_plan']['population_cases'])
        need(row['ordinal']in ords and row['model_id']in task['source_models'] and row['run_id']==task['source_runs'][row['model_id']] and row['attempt']==0,'frozen_panel_four_exact_runs')
    refs=task['readset']['objects'];need(len(refs)==425 and len({r['uri']for r in refs})==425,'one_combined_unique_readset')
    transport=borrowed.ExactReads(store,refs,deadline_epoch);work.mkdir(parents=True,exist_ok=False);out=borrowed.OutputBudget(output)
    normalize,restore,gate=b.source_functions();buckets={};selected=0
    controls={'original','T-s0-u20000-EMA','T-s1-u20000-EMA'}
    for name,count in [('old_cache',106958),('merged_delta',7234)]:
        ref=spec['caches'][name];path=work/(name+'.jsonl.gz');transport.download(ref['uri'],path,ref)
        rawrows=(x for x in b.stream_cache(path,count) if x['ordinal']in ords and x['model_id']in controls)
        for row in b.cache_rows(docs['historical_plan'],name,f.tagged_cache_rows(rawrows,ref),restore,spec['case_list_identity']):
            need(time.time()<deadline_epoch,'science_deadline');b.validate_endpoint(row,spec['models'],docs['historical_plan']['population_cases']);b.add_endpoint(buckets,borrowed.compact_endpoint(row));selected+=1
    summary=borrowed.exact((inputs/'NEW-SUMMARIES-v1.json.gz').read_bytes(),task['NEW-SUMMARIES-v1.json.gz']);overlay=SummaryOverlay(summary,endpoints,transport,borrowed)
    for raw in sorted(endpoints,key=lambda r:(r['model_id'],r['ordinal'])):
        need(time.time()<deadline_epoch,'science_deadline');row=b.materialize_t30(raw,spec['models'][raw['model_id']],overlay,normalize);row['population_identity']=spec['case_list_identity'];b.add_endpoint(buckets,borrowed.compact_endpoint(restore(row)))
    need(len(overlay.seen)==423 and len(transport.seen)==425 and transport.bytes==task['readset']['exact_total_bytes'],'exact_once_each_input')
    weights={r['ordinal']:r for r in task['weights']['rows']if r['panel']=='Plus'};need(set(weights)==ords,'complete_fixed_weights')
    traces=[];groups=[];target_groups=[]
    for contrast in task['contrasts']:
        cells=Counter();reasons=Counter();target={t:{'known_weight':Fraction(0),'weighted_delta_known':Fraction(0),'qualified':0,'reasons':Counter()}for t in ['mapped_task_macro','unmapped_layout_descriptive']}
        for o in panel['ordinals']:
            a=buckets.get(('Plus',contrast['control'],o));z=buckets.get(('Plus',contrast['treatment'],o))
            if a and z:need(a['case_definition_sha256']==z['case_definition_sha256'] and a['population_identity']==z['population_identity'],'exact_same_case_population')
            diagnostic=b.pair_diagnostics(a,z,gate);why=diagnostic['reason'];reasons[why]+=1;wrow=weights[o];entry=target[wrow['target']];entry['reasons'][why]+=1;cell=None
            if why=='qualified_recorded_pair':
                cell=('S'if a['success']else'F')+('S'if z['success']else'F');cells[cell]+=1;w=Fraction(wrow['weight_numerator'],wrow['weight_denominator']);entry['known_weight']+=w;entry['weighted_delta_known']+=w*(int(z['success'])-int(a['success']));entry['qualified']+=1
            traces.append({'contrast':contrast['id'],'ordinal':o,'diagnostic':diagnostic,'cell':cell,'control_source':f.source_trace(a),'treatment_source':f.source_trace(z),'target':wrow['target'],'weight_numerator':wrow['weight_numerator'],'weight_denominator':wrow['weight_denominator']})
        q=sum(cells.values());net=cells['FS']-cells['SF'];groups.append(dict(contrast=contrast['id'],control=contrast['control'],treatment=contrast['treatment'],new_cohort_n=1053,qualified_pairs=q,SS=cells['SS'],SF=cells['SF'],FS=cells['FS'],FF=cells['FF'],gains=cells['FS'],harms=cells['SF'],net_qualified_count=net,delta_SR=None,qualified_subset_delta_SR=net/q if q else None,pairing_reasons=dict(reasons),strict_completion_bounds=[(net-(1053-q))/1053,(net+(1053-q))/1053]))
        for name,v in target.items():
            known=v['known_weight'];delta=v['weighted_delta_known'];missing=1-known;need(0<=known<=1,'valid_weight_support')
            target_groups.append(dict(contrast=contrast['id'],target=name,qualified=v['qualified'],known_weight=float(known),unidentified_weight=float(missing),weighted_delta_known=float(delta),full_target_bounds=[float(delta-missing),float(delta+missing)],qualified_subset_mean=float(delta/known)if known else None,bounds_are_not_CI=True,reasons=dict(v['reasons'])))
    with out.gzip('NEW-PAIR-TRACE-v1.jsonl.gz')as stream:
        for row in traces:stream.write(borrowed.canonical(row)+b'\n')
    with out.gzip('NORMALIZED-NEW-v1.jsonl.gz')as stream:
        for key,row in sorted(buckets.items()):stream.write(borrowed.canonical(row)+b'\n')
    result={'schema':'paper3-added-endpoint-pairs-v1','analysis_kind':task['analysis_kind'],'status':'computed_unpublished','model_id':task['model_id'],'run_id':task['run_id'],'parent_T30_total':3626,'parent_snapshot':task['parent_snapshot'],'new_endpoint_identity':task['new_endpoint_identity'],'new_count':423,'new_trace_rows':len(traces),'new_cohort_pairs':groups,'fixed_weighted_targets':target_groups,'new_OUTCOME_GETs':423,'old_OUTCOME_GETs':0,'baseline_GETs':2,'baseline_rows_scanned':114192,'baseline_rows_selected':selected,'input_objects':len(transport.seen),'input_bytes':transport.bytes,'input_spec':borrowed.file_identity(inputs/'INPUT-SPEC-v1.json'),'worker_source':borrowed.file_identity(__file__),'q0_runtime_noise_gate_changed':False,'GPU':0,'model_calls':0,'full_population_evaluation_complete':False,'CI':None,'training_seeds_pooled':False,'cache_contents':'All already-available T20/Original endpoints on fixed1053 plus new423 T30; other old T30 endpoints not yet included. No full historical cache republished.','cache_counts':dict(Counter(row['model_id']for row in buckets.values())),'fixed_panel_sha256':task['panel_ref']['sha256'],'fixed_weights_sha256':task['weights_ref']['sha256'],'interpretation':'Partial support; missing old T30 in this cache is not absence of its already adopted outcome. Join existing derived sources before any global scientific decision.'}
    out.json('INPUT-RECEIPT-v1.json',{'schema':'paper3-added-endpoint-input-receipt-v1','readset':refs,'verified_objects':len(transport.seen),'verified_bytes':transport.bytes,'new_OUTCOME_GETs':423,'old_OUTCOME_GETs':0,'baseline_cache_GETs':2,'summary_cloud_GETs':0,'input_spec':result['input_spec'],'worker_source':result['worker_source']});out.json('RESULT-v1.json',result,reserve=False);return result
