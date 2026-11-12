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


def run(inputs,dependency_root,borrowed_worker_path,store,work,output,deadline_epoch):
    from collections import defaultdict
    need(platform.system()=='Linux' and 0<deadline_epoch-time.time()<=3600,'bounded_cloud_CPU_only')
    inputs,work=Path(inputs),Path(work);task=json.loads((inputs/'INPUT-SPEC-v1.json').read_bytes());need(task['analysis_kind']=='fixed_RSS_q0_source_audit' and task['new_count']==107,'exact107_q0_rejected_floor_pairs')
    borrowed=load_borrowed(borrowed_worker_path,task['borrowed_worker']);need(borrowed.file_identity(__file__)==task['worker_source'],'actual_worker_SHA');deps=json.loads(borrowed.exact((inputs/'DEPENDENCIES-v1.json').read_bytes(),task['DEPENDENCIES-v1.json']));b,_repair,f,paths=borrowed.dependencies(deps,dependency_root);spec,docs=b.load_controls(paths['frozen-bridge/inputs-2648-v1/INPUT-SPEC-v1.json'].parent)
    endpoints=json.loads(bounded_gzip(borrowed.exact((inputs/'NEW-ENDPOINTS-v1.json.gz').read_bytes(),task['NEW-ENDPOINTS-v1.json.gz']),8*1024**2));expected={(r['model_id'],r['ordinal'],r['outcome_canonical_sha256'])for r in task['new_endpoint_identity']};need(len(endpoints)==len(expected)==107 and expected=={(r['model_id'],r['ordinal'],r['outcome_canonical_sha256'])for r in endpoints},'unique_exact107_endpoints')
    refs=task['readset']['objects'];need(len(refs)==108 and sum(r['bytes']for r in refs)==task['readset']['exact_total_bytes'],'exact_oneJDcache_plus107OUTCOME');transport=borrowed.ExactReads(store,refs,deadline_epoch);work.mkdir(parents=True,exist_ok=False);out=borrowed.OutputBudget(output);normalize,restore,gate=b.source_functions();pairs={(r['control_model'],r['ordinal']):r for r in task['selected_pairs']};need(len(pairs)==107,'one_control_per_pair')
    cache=spec['caches']['merged_delta'];need(cache==task['baseline_cache'],'actual_pinned_JD_cache');path=work/'JD-delta.jsonl.gz';transport.download(cache['uri'],path,cache)
    selected=(r for r in b.stream_cache(path,7234)if(r['model_id'],r['ordinal'])in pairs);controls={}
    for r in b.cache_rows(docs['historical_plan'],'merged_delta',f.tagged_cache_rows(selected,cache),restore,spec['case_list_identity']):
        b.validate_endpoint(r,spec['models'],docs['historical_plan']['population_cases']);controls[(r['model_id'],r['ordinal'])]=r
    need(set(controls)==set(pairs),'all107_parent_sources_located')
    summary=borrowed.exact((inputs/'NEW-SUMMARIES-v1.json.gz').read_bytes(),task['NEW-SUMMARIES-v1.json.gz']);overlay=SummaryOverlay(summary,endpoints,transport,borrowed);traces=[];fields_cache=[];counts=Counter();groups=defaultdict(Counter)
    for raw in sorted(endpoints,key=lambda r:(r['model_id'],r['ordinal'])):
        need(time.time()<deadline_epoch,'absolute_science_deadline');b.validate_endpoint(raw,spec['models'],docs['historical_plan']['population_cases']);r=b.materialize_t30(raw,spec['models'][raw['model_id']],overlay,normalize);r['population_identity']=spec['case_list_identity'];r=restore(r);seed=0 if '-s0-'in r['model_id']else 1;a=controls[(f'T-s{seed}-u20000-EMA',r['ordinal'])];pair=pairs[(a['model_id'],r['ordinal'])];need(a['outcome_canonical_sha256']==pair['control_outcome_SHA']and r['outcome_canonical_sha256']==pair['treatment_outcome_SHA'],'same_preselected_pair_actual_identity')
        diag=b.pair_diagnostics(a,r,gate);need(diag==pair['diagnostic'] and diag['reason']=='q0_input_difference','original_gate_reproduced_no_override')
        af=a.get('input',{}).get('fields');rf=r.get('input',{}).get('fields');need(isinstance(af,dict)and isinstance(rf,dict),'actual_full_q0_fields_present')
        equal=[];different=[];missing=[];typed=[]
        for k in sorted(set(af)|set(rf)):
            x,y=af.get(k),rf.get(k)
            if x is None or y is None:missing.append(k)
            elif borrowed.canonical(x)==borrowed.canonical(y):equal.append(k)
            else:
                different.append(k)
                if isinstance(x,dict)and isinstance(y,dict)and 'hash_domain'in x and 'hash_domain'in y:
                    typed.append({'field':k,'left_domain':x['hash_domain'],'right_domain':y['hash_domain'],'dtype_shape_domain_bytes_equal':all(x.get(z)==y.get(z)for z in ['dtype','shape','hash_domain','order','bytes']),'payload_SHA_equal':x.get('sha256')==y.get('sha256')})
        counts.update(different);case=raw['official_case'];g=groups[(seed,case['suite'])];g['pairs']+=1;g.update(different)
        traces.append(dict(seed=seed,ordinal=r['ordinal'],suite=case['suite'],category=case['category'],control_model=a['model_id'],treatment_model=r['model_id'],diagnostic=diag,fields_equal=equal,fields_different=different,fields_missing=missing,typed_array_identity_differences=typed,control_run=a['run_id'],treatment_run=r['run_id'],source_trace_control=f.source_trace(a),source_trace_treatment=f.source_trace(r)))
        fields_cache.append(dict(seed=seed,ordinal=r['ordinal'],control_fields=af,treatment_fields=rf,control_signature=a['input']['signature'],treatment_signature=r['input']['signature'],control_outcome_SHA=a['outcome_canonical_sha256'],treatment_outcome_SHA=r['outcome_canonical_sha256']))
    need(len(transport.seen)==108 and transport.bytes==task['readset']['exact_total_bytes']and len(overlay.seen)==107,'all_exact_refs_once')
    with out.gzip('NEW-PAIR-TRACE-v1.jsonl.gz')as stream:
        for row in traces:stream.write(borrowed.canonical(row)+b'\n')
    with out.gzip('NORMALIZED-NEW-v1.jsonl.gz')as stream:
        for row in fields_cache:stream.write(borrowed.canonical(row)+b'\n')
    result={'schema':'paper3-added-endpoint-pairs-v1','analysis_kind':task['analysis_kind'],'status':'computed_unpublished','model_id':task['model_id'],'run_id':task['run_id'],'parent_T30_total':3626,'parent_snapshot':task['parent_snapshot'],'new_endpoint_identity':task['new_endpoint_identity'],'new_count':107,'new_trace_rows':107,'new_OUTCOME_GETs':107,'old_OUTCOME_GETs':0,'baseline_GETs':1,'input_objects':108,'input_bytes':transport.bytes,'input_spec':borrowed.file_identity(inputs/'INPUT-SPEC-v1.json'),'worker_source':borrowed.file_identity(__file__),'q0_runtime_noise_gate_changed':False,'GPU':0,'model_calls':0,'full_population_evaluation_complete':False,'CI':None,'training_seeds_pooled':False,'field_difference_counts':dict(counts),'groups':[dict(seed=k[0],suite=k[1],**dict(v))for k,v in sorted(groups.items())],'all107_original_rejections_reproduced':True,'missing_fields_in_any_pair':sum(bool(x['fields_missing'])for x in traces),'physical_or_pixel_magnitude_measured':False,'hardware_causal_attribution':False,'cache_contents':'Exact full q0 field dictionaries for107 preselected input-discordant pairs; metadata hashes only, no arrays decoded.'}
    out.json('INPUT-RECEIPT-v1.json',{'schema':'paper3-added-endpoint-input-receipt-v1','readset':refs,'verified_objects':108,'verified_bytes':transport.bytes,'new_OUTCOME_GETs':107,'old_OUTCOME_GETs':0,'baseline_cache_GETs':1,'input_spec':result['input_spec'],'worker_source':result['worker_source']});out.json('RESULT-v1.json',result,reserve=False);return result
