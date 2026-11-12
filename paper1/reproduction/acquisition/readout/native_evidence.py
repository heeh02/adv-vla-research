"""Existing local receipts for all 536 NATIVE metadata member-size gates.

The prior 241 successful reads prove a <=2 MiB bound, rather than an invented
exact size. The other 295 have actual manifest byte counts and source hashes.
No cloud calls or array reads occur here.
"""
from __future__ import annotations
import csv,hashlib,io,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
Q=HERE.parent
FINAL=Q/'plus-valid-mask-analysis-v1/full536-extension-v1/final-readset-v1/FINAL-READSET-v1.csv'
DIAGNOSTIC=Q/'plus-valid-mask-analysis-v1/failure-diagnostic-v1/NATIVE-INDEX-READBACK-v1.json'
PRIOR_RESULT=Q/'shared-channel-structure-v1/run-v1/readback/RESULT-v1.json'
READSET=HERE/'FINAL_READSET-v1.json'
OUTPUT=HERE/'NATIVE-SIZE-EVIDENCE-v2.json'
CAP=4*1024*1024
OLD_CAP=2*1024*1024
FINAL_SHA='20d6cbcdce76b7664defb1c60ce448cae04b623d0637257a36a00b6b15263a05'
PRIOR_SHA='d031ff40646a4383b5ec2e60aaaf3c0129ae4350f41b08e038098d2038e4cb84'

def need(ok,code):
    if not ok:raise RuntimeError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def rows(raw):return list(csv.DictReader(io.StringIO(raw.decode())))

def compute():
    final_raw=FINAL.read_bytes();prior_raw=PRIOR_RESULT.read_bytes()
    diagnostic_raw=DIAGNOSTIC.read_bytes();readset_raw=READSET.read_bytes()
    need(sha(final_raw)==FINAL_SHA and sha(prior_raw)==PRIOR_SHA,'actual_source_and_prior_run_SHA')
    final={r['case_id']:r for r in rows(final_raw)}
    case_rows={r['case_id']:r for r in json.loads(readset_raw)['cases']}
    need(len(final)==len(case_rows)==536 and set(final)==set(case_rows),'same_536_case_set')
    diagnostic=json.loads(diagnostic_raw)
    observed={r['case_id']:r for r in diagnostic['cases']}
    need(len(observed)==12,'exact_twelve_prior_native_overlimit')
    prior=json.loads(prior_raw)
    read={r['case_id'] for r in prior['case_read_proofs']}
    failed={r['case_id'] for r in prior['technical_read_failures']}
    need(len(read)==491 and len(failed)==45 and read.isdisjoint(failed)
         and read|failed==set(case_rows),'prior_actual_491_plus_45_receipt')
    result=[]
    for case_id in sorted(case_rows):
        source=final[case_id];case=case_rows[case_id]
        need(source['source_index_sha256']==case['index_sha256']
             and source['source_manifest_sha256']==case['manifest_sha256']
             and source['source_package_sha256']==case['source_package_sha256'],
             'same_exact_source_'+case_id)
        row={'case_id':case_id,'cohort':case['cohort'],
             'source_package_sha256':case['source_package_sha256'],
             'source_index_sha256':case['index_sha256'],
             'source_manifest_sha256':case['manifest_sha256']}
        if source['native_bytes']:
            size=int(source['native_bytes'])
            need(0<size<=CAP and len(source['native_sha256'])==64,
                 'new_actual_Native_limit_and_SHA_'+case_id)
            row.update(size_proof='actual_frozen_source_manifest',native_bytes=size,
                       native_sha256=source['native_sha256'])
        elif case_id in observed:
            original=observed[case_id];size=int(original['native_bytes'])
            need(case['cohort']=='v3' and case_id in failed
                 and original['source_index_sha256']==case['index_sha256']
                 and original['source_manifest_sha256']==case['manifest_sha256']
                 and OLD_CAP<size<=CAP and len(original['native_sha256'])==64,
                 'old_twelve_actual_Native_limit_and_SHA_'+case_id)
            row.update(size_proof='actual_old_diagnostic_manifest',native_bytes=size,
                       native_sha256=original['native_sha256'])
        else:
            need(case['cohort']=='v3' and case_id in read,
                 'old_successful_bounded_native_read_'+case_id)
            row.update(size_proof='prior_v1_header_SHA_verified_read_bound',
                       native_bytes_upper_bound=OLD_CAP,
                       exact_native_bytes_and_SHA_not_recorded=True)
        result.append(row)
    kinds={kind:sum(r['size_proof']==kind for r in result)for kind in
           ('actual_frozen_source_manifest','actual_old_diagnostic_manifest','prior_v1_header_SHA_verified_read_bound')}
    need(kinds=={'actual_frozen_source_manifest':283,'actual_old_diagnostic_manifest':12,
                 'prior_v1_header_SHA_verified_read_bound':241},'actual_536_native_size_proof_classes')
    actual_max=max(r['native_bytes']for r in result if 'native_bytes'in r)
    need(actual_max==3188932 and actual_max<=CAP and OLD_CAP<CAP,
         'actual_max_3188932_fits_fixed_4MiB')
    return {'schema':'plus-shared-channel-native-size-evidence-v2',
            'source_inputs':{'final_source_csv_sha256':sha(final_raw),
                             'old_native_diagnostic_sha256':sha(diagnostic_raw),
                             'prior_v1_result_sha256':sha(prior_raw),
                             'actual_readset_sha256':sha(readset_raw)},
            'planned':536,'proof_classes':kinds,'actual_max_native_bytes':actual_max,
            'reader_cap_bytes':CAP,'unknown_exact_prior_241_not_invented':True,
            'rows':result,'cloud_calls':0,'original_array_payloads_local':0}

def encoded():return (json.dumps(compute(),sort_keys=True,separators=(',',':'))+'\n').encode()

if __name__=='__main__':
    need(not OUTPUT.exists(),'immutable_native_size_evidence_exists')
    raw=encoded();OUTPUT.write_bytes(raw)
    print(json.dumps({'bytes':len(raw),'sha256':sha(raw),
                      'classes':compute()['proof_classes'],'actual_max_native_bytes':3188932,
                      'reader_cap_bytes':CAP,'cloud_calls':0},sort_keys=True))
