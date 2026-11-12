"""Cloud-only consumer of seven pinned scalar JSON caches; no cloud SDK."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
import time
import traceback
import analysis as A

HERE = Path(__file__).resolve().parent
RESULT = 'dev16-bridge-readout-result.json'
INPUT_NAMES = ('FIELD-COMPARISONS.json', 'FEATURE-COMPARISONS.json',
               'ENDPOINT-COMPARISONS.json', 'dev16-feature-bridge-result.json',
               'QA.json', 'PROTOCOL.json', 'ARCHIVE-RECEIPT.json')
OUTPUT_NAMES = ('SUMMARY.json', 'QA.json', 'FULL-SUPPORT.csv.gz', 'PROTOCOL.json',
                'F1-BRIDGE-ERROR.pdf', 'F1-BRIDGE-ERROR.png', 'F1-BRIDGE-ERROR.svg', RESULT)


def ident(path):
    raw = Path(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    with Path(path).open('xb') as f:
        f.write((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode())
    return ident(path)


def load(stage, manifest, *, synthetic=False):
    A.need(synthetic or sys.platform.startswith('linux'), 'actual_scalar_readout_cloud_Linux_only')
    refs = manifest['input_objects']
    A.need(len(refs) == manifest['input_object_count'] == 7 and
           {r['path'] for r in refs} == set(INPUT_NAMES), 'exact7_input_objects')
    A.need(manifest['input_body_bytes'] == sum(r['bytes'] for r in refs) <= 2*1024**2,
           'bounded_exact_input_bytes')
    stage=Path(stage)
    A.need({p.name for p in stage.iterdir()} == set(INPUT_NAMES), 'exact_stage_roster')
    docs={}
    for r in refs:
        p=stage/r['path']
        A.need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 2*1024**2,
               'bounded_regular_JSON_input')
        A.need(ident(p) == {k:r[k] for k in ('bytes','sha256')}, 'input_whole_SHA')
        docs[r['path']]=json.loads(p.read_bytes())
    archive=docs['ARCHIVE-RECEIPT.json']; producer=docs['dev16-feature-bridge-result.json']
    A.need(archive['run_id'] == 'jinwen-paper2-dev16-feature-bridge-20261020-v2' and
           archive['archive_complete'] is True and archive['dev16_feature_bridge_analysis_complete'] is True
           and archive['science_archive']['complete_for_observed_files'] is True and
           archive['science_archive']['errors'] == [], 'complete_parent_archive')
    bindings={r['name']:r for r in archive['science_archive']['objects']}
    produced={r['path']:r for r in producer['outputs']}
    for r in refs:
        if r['path']=='ARCHIVE-RECEIPT.json': continue
        A.need(r['path'] in bindings and
               {k:bindings[r['path']][k] for k in ('bytes','sha256','uri')} ==
               {k:r[k] for k in ('bytes','sha256','uri')}, 'parent_archive_binding')
        if r['path'] != 'dev16-feature-bridge-result.json':
            A.need(r['path'] in produced and
                   {k:produced[r['path']][k] for k in ('bytes','sha256')} ==
                   {k:r[k] for k in ('bytes','sha256')}, 'producer_binds_cached_outputs')
    return A.aggregate(docs[INPUT_NAMES[0]], docs[INPUT_NAMES[1]], docs[INPUT_NAMES[2]],
                       producer, docs['QA.json'], docs['PROTOCOL.json'], synthetic=synthetic)


def support_csv(rows, path):
    columns=['source_kind','root_id','query_index']+sorted(set().union(*(set(r) for r in rows))-
              {'source_kind','root_id','query_index'})
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
                             if isinstance(v,(dict,list)) else v for k,v in row.items()})
    return len(rows)


def run(stage, out, source_sha, deadline):
    A.need(sys.platform.startswith('linux'), 'actual_scalar_readout_cloud_Linux_only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    A.need(not out.is_symlink() and not any(out.iterdir()), 'new_empty_owned_output')
    started=time.monotonic();phase='source'
    result={'schema':'paper2-dev16-bridge-readout-result-v1','status':'INCOMPLETE',
            'synthetic':False,'input_qualified':False,'analysis_complete':False,'render_complete':False,
            'new_model_calls':0,'new_env_steps':0,'new_fits':0,'training_updates':0,
            'JPFS_new_bytes':0,'GPU_count':0,'native_qualification_reexecuted':False,
            'raw_neural_or_model_objects_read':False}
    def check():A.need(time.time()<deadline,'absolute_science_deadline')
    try:
        check();source=ident(HERE/'SOURCE-MANIFEST.json')
        A.need(source['sha256']==source_sha,'externally_pinned_source_manifest')
        manifest=json.loads((HERE/'SOURCE-MANIFEST.json').read_bytes())
        A.need(manifest['schema']=='paper2-dev16-bridge-readout-source-v1','source_schema')
        for row in manifest['files']:
            p=HERE/row['path']
            A.need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(HERE),
                   'owned_source_member')
            A.need(ident(p)=={k:row[k] for k in ('bytes','sha256')},'source_member_SHA')
        inputs=json.loads((HERE/'INPUTS.json').read_bytes())
        protocol=json.loads((HERE/'PROTOCOL.json').read_bytes())
        result.update(source_manifest=source,inputs_manifest=ident(HERE/'INPUTS.json'),
                      protocol=ident(HERE/'PROTOCOL.json'),scope=protocol['scope'])
        phase='scalar_cache_QA';data=load(Path(stage).resolve(strict=True),inputs)
        result.update(input_qualified=True,input_objects_verified=7,input_body_bytes=inputs['input_body_bytes'])
        check();phase='render'
        import renderer
        rendered=renderer.render(data,out);check()
        summary=data['summary'];summary.update(source_inputs=inputs['input_objects'],
                                                caption=rendered['caption'],figure=rendered['plotted_values'])
        write(out/'SUMMARY.json',summary)
        rows=support_csv(data['support'],out/'FULL-SUPPORT.csv.gz')
        A.need(rows==1000,'all1000_support_rows')
        qa={'schema':'paper2-dev16-bridge-readout-QA-v1','pass':True,'synthetic':False,
            'counts':{**A.COUNTS,'CSV_rows':rows},'source_inputs':inputs['input_objects'],
            'layout':rendered['layout'],'all8_per_group':True,
            'strict_ratio_counts_use_original_scalars':True,'native_qualification_reexecuted':False,
            'arithmetic_check_tolerance':5e-12,'equivalence_threshold':None,
            'equal_payload_RMS_last_bit_differences':data['equal_payload_RMS_last_bit_differences'],
            'equal_payload_RMS_last_bit_difference_count':len(data['equal_payload_RMS_last_bit_differences']),
            'scalar_RMS_tolerance_changes_native_equality_or_ratio_counts':False,
            'new_model_calls':0,'new_env_steps':0,'new_fits':0}
        write(out/'QA.json',qa)
        with (out/'PROTOCOL.json').open('xb') as f:f.write((HERE/'PROTOCOL.json').read_bytes())
        outputs=[{'path':p.name,**ident(p)} for p in sorted(out.iterdir())]
        A.need({r['path'] for r in outputs}==set(OUTPUT_NAMES)-{RESULT},'exact_output_roster')
        A.need(all(r['bytes']<=8*1024**2 for r in outputs) and
               sum(r['bytes'] for r in outputs)+1024**2<=16*1024**2,'bounded_outputs')
        check()
        result.update(status='DEV16_BRIDGE_READOUT_COMPLETE',analysis_complete=True,render_complete=True,
                      counts={**A.COUNTS,'CSV_rows':rows},outputs=outputs,output_names=list(OUTPUT_NAMES))
    except BaseException as err:
        result.update(status='FAILED_PRESERVED',analysis_complete=False,render_complete=False,
                      failure={'phase':phase,'type':type(err).__name__,
                        'code':str(err) if isinstance(err,ValueError) and len(str(err))<180 else None,
                        'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
                                  for f in traceback.extract_tb(err.__traceback__)[-8:]]})
    result['elapsed_seconds']=time.monotonic()-started
    write(out/RESULT,result)
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('stage','out','source-manifest-sha256','deadline-epoch'):p.add_argument('--'+name,required=True)
    a=p.parse_args();r=run(a.stage,a.out,a.source_manifest_sha256,float(a.deadline_epoch))
    print(json.dumps({'status':r['status'],'analysis_complete':r['analysis_complete'],
                      'failure':r.get('failure')}),flush=True)
    return 0 if r['analysis_complete'] else 2


if __name__=='__main__':raise SystemExit(main())
