"""Cloud-only cached-score audit and rendering. Five GETs external; no refit."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import traceback
import zipfile
import numpy as np
import audit as A

HERE=Path(__file__).resolve().parent
RESULT='signed-figures-result.json'
INPUT_NAMES={'signed-transfer-result.json','evaluation.json.gz','predictions.npz','fit-arrays.npz','fit-manifest.json'}
FIGURE_IDS=('F1-CAPACITY-TRANSFER','F2-LOCAL-STRUCTURE','F3-ROOT-COUNTEREXAMPLES','F4-PAIRED-INTERACTION')
OUTPUT_NAMES=tuple(f+'.'+ext for f in FIGURE_IDS for ext in ('png','pdf','svg'))+(
    'SUPPLEMENT-ATLAS.pdf','ATLAS-CONTACT-01.png','ATLAS-CONTACT-02.png',
    'FULL-SCORES.csv.gz','FIGURE-VALUES.json','QA.json','CAPTIONS.json','RENDER-MANIFEST.json',RESULT)
PRODUCER_SOURCE='32807f1f87df62278adad94fe7a336b6ebf0ee70a12d8693bef74e2eae53b019'


def ident(path):
    p=Path(path);h=hashlib.sha256();size=0
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
    return {'bytes':size,'sha256':h.hexdigest()}


def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,float):
        A.need(not math.isinf(x),'no_infinite_report')
        return None if math.isnan(x)else x
    if isinstance(x,dict):return {str(k):serial(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v)for v in x]
    return x


def write(path,value):
    raw=(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with Path(path).open('xb')as f:f.write(raw)
    return ident(path)


def load_npz(path):
    with zipfile.ZipFile(path)as z:
        infos=z.infolist();A.need(len(infos)<=512 and len({i.filename for i in infos})==len(infos),'bounded_unique_npz_members')
        A.need(sum(i.file_size for i in infos)<=128*1024**2 and all(i.file_size<=64*1024**2 and
            i.filename.endswith('.npy')and '/'not in i.filename and '\\'not in i.filename for i in infos),'bounded_flat_npz_arrays')
    with np.load(path,allow_pickle=False)as z:
        result={k:np.array(z[k],copy=True)for k in z.files}
    A.need(all(v.dtype.kind in 'biuf'and np.isfinite(v).all()for v in result.values()),'finite_numeric_cache_arrays')
    return result


def run(stage,out,source_sha,deadline):
    A.need(sys.platform.startswith('linux'),'actual_cached_arrays_and_render_cloud_Linux_only')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    A.need(not out.is_symlink()and not any(out.iterdir()),'new_empty_owned_output')
    result={'schema':'paper2-signed-figures-result-v1','status':'INCOMPLETE','analysis_complete':False,
        'input_qualified':False,'synthetic':False,'new_model_calls':0,'new_env_steps':0,'new_fits':0,
        'training_updates':0,'JPFS_new_bytes':0,'original_1500_objects_read':False,
        'qualified_feature_cache_read':False,'GPU_auto_GO':False}
    started=time.monotonic();phase='source'
    def check():A.need(time.time()<deadline,'absolute_science_deadline')
    try:
        check();ref=ident(HERE/'SOURCE-MANIFEST.json');A.need(ref['sha256']==source_sha,'externally_bound_source_manifest')
        manifest=json.loads((HERE/'SOURCE-MANIFEST.json').read_bytes())
        A.need(manifest['schema']=='paper2-signed-figures-source-v1','source_schema')
        for row in manifest['files']:
            p=HERE/row['path'];A.need(p.is_file()and not p.is_symlink()and p.resolve().is_relative_to(HERE),'owned_source_member')
            A.need(ident(p)=={k:row[k]for k in ('bytes','sha256')},'source_member_SHA');check()
        inputs=json.loads((HERE/'INPUTS.json').read_bytes());contract=json.loads((HERE/'FIGURE-CONTRACT.json').read_bytes())
        refs=inputs['input_objects'];A.need(len(refs)==5 and {r['path']for r in refs}==INPUT_NAMES,'only_five_original_caches')
        A.need(inputs['input_body_bytes']==23044898,'exact_cache_body_bytes')
        result.update(source_manifest=ref,inputs_manifest=ident(HERE/'INPUTS.json'),scope=contract['scope'],
            figure_contract=ident(HERE/'FIGURE-CONTRACT.json'))
        stage=Path(stage).resolve(strict=True);phase='five_input_SHA'
        for row in refs:
            p=stage/row['path'];A.need(p.is_file()and not p.is_symlink(),'regular_staged_cache')
            A.need(ident(p)=={k:row[k]for k in ('bytes','sha256')},'cache_whole_bytes_SHA');check()
        producer=json.loads((stage/'signed-transfer-result.json').read_bytes())
        A.need(producer['source_manifest']['sha256']==PRODUCER_SOURCE,'fixed_original_producer_source')
        bound={r['path']:r for r in producer['outputs']}
        for row in refs:
            if row['path']=='signed-transfer-result.json':continue
            A.need({k:bound[row['path']][k]for k in ('bytes','sha256')}=={k:row[k]for k in ('bytes','sha256')},'cache_bound_by_producer_result')
        A.need(producer['fit_identity']==ident(stage/'fit-manifest.json'),'fit_manifest_producer_binding')
        phase='decode_caches'
        with gzip.open(stage/'evaluation.json.gz','rb')as f:raw=f.read(64*1024**2+1)
        A.need(len(raw)<=64*1024**2,'bounded_evaluation_JSON_expansion')
        evaluation=json.loads(raw);fit=json.loads((stage/'fit-manifest.json').read_bytes())
        predictions=load_npz(stage/'predictions.npz');arrays=load_npz(stage/'fit-arrays.npz');check()
        phase='independent_score_QA';data=A.verify(producer,evaluation,predictions,arrays,fit,check=check)
        data['qa']['input_objects']=refs;write(out/'QA.json',data['qa'])
        result.update(input_qualified=True,input_objects_verified=5,input_body_bytes=23044898,
            QA_passed=True,legacy_QA_independently_reexecuted=False)
        del predictions,arrays,evaluation;check()
        phase='render_all_declared_support'
        import renderer
        rendered=renderer.render(data,out,check=check)
        write(out/'CAPTIONS.json',rendered['captions'])
        write(out/'FIGURE-VALUES.json',{'schema':'signed-figures-values-v1','synthetic':False,
            'source_inputs':refs,'main':rendered['figures'],'all37_all54_groups':'FULL-SCORES.csv.gz',
            'CSV_rows':rendered['CSV_rows'],'model_names':data['names'],'dimensions':list(A.DIMENSIONS),
            'main_primary_scale':'Original DEV_sigma','undefined':'null or NA; never replaced by zero'})
        render_manifest={'schema':'signed-figures-render-manifest-v1','synthetic':False,'source_manifest':ref,
            'inputs_manifest':result['inputs_manifest'],'figure_contract':result['figure_contract'],
            'QA':ident(out/'QA.json'),'main_layouts':rendered['layouts'],'atlas_pages':rendered['atlas_pages'],
            'atlas_page_count':len(rendered['atlas_pages']),'CSV_rows':rendered['CSV_rows'],
            'minimum_font_points':7,'all_text_bbox_checks_passed':True,
            'main_figures':list(FIGURE_IDS),'captions':ident(out/'CAPTIONS.json'),
            'figure_values':ident(out/'FIGURE-VALUES.json'),'outputs':[]}
        for p in sorted(out.iterdir()):
            A.need(p.name in OUTPUT_NAMES and p.is_file()and not p.is_symlink(),'declared_regular_figure_output')
            record={'path':p.name,**ident(p)};A.need(record['bytes']<=8*1024**2,'per_output8MiB')
            render_manifest['outputs'].append(record)
        write(out/'RENDER-MANIFEST.json',render_manifest)
        result.update(status='SIGNED_FIGURES_COMPLETE',analysis_complete=True,render_complete=True,
            main_figures=4,atlas_pages=30,contact_sheets=2,CSV_rows=rendered['CSV_rows'],
            output_names=list(OUTPUT_NAMES),outputs=[{'path':p.name,**ident(p)}for p in sorted(out.iterdir())],
            no_fit_or_model_calls=True,posthoc_CONF_not_new_blind_test=True)
        A.need({p.name for p in out.iterdir()}==set(OUTPUT_NAMES)-{RESULT},'complete_exact_output_roster')
        A.need(sum(p.stat().st_size for p in out.iterdir())+2*1024**2<=32*1024**2,'all_outputs_plus_result_under32MiB')
        check()
    except BaseException as error:
        result.update(status='FAILED_PRESERVED',analysis_complete=False,render_complete=False,
            failure={'phase':phase,'type':type(error).__name__,
                'code':str(error)if isinstance(error,(ValueError,RuntimeError))and len(str(error))<180 and
                  all(c.isalnum()or c in '_-'for c in str(error))else None,
                'frames':[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno}for f in traceback.extract_tb(error.__traceback__)[-10:]]})
    result['elapsed_seconds']=time.monotonic()-started
    write(out/RESULT,result);A.need((out/RESULT).stat().st_size<=2*1024**2,'result2MiB')
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('stage','out','source-manifest-sha256','deadline-epoch'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=run(a.stage,a.out,a.source_manifest_sha256,float(a.deadline_epoch))
    print(json.dumps({'status':result['status'],'analysis_complete':result['analysis_complete'],'failure':result.get('failure')}),flush=True)
    return 0 if result['analysis_complete']else 2


if __name__=='__main__':raise SystemExit(main())
