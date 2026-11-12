"""Cloud-only numeric diagnostics for two fixed q0 witnesses; no model/environment."""
import hashlib, importlib.util, io, json, os, platform, tarfile, time, urllib.request, zipfile
from pathlib import Path
from urllib.parse import urlsplit


def need(ok, reason):
    if not ok: raise ValueError(reason)


def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def load_borrowed(path,ref):
    spec=importlib.util.spec_from_file_location('_witness_borrowed',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    need(m.file_identity(path)=={k:ref[k]for k in ('bytes','sha256')},'borrowed_source_SHA');return m


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs): raise ValueError('redirect_disallowed')


class RangeReader:
    def __init__(self,urls,deadline):
        self.urls=urls;self.deadline=deadline;self.bytes=0;self.receipts=[];self.seen=set()
        self.opener=urllib.request.build_opener(NoRedirect())
    def get(self,archive,start,size):
        uri=archive['uri'];key=(uri,start,size)
        need(uri in self.urls and key not in self.seen,'exact_once_range')
        need(0<=start<start+size<=archive['bytes'] and size<=404312,'bounded_member_range')
        need(time.time()<self.deadline and self.bytes+size<=1619296,'bounded_total_read')
        url=self.urls[uri];p=urlsplit(url)
        need(p.scheme=='https' and not p.username and not p.password and p.hostname and (p.hostname.endswith('.bcebos.com') or p.hostname.endswith('.jdcloud-oss.com')),'signed_official_object_url')
        self.seen.add(key)
        req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+size-1}','Accept-Encoding':'identity'},method='GET')
        try:
            with self.opener.open(req,timeout=min(60,max(1,self.deadline-time.time())))as r:
                need(r.status==206,'range_206_required_no_full_download')
                need(r.headers.get('Content-Range')==f'bytes {start}-{start+size-1}/{archive["bytes"]}','exact_content_range')
                need(int(r.headers.get('Content-Length','-1'))==size,'range_content_length')
                need(r.headers.get('Content-Encoding','identity')=='identity','unencoded_range')
                raw=r.read(size+1);self.bytes+=len(raw);need(len(raw)==size,'exact_range_bytes')
        except ValueError: raise
        except Exception: raise ValueError('bounded_range_transport_failed')from None
        self.receipts.append({'archive_uri':uri,'start':start,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
        return raw


def read_member(reader,ref):
    t=ref['target'];need(t['offset_data']==t['offset_header']+512,'adjacent_tar_header')
    header=reader.get(ref['archive'],t['offset_header'],512)
    if 'header_sha256'in t:need(hashlib.sha256(header).hexdigest()==t['header_sha256'],'exact_header_SHA')
    info=tarfile.TarInfo.frombuf(header,'utf-8','strict')
    need(info.isfile() and info.name==t['member'] and info.size==t['bytes'],'actual_tar_member_binding')
    need('/native-evidence/'in info.name and info.name.endswith('/query-0000-input.npz'),'actual_case_not_bridge')
    raw=reader.get(ref['archive'],t['offset_data'],t['bytes'])
    need(hashlib.sha256(raw).hexdigest()==t['sha256'],'exact_native_input_SHA')
    return raw


def array_identity(a):
    need(not a.dtype.hasobject,'no_object_array');h={'shape':list(a.shape),'dtype':a.dtype.str,'order':'C'}
    return {**h,'bytes':a.nbytes,'hash_domain':'plus-policy-native-array-v1','sha256':hashlib.sha256(canonical(h)+b'\0'+a.tobytes(order='C')).hexdigest()}


MAPPING={'physical_q0':'physical_sim_state','main_rgb':'agentview_image','wrist_rgb':'robot0_eye_in_hand_image','eef_position':'robot0_eef_pos','eef_quaternion':'robot0_eef_quat','gripper_qpos':'robot0_gripper_qpos','inference_noise':'noise'}


def load_arrays(raw,fields):
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(raw))as z:
        need(len(z.infolist())<32 and sum(v.file_size for v in z.infolist())<2*1024**2,'bounded_nonpickle_npz')
        need(len(set(z.namelist()))==len(z.namelist()),'unique_npz_keys')
    with np.load(io.BytesIO(raw),allow_pickle=False)as z:
        arr={k:z[k]for k in z.files}
    for key,a in arr.items():
        need(not a.dtype.hasobject and a.dtype.kind in 'buifSU','native_numeric_or_text_metadata_only')
        if a.dtype.kind in 'buif':need(np.isfinite(a).all(),'finite_numeric_native_arrays')
    for field,key in MAPPING.items():
        need(key in arr and array_identity(arr[key])==fields[field],'native_array_field_identity:'+field)
    need(array_identity(arr['sensor_rng_keys'])==fields['sensor_numpy_rng']['keys'],'sensor_keys_identity')
    need(array_identity(arr['sensor_rng_cached_gaussian'])==fields['sensor_numpy_rng']['cached_gaussian'],'sensor_gaussian_identity')
    return arr


def compare(a,b):
    import numpy as np
    need(a.keys()==b.keys(),'same_npz_keys');stats={}
    for key in sorted(a):
        x,y=a[key],b[key]
        if x.dtype.kind in 'SU' and y.dtype.kind in 'SU':
            stats[key]={'kind':'auxiliary_text_metadata','T20_shape':list(x.shape),'T30_shape':list(y.shape),'T20_dtype':x.dtype.str,'T30_dtype':y.dtype.str,'T20_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'T30_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'exact_bytes_equal':x.tobytes()==y.tobytes(),'not_interpreted_as_numeric':True}
            continue
        need(x.shape==y.shape and x.dtype==y.dtype,'matching_native_array_ABI')
        d=y.astype(np.float64)-x.astype(np.float64);changed=x!=y
        s={'shape':list(x.shape),'dtype':x.dtype.str,'elements':int(x.size),'changed_elements':int(changed.sum()),'max_abs':float(np.max(np.abs(d)))if d.size else 0.,'rms':float(np.sqrt(np.mean(d*d)))if d.size else 0.,'mean_signed':float(np.mean(d))if d.size else 0.,'exact_bytes_equal':x.tobytes()==y.tobytes()}
        if x.ndim==3 and x.shape[-1]==3:
            s.update(changed_pixels=int(np.any(changed,axis=-1).sum()),pixel_count=int(x.shape[0]*x.shape[1]),changed_pixel_fraction=float(np.any(changed,axis=-1).mean()))
        elif x.size<=128:
            ids=np.flatnonzero(changed);s['changed_coordinates']=[{'flat_index':int(i),'T20':float(x.reshape(-1)[i]),'T30':float(y.reshape(-1)[i]),'delta':float(d.reshape(-1)[i])}for i in ids]
        stats[key]=s
    return stats


def plot(pairs,rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    fig,axs=plt.subplots(4,3,figsize=(10,12),layout='constrained')
    for j,(a,b)in enumerate(pairs):
        for k,key in enumerate(['agentview_image','robot0_eye_in_hand_image']):
            r=j*2+k;diff=np.max(np.abs(b[key].astype(float)-a[key].astype(float)),axis=2)
            for col,img in enumerate([a[key],b[key],diff]):
                ax=axs[r,col]
                if col==2:
                    im=ax.imshow(img,cmap='magma',vmin=0,vmax=255);fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
                else:ax.imshow(img)
                ax.set_xticks([]);ax.set_yticks([])
                name=['T20 / JD','T30 floor / BOS','Absolute difference (max RGB; 0–255)'][col]
                ax.set_title(name,fontsize=9)
            s=rows[j]['arrays'][key];axs[r,0].set_ylabel(f'p{rows[j]["ordinal"]}: {"main" if k==0 else "wrist"}',fontsize=10)
            axs[r,2].set_xlabel(f'{s["changed_pixels"]}/{s["pixel_count"]} pixels differ; max={s["max_abs"]:g}; RMS={s["rms"]:.4g}')
    fig.suptitle('Actual first-query inputs: two fixed diagnostic witnesses',fontsize=15)
    fig.supxlabel('Original stored orientation; no resampling or contrast enhancement. Differences are diagnostics, not attention.\nTwo selected witnesses do not estimate prevalence or establish a platform cause.',fontsize=10)
    fig.savefig(path/'Q0-WITNESSES.png',dpi=160);fig.savefig(path/'Q0-WITNESSES.svg');plt.close(fig)


def run(inputs,dependency_root,borrowed_worker_path,store,work,output,deadline_epoch):
    need(platform.system()=='Linux','scientific_arrays_cloud_only');inputs=Path(inputs);task=json.loads((inputs/'INPUT-SPEC-v1.json').read_bytes())
    need(task['analysis_kind']=='fixed_two_q0_numeric_witnesses' and task['new_count']==0,'diagnostic_only')
    borrowed=load_borrowed(borrowed_worker_path,task['borrowed_worker']);need(borrowed.file_identity(__file__)==task['worker_source'],'worker_source_SHA')
    plan=task['witness_plan'];need([w['ordinal']for w in plan['witnesses']]==[4770,4794],'fixed_outcome_blind_witnesses')
    urls=json.loads(os.environ.pop('ADV_VLA_EXACT_READ_URLS'));need(set(urls)=={r['archive']['uri']for w in plan['witnesses']for r in [w['JD'],w['BOS']]},'exact_four_signed_object_scopes')
    reader=RangeReader(urls,deadline_epoch);pairs=[];rows=[]
    try:
        for w in plan['witnesses']:
            a,b=[load_arrays(read_member(reader,w[k]),w[k]['expected_fields'])for k in ('JD','BOS')]
            rows.append({'ordinal':w['ordinal'],'case':w['case'],'selection':w['selection'],'arrays':compare(a,b),'source_members':{k:w[k]['target']for k in ('JD','BOS')},'all_selected_array_identities_match_previous_hash_audit':True});pairs.append((a,b))
    finally:urls.clear()
    need(reader.bytes==task['readset']['exact_total_bytes']==1619296 and len(reader.receipts)==8,'exact_four_members_and_headers')
    out=borrowed.OutputBudget(output);plot(pairs,rows,out.directory)
    for name in ['NEW-PAIR-TRACE-v1.jsonl.gz','NORMALIZED-NEW-v1.jsonl.gz']:
        with out.gzip(name)as s:
            for row in rows:s.write(canonical(row)+b'\n')
    result={'schema':'paper3-added-endpoint-pairs-v1','analysis_kind':task['analysis_kind'],'model_id':task['model_id'],'run_id':task['run_id'],'new_count':0,'new_endpoint_identity':[],'new_trace_rows':2,'new_OUTCOME_GETs':0,'old_OUTCOME_GETs':0,'GPU':0,'model_calls':0,'new_environment_steps':0,'full_population_evaluation_complete':False,'CI':None,'input_objects':4,'input_bytes':reader.bytes,'input_spec':borrowed.file_identity(inputs/'INPUT-SPEC-v1.json'),'witnesses':rows,'original_pairing_threshold_changed':False,'whole_archive_SHA_not_recomputed':True,'exact_member_SHA_and_tar_header_checked':True,'generalization_to107':False,'platform_causality_established':False}
    out.json('INPUT-RECEIPT-v1.json',{'verified_objects':4,'verified_bytes':reader.bytes,'new_OUTCOME_GETs':0,'old_OUTCOME_GETs':0,'input_spec':result['input_spec'],'ranges':reader.receipts,'raw_arrays_on_Mac':False,'full_archive_GETs':0});out.json('RESULT-v1.json',result,reserve=False);return result
