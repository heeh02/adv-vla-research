"""One v3 case render audit from exact BOS tar Range reads. No model or GPU."""
from __future__ import annotations
import argparse, hashlib, io, json, math, os, pathlib, re, tarfile, time, urllib.parse, urllib.request

RUN = "jinwen-adv-vla-plus-repaired-render-validation-20261010-v1"
OUTPUT = "redacted-82b1d017976a"
SAMPLE_QUERIES = (0, 1, 5)
MAX_SELECTED_BYTES = 8 * 1024 * 1024

def need(ok, code):
    if not ok: raise RuntimeError(code)

def digest(raw): return hashlib.sha256(raw).hexdigest()

def read_json(path):
    raw = pathlib.Path(path).read_bytes(); need(len(raw) <= 2 * 1024 * 1024, "small_local_json")
    return json.loads(raw), {'bytes': len(raw), 'sha256': digest(raw)}

def selected_names(manifest, reference_y, treatment_y):
    available = {row['path'] for row in manifest}
    names, unreached = [], []
    for arm in ('reference', 'treatment'):
        y = reference_y if arm == 'reference' else treatment_y
        need(type(y.get('queries')) is int and y['queries'] >= 0, 'actual_Y_query_count')
        for q in SAMPLE_QUERIES:
            pair = [f'{arm}/Y/query-{q:04d}-input.npz', f'{arm}/Y/query-{q:04d}-actions.npz']
            if q >= y['queries']:
                need(not set(pair) & available, 'post_terminal_query_member_unexpected')
                unreached.append({'arm': arm, 'query': q, 'reason': 'not_reached'})
                continue
            need(set(pair) <= available, 'actual_Y_query_member_missing')
            names.extend(pair)
        need(f'{arm}/Y/terminal.npz' in available, 'actual_Y_terminal_member_missing')
        names.append(f'{arm}/Y/terminal.npz')
    return names, unreached

def offsets(manifest, index, names):
    need(index['schema'] == 'plus-runtime-restoration-capsule-v1' and index['archive_includes_models'] is False, 'case_archive_identity')
    need(index['members'] == len(manifest) and len(manifest) > 0, 'actual_member_count')
    need([r['path'] for r in manifest] == sorted(r['path'] for r in manifest), 'sorted_manifest')
    cursor, mapped = 0, {}
    for row in manifest:
        name = row['path']; need(re.fullmatch(r'[A-Za-z0-9_./-]+', name) and '..' not in name.split('/'), 'safe_tar_name')
        info = tarfile.TarInfo(name); info.size = row['bytes']; info.mode = row['mode']; info.mtime = info.uid = info.gid = 0
        header = info.tobuf(format=tarfile.PAX_FORMAT)
        need(len(header) == 512, 'unexpected_PAX_extended_header')
        mapped[name] = {'header_offset': cursor, 'payload_offset': cursor + 512, **row}
        cursor += 512 + math.ceil(row['bytes'] / 512) * 512
    tar_bytes = math.ceil((cursor + 1024) / 10240) * 10240
    need(tar_bytes == index['archive']['bytes'] == sum(p['bytes'] for p in index['parts']), 'archive_layout_size')
    parts = index['parts']; need(parts and all(p['bytes'] > 0 for p in parts), 'actual_parts_required')
    part_cursor = 0
    for part in parts:
        need(part['offset'] == part_cursor, 'exact_contiguous_part_offsets')
        part_cursor += part['bytes']
    need(part_cursor == index['archive']['bytes'], 'actual_part_total')
    need(sum(mapped[n]['bytes'] for n in names) <= MAX_SELECTED_BYTES, 'small_selection_only')
    return mapped

class RangeReader:
    def __init__(self, signer, parts, deadline):
        self.signer, self.parts, self.deadline = signer, parts, deadline
        self.range_requests = []
        self.opener = urllib.request.build_opener(type('NoRedirect', (urllib.request.HTTPRedirectHandler,), {'redirect_request': lambda self, *a, **k: None})())

    def one(self, part, start, length):
        need(time.time() < self.deadline - 5, 'range_deadline')
        need(0 <= start < part['bytes'] and 0 < length <= part['bytes'] - start and length <= 2 * 1024 * 1024, 'bounded_part_range')
        uri = urllib.parse.urlsplit(part['uri']); need(uri.scheme == 'bos' and uri.netloc == 'redacted-417e8f2cb1dd' and uri.path.startswith('/research-storage/'), 'exact_BOS_part')
        key = uri.path.lstrip('/'); path = '/' + '/'.join(urllib.parse.quote(x, safe='-_.~') for x in key.split('/'))
        byte_range = f'bytes={start}-{start + length - 1}'
        headers = self.signer._headers('GET', path, {'content-length':'0', 'range':byte_range, 'accept-encoding':'identity'})
        req = urllib.request.Request('https://' + self.signer._host + path, headers=headers, method='GET')
        try:
            with self.opener.open(req, timeout=min(30, self.deadline - time.time())) as response:
                need(response.status == 206 and response.headers.get('Content-Range') == f'bytes {start}-{start + length - 1}/{part["bytes"]}', 'actual_HTTP206_ContentRange')
                need(response.headers.get('Content-Length') == str(length) and response.headers.get('Content-Encoding') in (None, 'identity'), 'actual_range_length_encoding')
                raw = response.read(length + 1)
        except Exception as exc:
            raise RuntimeError('BOS_Range_'+type(exc).__name__) from None
        need(len(raw) == length, 'range_exact_byte_count')
        self.range_requests.append({'part_sha256':part['sha256'],'start':start,'length':length,'sha256':digest(raw),'HTTP206':True})
        return raw

    def read(self, start, length):
        need(0 <= start and 0 < length <= MAX_SELECTED_BYTES, 'bounded_archive_range')
        out = bytearray(); pos = start; remaining = length
        while remaining:
            found = [p for p in self.parts if p['offset'] <= pos < p['offset'] + p['bytes']]
            need(len(found) == 1, 'range_part_gap_or_overlap')
            part = found[0]; within = pos - part['offset']; take = min(remaining, part['bytes'] - within, 2 * 1024 * 1024)
            out.extend(self.one(part, within, take)); pos += take; remaining -= take
        return bytes(out)

def restore(reader, mapped, name):
    row = mapped[name]; header = reader.read(row['header_offset'], 512)
    need(header[257:263] == b'ustar\0', 'actual_tar_ustar_header')
    info = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
    need(info.isfile() and info.name == name and info.size == row['bytes'] and info.mode == row['mode'], 'actual_tar_member_identity')
    payload = reader.read(row['payload_offset'], row['bytes'])
    need(digest(payload) == row['sha256'], 'actual_member_payload_SHA')
    return payload, {'name': name, 'bytes': row['bytes'], 'sha256': row['sha256'], 'header_offset': row['header_offset'], 'actual_tar_header_sha256': digest(header), 'actual_header_verified': True}

def numeric_summary(a):
    import numpy as np
    if a.dtype.kind not in 'biuf' or a.size > 10_000_000: return None
    v = np.asarray(a, dtype=np.float64)
    return {'min': float(v.min()), 'max': float(v.max()), 'variance': float(v.var()), 'mean':float(v.mean())}

def inspect_npz(raw, name):
    import numpy as np
    result = {'name': name, 'fields': {}, 'rgb': {}, 'state': {}, 'actions': {}}
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        for key in archive.files:
            if key == '__schema__': continue
            a = archive[key]; entry = {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':digest(a.tobytes(order='C'))}
            image = a.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] in (3,4) and 24 <= a.shape[0] <= 2048 and 24 <= a.shape[1] <= 2048
            state = key in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','physical_sim_state')
            action = key in ('raw_actions','actions')
            if image:
                entry.update(numeric_summary(a));entry.update(all_zero=bool(np.all(a==0)),near_black_fraction=float(np.mean(a[...,:3]<5)),saturated_fraction=float(np.mean(a[...,:3]>250)))
                result['rgb'][key] = entry
            elif state:
                entry.update(numeric_summary(a));
                if a.size <= 8 and a.dtype.kind in 'biuf': entry['values'] = [float(v) for v in a.reshape(-1)]
                result['state'][key] = entry
            elif action:
                entry.update(numeric_summary(a)); result['actions'][key] = entry
            result['fields'][key] = {'shape':entry['shape'],'dtype':entry['dtype'],'sha256':entry['sha256']}
    return result

def contact_sheet(items, out):
    import numpy as np
    from PIL import Image, ImageDraw
    selected=[]
    for name, raw in items.items():
        if not (name.endswith('-input.npz') or name.endswith('terminal.npz')): continue
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            for key in archive.files:
                a=archive[key]
                if a.dtype == np.uint8 and a.ndim == 3 and a.shape[-1] in (3,4) and a.shape[0] >= 24 and a.shape[1] >= 24:
                    selected.append((name,key,Image.fromarray(a[...,:3]).convert('RGB')))
    need(selected, 'no_actual_RGB_fields')
    width=320; row_height=220; sheet=Image.new('RGB',(width*2,row_height*math.ceil(len(selected)/2)),(255,255,255));draw=ImageDraw.Draw(sheet)
    for i,(name,key,img) in enumerate(selected):
        x=(i%2)*width; y=(i//2)*row_height
        img.thumbnail((width-16,row_height-38))
        sheet.paste(img,(x+8,y+8))
        label=name.split('/')[0]+' '+name.split('/')[-1].replace('-input.npz','').replace('.npz','')+' '+key
        draw.text((x+8,y+row_height-24),label[:52],fill=(0,0,0))
    sheet.save(out,format='PNG')

def run(root, deadline):
    import numpy as np
    from bos_stdlib import BOSWriter
    root=pathlib.Path(root); need(root.as_posix()=='/tmp/Jinwen/adv-vla/'+RUN and root.is_dir() and not root.is_symlink(), 'owned_private_tmp')
    index,idxref=read_json(root/'package/metadata/INDEX-v1.json'); manifest,manref=read_json(root/'package/metadata/MANIFEST-v1.json')
    request,_=read_json(root/'package/metadata/REQUEST-v1.json')
    need(request['run_id']==RUN and request['case_scope'].startswith('redacted-90e00ae5ddb1')
         and idxref==request['index'] and manref==request['manifest'], 'bound_v3_case_index_manifest_SHA')
    native,native_ref=read_json(root/'package/metadata/NATIVE-FILES-v1.json')
    reference_y,ref_y_sha=read_json(root/'package/metadata/REFERENCE-Y-v1.json')
    treatment_y,treat_y_sha=read_json(root/'package/metadata/TREATMENT-Y-v1.json')
    need(ref_y_sha==request['reference_y'] and treat_y_sha==request['treatment_y'], 'actual_v3_Y_source_SHA')
    need(native_ref==request['native'], 'actual_v3_native_manifest_SHA')
    names,unreached=selected_names(manifest,reference_y,treatment_y)
    mapped=offsets(manifest,index,names); writer=BOSWriter(OUTPUT,deadline-30); reader=RangeReader(writer,index['parts'],deadline-60)
    restored={}; proofs=[]
    for name in names:
        raw,proof=restore(reader,mapped,name);restored[name]=raw;proofs.append(proof)
    need(native_ref==request['native'] and native_ref=={k:mapped['NATIVE-FILES-v1.json'][k] for k in ('bytes','sha256')}, 'native_metadata_prior_readback_SHA')
    native_rows={r['path']:r for r in native['files']}
    for proof in proofs:
        row=native_rows[proof['name']];need(row['bytes']==proof['bytes'] and row['sha256']==proof['sha256'], 'native_files_and_archive_member_identity')
    analyses={name:inspect_npz(raw,name) for name,raw in restored.items() if name.endswith('.npz')}
    images={name:analysis['rgb'] for name,analysis in analyses.items() if analysis['rgb']}
    comparisons=[]
    for key in sorted({k for fields in images.values() for k in fields}):
        series=[(name,images[name][key]['sha256']) for name in sorted(images) if key in images[name]]
        comparisons.append({'rgb_key':key,'series_sha256':series,'unique_images':len(set(h for _,h in series)),
                            'reference_unique':len({h for n,h in series if n.startswith('reference/')}),
                            'treatment_unique':len({h for n,h in series if n.startswith('treatment/')}),
                            'sampled_pair_equal':{str(q):(
                                images[f'reference/Y/query-{q:04d}-input.npz'][key]['sha256']
                                == images[f'treatment/Y/query-{q:04d}-input.npz'][key]['sha256'])
                                if key in images.get(f'reference/Y/query-{q:04d}-input.npz',{})
                                and key in images.get(f'treatment/Y/query-{q:04d}-input.npz',{}) else None
                                for q in SAMPLE_QUERIES},
                            'reference_terminal_equals_treatment_terminal':images.get('reference/Y/terminal.npz',{}).get(key,{}).get('sha256')==images.get('treatment/Y/terminal.npz',{}).get(key,{}).get('sha256') if key in images.get('reference/Y/terminal.npz',{}) and key in images.get('treatment/Y/terminal.npz',{}) else None})
    changes=[]
    for arm in ('reference','treatment'):
        for field in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','physical_sim_state','actions','raw_actions'):
            series=[]
            for name,analysis in sorted(analyses.items()):
                if name.startswith(arm+'/Y/'):
                    row=analysis['state'].get(field) or analysis['actions'].get(field)
                    if row:series.append((name,row['sha256']))
            if series:changes.append({'arm':arm,'field':field,'sampled_hashes':series,'unique_samples':len(set(h for _,h in series)),
                                      'first_equals_last':series[0][1]==series[-1][1]})
    need(all(k in reference_y and k in treatment_y for k in ('status','policy_steps','queries','success')),
         'actual_v3_Y_fields_required')
    output=root/'output';output.mkdir(exist_ok=True);contact_sheet(restored,output/'CONTACT-SHEET-v1.png')
    report={'schema':'plus-repaired-render-validation-v1','run_id':RUN,'case_scope':request['case_scope'],
            'source_case_commit':request['case_commit'],'source_index':idxref,'source_manifest':manref,'native_files_readback':native_ref,
            'selected_members':proofs,'range_requests':reader.range_requests,'npz_fields':analyses,'rgb_comparisons':comparisons,'sampled_state_action_changes':changes,
            'reference_result':{'status':reference_y['status'],'steps':reference_y['policy_steps'],'queries':reference_y['queries'],'success':reference_y['success'],'source':ref_y_sha},
            'treatment_result':{'status':treatment_y['status'],'steps':treatment_y['policy_steps'],'queries':treatment_y['queries'],'success':treatment_y['success'],'source':treat_y_sha},
            'only_selected_RGB_and_small_state_decoded':True,'full_archive_retrieved':False,'model_loaded':False,'GPU':0,
            'unreached_queries':unreached,'interpretation_limit':'only reached q0,q1,q5 and terminal sampled; no all-query rendering claim'}
    path=output/'AUDIT-v1.json';path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    receipt={'schema':'plus-repaired-render-validation-delivery-v1','run_id':RUN,'case_scope':request['case_scope'],'source_archive_sha256':index['archive']['sha256'],
             'files':[writer.put(path,'results/'+path.name) for path in (path,output/'CONTACT-SHEET-v1.png')],
             'selected_members_SHA_verified':len(proofs),'all_selected_actual_tar_headers_verified':True,'GPU':0,'model_loads':0}
    receipt_path=output/'DELIVERY-v1.json';receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');writer.put(receipt_path,'results/DELIVERY-v1.json')
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-cloud',action='store_true');p.add_argument('--root');p.add_argument('--deadline',type=float);a=p.parse_args()
    if not a.execute_cloud: print(json.dumps({'status':'dry_repaired_render_worker','cloud_calls':0,'GPU':0}));return 0
    need(a.root and a.deadline and time.time()<a.deadline-120,'bounded_cloud_runtime')
    print(json.dumps(run(a.root,a.deadline)),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
