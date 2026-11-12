"""Exact cloud-only case member reader using the already verified Range contract."""
from __future__ import annotations
import io
import json
import math
from pathlib import Path
import tarfile
import time
from urllib.parse import quote,urlsplit
import urllib.request
import zipfile
import numpy as np
import range_source
import tar_source
import science_core as S

range_source.MAX_SELECTED_BYTES=64*1024*1024

def need(ok,code):
    if not ok:raise RuntimeError(code)

def _url(uri):
    parsed=urlsplit(uri)
    need(parsed.scheme=='bos'and parsed.netloc=='redacted-417e8f2cb1dd'
         and parsed.path.startswith('/research-storage/adv-vla/')
         and not parsed.query and not parsed.fragment,'exact_Jinwen_BOS_URI')
    path='/'+'/'.join(quote(x,safe='-_.~')for x in parsed.path.lstrip('/').split('/'))
    return parsed,path

def get_small(signer,uri,expected_sha,limit=2*1024*1024):
    """GET an exact metadata object; no listing, redirects or large payloads."""
    parsed,path=_url(uri)
    req=urllib.request.Request('https://'+signer._host+path,
        headers=signer._headers('GET',path,{'content-length':'0','accept-encoding':'identity'}),method='GET')
    opener=urllib.request.build_opener(type('NoRedirect',(urllib.request.HTTPRedirectHandler,),
        {'redirect_request':lambda self,*a,**k:None})())
    with opener.open(req,timeout=30)as response:
        size=int(response.headers.get('Content-Length','-1'))
        need(response.status==200 and 0<size<=limit and response.headers.get('Content-Encoding')in(None,'identity'),
             'bounded_metadata_response')
        raw=response.read(limit+1)
    need(len(raw)==size and tar_source.sha(raw)==expected_sha,'metadata_whole_SHA')
    return raw

class CaseReader:
    def __init__(self,signer,case_row,member_rows,deadline):
        self.signer,self.row,self.deadline=signer,case_row,deadline
        self.scope=case_row['case_scope']
        index_raw=get_small(signer,case_row['index_uri'],case_row['index_sha256'])
        index=json.loads(index_raw)
        manifest_ref=index['file_manifests'][0]
        need(manifest_ref['uri']==case_row['manifest_uri'] and manifest_ref['sha256']==case_row['manifest_sha256']
             and manifest_ref['bytes']<=2*1024*1024,'bound_manifest_identity')
        manifest_raw=get_small(signer,manifest_ref['uri'],manifest_ref['sha256'])
        need(len(manifest_raw)==manifest_ref['bytes'],'manifest_bytes')
        manifest=[json.loads(line)for line in manifest_raw.splitlines()if line.strip()]
        self.mapped=tar_source.build_map(index,manifest,self.scope)
        self.range=range_source.RangeReader(signer,index['parts'],deadline)
        self.members={r['member']:r for r in member_rows}
        for name,row in self.members.items():
            need(name in self.mapped and self.mapped[name]['row']['bytes']==int(row['bytes'])
                 and self.mapped[name]['row']['sha256']==row['sha256'],'published_member_source_identity')
        native_raw=self.member('NATIVE-FILES-v1.json',limit=4*1024*1024)
        native=json.loads(native_raw)
        self.native={r['path']:r for r in native['files']}
        for name,row in self.members.items():
            need(name in self.native and self.native[name]['bytes']==int(row['bytes'])
                 and self.native[name]['sha256']==row['sha256'],'producer_native_manifest_identity')
        self.index=index

    def member(self,name,limit=64*1024*1024):
        need(name in self.mapped and self.mapped[name]['row']['bytes']<=limit,'selected_member_only')
        raw,proof=tar_source.read_one(self.range,self.mapped,name)
        need(proof['actual_header_verified']and proof['sha256']==self.mapped[name]['row']['sha256'],
             'real_tar_header_and_member_SHA')
        return raw

    def json_member(self,name,limit=256*1024):
        return json.loads(self.member(name,limit))

    def fields(self,name,names):
        need(name in self.members and name in self.native,'indexed_NPZ_only')
        raw=self.member(name)
        proof=self.native[name]
        selected=tuple(names)
        need(len(selected)==len(set(selected))and set(selected)<=set(proof['fields']),'selected_field_inventory')
        result={}
        with zipfile.ZipFile(io.BytesIO(raw))as archive:
            infos=archive.infolist()
            need(len(infos)==len({x.filename for x in infos})
                 and sum(x.file_size for x in infos)<=96*1024*1024,
                 'bounded_unique_NPZ_fields')
            expected={key+'.npy'for key in proof['fields']}|{'__schema__.npy'}
            need({x.filename for x in infos}==expected,'complete_NPZ_field_inventory')
            with archive.open('__schema__.npy')as stream:
                embedded=json.loads(np.lib.format.read_array(stream,allow_pickle=False).item())
            need(embedded['fields']==proof['fields'],'producer_embedded_schema_identity')
            for key in selected:
                with archive.open(key+'.npy')as stream:
                    stored=np.lib.format.read_array(stream,allow_pickle=False)
                result[key]=S.decode_field(stored,proof['fields'][key])
        return result
