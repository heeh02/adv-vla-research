"""Two fixed witnesses: original archive metadata only, never scientific arrays."""
from pathlib import Path
import sys,json,gzip,hashlib,time
from urllib.parse import urlsplit
H=Path(__file__).resolve().parent;C=H.parents[2];ROOT=C.parents[2]
sys.path.insert(0,str(ROOT/'existing-data-synthesis-20261018-v1/behavior'))
# Explicit resolved path for the established original metadata validator.
sys.path.insert(0,'/Users/researcher/Adv-VLA-π0.5/docs/experiments/existing-data-synthesis-20261018-v1/behavior')
import jd_endpoint_adapter as A
sys.path.insert(0,'/Users/researcher/one_vla_latest_main/src')
from one_vla.cloud.profile import CloudProfile
from one_vla.cloud.jdcloud.object_store import JdcloudObjectStore
plan=json.loads(gzip.decompress((C/'comparison-cloud-bridge-v1/inputs-2648-v1/HISTORICAL-PLAN-v1.json.gz').read_bytes()))
selected=json.loads((H.parent/'RAW-WITNESS-SELECTION-v1.json').read_text())['selected'];wanted={r['ordinal']for r in selected};assert wanted=={4770,4794}
arts=[a for a in plan['JD_artifacts']if any(r['ordinal']in wanted and r['model_id']=='T-s0-u20000-EMA'for r in a['rows'])];assert len(arts)==1;art=arts[0];rr=[r for r in art['rows']if r['ordinal']in wanted];assert len(rr)==2
allow={art['prefix']+'_manifest.json',art['prefix']+'payload/LOCAL_RESULT_INDEX-v1.json'}
for r in rr:
 base=r['summary_relative'].rsplit('/',1)[0];allow.update(art['prefix']+'payload/'+n for n in [r['summary_relative'],base+'/CASE_PACK_MANIFEST-v1.json'])
class Store:
 def __init__(self):self.s3=JdcloudObjectStore(CloudProfile.from_yaml(Path('/Users/researcher/one_vla_latest_main/configs/cloud/jdcloud-cn-north-1.yaml')))._s3();self.refs=[];self.total=0
 def get(self,uri,*,limit):
  assert uri in allow and uri.endswith('.json')and limit<=8*1024**2;dest=H/'objects'/ (hashlib.sha256(uri.encode()).hexdigest()+'.json');dest.parent.mkdir(exist_ok=True)
  if dest.exists():b=dest.read_bytes();reused=True
  else:
   u=urlsplit(uri);res=self.s3.get_object(Bucket=u.netloc,Key=u.path.lstrip('/'));body=res['Body']
   try:
    assert res['ContentLength']<=limit and self.total+res['ContentLength']<=16*1024**2;b=body.read(limit+1);assert len(b)==res['ContentLength'];dest.write_bytes(b);self.total+=len(b);reused=False
   finally:body.close()
  self.refs.append({'uri':uri,'path':str(dest),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'local_reuse':reused});return b
store=Store();mf,index=A.initialize_artifact(store,art);out=[]
for r in rr:
 prefix=art['prefix'];rel=r['summary_relative'];base=rel.rsplit('/',1)[0];raw=store.get(prefix+'payload/'+rel,limit=8*1024**2);A.exact(raw,r['summary_identity'],'root_summary_SHA');A.exact(raw,mf[rel],'artifact_summary_SHA');summary=json.loads(raw)
 packraw=store.get(prefix+'payload/'+base+'/CASE_PACK_MANIFEST-v1.json',limit=8*1024**2);A.exact(packraw,mf[base+'/CASE_PACK_MANIFEST-v1.json'],'artifact_pack_manifest_SHA');A.exact(packraw,summary['pack_manifest'],'summary_pack_manifest_SHA');pm=json.loads(packraw);assert pm['archive']==mf[base+'/CASE_PACK-v1.tar'];offset=0;members=[]
 for name,ref in sorted(pm['members'].items()):
  if 'query-0000-input' in name or 'RUNTIME' in name or 'OUTCOME' in name or 'COMMIT' in name:members.append({'member':name,'offset_header':offset,'offset_data':offset+512,**ref})
  offset+=512+((ref['bytes']+511)//512)*512
 out.append({'ordinal':r['ordinal'],'model_id':r['model_id'],'case':r['reserved']['case'],'native_index':r['index'],'archive':{'uri':prefix+'payload/'+base+'/CASE_PACK-v1.tar',**pm['archive']},'selected_members':members,'all_member_count':len(pm['members']),'query0_input_matches':[m for m in members if 'query-0000-input'in m['member']],'header_byte_ranges_not_yet_verified':True,'scientific_array_GETs':0})
result={'status':'TWO_JD_WITNESS_METADATA_LOCATED','witnesses':out,'GET_bytes':store.total,'refs':store.refs,'full_tar_GETs':0,'scientific_array_GETs':0,'new_model_environment_calls':0};(H/'JD-LOCATORS-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(out),'GET_bytes':store.total,'q0_input_files':[len(r['query0_input_matches'])for r in out]}))
