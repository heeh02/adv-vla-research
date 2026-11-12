"""Whole-population released CONF -> unchanged SR and first-query CLI readouts."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE_PREFIX='redacted-5eae3cf06a88'
OUTPUT_PREFIX='redacted-7555e42975b3'


def need(ok,code):
    if not ok:raise ValueError(code)
def encoded(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def ident(raw):return{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def exact_small(path,ref,cap=4*1024**2):
    p=Path(path);need(p.is_file()and not p.is_symlink()and p.stat().st_size<=cap,'small_regular_metadata')
    raw=p.read_bytes();need(ident(raw)=={k:ref[k]for k in('bytes','sha256')},'small_metadata_SHA');return raw


def validate_release(release,protocol_identity,g1_identity,admitted):
    """Operational all-population unblinding barrier; not a scientific gate."""
    need(release.get('schema')=='cf-repair-CONF-outcome-release-v1'
         and release.get('protocol_identity')==protocol_identity
         and release.get('G1_attachment_identity')==g1_identity,
         'actual_global_release_binds_original_PROTOCOL_and_G1')
    need(release.get('all_admitted_population_closed')is True
         and release.get('no_partial_shard_unblinding')is True,'no_outcome_read_before_all_population_closes')
    rows=release.get('admitted_rows');closure=release.get('row_closure')
    need(isinstance(rows,list)and len(rows)==len(set(rows))and set(rows)==set(admitted)
         and isinstance(closure,dict)and set(closure)==set(admitted),'release_covers_exact_all_admitted_CONF_population')
    for row,state in closure.items():
        need(isinstance(state,dict)and state.get('status')in('COLLECTION_TERMINAL','PREDEFINED_STOP_RECORDED'),
             'each_admitted_variant_terminal_or_original_stop_recorded')
        ref=state.get('evidence_identity')
        need(isinstance(ref,dict)and type(ref.get('bytes'))is int and ref['bytes']>0
             and isinstance(ref.get('sha256'),str)and len(ref['sha256'])==64
             and all(c in'0123456789abcdef'for c in ref['sha256'])
             and set(state).isdisjoint({'success','outcome','success_rate','harm','recovery'}),
             'technical_closure_identity_without_outcome_values')


def validate_bound_sources(source,spec):
    source=Path(source)
    praw=exact_small(source/'PROTOCOL-v4.json',spec['protocol_identity']);p=json.loads(praw)
    need(p.get('schema')=='cf-repair-PROTOCOL-v4'and p.get('stage1B_frozen')is True
         and p.get('G1_actual_decisions')is None and p.get('fixture_only')is not True,'actual_original_immutable_1B')
    freeze=json.loads(exact_small(source/'FREEZE-v1.json',spec['freeze_identity']))
    need(freeze.get('protocol_identity')==ident(praw)and freeze.get('frozen_before_effects')is True,
         'original_external_freeze')
    araw=exact_small(source/'ASSIGNMENTS-133-v1.json',p['bindings']['assignment_table'])
    for name,binding in [('extract_confirm.py','confirm_scalar_extractor_code'),
            ('analyze_cases.py','analysis_code'),('analyze_confirm_commands.py','confirm_command_analysis_code')]:
        exact_small(source/name,p['bindings'][binding])
    A=load('_CONF_original_primary_and_G1_loader',source/'analyze_cases.py')
    graw=exact_small(source/'G1-DECISION-v1.json',spec['G1_attachment_identity'])
    g1,gid=A.load_g1_attachment(source/'G1-DECISION-v1.json',ident(graw)['sha256'],protocol=p,
        protocol_identity=ident(praw),analysis_identity={k:p['bindings']['analysis_code'][k]for k in('bytes','sha256')})
    E=load('_CONF_frozen_extractor_for_admission',source/'extract_confirm.py')
    assignment=json.loads(araw);admitted=E.admitted_rows(p,assignment,g1)
    release=json.loads(exact_small(source/'CONF-OUTCOME-RELEASE-v1.json',spec['release_identity']))
    validate_release(release,ident(praw),gid,admitted)
    return p,assignment,ident(praw),ident(araw),g1,gid


def validate_jobs(spec,p,assignment,g1,E):
    need(spec.get('schema')=='cf-repair-confirm-CPU-input-v1','exact_confirm_CPU_spec')
    admitted=E.admitted_rows(p,assignment,g1);seen=set();names=set()
    for j in spec['jobs']:
        name=j['source_job'];rows=j['shard_rows'];prefix=j['run_prefix']
        need(isinstance(name,str)and name.startswith('jinwen-cf-repair-conf-')and '/'not in name
             and prefix==SOURCE_PREFIX+name+'/','exact_owned_CONF_source_prefix')
        need(name not in names and rows and len(rows)==len(set(rows))and set(rows)<=set(admitted)
             and not seen.intersection(rows),'one_authoritative_attempt_per_admitted_CONF_row')
        names.add(name);seen.update(rows)
        need(j['protocol_sha256']==spec['protocol_identity']['sha256']
             and j['G1_attachment_identity']==spec['G1_attachment_identity'],'source_original_PROTOCOL_G1')
        if j.get('archive_manifest')is None:continue
        need(j['archive_manifest']['uri']==prefix+'ARCHIVE-MANIFEST-v1.json'
             and j['source_input_spec']['uri']==prefix+'control/INPUT-SPEC-v1.json','exact_CONF_initial_refs')
    need(0<len(admitted)<=108 and spec['max_GETs']==2700 and spec['input_byte_cap']==4*1024**3,
         'fixed_admitted_population_bounded_read_budget')


def restore_inputs(source,local,spec,reader,E):
    p,assignment,pid,aid,g1,gid=validate_bound_sources(source,spec)
    validate_jobs(spec,p,assignment,g1,E)
    jobs=[]
    for number,j in enumerate(spec['jobs']):
        root=Path(local)/'input'/('%02d'%number)/'CONF';root.mkdir(parents=True)
        row={'source_job':j['source_job'],'root':str(root),'protocol_sha256':pid['sha256'],
            'G1_attachment_identity':gid,'shard_rows':j['shard_rows'],'manifest_identity':None};jobs.append(row)
        if j.get('archive_manifest')is None:continue
        ip=reader(j['source_input_spec'],root.parent/'SOURCE-INPUT-SPEC-v1.json','native_input_metadata')
        inp=json.loads(ip.read_bytes());job=inp['job_fields']
        need(inp.get('schema')=='cf-repair-baidu-stage23-input-v1'and job.get('schema')=='cf-repair-stage23-job-v1'
             and job.get('job_id')=='CONF'and job.get('shard_rows')==j['shard_rows']
             and {k:job['stage23_protocol'][k]for k in('bytes','sha256')}==pid
             and {k:job['stage23_G1'][k]for k in('bytes','sha256')}==gid,'native_CONF_original_protocol_G1_rows')
        ap=reader(j['archive_manifest'],root.parent/'ARCHIVE-MANIFEST-v1.json','archive_metadata')
        archive=json.loads(ap.read_bytes());need(archive.get('schema')=='cf-repair-Baidu-archive-v1','actual_parent_archive')
        bypath={r['path']:r for r in archive['files']};need(len(bypath)==len(archive['files']),'unique_archive_members')
        def remote(path):
            value=bypath.get('CONF/'+path);need(value is not None,'declared_member_missing_from_archive')
            ref=value['remote'];need(ref['uri']==j['run_prefix']+'results/CONF/'+path
                and {k:value[k]for k in('bytes','sha256')}=={k:ref[k]for k in('bytes','sha256')},'exact_inner_outer_CONF_identity')
            return ref
        if 'CONF/MANIFEST-v1.json'not in bypath:continue
        mp=reader(remote('MANIFEST-v1.json'),root/'MANIFEST-v1.json','job_manifest')
        manifest=json.loads(mp.read_bytes());E.members(manifest);row['manifest_identity']=ident(mp.read_bytes())
        first=E.episode_requests(manifest,p,assignment,j['shard_rows'],g1)
        for item in first['requests']:
            ref=remote(item['path']);need({k:item[k]for k in('bytes','sha256')}=={k:ref[k]for k in('bytes','sha256')},'episode_SHA_chain')
            reader(ref,root/E.safe(item['path']),'science_episode_metadata')
        documents=E.load_episode_documents(root,first['requests'])
        plans=E.first_plan_requests(manifest,documents,p,assignment,j['shard_rows'],g1)
        for item in plans['requests']:
            ref=remote(item['path']);need({k:item[k]for k in('bytes','sha256')}=={k:ref[k]for k in('bytes','sha256')},'q0_plan_SHA_chain')
            need(item['path'].startswith('SHARDS/')and item['path'].endswith('-first-query-plan.npz'),'q0_plan_only')
            reader(ref,root/E.safe(item['path']),'first_query_plan_npz')
    return{'schema':'cf-repair-confirm-scalar-index-v1','protocol_identity':pid,'assignment_input_identity':aid,'jobs':jobs}


def run(source,local,descriptor,writer,namespace):
    source,local=Path(source),Path(local);B=load('_CONF_exact_transport',source/'base_transport.py')
    E=load('_CONF_original_extractor',source/'extract_confirm.py')
    need(sys.platform.startswith('linux')and descriptor['resources']['GPU']==0
         and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('NVIDIA_VISIBLE_DEVICES')=='void',
         'cloud_CPU_only')
    need(descriptor['output_prefix']==OUTPUT_PREFIX+descriptor['run_id']+'/','owned_CONF_readout_prefix')
    deadline=min(descriptor['allocation_deadline_epoch']-120,float(os.environ['ONE_VLA_BOS_STS_EXPIRATION_EPOCH'])-180)
    out=local/'output';out.mkdir();reads=[];total=0
    result={'schema':'cf-repair-confirm-CPU-result-v1','run_id':descriptor['run_id'],'status':'FAILED_PRESERVED',
        'GPU':0,'model_queries':0,'environment_steps':0,'archive_complete':False,'no_partial_population_unblinding':True}
    def reader(ref,destination,role):
        nonlocal total
        need(time.time()<deadline and len(reads)<2700 and total+ref['bytes']<=4*1024**3,'bounded_confirm_CPU_reads')
        need(role in('input_spec','native_input_metadata','archive_metadata','job_manifest','science_episode_metadata','first_query_plan_npz'),
             'exact_CONF_read_roles')
        if role!='first_query_plan_npz':need(ref['bytes']<=4*1024**2,'small_metadata_cap')
        r=B.Reader(writer,[ref],deadline,max_gets=1,byte_cap=ref['bytes']);path=r.get(ref,destination)
        reads.append({'role':role,**ref});total+=ref['bytes'];return path
    try:
        ref=descriptor['inputs']['spec_ref'];spec=json.loads(reader(ref,local/'INPUT-SPEC-v1.json','input_spec').read_bytes())
        need(spec['run_id']==descriptor['run_id'],'readout_run_identity')
        index=restore_inputs(source,local,spec,reader,E)
        indexpath=out/'CONF-INPUT-INDEX-v1.json';indexpath.write_bytes(encoded(index))
        sr=out/'CONF-SUCCESS-SCALARS-v1.json';cmd=out/'CONF-Q0-COMMAND-SCALARS-v1.json'
        g1path=source/'G1-DECISION-v1.json';gsha=spec['G1_attachment_identity']['sha256']
        common=['--protocol',str(source/'PROTOCOL-v4.json'),'--assignments',str(source/'ASSIGNMENTS-133-v1.json'),
                '--g1-input',str(g1path),'--g1-sha',gsha]
        base=['--base-analysis',str(source/'analyze_cases.py')]
        commands=[['extract_confirm.py','--cloud-array-read','--index',str(indexpath),
                    '--sr-output',str(sr),'--command-output',str(cmd),*base],
                  ['analyze_cases.py','--input',str(sr),'--output',str(out/'CONF-SUCCESS-READOUT-v1.json')],
                  ['analyze_confirm_commands.py','--input',str(cmd),'--output',str(out/'CONF-Q0-COMMAND-READOUT-v1.json'),*base]]
        for script,*args in commands:
            with(out/(script+'.stdout')).open('xb')as stdout,(out/(script+'.stderr')).open('xb')as stderr:
                subprocess.run([sys.executable,'-I','-B',str(source/script),*args,*common],check=True,
                    stdout=stdout,stderr=stderr,cwd=local,timeout=max(1,deadline-time.time()),
                    env={'PATH':os.defpath,'PYTHONHASHSEED':'0'})
        sr_doc=json.loads(sr.read_bytes());cmd_doc=json.loads(cmd.read_bytes())
        p,assignment,_,_,g1,_=validate_bound_sources(source,spec)
        count=len(E.admitted_rows(p,assignment,g1))*12
        need(len(sr_doc['records'])==len(cmd_doc['records'])==count,'complete_admitted_population_both_endpoints')
        result.update(status='COMPLETE',protocol_identity=spec['protocol_identity'],G1_attachment_identity=spec['G1_attachment_identity'],
            release_identity=spec['release_identity'],success_records=count,q0_command_records=count,
            primary_success_and_secondary_command_six_comparisons_distinct=True,
            H_flow_raw32_noise_or_repeats_materialized=False)
    except BaseException as e:result.update(error_type=type(e).__name__,safe_error_code='CONF_CPU_READOUT_FAILED_OUTPUTS_PRESERVED')
    result.update(finished_epoch=time.time(),physical_GETs=len(reads),input_bytes=total)
    (out/'READS-v1.json').write_bytes(encoded({'reads':reads,'only_exact_objects':True,'no_cloud_listing':True}))
    outputs=[]
    for path in sorted(out.iterdir()):
        need(path.is_file()and path.stat().st_size<=4*1024**2,'small_readout_output')
        ref=writer.put(path,'results/'+path.name);need(ident(path.read_bytes())=={k:ref[k]for k in('bytes','sha256')},'published_output_SHA')
        outputs.append({'path':path.name,**ref})
    result.update(archive_complete=True,outputs=outputs)
    dest=local/'RESULT-v1.json';dest.write_bytes(encoded(result));writer.put(dest,'summary/RESULT-v1.json');return result
