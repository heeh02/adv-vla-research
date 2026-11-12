"""Tiny synthetic CONF fixtures only, including a global-release barrier."""
from pathlib import Path
import copy
import ast
import io
import json
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

HERE=Path(__file__).resolve().parent
TAKEOVER=HERE.parents[1]
sys.path.insert(0,str(HERE))
import cloud_readout as C
import export_candidate as X


def npz(value):
    h=repr({'descr':'<f8','fortran_order':False,'shape':(1,10,7)});h+=' '*(-(10+len(h)+1)%64)+'\n';raw=h.encode()
    arr=b'\x93NUMPY\x01\x00'+len(raw).to_bytes(2,'little')+raw+struct.pack('<70d',*([value]*70))
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w')as z:
        z.writestr('plan_adapter7.npy',arr);z.writestr('H_T__bf16_bits.npy',b'NEVER_DECODE')
    return out.getvalue()


class Fixture:
    def __init__(self,root,*,sensor_admitted=True):
        self.root=Path(root);self.source=self.root/'source';self.source.mkdir();self.local=self.root/'local';self.local.mkdir()
        # All synthetic identities remain in this TemporaryDirectory. No actual
        # CONF or DEV effects, G1 file or release declaration is loaded.
        praw=(TAKEOVER/'protocol/ONE-B-FROZEN-v1/PROTOCOL-v4.json').read_bytes();self.protocol=json.loads(praw)
        araw=(TAKEOVER/'protocol/ONE-B-CANDIDATE/ASSIGNMENTS-133-v1.json').read_bytes();self.assignment=json.loads(araw)
        for name,path in [('extract_confirm.py',TAKEOVER/'protocol/CONFIRM-CPU/extract_confirm.py'),
                          ('analyze_cases.py',TAKEOVER/'protocol/ONE-B-CANDIDATE/analyze_cases.py'),
                          ('analyze_confirm_commands.py',TAKEOVER/'protocol/CONFIRM-COMMAND-CPU/analyze_confirm_commands.py')]:
            (self.source/name).write_bytes(path.read_bytes())
        (self.source/'PROTOCOL-v4.json').write_bytes(praw);(self.source/'ASSIGNMENTS-133-v1.json').write_bytes(araw)
        fraw=C.encoded({'protocol_identity':C.ident(praw),'frozen_before_effects':True})
        (self.source/'FREEZE-v1.json').write_bytes(fraw)
        families=tuple(self.protocol['stage3']['rows']);decisions={f:{'advance':True,'point':.5,'interval':[.4,.6],
            'controls_not_a_gate':True}for f in families}
        if not sensor_admitted:decisions['Sensor Noise'].update(advance=False,point=0.,interval=[-.1,.1])
        self.g1={'schema':'cf-repair-G1-G2-scalar-readout-v1','stage':2,'protocol_identity':C.ident(praw),
            'analysis_code_identity':{k:self.protocol['bindings']['analysis_code'][k]for k in('bytes','sha256')},
            'input_status':'actual_frozen_scalar_input','G1_decisions':decisions,
            'comparisons':[{'family':f,'lhs':'R_resp','rhs':arm,'point':decisions[f]['point'],'sampling_interval':decisions[f]['interval']}
                for f in families for arm in('N_T','R_clean','R_rand')]}
        graw=C.encoded(self.g1);(self.source/'G1-DECISION-v1.json').write_bytes(graw)
        self.E=C.load('_synthetic_original_CONF_extract',self.source/'extract_confirm.py')
        self.A=C.load('_synthetic_original_primary',self.source/'analyze_cases.py')
        self.Q=C.load('_synthetic_original_secondary',self.source/'analyze_confirm_commands.py')
        admitted=self.E.admitted_rows(self.protocol,self.assignment,decisions)
        self.release={'schema':'cf-repair-CONF-outcome-release-v1','protocol_identity':C.ident(praw),
            'G1_attachment_identity':C.ident(graw),'all_admitted_population_closed':True,'no_partial_shard_unblinding':True,
            'admitted_rows':sorted(admitted),'row_closure':{r:{'status':'PREDEFINED_STOP_RECORDED',
                'evidence_identity':{'bytes':1,'sha256':'0'*64}}for r in admitted}}
        rraw=C.encoded(self.release);(self.source/'CONF-OUTCOME-RELEASE-v1.json').write_bytes(rraw)
        self.rows=[rows[0]for family,rows in self.protocol['stage3']['rows'].items()if decisions[family]['advance']];self.map={};self.reads=[]
        run='jinwen-cf-repair-conf-synthetic-0000000000000001';prefix=C.SOURCE_PREFIX+run+'/'
        member=[]
        def add(path,raw):
            self.map[prefix+'results/CONF/'+path]=raw
            return{'path':path,**C.ident(raw),'sealed':False,'sealed_fields':[]}
        byid={r['row_id']:r for r in self.assignment['rows']}
        for row in self.rows:
            for label in(0,1):
                for arm,val in zip(self.E.NATURAL,(0,1,.5,.8,.9,.1)):
                    sample=self.E.sample(row,label,arm);path='SHARDS/'+sample+'-first-query-plan.npz'
                    ref=add(path,npz(val));member.append(ref)
                    doc={'schema':'cf-repair-science-episode-v1','unit':{'row_id':row,'family':byid[row]['family']},
                        'noise_label':label,'arm':arm,'execution_status':'complete','behavior':{'success':arm=='R_resp'},
                        'qualified_anchor_records':[{'query':0}]if arm in('N_T','R_resp')else[],
                        'first_query_plan_values':{'role':'complete_generated_first_plan_before_any_control_is_executed',
                            'arrays':{'relative':path,'bytes':ref['bytes']}}}
                    member.append(add('SCIENCE-EPISODES/'+sample+'.json',C.encoded(doc)))
        manifest={'schema':'cf-repair-1a-results-manifest-v1','job_id':'CONF','files':member}
        mraw=C.encoded(manifest);self.map[prefix+'results/CONF/MANIFEST-v1.json']=mraw
        ar={'schema':'cf-repair-Baidu-archive-v1','files':[{'path':'CONF/'+r['path'],**{k:r[k]for k in('bytes','sha256')},
            'remote':{'uri':prefix+'results/CONF/'+r['path'],**{k:r[k]for k in('bytes','sha256')}}}for r in member]+[
            {'path':'CONF/MANIFEST-v1.json',**C.ident(mraw),'remote':{'uri':prefix+'results/CONF/MANIFEST-v1.json',**C.ident(mraw)}}]}
        arcraw=C.encoded(ar);self.map[prefix+'ARCHIVE-MANIFEST-v1.json']=arcraw
        native={'schema':'cf-repair-baidu-stage23-input-v1','job_fields':{'schema':'cf-repair-stage23-job-v1','job_id':'CONF',
            'shard_rows':self.rows,'stage23_protocol':C.ident(praw),'stage23_G1':C.ident(graw)}}
        iraw=C.encoded(native);self.map[prefix+'control/INPUT-SPEC-v1.json']=iraw
        self.spec={'schema':'cf-repair-confirm-CPU-input-v1','run_id':'synthetic','protocol_identity':C.ident(praw),
            'G1_attachment_identity':C.ident(graw),'freeze_identity':C.ident(fraw),'release_identity':C.ident(rraw),
            'max_GETs':2700,'input_byte_cap':4*1024**3,'jobs':[{'source_job':run,'run_prefix':prefix,
                'protocol_sha256':C.ident(praw)['sha256'],'G1_attachment_identity':C.ident(graw),'shard_rows':self.rows,
                'source_input_spec':{'uri':prefix+'control/INPUT-SPEC-v1.json',**C.ident(iraw)},
                'archive_manifest':{'uri':prefix+'ARCHIVE-MANIFEST-v1.json',**C.ident(arcraw)}}]}

    def read(self,ref,destination,role):
        raw=self.map[ref['uri']];assert C.ident(raw)=={k:ref[k]for k in('bytes','sha256')}
        self.reads.append((role,ref['uri']));dest=Path(destination);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);return dest
    def change_release(self):
        raw=C.encoded(self.release);(self.source/'CONF-OUTCOME-RELEASE-v1.json').write_bytes(raw);self.spec['release_identity']=C.ident(raw)


class Tests(unittest.TestCase):
    def test_original_three_CLIs_distinct_outputs_and_full_admitted_support(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Fixture(tmp);idx=C.restore_inputs(f.source,f.local,f.spec,f.read,f.E)
            ip=f.local/'INDEX.json';ip.write_bytes(C.encoded(idx));sr=f.local/'SR.json';cmd=f.local/'CMD.json'
            common=['--protocol',str(f.source/'PROTOCOL-v4.json'),'--assignments',str(f.source/'ASSIGNMENTS-133-v1.json'),
                '--g1-input',str(f.source/'G1-DECISION-v1.json'),'--g1-sha',f.spec['G1_attachment_identity']['sha256']]
            base=['--base-analysis',str(f.source/'analyze_cases.py')]
            original=zipfile.ZipFile.read;seen=[]
            def only_plan(z,info,*a,**kw):
                name=info.filename if hasattr(info,'filename')else info;self.assertEqual(name,'plan_adapter7.npy');seen.append(name);return original(z,info,*a,**kw)
            with patch.object(sys,'platform','linux'),patch.object(zipfile.ZipFile,'read',only_plan):
                f.E.main(['--cloud-array-read','--index',str(ip),'--sr-output',str(sr),'--command-output',str(cmd),*base,*common])
            f.A.main(['--input',str(sr),'--output',str(f.local/'SR-READOUT.json'),*common])
            f.Q.main(['--input',str(cmd),'--output',str(f.local/'CMD-READOUT.json'),*base,*common])
            self.assertEqual(len(json.loads(sr.read_bytes())['records']),1296)
            self.assertEqual(len(json.loads(cmd.read_bytes())['records']),1296)
            self.assertEqual(len(seen),24)
            self.assertTrue(all('SCIENCE-REPEATS'not in u and '/EPISODES/'not in u for _,u in f.reads))
            self.assertEqual(json.loads((f.local/'CMD-READOUT.json').read_bytes())['schema'],'cf-repair-confirm-command-readout-v1')
            self.assertEqual(json.loads((f.local/'SR-READOUT.json').read_bytes())['schema'],'cf-repair-G1-G2-scalar-readout-v1')

    def test_single_finished_shard_cannot_open_outcomes(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Fixture(tmp);f.release['all_admitted_population_closed']=False;f.change_release()
            with self.assertRaisesRegex(ValueError,'all_population_closes'):C.restore_inputs(f.source,f.local,f.spec,f.read,f.E)
            self.assertFalse(f.reads)

    def test_release_missing_one_planned_row_rejected_before_reads(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Fixture(tmp);del f.release['row_closure'][f.release['admitted_rows'][-1]];f.change_release()
            with self.assertRaisesRegex(ValueError,'exact_all_admitted'):C.restore_inputs(f.source,f.local,f.spec,f.read,f.E)
            self.assertFalse(f.reads)

    def test_duplicate_attempt_rejected_before_outcomes(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Fixture(tmp);f.spec['jobs'].append(copy.deepcopy(f.spec['jobs'][0]))
            with self.assertRaisesRegex(ValueError,'one_authoritative_attempt'):C.restore_inputs(f.source,f.local,f.spec,f.read,f.E)
            self.assertFalse(f.reads)

    def test_missing_manifest_keeps_every_admitted_unknown(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Fixture(tmp);f.spec['jobs'][0]['archive_manifest']=None
            idx=C.restore_inputs(f.source,f.local,f.spec,f.read,f.E)
            sr,cmd=f.E.extract_index(idx,f.protocol,f.assignment,f.g1['G1_decisions'],
                protocol_sha256=f.spec['protocol_identity']['sha256'],g1_identity=f.spec['G1_attachment_identity'])
            self.assertEqual(len(sr['records']),1296);self.assertEqual(len(cmd['records']),1296)
            self.assertTrue(all(r['success']is None for r in sr['records']));self.assertFalse(f.reads)

    def test_unadmitted_family_excluded_without_shrinking_comparison_family(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Fixture(tmp,sensor_admitted=False);idx=C.restore_inputs(f.source,f.local,f.spec,f.read,f.E)
            sr,cmd=f.E.extract_index(idx,f.protocol,f.assignment,f.g1['G1_decisions'],
                protocol_sha256=f.spec['protocol_identity']['sha256'],g1_identity=f.spec['G1_attachment_identity'])
            self.assertEqual(len(sr['records']),576);self.assertEqual(len(cmd['records']),576)
            self.assertTrue(all(r['family']=='Camera Viewpoints'for r in sr['records']))
            readout=f.Q.analyze_commands(cmd,f.protocol,f.assignment,g1_decisions=f.g1['G1_decisions'],base=f.A)
            self.assertEqual(len(readout['comparisons']),3)
            self.assertTrue(all(r['simultaneous_family_size']==6 for r in readout['comparisons']))

    def test_export_only_after_global_release_and_CPU_masks(self):
        with tempfile.TemporaryDirectory()as tmp:
            f=Fixture(tmp);path=f.root/'JOBS.json';path.write_bytes(C.encoded({'schema':'cf-repair-confirm-authoritative-jobs-v1',
                'protocol_identity':f.spec['protocol_identity'],'G1_attachment_identity':f.spec['G1_attachment_identity'],'jobs':f.spec['jobs']}))
            a=SimpleNamespace(protocol=f.source/'PROTOCOL-v4.json',protocol_sha=f.spec['protocol_identity']['sha256'],
                freeze=f.source/'FREEZE-v1.json',freeze_sha=f.spec['freeze_identity']['sha256'],
                g1=f.source/'G1-DECISION-v1.json',g1_sha=f.spec['G1_attachment_identity']['sha256'],
                release=f.source/'CONF-OUTCOME-RELEASE-v1.json',release_sha=f.spec['release_identity']['sha256'],
                assignments=f.source/'ASSIGNMENTS-133-v1.json',jobs_index=path,run_id='jinwen-cf-repair-confirm-readout-synthetic01',
                allocation_deadline=time.time()+900,out=f.root/'EXPORT')
            result=X.export(a);plan=json.loads((a.out/'ROOT-PLAN-v1.json').read_bytes())
            self.assertEqual(plan['GPU'],0);self.assertEqual(plan['public_env']['CUDA_VISIBLE_DEVICES'],'-1')
            self.assertTrue(result['READ_resources'][-1].endswith('results/CONF/*'))
            self.assertFalse(result['ready_for_cloud_submit'])

    def test_provider_empty_NVIDIA_mask_hydrated_only_after_zero_GPU_descriptor(self):
        tree=ast.parse((HERE/'bootstrap_cpu.py').read_bytes())
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='main')
        first=next(i for i,n in enumerate(main.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='cpu_masks'for t in n.targets))
        last=next(i for i,n in enumerate(main.body[first:],first)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
            and any(isinstance(a,ast.Constant)and a.value=='explicit_nonempty_CPU_masks'for a in n.value.args))
        before=ast.unparse(ast.Module(body=main.body[:first],type_ignores=[]))
        self.assertIn("d['resources']['GPU'] == 0",before)
        fake=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES':'-1','NVIDIA_VISIBLE_DEVICES':'','UNRELATED_SECRET':'not-read'})
        env={'d':{'resources':{'GPU':0}},'os':fake,'need':C.need}
        exec(compile(ast.Module(body=main.body[first:last+1],type_ignores=[]),'<CPU-mask-fixture>','exec'),env)
        self.assertEqual(fake.environ['NVIDIA_VISIBLE_DEVICES'],'void')
        self.assertEqual(env['cpu_masks']['original_NVIDIA_VISIBLE_DEVICES'],'')
        self.assertNotIn('UNRELATED_SECRET',str(env['cpu_masks']))


if __name__=='__main__':unittest.main()
