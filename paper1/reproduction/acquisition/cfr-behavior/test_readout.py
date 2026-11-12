"""Small synthetic software tests only, never native/scientific qualification."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

M=module("behavior",HERE/"readout.py")
B=module("frozen_primary",ROOT/"protocol/ONE-B-CANDIDATE/analyze_cases.py")
F=module("frozen_extractor",ROOT/"protocol/CONFIRM-CPU/extract_confirm.py")

def npy(values,shape):
    h=repr({"descr":"<f8","fortran_order":False,"shape":shape}).encode("latin1")+b"\n"
    return b"\x93NUMPY\x01\x00"+struct.pack("<H",len(h))+h+struct.pack("<"+str(len(values))+"d",*values)

def fixture(folder):
    rid="synthetic-row";family=M.FAMILIES[0]
    assignment={"rows":[{"row_id":rid,"family":family,"task_id":"task-A","split":"confirm"}]}
    protocol={"stage3":{"rows":{family:[rid],M.FAMILIES[1]:[]},"anchor_selection":[]},
              "statistics":{"bootstrap":{"seed":20261019,"replicates":10}}}
    g1={family:{"advance":True},M.FAMILIES[1]:{"advance":False}}
    files=[]
    def save_json(name,value):
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));
        files.append({"path":name,**M.ident(p),"sealed":False,"sealed_fields":[]})
    def save_npz(name,data):
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(p,"w") as z:
            for member,(values,shape) in data.items():z.writestr(member+".npy",npy(values,shape))
            # Must not be decoded by any secondary reader.
            z.writestr("H_T__bf16_bits.npy",b"not an NPY")
        files.append({"path":name,**M.ident(p),"sealed":False,"sealed_fields":[]})
        return {"relative":name,"bytes":p.stat().st_size}
    logs=[]
    for n in (0,1):
        for i,arm in enumerate(M.ARMS):
            sample=F.sample(rid,n,arm);plan=[float(i)]*70
            first=save_npz("SHARDS/"+sample+"-first-query-plan.npz",{"plan_adapter7":(plan,(1,10,7))})
            observations=[]
            for endpoint in (5,60,"terminal"):
                step=60 if endpoint=="terminal" else endpoint
                ref=save_npz("SHARDS/"+sample+"-"+str(endpoint)+".npz",{
                    "native_observation__robot0_eef_pos":([float(i+n),0.,0.],(3,))})
                observations.append({"phase":"terminal" if endpoint=="terminal" else "fixed_returned_control",
                    "control_step":step,"sim_time":step*.05,"arrays":ref,"geometry":{"status":"NA"}})
            doc={"schema":"cf-repair-science-episode-v1","sample_id":sample,"arm":arm,"noise_label":n,
                "unit":{"row_id":rid,"family":family},"execution_status":"complete",
                "included_in_primary_six_arm_contrasts":True,"actual_returned_control_steps":60,
                "query_count":1,"first_query_plan_values":{"arrays":first,
                    "role":"complete_generated_first_plan_before_any_control_is_executed"},
                "physical_observations":observations,"qualified_anchor_records":[],"planned_anchor_queries":[],
                "queries":[{"query":0,"issued_controls_start":0,"issued_controls_count":5,"execution_status":"issued_prefix_recorded"}],
                "behavior":{"success":i%2==0,"issued_adapter7":[plan[j*7:j*7+7] for j in range(5)]}}
            save_json("SCIENCE-EPISODES/"+sample+".json",doc)
            if arm in M.REPAIR:
                for j in range(10):
                    logs.append({"schema":"cf-repair-call-log-v1","sample_id":sample,"arm":arm,"episode":True,
                        "query_index":0,"call_index":j,"schedule_active":True,"x_j_sha256":"0"*64,
                        "tokens":{"W_actual_norm":[.1]*10,"actual_noop":[False]*10,"declared_noop":[False]*10,
                                  "H_prime_equals_H_T_bits":[False]*10}})
    p=folder/"CALL-LOG-v1.jsonl.gz"
    with gzip.open(p,"wt") as f:
        for r in logs:f.write(json.dumps(r)+"\n")
    files.append({"path":p.name,**M.ident(p),"sealed":False,"sealed_fields":[]})
    manifest={"schema":"cf-repair-1a-results-manifest-v1","job_id":"CONF","files":files}
    (folder/"MANIFEST-v1.json").write_text(json.dumps(manifest))
    gid={"bytes":1,"sha256":"1"*64}
    index={"schema":"cf-repair-confirm-scalar-index-v1","G1_attachment_identity":gid,"jobs":[{
        "shard_rows":[rid],"root":str(folder),"source_job":"synthetic-job","protocol_sha256":M.PROTOCOL_SHA,
        "G1_attachment_identity":gid,"manifest_identity":M.ident(folder/"MANIFEST-v1.json")}]}
    return index,protocol,assignment,g1,manifest


class SoftwareOnly(unittest.TestCase):
    def test_direction_negative_and_overshoot(self):
        t=[0.]*70;c=[1.]*70
        negative=M.direction_metrics([-1.]*70,c,t,"translation")
        over=M.direction_metrics([2.]*70,c,t,"translation")
        self.assertLess(negative["signed_along_clean"],0)
        self.assertAlmostEqual(over["positive_overshoot"],15**.5)
        self.assertAlmostEqual(over["off_target"],0)

    def test_zero_direction_is_NA_not_deleted(self):
        r=M.direction_metrics([1.]*70,[0.]*70,[0.]*70,"rotation")
        self.assertTrue(r["direction_NA"]);self.assertIsNone(r["signed_along_clean"])
        self.assertLess(r["error_gain"],0)

    def test_prefix_early_stop_and_identity(self):
        p=[float(i) for i in range(70)]
        d={"queries":[{"query":0,"issued_controls_start":0,"issued_controls_count":2}],
           "behavior":{"issued_adapter7":[p[:7],p[7:14]]}}
        r=M.issued_prefix(d,p)
        self.assertEqual(r["first_query_actual_controls"],2)
        self.assertEqual(r["plan_prefix_maxabs"],0)
        self.assertIn("not_mixed_unit_action_strength",r["L2_role"])

    def test_noncontiguous_issued_rejected(self):
        d={"queries":[{"query":0,"issued_controls_start":1,"issued_controls_count":0}],"behavior":{"issued_adapter7":[]}}
        with self.assertRaisesRegex(ValueError,"contiguous"):M.issued_prefix(d,None)

    def test_outcome_orientation_and_unknown_identification(self):
        r=M.outcome_components(0,1)
        self.assertEqual(r["harm"]["value"],1);self.assertEqual(r["recovery"]["value"],0)
        r=M.outcome_components(None,1)
        self.assertIsNone(r["harm"]["value"]);self.assertEqual(r["harm"]["upper"],1)
        self.assertEqual(r["recovery"]["value"],0)

    def test_only_small_allowlisted_members(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root);member=m["files"][0]
            ref={"relative":member["path"],"bytes":member["bytes"]}
            with self.assertRaisesRegex(ValueError,"no_neural"):M.read_array(root,{r["path"]:r for r in m["files"]},ref,"H_T__bf16_bits",(1,))

    def test_original_interface_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root)
            cases,logs=M.extract(i,p,a,g,B,F)
            self.assertEqual(len(cases),12);self.assertEqual(len(logs),80)
            out=M.analyze(cases,logs,p,a,g,B)
            self.assertEqual(len(out["motion"]),36)
            self.assertEqual(out["summary"]["geometry"]["status"],"NA")
            summary=next(r for r in out["summary"]["motion"] if r["arm"]=="R_resp" and r["endpoint"]==5)
            self.assertEqual(summary["six_arm_common_available_noise_cases"],2)
            self.assertEqual(summary["ratio_of_task_weighted_means"],2)
            dose=next(r["dose"] for r in out["behavior"] if r["arm"]=="R_resp")
            self.assertAlmostEqual(dose["all_generated_queries"]["token_L2_sum"],10.)

    def test_second_pass_plan_is_exact_and_no_initial_or_neural_decoding(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root)
            req=F.episode_requests(m,p,a,["synthetic-row"],g);docs=F.load_episode_documents(root,req["requests"])
            plan=M.secondary_requests(m,docs,p,a,["synthetic-row"],g,F)
            self.assertEqual(plan["exact_GET_objects"],49) #12 q0 +36 physical +calllog
            self.assertEqual(plan["exact_payload_bytes"],sum(r["bytes"] for r in plan["requests"]))
            self.assertFalse(plan["H_flow_raw32_noise_decoded"])

    def test_early_termination_does_not_supply_step60(self):
        d={"execution_status":"complete","actual_returned_control_steps":5,"physical_observations":[]}
        r=M.physical_rows(d,".",{})
        self.assertEqual(r[1]["status"],"STRUCTURAL_UNREACHED")
        self.assertIsNone(r[1]["position"])

    def test_technical_terminal_not_complete_endpoint(self):
        d={"execution_status":"technical_missing","physical_observations":[{"phase":"terminal","control_step":10}]}
        r=M.physical_rows(d,".",{})
        self.assertEqual(r[2]["status"],"TECHNICAL_PARTIAL_TERMINAL_NOT_ENDPOINT")

    def test_missing_source_dose_not_zero_and_support_retained(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root);i["jobs"][0]["manifest_identity"]=None
            cases,logs=M.extract(i,p,a,g,B,F);out=M.analyze(cases,logs,p,a,g,B)
            r=next(r for r in out["behavior"] if r["arm"]=="R_resp")
            self.assertIsNone(r["dose"]["all_generated_queries"])
            self.assertEqual(len(cases),12)
            r=out["summary"]["outcomes"][0]
            self.assertIsNone(r["components"]["harm"]["point"])
            self.assertEqual(r["components"]["harm"]["identification_bounds"],[0.,1.])

    def test_duplicate_scientific_attempt_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root);i["jobs"].append(dict(i["jobs"][0]))
            with self.assertRaisesRegex(ValueError,"authoritative_CONF_attempt"):M.extract(i,p,a,g,B,F)

    def test_q12_is_own_state_and_structural_absence(self):
        d={"arm":"R_resp","planned_anchor_queries":[12],"qualified_anchor_records":[],
           "execution_status":"complete","query_count":12}
        r=M.clean_anchors(d,".",{})
        self.assertEqual(r,[{"query":12,"status":"STRUCTURAL_UNREACHED"}])

    def test_original_hierarchy_noise_then_variant_task(self):
        rows=[{"row_id":"a","task_id":"A"},{"row_id":"b","task_id":"A"},{"row_id":"c","task_id":"B"}]
        values={(r["row_id"],0,n):v for r,v in zip(rows,(0.,0.,1.)) for n in (0,1)}
        result=M.summary_value(rows,values,B,0,{"seed":20261019,"replicates":10})
        self.assertEqual(result["point"],.5)

    def test_partial_technical_episode_dose_not_full_exposure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root);cases,logs=M.extract(i,p,a,g,B,F)
            cases[("synthetic-row",0,"R_resp")]["original_execution_status"]="technical_missing"
            out=M.analyze(cases,logs,p,a,g,B)
            r=next(r for r in out["behavior"] if r["arm"]=="R_resp" and r["noise_label"]==0)
            self.assertIsNone(r["dose"]["all_generated_queries"])
            self.assertGreater(r["dose"]["observed_logged_exposure"]["token_L2_sum"],0)

    def test_global_release_required_for_entire_admitted_population(self):
        pid={"bytes":1,"sha256":"0"*64};gid={"bytes":1,"sha256":"1"*64}
        r={"schema":"cf-repair-CONF-outcome-release-v1","protocol_identity":pid,"G1_attachment_identity":gid,
           "all_admitted_population_closed":True,"no_partial_shard_unblinding":True,
           "admitted_rows":["a"],"row_closure":{"a":{"status":"COLLECTION_TERMINAL",
               "evidence_identity":{"bytes":1,"sha256":"2"*64}}}}
        M.validate_release(r,pid,gid,{"a":{}})
        with self.assertRaisesRegex(ValueError,"all_admitted"):M.validate_release(r,pid,gid,{"a":{},"b":{}})

    def test_terminal_source_times_retained_not_aligned(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root);cases,logs=M.extract(i,p,a,g,B,F)
            out=M.analyze(cases,logs,p,a,g,B)
            r=next(r for r in out["motion"] if r["arm"]=="R_resp" and r["endpoint"]=="terminal")
            self.assertTrue(r["terminal_is_each_arms_own_endpoint"])
            self.assertEqual(r["source_observations"][0]["control_step"],60)

    def test_common_motion_support_is_conditional_not_missing_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root);cases,logs=M.extract(i,p,a,g,B,F)
            cases[("synthetic-row",0,"R_full")]["motion"][1].update(status="STRUCTURAL_UNREACHED",position=None)
            out=M.analyze(cases,logs,p,a,g,B)
            r=next(r for r in out["summary"]["motion"] if r["arm"]=="R_resp" and r["endpoint"]==60)
            self.assertEqual(r["six_arm_common_available_noise_cases"],1)
            self.assertEqual(r["full_plan_cases_outside_common_support"],1)
            self.assertIsNone(r["separation"]["full_population_point"])

    def test_zero_sampling_reference_retained_and_ratio_NA(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);i,p,a,g,m=fixture(root);cases,logs=M.extract(i,p,a,g,B,F)
            cases[("synthetic-row",1,"N_C")]["motion"][0]["position"]=[0.,0.,0.]
            out=M.analyze(cases,logs,p,a,g,B)
            r=next(r for r in out["summary"]["motion"] if r["arm"]=="R_resp" and r["endpoint"]==5)
            self.assertEqual(r["six_arm_common_available_noise_cases"],2)
            self.assertIsNone(r["ratio_of_task_weighted_means"])


if __name__=="__main__":unittest.main()
