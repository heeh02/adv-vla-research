"""Full ten-object reference linkage with synthetic typed arrays and archive receipts."""
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import pilot_contract as C
import reference_loader as L
import resume_core as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"reference_reader"))
import physical_codec as D
import pinned_writer as W
import reference_fixture as F
import validate as V


def fixture(root):
    folder, stage = root/"reference_reader", root/"stage"
    folder.mkdir()
    stage.mkdir()
    for name in ("validate.py", "physical_codec.py", "pinned_campaign.py", "pinned_writer.py"):
        shutil.copyfile(HERE/"reference_reader"/name, folder/name)
    campaign, native, qualified = F.fixture()
    source_manifest = {"bytes": 10, "sha256": "f"*64}
    input_manifest = {"bytes": 11, "sha256": "i"*64}
    source_zip = {"bytes": 12, "sha256": "z"*64}
    origin = "redacted-bb0bf1ca4c59"
    refs, raw_docs = [], {}
    def store(logical, raw):
        name = __import__('hashlib').sha256(logical.encode()).hexdigest()+('.json.gz' if logical.endswith('.gz') else '.json')
        ref = {"path": name, "logical_path": logical, "uri": origin+logical, **C.ident(raw)}
        (stage/name).write_bytes(raw)
        refs.append(ref)
        raw_docs[logical] = raw
        return ref
    store("science/PHYSICAL-CAMPAIGN.json.gz", gzip.compress(W.encoded(campaign), mtime=0))
    store("science/NATIVE-SUMMARY.json", C.encoded(native))
    store("science/INPUT-QUALIFICATION.json", C.encoded(qualified))
    result = {"synthetic": False, "completed_all_roles": False, "new_env_steps": 140,
              "new_model_calls": 0, "source_manifest": source_manifest, "inputs_identity": input_manifest}
    store("science/PHYSICAL-RESULT.json", C.encoded(result))
    runtime = {"run_id": "fake-original-native", "new_env_steps": 140, "new_model_calls": 0,
        "resources": C.RESOURCES, "input_body_bytes": 796390625, "verified_inputs": [{}]*243,
        "source_abi": {"source_manifest": source_manifest, "source_zip": source_zip,
            "files": [{"path": old, **C.ident(folder/name)} for name, old in
                      (("pinned_campaign.py", "campaign_core.py"), ("pinned_writer.py", "physical_writer.py"))]}}
    control = store("control/RUNTIME-RESULT.json", C.encoded(runtime))
    archived = [{"name": r["logical_path"].split('/')[-1], **{k: r[k] for k in ('uri','bytes','sha256')}}
                for r in refs if r["logical_path"].startswith('science/')]
    archived += [{"name": 'other'+str(i)} for i in range(5)]
    archive = {"run_id": runtime["run_id"], "archive_complete": True, "control_receipt": control,
        "science_archive": {"errors": [], "complete_for_observed_files": True,
                            "objects": archived, "observed_files": 9}}
    store("control/ARCHIVE-RECEIPT.json", C.encoded(archive))
    original_refs = list(refs)
    manifest = {"input_objects": original_refs, "source_run_id": runtime["run_id"],
                "source_manifest": source_manifest, "source_input_manifest": input_manifest, "source_zip": source_zip}
    (folder/"INPUTS.json").write_bytes(C.encoded(manifest))
    witness, _ = V.inspect_campaign(campaign, native, qualified)
    witness.update(original_campaign_reference=refs[0], original_source_manifest=source_manifest,
                   original_source_input_manifest=input_manifest)
    witness_raw = C.encoded(witness)
    store("witness/science/RESUME-WITNESS.json", witness_raw)
    store("witness/science/q11-resume-witness-result.json", C.encoded({
        "analysis_complete": True, "input_qualified": True, "new_environment_execution_authorized": False,
        "new_env_steps": 0, "new_model_calls": 0, "runtime_imports": 0}))
    wrt = store("witness/control/RUNTIME-RESULT.json", C.encoded({"run_id": "jinwen-paper2-q11-resume-witness-20261021-v1"}))
    war = {"run_id": "jinwen-paper2-q11-resume-witness-20261021-v1", "archive_complete": True,
        "control_receipt": wrt, "science_archive": {"errors": [], "objects": [
            {"name": r["logical_path"].split('/')[-1], **{k: r[k] for k in ('uri','bytes','sha256')}}
            for r in refs if r["logical_path"].startswith('witness/science/')]}}
    store("witness/control/ARCHIVE-RECEIPT.json", C.encoded(war))
    inputs = {"input_objects": refs, "reference_paths": {r['logical_path']: r['path'] for r in refs},
              "reference_input_manifest": manifest}
    return stage, inputs, witness_raw


class References(unittest.TestCase):
    def test_ten_objects_full_provenance_payload_hash_and_witness_reconstruction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage, inputs, witness = fixture(root)
            with patch.object(R, "WITNESS", C.ident(witness)):
                retained, raw, proof = L.load_reference(root, stage, inputs, lambda: None)
            self.assertEqual(raw, witness)
            self.assertTrue(proof["passed"])
            self.assertEqual(proof["records_verified"], 10)
            self.assertEqual(retained["roles"][1]["status"], "FAILED_NO_RETRY")
            self.assertFalse(proof["new_environment_authority_inferred"])

    def test_rewrapped_changed_witness_still_fails_pin_even_if_object_SHA_updated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage, inputs, witness = fixture(root)
            name = inputs["reference_paths"]["witness/science/RESUME-WITNESS.json"]
            changed = witness+b' '
            (stage/name).write_bytes(changed)
            row = next(r for r in inputs['input_objects'] if r['path']==name)
            row.update(C.ident(changed))
            with patch.object(R, "WITNESS", C.ident(witness)):
                with self.assertRaisesRegex(ValueError, "exact_actual_cloud_resume_witness"):
                    L.load_reference(root, stage, inputs, lambda: None)


if __name__ == "__main__":
    unittest.main()
