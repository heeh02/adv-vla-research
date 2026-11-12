"""Recheck retained native observations before any new environment construction."""
import importlib.util
import json
from pathlib import Path
import sys
from campaign_core import need
from resume_core import load_witness, witness_gate


def load_reference(package, stage, inputs, check):
    folder = Path(package)/"reference_reader"
    sys.path.insert(0, str(folder))
    import physical_codec as C
    spec = importlib.util.spec_from_file_location("_q11_retained_observation_validator", folder/"validate.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    refs = {r["path"]: r for r in inputs["input_objects"]}
    docs, raw_docs, decoded = {}, {}, None
    paths = inputs["reference_paths"]
    need(len(paths) == 10 and len(set(paths.values())) == 10, "ten_unique_reference_records")
    for logical, name in paths.items():
        check()
        ref, path = refs[name], stage/name
        need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= 2*1024**2,
             "bounded_reference_record")
        raw = path.read_bytes()
        need(C.identity(raw) == {k: ref[k] for k in ("bytes", "sha256")},
             "retained_reference_whole_SHA")
        raw_docs[logical] = raw
        if logical.endswith(".gz"):
            docs[logical], decoded = C.load(path)
        else:
            docs[logical] = C.json_bytes(raw)
    manifest = C.json_bytes((folder/"INPUTS.json").read_bytes())
    need(manifest == inputs["reference_input_manifest"], "exact_previous_reference_input_manifest")
    validator.provenance(docs, manifest, folder)
    retained = docs["science/PHYSICAL-CAMPAIGN.json.gz"]
    reconstructed, checks = validator.inspect_campaign(retained, docs["science/NATIVE-SUMMARY.json"],
                                                      docs["science/INPUT-QUALIFICATION.json"])
    raw_witness = raw_docs["witness/science/RESUME-WITNESS.json"]
    witness = load_witness(raw_witness)
    witness_gate(witness)
    need(all(witness.get(k) == v for k, v in reconstructed.items()),
         "all_cloud_witness_fields_recomputed_from_retained_records")
    need(witness["original_campaign_reference"] == next(r for r in manifest["input_objects"]
             if r["logical_path"] == "science/PHYSICAL-CAMPAIGN.json.gz") and
         witness["original_source_manifest"] == manifest["source_manifest"] and
         witness["original_source_input_manifest"] == manifest["source_input_manifest"],
         "exact_original_scientific_source_and_input_scope")
    ar, rt = docs["witness/control/ARCHIVE-RECEIPT.json"], docs["witness/control/RUNTIME-RESULT.json"]
    result = docs["witness/science/q11-resume-witness-result.json"]
    need(ar["archive_complete"] is True and ar["science_archive"]["errors"] == [] and
         ar["run_id"] == rt["run_id"] == "jinwen-paper2-q11-resume-witness-20261021-v1" and
         result["analysis_complete"] is True and result["input_qualified"] is True and
         result["new_environment_execution_authorized"] is False and
         result["new_env_steps"] == result["new_model_calls"] == result["runtime_imports"] == 0,
         "qualified_existing_data_reader_not_execution_authority")
    need(all(ar["control_receipt"][k] == C.identity(raw_docs["witness/control/RUNTIME-RESULT.json"])[k]
             for k in ("bytes", "sha256")), "witness_runtime_archive_identity")
    for name in ("RESUME-WITNESS.json", "q11-resume-witness-result.json"):
        row = next(r for r in ar["science_archive"]["objects"] if r["name"] == name)
        ref = refs[paths["witness/science/"+name]]
        need(all(row[k] == ref[k] for k in ("uri", "bytes", "sha256")),
             "witness_independent_archive_readback_SHA")
    proof = {"schema": "paper2-q11-native-reference-requalification-v1", "passed": True,
        "records_verified": 10, "typed_arrays_verified": decoded["arrays"],
        "trajectory_identities_recomputed": 2, "witness_fields_recomputed": len(reconstructed),
        "old_unmixed_group_comparisons_exact": checks["old_unmixed_group_comparisons_exact"],
        "original_N_status": witness["original_N_attempt_status"],
        "original_N_close_failure": witness["original_N_close_failure"],
        "old_run_status_modified": False, "new_environment_authority_inferred": False,
        "cross_job_parity_established": False, "reference_witness_identity": C.identity(raw_witness),
        "original_campaign_reference": witness["original_campaign_reference"]}
    return retained, raw_witness, proof
