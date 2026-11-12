"""Actual metadata consumer for the new explicit-init Q0 contract; no model I/O."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
import trusted_init

SCHEMA='plus-explicit-init-runtime-source-gate-v1'
COMMIT='4976dc30028e805ff8094b55501d532c48fec182'
CLASS_SHA='faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6'
BASE_INDEX_SHA='63e0e0ef0a10ef9600a6080769f3b15d8b6a5c2d302b6d2539708c9f1cb36509'
BASE_ARCHIVE_SHA='71277ae2801157f68a098ff811cf6b4fd2406f8fd5927a3130fc74b3679fc26e'
V6_RESULT_SHA='40acab64526159f46e8082cedcec4496fc81f647ac176c2ed966ff967effc2a1'
VARIANT_SHA='4c14a2cbf43f300a8111523f0a4bb9d2dd20ca7d5282f2bedf04884e006710df'

def need(ok,message):
    if not ok:raise ValueError(message)

def object_ref(ref):
    u=urlsplit(ref['uri']);need(u.scheme=='bos' and u.netloc=='redacted-417e8f2cb1dd' and u.path.startswith('/research-storage/'),'BOS/Jinwen object required')
    need(type(ref['bytes']) is int and ref['bytes']>0 and len(ref['sha256'])==64 and set(ref['sha256'])<=set('0123456789abcdef'),'object byte identity missing')

def validate_gate(path,expected):
    need(trusted_init.identity(path)=={k:expected[k] for k in ('bytes','sha256')},'Q0 gate byte identity differs')
    need(Path(path).stat().st_size<=1024**2,'Q0 gate metadata bound')
    value=json.loads(Path(path).read_bytes())
    need(value.get('schema')==SCHEMA and value.get('status')=='qualified_explicit_init_runtime_source','explicit-init Q0 producer not qualified')
    need(value.get('official_commit')==COMMIT and value.get('classification_sha256')==CLASS_SHA,'official source identity differs')
    need(type(value.get('resolved_variant_count')) is int and value['resolved_variant_count']==10030 and value.get('unresolved_variant_ids')==[], 'all10030 actual compatible selectors required')
    for field in ('actual_cloud_operator_import','all_source_files_reverified','all_native_init0_and_released_prompts_match_v6','global_torch_load_unchanged','restored_base_whole_and_all_members_verified','default_incompatibility_evidence_preserved','actual_environment_reuse_verified'):
        need(value.get(field) is True,'actual Q0 evidence missing: '+field)
    need(value.get('protocol_mode')=='released_task_language','released prompt protocol changed')
    contract=value['init_contract'];need(contract.get('id')==trusted_init.CONTRACT_ID and contract.get('loader_options')==trusted_init.OPTIONS,'trusted CPU loader contract differs')
    need(contract.get('loader_source')==trusted_init.identity(Path(trusted_init.__file__)),'consumer/producer init source differs')
    need(type(contract.get('unique_init_files_verified')) is int and contract['unique_init_files_verified']==1565 and contract.get('default_loader_retried') is False,'expected unique init population or historical-default boundary differs')
    need(value['base_capsule_index']['sha256']==BASE_INDEX_SHA and value['base_capsule_index']['bytes']==6858 and value['base_archive']=={'bytes':774000640,'sha256':BASE_ARCHIVE_SHA},'verified v6 base capsule differs')
    need(value['original_v6_default_failure']['sha256']==V6_RESULT_SHA and value['original_v6_default_failure']['bytes']==1919528 and type(value['original_v6_default_failure'].get('default_unresolved_count')) is int and value['original_v6_default_failure']['default_unresolved_count']==10030,'v6 failure evidence omitted or changed')
    need(value['variant_reference']['sha256']==VARIANT_SHA and value['variant_reference']['member']=='reports/VARIANT_RUNTIME_INDEX-v1.jsonl','actual native init reference differs')
    for field in ('base_capsule_index','original_v6_default_failure','assets_keeper'):object_ref(value[field])
    need(value.get('EGL_qualified') is False and value.get('model_forward') is False and value.get('actual_environment_reset') is False,'CPU source gate cannot assert Q1 success')
    return value
