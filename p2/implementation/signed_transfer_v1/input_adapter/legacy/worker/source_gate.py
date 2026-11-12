"""One precise root-reviewed T09 diagnostic source variant; no IO/fit/model."""
import hashlib,json
REGISTRY = {'schema': 'CONF-readout-approved-source-variant-v1', 'status': 'ROOT_APPROVED_DIAGNOSTIC_SOURCE_IDENTITY_ONLY', 'baseline_python_count': 286, 'baseline_python_files_sha256': 'a3b74c9f21a10230c960adcd2ccd7849937340df863d25bf13409597e9226aae', 'runs': {'CCONF-T06': {'run_id': 'jinwen-state-response-cont-20261013-conf-20261014-v3-t06', 'source_manifest': {'bytes': 65552, 'sha256': '567d53a627991fb86cd39fb4cc0793c928e0810ed1639a888fc33e59a584404f'}}, 'CCONF-T07': {'run_id': 'jinwen-state-response-cont-20261013-conf-20261014-v3-t07', 'source_manifest': {'bytes': 65552, 'sha256': 'b28618d1362125aebdc91e58de93d2d034009b2b0fb09206c02aceee61dce9cf'}}, 'CCONF-T08': {'run_id': 'jinwen-state-response-cont-20261013-conf-20261014-v3-t08', 'source_manifest': {'bytes': 65552, 'sha256': '613cc5dec33db58e90f322d572eee94800fff952d185b3538269e7a114e912c3'}}, 'CCONF-T09': {'run_id': 'jinwen-state-response-cont-20261013-conf-20261014-v4-t09', 'source_manifest': {'bytes': 65744, 'sha256': 'd5ecf2e801d34160acd021b5fed37ce6e5c11fd94a73844c3a2968cfeee701e2'}}}, 'approved_source_variants': [{'shard_id': 'CCONF-T09', 'run_id': 'jinwen-state-response-cont-20261013-conf-20261014-v4-t09', 'source_manifest_sha256': 'd5ecf2e801d34160acd021b5fed37ce6e5c11fd94a73844c3a2968cfeee701e2', 'replacements': [{'path': 'repo/docs/experiments/state-dependent-response-use-20261013-v1/continuous/native-pairing/bridge.py', 'previous': {'bytes': 9410, 'sha256': 'f0886e76370d6c5440e549915266cdbe57dfd652e6d60ec79e87ad8aa2c2d865'}, 'approved': {'bytes': 10305, 'sha256': '73255baec71839f66dd59a166ac0f4ce457b73d10db7a2d687bb741ac6de76e6'}}, {'path': 'study/continuous/native-pairing/bridge.py', 'previous': {'bytes': 9410, 'sha256': 'f0886e76370d6c5440e549915266cdbe57dfd652e6d60ec79e87ad8aa2c2d865'}, 'approved': {'bytes': 10305, 'sha256': '73255baec71839f66dd59a166ac0f4ce457b73d10db7a2d687bb741ac6de76e6'}}, {'path': 'study/continuous/task-heldout-confirmation-v1/worker/science.py', 'previous': {'bytes': 19814, 'sha256': '9bad7b37c58fc1bdfd3228de59992b4caacbfb03f9e7d810a79b326eaaf591b8'}, 'approved': {'bytes': 20446, 'sha256': '98e74e98f2ee6ac6aa189eb4adb08e647ef2cf2ab1e8d4223875ad99f3909282'}}], 'repair_proof': {'root_review': {'path': 'root/ROOT-REPAIR-REVIEW-v4.md', 'bytes': 826, 'sha256': '46947a8649da7f6663567fe618b28574de147abc63f741dfc5b4423db9e115f0'}, 'source_delta': {'path': 'root/campaign-v4/SOURCE-REPAIR-DELTA-v4.json', 'bytes': 2456, 'sha256': 'cef06020d7058791d80700c88ccba5e798faedbec31d0f58a9866e760b4fc38f'}, 'software_QA': {'path': 'execution-v4/SOFTWARE-QA-v4.json', 'bytes': 1552, 'sha256': '9bfadd2a6d4325f4ca44843f876e0afe1dd16b9a58487c876dd7dce3f4d3fd9c'}, 'source_diff': {'path': 'execution-v4/SOURCE-DIFF-v4.patch', 'bytes': 3991, 'sha256': '1fc030e21615e7588425f42562571aaf848e100b21ebd1a462e370fb8f3386d9'}, 'AST_test': {'path': 'execution-v4/test_repair.py', 'bytes': 6852, 'sha256': '554e799719d6ae171df8dd66624846915ab482673deaf588cad886673cff8305'}}}], 'scientific_metric_fit_changes': False, 'T09_v3_failure_must_be_retained': True, 'T09_v4_completion_not_proof_original_sensor_error_fixed': True}

def need(ok,code):
 if not ok:raise ValueError(code)
def normalized(rows):return sorted([{k:r[k]for k in ('path','bytes','sha256')}for r in rows],key=lambda r:r['path'])
def digest(rows):return hashlib.sha256((json.dumps(normalized(rows),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
def verify(manifest,shard,plan):
 sid=shard['shard_id'];need(sid in REGISTRY['runs'],'only_predeclared_four_shards')
 bound=REGISTRY['runs'][sid]
 need(shard['source_run_id']==manifest['run_id']==bound['run_id'],'exact_selected_source_run')
 need(all(shard['source_manifest'][k]==bound['source_manifest'][k]for k in ('bytes','sha256')),'exact_selected_source_manifest')
 baseline=plan['shared_scientific_files'];need(len(baseline)==REGISTRY['baseline_python_count']and digest(baseline)==REGISTRY['baseline_python_files_sha256'],'complete_original_python_source_baseline')
 variants=plan.get('approved_source_variants',[]);need(len(variants)==1,'exact_one_explicit_source_variant')
 actual,expected=variants[0],REGISTRY['approved_source_variants'][0]
 need(all(actual.get(k)==expected[k]for k in ('shard_id','run_id','source_manifest_sha256','replacements')),'only_exact_root_approved_T09_changes')
 proofs=actual.get('repair_proof',{});need(set(proofs)==set(expected['repair_proof']),'all_five_review_proof_identities')
 for name,ref in expected['repair_proof'].items():
  need(all(proofs[name].get(k)==ref[k]for k in ('bytes','sha256')),'exact_root_review_proof_'+name)
 # The IO prepare stage verifies the actual packaged proof bytes. These pinned
 # identities cannot be replaced by a plan-supplied approval boolean.
 target={r['path']:{k:r[k]for k in ('bytes','sha256')}for r in baseline}
 if sid==expected['shard_id']:
  for r in expected['replacements']:
   need(target.get(r['path'])==r['previous'],'exact_previous_source_before_replacement');target[r['path']]=r['approved']
 current={r['path']:{k:r[k]for k in ('bytes','sha256')}for r in manifest['files']if r['path'].endswith('.py')}
 need(len(current)==sum(r['path'].endswith('.py')for r in manifest['files'])and current==target,'all_python_sources_exact_no_wildcard_or_missing')
 return {'source_run_id':bound['run_id'],'source_manifest_sha256':bound['source_manifest']['sha256'],'approved_variant_used':sid==expected['shard_id'],'all_python_files_verified':len(current),'metrics_or_fit_changed':False}
