"""Strict metadata-only Plus adapter contract; no runtime/cloud imports."""
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'src/openpi/eval/campaign_policy.py').is_file())
N = HERE.parent
UPSTREAM = ROOT / 'docs/experiments/libero-light-v1/libero-plus-review-20260927/references/upstream-src'
COMMIT = '4976dc30028e805ff8094b55501d532c48fec182'
CLASS_SHA = 'faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
COUNTS = dict(zip(SUITES, (2402, 2518, 2591, 2519)))
BUDGETS = dict(zip(SUITES, (280, 280, 300, 520)))
ORIGINAL = 'sha256:5df68cd8bc720a4a80f1959006fab1a911bace270cc2545d8556eb8ea25038c1'
PROMPT_MODES = ('released_task_language', 'resolved_bddl_language')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def identity(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular source file required')
    h = hashlib.sha256()
    size = 0
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
            size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def read(path):
    p = Path(path)
    need(p.stat().st_size <= 16*1024**2, 'bounded metadata only')
    return json.loads(p.read_bytes())


def exact(path, ref):
    need(identity(path) == {k: ref[k] for k in ('bytes', 'sha256')}, 'file byte identity differs')


def catalog(path=None):
    p = Path(path) if path else UPSTREAM / 'libero/libero/benchmark/task_classification.json'
    need(identity(p)['sha256'] == CLASS_SHA, 'fixed official classification SHA differs')
    value = read(p)
    for suite in SUITES:
        entries = value[suite]
        need(len(entries) == COUNTS[suite], 'full official suite size differs')
        need([e['id'] for e in entries] == list(range(1, COUNTS[suite]+1)), 'one-based official ordering differs')
        need(len({e['name'] for e in entries}) == len(entries), 'duplicate official variant names')
    need(sum(e['difficulty_level'] is None for s in SUITES for e in value[s]) == 121, 'unlabeled entries must remain')
    return value


def make_case(suite, task_id, classification):
    need(suite in SUITES and type(task_id) is int and 0 <= task_id < COUNTS[suite], 'official variant id outside suite')
    e = classification[suite][task_id]
    return {'suite': suite, 'task_id': task_id, 'classification_id': e['id'], 'task_name': e['name'],
            'category': e['category'], 'difficulty_level': e['difficulty_level'], 'episode_idx': 0,
            'trial_start_index': 0, 'num_trials_per_task': 1, 'env_seed': 0, 'max_policy_steps': BUDGETS[suite]}


def validate_case(case, classification):
    for key in ('task_id', 'classification_id', 'episode_idx', 'trial_start_index', 'num_trials_per_task', 'env_seed', 'max_policy_steps'):
        need(type(case.get(key)) is int, 'native JSON integer required: '+key)
    need(case.get('difficulty_level') is None or type(case['difficulty_level']) is int, 'difficulty must retain integer/null identity')
    need(case == make_case(case['suite'], case['task_id'], classification), 'case must exactly match official variant and fixed trial/budget')


def object_ref(ref):
    need(set(ref) == {'uri', 'bytes', 'sha256'}, 'exact immutable object reference fields required')
    u = urlsplit(ref['uri'])
    need(u.scheme == 'bos' and u.netloc == 'redacted-417e8f2cb1dd' and not u.query and not u.fragment
         and u.path.startswith(('/research-storage/adv-vla/', '/team3_data/chekpoints/Jinwen/'))
         and '..' not in Path(u.path).parts, 'exact Jinwen BOS object required')
    need(type(ref['bytes']) is int and ref['bytes'] > 0 and re.fullmatch('[a-f0-9]{64}', ref['sha256']), 'object byte/SHA identity required')


def model_identity(model):
    need(set(model) == {'id', 'parameter_role', 'checkpoint', 'original_artifact_id'}, 'exact model identity fields required')
    need(model['original_artifact_id'] == ORIGINAL, 'original pi05_libero identity differs')
    if model['id'] == 'original':
        need(model['parameter_role'] == 'original', 'original role differs')
    else:
        need(re.fullmatch('[CUL]-s[01]-u(05000|10000)-EMA', model['id']) and model['parameter_role'] == 'EMA', 'trained model role differs')
    object_ref(model['checkpoint'])


def episode_id(case, model, protocol_mode='released_task_language'):
    model_identity(model)
    need(protocol_mode in PROMPT_MODES, 'explicit supported prompt protocol required')
    return f"plus-{COMMIT[:12]}--{protocol_mode}--{model['id']}-{model['checkpoint']['sha256']}--{case['suite']}-v{case['classification_id']:05d}-t0"


def validate_request(request, classification):
    need(set(request) == {'schema', 'run_id', 'official_commit', 'classification_sha256', 'model', 'cases', 'attempt', 'purpose', 'protocol_mode'}, 'request fields differ')
    need(request['schema'] == 'plus-policy-Q1-request-v1' and request['purpose'] == 'qualification_not_formal_score', 'Q1 request only')
    need(re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', request['run_id']), 'safe unique run id required')
    need(request['official_commit'] == COMMIT and request['classification_sha256'] == CLASS_SHA, 'official source identity differs')
    need(request['protocol_mode'] in PROMPT_MODES, 'named prompt protocol required')
    model_identity(request['model'])
    need(request['model']['id'] == 'original', 'Q1 first qualifies released original only')
    need(type(request['attempt']) is int and request['attempt'] >= 0, 'explicit attempt required; no automatic retry')
    need(type(request['cases']) is list and 0 < len(request['cases']) <= 28, 'bounded Q1 cases required')
    for case in request['cases']:
        validate_case(case, classification)
    need(len({episode_id(c, request['model'], request['protocol_mode']) for c in request['cases']}) == len(request['cases']), 'duplicate protocol/model/variant identity')


def validate_admission(admission, request, request_path, now):
    need(admission.get('schema') == 'plus-policy-root-admission-v1' and admission.get('execute') is True,
         'explicit exact root admission required')
    need(admission.get('provider') == 'baidu' and admission.get('owner') == 'Jinwen' and admission.get('run_id') == request['run_id'], 'cloud owner/run binding differs')
    need(admission.get('request_sha256') == identity(request_path)['sha256'], 'exact request admission differs')
    start, work, end = (admission.get(k) for k in ('created_epoch', 'science_deadline_epoch', 'allocation_deadline_epoch'))
    need(all(type(t) in (int, float) and math.isfinite(t) for t in (start, work, end)), 'finite absolute deadlines required')
    need(start <= now < work <= end-600 and 0 < end-start <= 3600, 'one-hour maximum with 600s archive reserve required')
    need(type(admission.get('case_timeout_seconds')) is int and 0 < admission['case_timeout_seconds'] <= 1800, 'bounded case timeout required')
    need(admission.get('cloud_parent_supervisor_armed') is True and admission.get('resources') == {'GPU': 1, 'CPU': 8, 'memory_GiB': 64}, 'actual cloud supervisor/resource admission required')
    prefix = admission.get('output_prefix', '')
    need(prefix == f"redacted-1732b90171c8'run_id']}/", 'unique exact Jinwen output prefix required')
    return admission
