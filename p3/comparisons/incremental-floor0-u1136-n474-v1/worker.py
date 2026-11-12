"""Bounded existing-metadata CPU increment. No network, Create, Stop or PUT API.

Root supplies a read-only exact store or cloud-staged object map. Frozen source
files remain byte-identical. Old gate labels are reused only for unchanged
endpoints; all newly affected pairs run the original input gate directly.
"""
import argparse
import ast
import base64
import copy
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import platform
import sys
import time
import types
from collections import Counter
from pathlib import Path

MODEL = 'T-floor-s0-u30000-EMA'
RUN = 'jinwen-t30-eval-floor-s0-plus-u1136-20261019-v10'
NEW = set(range(1136, 1610))
NEW_COUNT = 474
OLD_T30_COUNT = 3152
POPULATION_COUNT = 10030
MAX_INPUT = 400 * 1024**2
MAX_OUTPUT = 32 * 1024**2
SUMMARY_RESERVE = 256 * 1024


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def file_identity(path):
    h = hashlib.sha256()
    n = 0
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            n += len(chunk)
            h.update(chunk)
    return {'bytes': n, 'sha256': h.hexdigest()}


def exact(raw, ref):
    need(identity(raw) == {k: ref[k] for k in ('bytes', 'sha256')}, 'exact_input_SHA')
    return raw


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dependencies(manifest, root=None):
    paths = {}
    for row in manifest['files']:
        name = row['package_path']
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'relative_dependency')
        p = Path(root) / name if root else Path(row['source_path'])
        need(file_identity(p) == {k: row[k] for k in ('bytes', 'sha256')}, 'frozen_dependency_SHA:' + name)
        paths[name] = p
    r = load_module('recover', paths['frozen-repair/recover.py'])
    f = load_module('_increment_trace_source', paths['frozen-repair/future_trace_overlay.py'])
    b = r.bridge(paths['frozen-bridge/bridge.py'].parent)
    return b, r, f, paths


class ExactReads:
    def __init__(self, store, refs, deadline):
        self.store, self.deadline = store, deadline
        self.refs = {r['uri']: r for r in refs}
        need(len(self.refs) == len(refs), 'unique_exact_readset')
        self.expected = sum(r['bytes'] for r in refs)
        need(self.expected <= MAX_INPUT, 'input_cap400MiB')
        need(all(r['uri'].startswith('redacted-24fc043a6cae')
                 and type(r['bytes']) is int and r['bytes'] > 0 for r in refs), 'bounded_Jinwen_readset')
        self.seen, self.bytes = set(), 0

    def begin(self, uri):
        need(time.time() < self.deadline, 'absolute_science_deadline')
        need(uri in self.refs and uri not in self.seen, 'unlisted_or_repeated_GET')
        ref = self.refs[uri]
        need(self.bytes + ref['bytes'] <= MAX_INPUT, 'input_cap400MiB')
        self.seen.add(uri)
        self.bytes += ref['bytes']
        return ref

    def get(self, uri, limit):
        ref = self.begin(uri)
        need(limit == ref['bytes'], 'exact_GET_limit')
        return exact(self.store.get(uri, limit=limit), ref)

    def download(self, uri, path, ref):
        expected = self.begin(uri)
        need(all(ref[k] == expected[k] for k in ('bytes', 'sha256')), 'exact_download_ref')
        self.store.download(uri, path, expected)
        need(file_identity(path) == {k: expected[k] for k in ('bytes', 'sha256')}, 'whole_download_SHA')


class OutputBudget:
    def __init__(self, directory, maximum=MAX_OUTPUT):
        self.directory = Path(directory)
        self.maximum = maximum
        self.directory.mkdir(parents=True, exist_ok=False)
        self.used = 0

    def charge(self, count, reserve=True):
        cap = self.maximum - (SUMMARY_RESERVE if reserve else 0)
        need(self.used + count <= cap, 'output_cap32MiB')
        self.used += count

    def json(self, name, value, reserve=True):
        raw = canonical(value) + b'\n'
        self.charge(len(raw), reserve)
        with (self.directory / name).open('xb') as f:
            f.write(raw)

    def gzip(self, name, max_file_bytes=None):
        owner = self
        raw = (self.directory / name).open('xb')
        written = 0
        class Sink:
            def write(self, data):
                nonlocal written
                need(max_file_bytes is None or written+len(data)<=max_file_bytes,
                     'new474_delta_compressed_cap24MiB')
                owner.charge(len(data))
                written += len(data)
                return raw.write(data)
            def flush(self):
                raw.flush()
        gz = gzip.GzipFile(filename='', mode='wb', fileobj=Sink(), mtime=0, compresslevel=6)
        class Stream:
            def __enter__(self):
                return gz
            def __exit__(self, *args):
                try:
                    gz.close()
                finally:
                    raw.close()
        return Stream()


def compact_endpoint(row):
    """Lossless for the frozen pair gate, not a 22-field attribution cache.

    q0 and runtime signatures, all typed noise values and query-key presence
    are copied without hashing them again into a substitute representation.
    Omitted fields stay reachable through exact original object/row provenance.
    """
    required = ('panel model_id checkpoint_sha256 ordinal case_definition_sha256 '
                'population_identity run_id attempt adopted success status outcome_canonical_sha256').split()
    result = {k: copy.deepcopy(row[k]) for k in required}
    for key in ('id', 'summary', 'outcome', 'source_package_sha256', 'cache_provenance', 'reason'):
        if key in row:
            result[key] = copy.deepcopy(row[key])
    inp = row.get('input', {})
    keys = ('status', 'signature', 'runtime_signature', 'recorded_noise_by_query', 'error_type', 'reason')
    result['input'] = {k: copy.deepcopy(inp[k]) for k in keys if k in inp}
    result['input_projection'] = {
        'schema': 'original-pair-gate-sufficient-input-v1',
        'normalized_input_before_projection_sha256': hashlib.sha256(canonical(inp)).hexdigest(),
        'omitted_original_keys': sorted(set(inp) - set(result['input'])),
        'field_attribution_supported': False,
        'missing_fields_not_fabricated': True,
    }
    return result


def source_trace(row, helper):
    if row is None:
        return None
    # This frozen candidate explicitly handles the actual Original legacy ABI:
    # summary:null plus immutable cache object / row identity, never a fake URI.
    return helper.source_trace(row)


def frozen_single_row(b):
    """Extract the three untouched single-statistic statements from write_tables."""
    tree=ast.parse(Path(b.__file__).read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='write_tables')
    candidates=[n for n in ast.walk(fn) if isinstance(n,ast.For) and
                isinstance(n.target,ast.Tuple) and ast.dump(n.target)==
                ast.dump(ast.parse('for m, model in spec["models"].items():\n pass').body[0].target)]
    need(len(candidates)==1,'exact_single_statistics_loop')
    body=candidates[0].body
    need([n.targets[0].id for n in body[:4]]==['rows','yes','missing','row'], 'exact_single_statements')
    arguments=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in
        ('kind','label','ords','m','buckets','spec')],kwonlyargs=[],kw_defaults=[],defaults=[])
    new=ast.FunctionDef(name='_single_from_frozen_source',args=arguments,
        body=copy.deepcopy(body[:4])+[ast.Return(value=ast.Name(id='row',ctx=ast.Load()))],decorator_list=[])
    namespace=dict(vars(b))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new],type_ignores=[])),
                 str(b.__file__), 'exec'),namespace)
    return namespace[new.name]


class MergedCells:
    def __init__(self, b, spec, cases, old_endpoints, old_qualification, old_t30, baseline, new_rows):
        self.b, self.spec, self.cases = b, spec, cases
        self.old_qualification = old_qualification
        self.contrasts = spec['comparison']['contrasts']
        self.population = spec['case_list_identity']
        self.buckets = {}
        for (m, o), endpoint in old_endpoints.items():
            if endpoint is not None:
                self.buckets['Plus', m, o] = dict(endpoint, model_id=m, ordinal=o,
                    case_definition_sha256=b.digest(cases[o]), population_identity=self.population)
        self.sources = {(r['model_id'], r['ordinal']): r for r in old_t30}
        for row in baseline:
            key = (row['model_id'], row['ordinal'])
            need(key in old_endpoints and old_endpoints[key] is not None, 'baseline_old_presence')
            need(all(row[k] == old_endpoints[key][k] for k in ('success', 'status')), 'baseline_old_outcome')
            need(key not in self.sources, 'baseline_cache_duplicate')
            self.sources[key] = row
            self.buckets['Plus', *key] = row
        expected = {k for k, e in old_endpoints.items() if e is not None and k[0] in b.BASELINES}
        need({k for k in self.sources if k[0] in b.BASELINES} == expected, 'all_old_baselines_present')
        self.changed = set()
        for row in new_rows:
            key = (row['model_id'], row['ordinal'])
            need(key not in self.sources and old_endpoints.get(key) is None, 'new_case_already_adopted')
            need(key not in self.changed, 'duplicate_new_case')
            self.changed.add(key)
            self.sources[key] = row
            self.buckets['Plus', *key] = row
        self.direct = {}
        _, _, self.gate = b.source_functions()
        for c in self.contrasts:
            for m, o in self.changed:
                if m not in (c['control'], c['treatment']):
                    continue
                a = self.buckets.get(('Plus', c['control'], o))
                z = self.buckets.get(('Plus', c['treatment'], o))
                if a and z:
                    need(all(a[k] == z[k] for k in ('case_definition_sha256', 'population_identity')), 'different_population_case')
                    need('input' in a and 'input' in z, 'new_pair_requires_actual_both_inputs')
                self.direct[c['id'], o] = b.pair_diagnostics(a, z, self.gate)

    def label(self, contrast, ordinal):
        key = contrast['id'], ordinal
        return copy.deepcopy(self.direct[key] if key in self.direct else self.old_qualification[key])

    def pair_group(self, ordinals, contrast):
        def dispatch(a, z, ignored_gate):
            if not a and not z:
                return self.b.pair_diagnostics(a, z, self.gate)
            o = a['ordinal'] if a else z['ordinal']
            return self.label(contrast, o)
        original = self.b.compare_group
        fn = types.FunctionType(original.__code__, dict(original.__globals__, pair_diagnostics=dispatch),
                                original.__name__, original.__defaults__, original.__closure__)
        need(fn.__code__ is original.__code__, 'unchanged_statistic_code_object')
        return fn(ordinals, contrast['control'], contrast['treatment'], self.buckets, self.gate)


def write_increment_tables(merged, old_singles, old_pairs, old_groups, output, trace_helper, table_refs, deadline):
    """Run untouched single statements and original compare_group code object.

    Unchanged contrasts/groups must equal archived values before output.
    """
    b, spec = merged.b, merged.spec
    single_row=frozen_single_row(b)
    all_singles, all_pairs = [], []
    def encoded_row(row, columns):
        buffer = io.StringIO(newline='')
        w = csv.DictWriter(buffer, fieldnames=columns)
        w.writerow(row)
        return buffer.getvalue().encode()
    with output.gzip('SINGLE-GROUPS-v1.csv.gz') as sf, output.gzip('PAIRED-GROUPS-v1.csv.gz') as pf:
        sw = pw = None
        for kind, label, ordinals in old_groups:
            need(time.time() < deadline, 'absolute_science_deadline')
            for m in spec['models']:
                row=single_row(kind,label,ordinals,m,merged.buckets,spec)
                if not any((m,o) in merged.changed for o in ordinals):
                    for k,v in old_singles[m,kind,label].items():
                        need(abs(row[k]-v)<=1e-12 if type(v)is float else row[k]==v,'unchanged_old_single_cell:'+k)
                need(row['planned'] == len(ordinals), 'original_denominator_unchanged')
                if sw is None:
                    sw = list(row)
                    sf.write((','.join(sw)+'\r\n').encode())
                sf.write(encoded_row(row, sw))
                if kind == 'all':
                    all_singles.append(row)
            for c in merged.contrasts:
                values = merged.pair_group(ordinals, c)
                row = dict(panel='Plus', seed=spec['comparison']['models'][c['treatment']]['seed'],
                           **c, group_kind=kind, group=label, **values)
                affected = any((m, o) in merged.changed for m in (c['control'], c['treatment']) for o in ordinals)
                if not affected:
                    # Archived float string conversion retains <=1e-12 precision.
                    for k, v in old_pairs[c['id'], kind, label].items():
                        need(abs(row[k]-v) <= 1e-12 if type(v) is float else row[k] == v,
                             'unchanged_old_pair_cell:' + k)
                if kind == 'all':
                    all_pairs.append(copy.deepcopy(row))
                row['reasons'] = canonical(row['reasons']).decode()
                row['diagnostics'] = canonical(row['diagnostics']).decode()
                if pw is None:
                    pw = list(row)
                    pf.write((','.join(pw)+'\r\n').encode())
                pf.write(encoded_row(row, pw))
    count = 0
    with output.gzip('PAIRED-CASE-QUALIFICATION-DELTA-v1.jsonl.gz') as f:
        for c in merged.contrasts:
            for o, case in enumerate(merged.cases):
                need(time.time() < deadline, 'absolute_science_deadline')
                direct = (c['id'], o) in merged.direct
                if not direct: continue  # Old120360-row trace remains its immutable parent artifact.
                row = dict(contrast=c['id'], panel='Plus', ordinal=o,
                           case_definition_sha256=b.digest(case), population_identity=merged.population,
                           **merged.label(c, o),
                           qualification_evidence='direct_original_gate_new_endpoint' if direct else 'verified_archived_singleton_gate_label',
                           control_source=source_trace(merged.sources.get((c['control'], o)), trace_helper),
                           treatment_source=source_trace(merged.sources.get((c['treatment'], o)), trace_helper))
                if not direct:
                    row['original_table_row'] = {'object': table_refs[1], 'contrast': c['id'],
                        'group_kind': 'task', 'group': canonical([case.get(k) for k in ('suite','task_id','task_name')]).decode()}
                f.write(canonical(row)+b'\n')
                count += 1
    return all_singles, all_pairs, count


def publish_verified(callback, phase, files, prefix):
    """Root-owned writer must return exact whole-object readback receipts.

    This module has no uploader or credentials. No callback means no claim of
    cloud durability and no attempt to spend the remaining output budget on a
    cache before the already computed scientific outputs are durable.
    """
    if callback is None:
        return None
    receipts = callback(phase, files)
    need(isinstance(receipts, list) and len(receipts) == len(files), 'complete_phase_publication_receipts')
    by_name = {r['relative']: r for r in receipts}
    need(len(by_name) == len(files), 'unique_phase_publication_receipts')
    for path in files:
        row = by_name[path.name]
        need(row['uri'] == prefix + path.name and row.get('whole_GET_SHA_verified') is True,
             'exact_phase_BOS_readback')
        need(file_identity(path) == {k: row[k] for k in ('bytes','sha256')}, 'published_phase_SHA')
    return receipts


def extend_prior_snapshot(b, spec, docs, inputs, task):
    """Use the already accepted504 metadata addition; no old OUTCOME read."""
    need(spec['expected_T30_unique'] == 2648, 'original2648_control_root')
    packed = exact((inputs/'OLD-FLOOR1-504-ENDPOINTS-v1.json.gz').read_bytes(), task['prior_floor1_endpoints'])
    extra = json.loads(gzip.decompress(packed))
    need(len(extra) == 504 and {r['ordinal'] for r in extra} == set(range(504, 1008)),
         'same_prior_floor1_504')
    spec, docs = copy.deepcopy(spec), copy.deepcopy(docs)
    keys = {(r['model_id'], r['ordinal']) for r in docs['t30_endpoints']}
    for row in extra:
        need(row['model_id'] == 'T-floor-s1-u30000-EMA'
             and row['run_id'] == 'jinwen-t30-eval-floor-s1-plus-u504-20261019-v7'
             and row['attempt'] == 0, 'accepted_old_increment_identity')
        key = row['model_id'], row['ordinal']
        need(key not in keys, 'old3152_no_duplicate_endpoint')
        keys.add(key)
        b.validate_endpoint(row, spec['models'], docs['historical_plan']['population_cases'])
    docs['t30_endpoints'].extend(extra)
    spec['expected_T30_unique'] = OLD_T30_COUNT
    need(len(docs['t30_endpoints']) == OLD_T30_COUNT, 'old3152_metadata_only_snapshot')
    return spec, docs


class SummarySourceStore:
    """Original materialize_t30 reads exact packaged summary bytes then cloud OUTCOME."""
    def __init__(self, raw, new_rows, transport):
        encoded_rows = json.loads(raw)
        self.summaries = {uri: base64.b64decode(value, validate=True)
                          for uri, value in encoded_rows.items()}
        self.refs = {row['summary']['uri']: row['summary'] for row in new_rows}
        need(set(self.summaries) == set(self.refs) and len(self.refs) == NEW_COUNT,
             'exact474_packaged_summaries')
        for uri, payload in self.summaries.items():
            exact(payload, self.refs[uri])
        self.transport, self.seen = transport, set()

    def get(self, uri, limit):
        if uri not in self.summaries:
            return self.transport.get(uri, limit)
        need(uri not in self.seen and limit == self.refs[uri]['bytes'],
             'one_exact_packaged_summary_read')
        self.seen.add(uri)
        return self.summaries[uri]


def sign(net, qualified):
    return None if qualified == 0 else ('positive' if net > 0 else 'negative' if net < 0 else 'zero')


def coverage_change(merged, old_singles, old_pairs, groups):
    """Observed old/new/additional support only, no population/causal inference."""
    primary = {'floor_s0_T30_minus_T20', 'floor_s0_T30_minus_Original'}
    rows = []
    for kind, label, ordinals in groups:
        if kind not in ('all', 'environment', 'environment_difficulty', 'environment_seed'):
            continue
        added = sum((MODEL, o) in merged.changed for o in ordinals)
        for contrast in merged.contrasts:
            if contrast['id'] not in primary:
                continue
            old = old_pairs[contrast['id'], kind, label]
            new = merged.pair_group(ordinals, contrast)
            fields = ('planned', 'both_endpoints', 'qualified_pairs', 'SS', 'SF', 'FS', 'FF',
                      'gains', 'harms', 'net_qualified_count', 'unresolved_or_rejected',
                      'qualified_subset_delta_SR', 'completion_lower', 'completion_upper',
                      'reasons', 'diagnostics')
            increment = {k: new[k]-old[k] for k in
                         ('both_endpoints', 'qualified_pairs', 'SS', 'SF', 'FS', 'FF',
                          'gains', 'harms', 'net_qualified_count')}
            need(all(increment[k] >= 0 for k in
                     ('both_endpoints', 'qualified_pairs', 'SS', 'SF', 'FS', 'FF', 'gains', 'harms')),
                 'new_support_does_not_replace_old_results')
            old_sign = sign(old['net_qualified_count'], old['qualified_pairs'])
            new_sign = sign(new['net_qualified_count'], new['qualified_pairs'])
            rows.append(dict(contrast=contrast['id'], group_kind=kind, group=label,
                added_closed_endpoint_count=added,
                old={k: old[k] for k in fields}, cumulative={k: new[k] for k in fields},
                added_qualified_cohort=increment,
                added_qualified_cohort_delta=(increment['net_qualified_count']/increment['qualified_pairs']
                                             if increment['qualified_pairs'] else None),
                old_observed_sign=old_sign, cumulative_observed_sign=new_sign,
                sign_changed_with_positive_support=(old_sign is not None and new_sign is not None
                                                   and old_sign != new_sign),
                newly_qualified_group=old['qualified_pairs'] == 0 < new['qualified_pairs'],
                qualification_selection_and_coverage_not_held_constant=True))
    return dict(schema='paper3-floor0-added474-coverage-change-v1',
        question='Does the new task/environment coverage change observed floor0 gain/harm versus same-seed T20 and Original?',
        old_T30=OLD_T30_COUNT, added_T30=NEW_COUNT, cumulative_T30=OLD_T30_COUNT+NEW_COUNT,
        rows=rows, no_new_effect_claim_when_added_qualified_count_zero=True,
        original_group_labels_and_nulls_retained=True, parent_task_family_not_inferred_from_singleton_variant_labels=True,
        full_population_evaluation_complete=False, population_effect_identified=False,
        CI=None, training_seeds_pooled=False, plots_created=0)


def run(inputs, dependency_root, store, work, out, deadline_epoch, publish_phase=None):
    need(platform.system() != 'Darwin', 'cloud_CPU_only_no_Mac_data')
    need(0 < deadline_epoch-time.time() <= 3600, 'absolute_deadline_at_most1h')
    inputs, work = Path(inputs), Path(work)
    task = json.loads((inputs/'INPUT-SPEC-v1.json').read_bytes())
    need(task['schema'] == 'paper3-floor0-u1136-added474-input-v1', 'increment474_schema')
    need(file_identity(__file__) == task['worker_source'], 'increment_worker_source_SHA')
    manifest = json.loads(exact((inputs/'DEPENDENCIES-v1.json').read_bytes(), task['dependencies']))
    b, r, f, paths = dependencies(manifest, dependency_root)
    controls = paths['frozen-bridge/inputs-2648-v1/INPUT-SPEC-v1.json'].parent
    spec, docs = b.load_controls(controls)
    spec, docs = extend_prior_snapshot(b, spec, docs, inputs, task)
    need(spec['case_list_identity'] == task['case_list_identity'], 'same_original10030_population')
    new_rows = json.loads(gzip.decompress(exact((inputs/'NEW-ENDPOINTS-v1.json.gz').read_bytes(), task['new_endpoints'])))
    need(len(new_rows) == NEW_COUNT and {x['ordinal'] for x in new_rows} == NEW, 'exact474_all_new_to3152')
    for row in new_rows:
        need(row['model_id'] == MODEL and row['run_id'] == RUN and row['attempt'] == 0,
             'exact_original_new_case_identity')
        b.validate_endpoint(row, spec['models'], docs['historical_plan']['population_cases'])
    reads = json.loads(exact((inputs/'EXACT-READSET-v1.json').read_bytes(), task['readset']))
    transport = ExactReads(store, reads['objects'], deadline_epoch)
    work.mkdir(parents=True, exist_ok=False)
    output = OutputBudget(out)
    snapshot_done, durable = False, None
    try:
        decoded = {}
        for row in task['prior_tables']:
            packed = transport.get(row['uri'], row['bytes'])
            with gzip.GzipFile(fileobj=io.BytesIO(packed)) as stream:
                raw = stream.read(row['decoded_limit_bytes']+1)
            need(len(raw) <= row['decoded_limit_bytes'], 'bounded3152_table_decode')
            decoded[row['name']] = raw
        singles, pairs, old_endpoints, qualification, groups = r.validate_tables(
            b, spec, docs, decoded['SINGLE-GROUPS-v1.csv.gz'], decoded['PAIRED-GROUPS-v1.csv.gz'])
        expected = json.loads(exact((inputs/'OLD-3152-RESULT-v1.json').read_bytes(), task['prior_result']))
        need(expected['T30_total'] == OLD_T30_COUNT and expected['snapshot_comparison_complete'] is True,
             'actual3152_completed_snapshot')
        for row in expected['single_endpoints']:
            r.same(singles[row['model'], 'all', 'ALL'], row)
        for row in expected['paired_contrasts']:
            r.same(pairs[row['id'], 'all', 'ALL'], row)
        del decoded
        normalize, restore, gate = b.source_functions()
        baseline = []
        for role, count in [('old_cache', 106958), ('merged_delta', 7234)]:
            cloud = spec['caches'][role];path = work/(role+'.jsonl.gz')
            transport.download(cloud['uri'], path, cloud)
            tagged = f.tagged_cache_rows(b.stream_cache(path, count), cloud)
            for row in b.cache_rows(docs['historical_plan'], role, tagged, restore, spec['case_list_identity']):
                need(time.time() < deadline_epoch, 'absolute_science_deadline')
                b.validate_endpoint(row, spec['models'], docs['historical_plan']['population_cases'])
                baseline.append(compact_endpoint(row))
        summary_blob = gzip.decompress(exact((inputs/'NEW-SUMMARIES-v1.json.gz').read_bytes(), task['new_summaries']))
        need(len(summary_blob) <= 4*1024**2, 'small_exact_summary_control_blob')
        source_store = SummarySourceStore(summary_blob, new_rows, transport)
        materialized = []
        for raw in new_rows:
            row = b.materialize_t30(raw, spec['models'][raw['model_id']], source_store, normalize)
            row['population_identity'] = spec['case_list_identity']
            materialized.append(compact_endpoint(restore(row)))
        need(len(source_store.seen) == NEW_COUNT, 'all474_local_summaries_verified_no_extra_GET')
        merged = MergedCells(b, spec, docs['historical_plan']['population_cases'], old_endpoints,
                            qualification, docs['t30_endpoints'], baseline, materialized)
        table_refs = [{k: row[k] for k in ('uri', 'bytes', 'sha256')} for row in task['prior_tables']]
        one, two, trace_count = write_increment_tables(merged, singles, pairs, groups, output, f,
                                                      table_refs, deadline_epoch)
        expected_delta = NEW_COUNT*sum(MODEL in (c['control'], c['treatment']) for c in merged.contrasts)
        need(trace_count == len(merged.direct) == expected_delta == 3*NEW_COUNT, 'only_three_affected_contrast_cells_per_added_endpoint')
        need(len(transport.seen) == len(transport.refs) and transport.bytes == transport.expected,
             'exact_cache_tables_and474_OUTCOMEs_once')
        output.json('QUALIFIED-COVERAGE-CHANGE-v1.json', coverage_change(merged, singles, pairs, groups))
        result = dict(schema='paper3-added474-incremental-comparison-result-v1', status='COMPLETE_EXACT_INCREMENT',
            previous_T30=OLD_T30_COUNT, new_T30=NEW_COUNT, T30_total=OLD_T30_COUNT+NEW_COUNT, unique_endpoints=len(merged.buckets),
            population_per_model=POPULATION_COUNT, single_endpoints=one, paired_contrasts=two,
            delta_trace_rows=trace_count, immutable_parent_trace=task['prior_trace'],
            trace_composition='Replace exactly delta contrast/ordinal keys in the3152 parent trace; all other keys retained.',
            direct_new_gate_cells=len(merged.direct), old_gate_labels_reused_only_for_unchanged_endpoints=True,
            old3152_OUTCOME_GETs=0, baseline_original_array_GETs=0,
            existing_baseline_normalized_cache_GETs=2, new_OUTCOME_GETs=NEW_COUNT,
            raw_input_bytes=transport.bytes, input_objects=len(transport.seen),
            GPU=0, model_calls=0, new_episodes=0, frozen_scientific_source_files_changed=False,
            training_seeds_pooled=False, CI=None, bounds_are_completion_bounds_not_CI=True,
            snapshot_comparison_complete=True, full_population_evaluation_complete=False,
            cache_status='PENDING_SEPARATE_NEW474_DELTA_ONLY',
            prior_optional_cache_failure_does_not_invalidate3152_tables=True)
        output.json('RESULT-v1.json', result)
        output.json('INPUT-RECEIPT-v1.json', dict(schema='paper3-added474-input-receipt-v1',
            objects=reads['objects'], verified_objects=len(transport.seen), verified_bytes=transport.bytes,
            controls=task, dependencies=manifest, old3152_OUTCOME_GETs=0,
            packaged_summary_reads=NEW_COUNT, new_summary_cloud_GETs_in_this_CPU_job=0))
        files = sorted(output.directory.iterdir())
        output.json('SCIENTIFIC-MANIFEST-v1.json', dict(schema='paper3-added474-scientific-manifest-v1',
            files=[dict(file=p.name, **file_identity(p)) for p in files],
            immutable_parent_trace=task['prior_trace'], delta_trace_rows=3*NEW_COUNT,
            full_population_evaluation_complete=False, old_artifacts_unchanged=True))
        files.append(output.directory/'SCIENTIFIC-MANIFEST-v1.json');snapshot_done = True
        durable = publish_verified(publish_phase, 'science', files, task['output_prefix'])
        cache_status='NOT_ATTEMPTED_STATISTICS_NOT_CLOUD_DURABLE'
        cache_error=None;cache_publication=None
        if durable is not None:
            try:
                cache=output.directory/'NORMALIZED-NEW474-v1.jsonl.gz'
                with output.gzip(cache.name,max_file_bytes=24*1024**2) as stream:
                    for row in sorted(materialized,key=lambda x:x['ordinal']):
                        need(time.time()<deadline_epoch,'absolute_science_deadline')
                        stream.write(canonical(row)+b'\n')
                output.json('CACHE-MANIFEST-v1.json',dict(
                    schema='paper3-new474-pair-gate-delta-cache-v1',
                    cache=dict(file=cache.name,**file_identity(cache)),
                    rows=NEW_COUNT,baseline_rows=0,old3152_rows=0,model_id=MODEL,scientific_run_id=RUN,
                    ordinals=[min(NEW),max(NEW)],case_list_identity=spec['case_list_identity'],
                    model_source=spec['models'][MODEL],input_spec=file_identity(inputs/'INPUT-SPEC-v1.json'),
                    worker_source=file_identity(__file__),typed_noise_values_preserved=True,
                    q0_and_runtime_signatures_preserved=True,field_attribution_supported=False,
                    missing_input_preserved=True,cache_max_compressed_bytes=24*1024**2,
                    new_model_calls=0,new_episodes=0,
                    reuse_rule='Verify complete cache/source/population/model/run/attempt/ordinal identities, then apply the original pair gate to each future pair.'))
                cache_publication=publish_verified(publish_phase,'cache',
                    [cache,output.directory/'CACHE-MANIFEST-v1.json'],task['output_prefix'])
                cache_status='COMPLETE_VERIFIED_NEW474_DELTA'
            except Exception as error:
                cache_status='NEW474_DELTA_INCOMPLETE_STATISTICS_PRESERVED'
                cache_error=dict(type=type(error).__name__,reason=str(error)[:200])
        delivery = dict(schema='paper3-added474-delivery-v1',
            snapshot_comparison_complete=True, snapshot_cloud_durable=durable is not None,
            snapshot_publication=durable, full_population_evaluation_complete=False,
            cache_status=cache_status,cache_error=cache_error,cache_publication=cache_publication,
            full_baseline_or_historical_cache_reemitted=False,
            previous_T30=OLD_T30_COUNT, new_T30=NEW_COUNT, T30_total=OLD_T30_COUNT+NEW_COUNT, old3152_OUTCOME_GETs=0,
            input_bytes=transport.bytes, input_objects=len(transport.seen), delta_trace_rows=3*NEW_COUNT,
            no_input_retry=True)
        output.json('DELIVERY-v1.json', delivery, reserve=False)
        publish_verified(publish_phase, 'delivery', [output.directory/'DELIVERY-v1.json'], task['output_prefix'])
        return delivery
    except Exception as error:
        output.json('FAILURE-v1.json', dict(schema='paper3-added474-increment-failure-v1',
            snapshot_comparison_complete=snapshot_done, snapshot_cloud_durable=durable is not None,
            full_population_evaluation_complete=False, error_type=type(error).__name__, reason=str(error)[:400],
            input_objects_attempted=len(transport.seen), input_bytes_charged=transport.bytes,
            completed_snapshot_files_preserved=snapshot_done), reserve=False)
        raise

