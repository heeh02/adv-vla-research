"""NSA independent recompute, cloud CPU worker (GPU=0, stdlib only).

Runs on the Baidu CPU instance. Streams (never persists beyond /tmp) the frozen plan,
the SEALED-LEDGER and the 287 per-root FORMAL-SIGNAL JSONs by exact uri/bytes/sha256,
recomputes exactly the quantities of recompute_outcomes.py (success rates, SS/SF/FS/FF
pairs, known harm/recovery, nine D/Q/R x condition contrasts with identification bounds)
and recompute_signals.py (L18 one_minus_squared_ratio, first-five returned translation/
rotation/gripper RMS both pair directions, actual L9 edit L2), with the SAME three-level
task-equal weighting. The SEALED-LEDGER 'results' rows replace the local per-root ledger
fragments: each row maps to (root ordinal via the frozen plan, visual_condition,
configuration, success) and is kept iff execution_status=='complete' and success is bool
(the accepted/complete attempt). Then it loads the FINAL-PRESENTATION-VIEW and writes a
comparison (signals to floating precision; SR bounds; contrasts; harm/recovery with the
definitional difference stated). Outputs + a RECEIPT with per-file bytes/sha are PUT to the
run output prefix; on any error a FAILED marker with the traceback is PUT instead.

Public config via RC_CFG_JSON env (no secrets). Scoped BOS STS via ONE_VLA_BOS_STS_* env.
No arrays are read; no CI; only available-support means with supports, and planned-population
identification bounds. Credentials stay in env/memory; never printed or written.
"""
import base64, datetime, hashlib, http.client, json, os, random, time, traceback
from collections import defaultdict
from urllib.parse import quote, urlsplit

CHUNK = 1024 * 1024
CONDS = ('clean', 'camera_plus15', 'headlight_off')
CONFIGS = ('Native', 'Response-small', 'Response-large', 'Clean-matched',
           'Random-1-matched', 'Random-2-matched', 'Random-3-matched')
# presentation_adapter.PAIRS, hardcoded so the worker stays self-contained stdlib.
PAIRS = ('same_input_edit_vs_Native', 'same_configuration_visual_vs_clean')
# Exact population descriptor of the paper sheet (ablation_values_view.py __main__).
ABLATION_POPULATION = {
    'closed_all21': 281, 'closed_partial': 6, 'missing_in_partial': 90,
    'known_cells': 5937, 'interrupted': 1, 'unrun': 0,
    'source': 'sealed ledger via FINAL-PRESENTATION-VIEW identification-bound widths (111 unknown cells '
              '= 90 in six partial roots + 21 in root 7); the local fragment walk counted 115 unknown '
              '(four camera arms of one partial root are known only in the sealed ledger)'}
RND_COEF = {'Random-1-matched': -1 / 3, 'Random-2-matched': -1 / 3, 'Random-3-matched': -1 / 3}
CONTRAST_COEFS = {
    'D': {'Response-large': 1.0, 'Native': -1.0},
    'Q': {'Response-large': 1.0, 'Clean-matched': -1.0},
    'R': {'Response-large': 1.0, **RND_COEF}}


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def req(name):
    v = os.environ.get(name)
    need(v and not any(c in v for c in "\r\n\0"), "missing_env:" + name)
    return v


def sha_hex(b):
    return hashlib.sha256(b).hexdigest()


def _pct(v, safe=""):
    return quote(v, safe=safe)


def _object_uri(key):
    return "/" + "/".join(_pct(p, safe="-_.~") for p in key.split("/"))


def _auth_header(method, path, headers):
    import hmac
    ak = req("ONE_VLA_BOS_STS_ACCESS_KEY_ID"); sk = req("ONE_VLA_BOS_STS_SECRET_ACCESS_KEY")
    ts = headers["x-bce-date"]
    ch = {k.lower(): " ".join(v.strip().split()) for k, v in headers.items()}
    keys = sorted(ch)
    ctext = "\n".join(f"{_pct(k, safe='-_.~')}:{_pct(ch[k], safe='-_.~')}" for k in keys)
    signed = ";".join(keys)
    canonical = "\n".join([method, path, "", ctext])
    prefix = f"bce-auth-v1/{ak}/{ts}/1800"
    signing_key = hmac.new(sk.encode(), prefix.encode(), hashlib.sha256).hexdigest()
    sig = hmac.new(signing_key.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    return f"{prefix}/{signed}/{sig}"


class BOS:
    def __init__(self, bucket):
        self.bucket = bucket
        ep = os.environ.get("ONE_VLA_BOS_ENDPOINT", "bj.bcebos.com").replace("https://", "").strip("/")
        self.host = ep if ep.startswith(bucket + ".") else bucket + "." + ep

    def _headers(self, method, key, extras=None):
        h = {"host": self.host,
             "x-bce-date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "x-bce-security-token": req("ONE_VLA_BOS_STS_SESSION_TOKEN"), **(extras or {})}
        h["authorization"] = _auth_header(method, _object_uri(key), h)
        return h

    def get_bytes(self, key, cap):
        conn = http.client.HTTPSConnection(self.host, 443, timeout=180)
        try:
            conn.request("GET", _object_uri(key), headers=self._headers("GET", key))
            r = conn.getresponse()
            need(r.status == 200, f"GET_{r.status}:{key}")
            d = hashlib.sha256(); n = 0; out = bytearray()
            while True:
                c = r.read(CHUNK)
                if not c:
                    break
                out.extend(c); d.update(c); n += len(c)
                need(n <= cap, f"over_cap:{key}:{n}")
            return bytes(out), n, d.hexdigest()
        finally:
            conn.close()

    def put_bytes(self, key, raw):
        conn = http.client.HTTPSConnection(self.host, 443, timeout=180)
        try:
            import hashlib as _h
            extras = {"content-length": str(len(raw)),
                      "content-md5": base64.b64encode(_h.md5(raw).digest()).decode(),
                      "content-type": "application/json"}
            conn.request("PUT", _object_uri(key), body=raw, headers=self._headers("PUT", key, extras))
            r = conn.getresponse(); r.read()
            need(r.status in (200, 204), f"PUT_{r.status}:{key}")
        finally:
            conn.close()


def get_exact(bos, ref, cap):
    """GET one object and verify whole-object bytes and sha256 against the ref."""
    key = urlsplit(ref["uri"]).path.lstrip("/")
    raw, n, s = bos.get_bytes(key, cap)
    need(n == ref["bytes"], f"bytes_differ:{key}:{n}vs{ref['bytes']}")
    need(s == ref["sha256"], f"sha_differs:{key}")
    return raw


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def ident(raw):
    return {"bytes": len(raw), "sha256": sha_hex(raw)}


# ------- porting recompute_outcomes.py (three-level task-equal weighting + identification bounds) -------
def task_of(root_id):
    return root_id.rsplit('-i', 1)[0]


def init_of(root_id):
    return root_id.split(':')[0]


def outcomes_recompute(plan, ledger):
    roots = {r['ordinal']: r for r in plan['roots']}
    rid2ord = {r['root_id']: r['ordinal'] for r in plan['roots']}
    need(len(rid2ord) == len(roots) == 288, 'plan_288_unique_roots')
    results = ledger['results']
    # Build Y by iterating in ordinal order, then ledger order within a root, so the
    # last-write-wins behaviour of recompute_outcomes.py over frag['results'] is preserved.
    by_ord_rows = defaultdict(list)
    for row in results:
        rid = row['root_id']
        need(rid in rid2ord, 'ledger_root_id_not_in_plan:' + rid)
        by_ord_rows[rid2ord[rid]].append(row)
    Y = {}
    rows_known = 0
    closed = set()
    conflicts = []  # grid cells where two complete rows disagree on success (replicate repeats)
    for o in sorted(roots):
        y = {(c, k): None for c in CONDS for k in CONFIGS}
        for r in by_ord_rows.get(o, []):
            if r['execution_status'] == 'complete' and isinstance(r['success'], bool):
                cell = (r['visual_condition'], r['configuration'])
                if cell in y and y[cell] is not None and y[cell] != r['success']:
                    conflicts.append({'ordinal': o, 'cell': list(cell)})
                y[cell] = r['success']
                rows_known += 1
                closed.add(o)
        Y[o] = y
    tasks = sorted({task_of(r['root_id']) for r in roots.values()})
    by_task = defaultdict(list)
    for r in roots.values():
        by_task[task_of(r['root_id'])].append(r['ordinal'])

    def task_equal(f):
        tv = []
        for t in tasks:
            inits = defaultdict(list)
            for o in by_task[t]:
                v = f(o)
                if v is not None:
                    inits[init_of(roots[o]['root_id'])].append(v)
            iv = [sum(v) / len(v) for v in inits.values()]
            if iv:
                tv.append(sum(iv) / len(iv))
        return (sum(tv) / len(tv) if tv else None), len(tv)

    def bounds(f_coef):
        lo_t = []; hi_t = []
        for t in tasks:
            inits = defaultdict(list)
            for o in by_task[t]:
                inits[init_of(roots[o]['root_id'])].append(f_coef(o))
            lo_i = [sum(v['lo'] for v in vs) / len(vs) for vs in inits.values()]
            hi_i = [sum(v['hi'] for v in vs) / len(vs) for vs in inits.values()]
            lo_t.append(sum(lo_i) / len(lo_i)); hi_t.append(sum(hi_i) / len(hi_i))
        return sum(lo_t) / len(tasks), sum(hi_t) / len(tasks)

    out = {'schema': 'nsa-independent-outcome-recompute-cloud-v1',
           'source': 'SEALED-LEDGER', 'closed_roots_used': len(closed),
           'known_main_rows': rows_known,
           'known_grid_cells': sum(1 for o in roots for c in CONDS for k in CONFIGS if Y[o][(c, k)] is not None),
           'replicate_conflicts': conflicts, 'tasks': tasks,
           'SR': {}, 'pairs_native_to_edited': {}, 'pairs_clean_to_visual': {}, 'contrasts': {}}
    for c in CONDS:
        for k in CONFIGS:
            m, nt = task_equal(lambda o, c=c, k=k: (None if Y[o][(c, k)] is None else float(Y[o][(c, k)])))
            lo, hi = bounds(lambda o, c=c, k=k: {'lo': float(Y[o][(c, k)]) if Y[o][(c, k)] is not None else 0.0,
                                                 'hi': float(Y[o][(c, k)]) if Y[o][(c, k)] is not None else 1.0})
            out['SR'][f'{c}|{k}'] = {'available_mean': m, 'tasks_with_support': nt, 'planned_bounds': [lo, hi]}

    def pair_stats(get_ref, get_tgt):
        cells = {}
        for lab, cond in (('SS', lambda r, t: r and t), ('SF', lambda r, t: r and not t),
                          ('FS', lambda r, t: (not r) and t), ('FF', lambda r, t: (not r) and (not t))):
            m, nt = task_equal(lambda o, cond=cond: (None if get_ref(o) is None or get_tgt(o) is None
                                                     else float(cond(get_ref(o), get_tgt(o)))))
            cells[lab] = m
        unknown, _ = task_equal(lambda o: float(get_ref(o) is None or get_tgt(o) is None))
        return {'cells_available_mean': cells, 'harm_available_mean': cells['SF'],
                'recovery_available_mean': cells['FS'], 'unknown_pair_mass_planned': unknown}

    for c in CONDS:
        for k in CONFIGS[1:]:
            out['pairs_native_to_edited'][f'{c}|{k}'] = pair_stats(
                lambda o, c=c: Y[o][(c, 'Native')], lambda o, c=c, k=k: Y[o][(c, k)])
    for k in CONFIGS:
        for c in CONDS[1:]:
            out['pairs_clean_to_visual'][f'{k}|{c}'] = pair_stats(
                lambda o, k=k: Y[o][('clean', k)], lambda o, c=c, k=k: Y[o][(c, k)])

    def contrast(c, coefs):
        def val(o):
            v = 0.0; lo = 0.0; hi = 0.0
            for k, w in coefs.items():
                yv = Y[o][(c, k)]
                if yv is None:
                    lo += w * (0.0 if w > 0 else 1.0); hi += w * (1.0 if w > 0 else 0.0)
                else:
                    lo += w * float(yv); hi += w * float(yv); v += w * float(yv)
            return {'lo': lo, 'hi': hi, 'known': all(Y[o][(c, k)] is not None for k in coefs), 'value': v}
        m, nt = task_equal(lambda o: (val(o)['value'] if val(o)['known'] else None))
        lo, hi = bounds(val)
        return {'available_mean': m, 'tasks_with_support': nt, 'planned_bounds': [lo, hi]}

    rnd = {'Random-1-matched': -1 / 3, 'Random-2-matched': -1 / 3, 'Random-3-matched': -1 / 3}
    for c in CONDS:
        out['contrasts'][f'D|{c}'] = contrast(c, {'Response-large': 1.0, 'Native': -1.0})
        out['contrasts'][f'Q|{c}'] = contrast(c, {'Response-large': 1.0, 'Clean-matched': -1.0})
        out['contrasts'][f'R|{c}'] = contrast(c, {'Response-large': 1.0, **rnd})
    return out, Y


# ------- porting recompute_signals.py (three-level available-support means from per-root signal JSONs) -------
def hmean(vals_by_root, roots):
    tasks = defaultdict(lambda: defaultdict(list))
    for o, v in vals_by_root.items():
        if v is None:
            continue
        rid = roots[o]['root_id']
        tasks[task_of(rid)][init_of(rid)].append(v)
    tv = []
    for t, inits in tasks.items():
        iv = [sum(v) / len(v) for v in inits.values()]
        tv.append(sum(iv) / len(iv))
    return (sum(tv) / len(tv) if tv else None), len(tv), sum(1 for v in vals_by_root.values() if v is not None)


def signals_recompute(bos, plan, signal_refs, phase):
    roots = {r['ordinal']: r for r in plan['roots']}
    per = {}
    def put(cond, cfg, metric, o, v):
        per.setdefault((cond, cfg, metric), {})[o] = v
    read_bytes = 0; rows_seen = 0; roots_read = 0
    for sref in signal_refs:
        o = sref['ordinal']
        phase['stage'] = f'signal_root_{o}'
        raw = get_exact(bos, sref, cap=8 * 1024 * 1024)
        read_bytes += len(raw)
        doc = json.loads(raw); del raw
        need(doc['ordinal'] == o and doc['root_id'] == roots[o]['root_id'], f'signal_identity:{o}')
        roots_read += 1
        for row in doc['rows']:
            if row.get('status') != 'SOURCE_VERIFIED_Q0_SIGNALS':
                continue
            rows_seen += 1
            cfg = row['configuration']; cond = row['visual_variant_id']
            red = (row.get('visual_response_reduction_vs_Native') or {}).get('L18_H_after_j0') or {}
            put(cond, cfg, 'L18_reduction', o, red.get('one_minus_squared_ratio'))
            put(cond, cfg, 'edit_l2', o, (row.get('actual_edit_budget') or {}).get('l2'))
            for label, key in (('vis_vs_clean', 'same_configuration_visual_vs_clean'),
                               ('edit_vs_native', 'same_input_edit_vs_Native')):
                pr = ((row.get(key) or {}).get('actual_returned_prefix') or {})
                groups = pr.get('groups') or {}
                for g, val in groups.items():
                    metric = (f'{label}:{g}:RMS_first5' if pr.get('fixed_first5_complete')
                              else f'{label}:{g}:RMS_commonK')
                    put(cond, cfg, metric, o, val.get('RMS'))
        del doc
    out = {'schema': 'nsa-independent-signal-recompute-cloud-v1', 'roots_read': roots_read,
           'bytes_streamed_not_persisted': read_bytes, 'rows_seen': rows_seen, 'metrics': {}}
    for (cond, cfg, metric), vals in sorted(per.items()):
        m, nt, nr = hmean(vals, roots)
        out['metrics'][f'{cond}|{cfg}|{metric}'] = {'mean': m, 'tasks': nt, 'roots': nr}
    return out


# ------- comparison against the FINAL-PRESENTATION-VIEW -------
VIEW_SIGNAL = {
    'L18_reduction': 'L18_response_reduction',
    'edit_l2': 'actual_L9_edit_l2',
    'vis_vs_clean:translation:RMS_first5': 'same_configuration_visual_vs_clean.returned_first5.translation_RMS',
    'vis_vs_clean:rotation:RMS_first5': 'same_configuration_visual_vs_clean.returned_first5.rotation_RMS',
    'vis_vs_clean:gripper:RMS_first5': 'same_configuration_visual_vs_clean.returned_first5.gripper_RMS',
    'edit_vs_native:translation:RMS_first5': 'same_input_edit_vs_Native.returned_first5.translation_RMS',
    'edit_vs_native:rotation:RMS_first5': 'same_input_edit_vs_Native.returned_first5.rotation_RMS',
    'edit_vs_native:gripper:RMS_first5': 'same_input_edit_vs_Native.returned_first5.gripper_RMS',
}
TOL = 1e-9


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL + 1e-6 * max(abs(a), abs(b))


def build_comparison(outc, sig, view):
    view_rows = {(r['condition'], r['configuration']): r for r in view['rows'] if r.get('row_type') == 'configuration'}
    view_cons = {c['name']: c for c in view['primary_contrasts']}
    cmp = {'schema': 'nsa-independent-recompute-cloud-comparison-v1',
           'view_outcome_main_status_counts': view['outcome_main_status_counts'],
           'view_outcome_status': view['outcome_status'],
           'definitional_notes': {
               'SR_bounds': 'identification bounds over the planned 288-root population with three-level '
                            'task-equal weighting (noise within initial state, initial state within task, 16 '
                            'tasks equal); unknown planned outcomes set to the worst/best legal value by sign.',
               'harm_recovery': 'recompute harm/recovery are available-support CONDITIONAL means (SF/FS indicator '
                                'under three-level weighting; unknown pair cells dropped and remaining noise repeats '
                                're-weighted within an initial state). view known_harm/known_recovery are the KNOWN '
                                'planned mass under fixed planned weights (unknown pair mass retained as its own term, '
                                'task_equal_rates SF/FS). They coincide only when no pair cell in a root is unknown.',
               'signals': 'recompute three-level available-support means vs view available_mean; identical source '
                          '(the same per-root FORMAL-SIGNAL JSONs).'},
           'signals': {}, 'SR_bounds': {}, 'contrasts': {}, 'harm_recovery': {},
           'summary': {}}
    # signals (cloud mean vs view available_mean)
    sig_match = sig_total = 0
    for c in CONDS:
        for k in CONFIGS:
            vr = view_rows[(c, k)]['signals']
            for rec_key, view_key in VIEW_SIGNAL.items():
                cm = sig['metrics'].get(f'{c}|{k}|{rec_key}', {}).get('mean')
                vm = vr.get(view_key, {}).get('available_mean')
                if cm is None and vm is None:
                    continue
                sig_total += 1
                ok = close(cm, vm)
                sig_match += int(ok)
                cmp['signals'][f'{c}|{k}|{rec_key}'] = {'cloud': cm, 'view': vm, 'match': ok}
    # SR bounds (cloud planned_bounds vs view SR lower/upper)
    sr_match = sr_total = 0
    for c in CONDS:
        for k in CONFIGS:
            lo, hi = outc['SR'][f'{c}|{k}']['planned_bounds']
            sr = view_rows[(c, k)]['outcomes']['SR']
            ok = close(lo, sr['lower']) and close(hi, sr['upper'])
            sr_total += 1; sr_match += int(ok)
            cmp['SR_bounds'][f'{c}|{k}'] = {'cloud': [lo, hi], 'view': [sr['lower'], sr['upper']],
                                            'cloud_available_mean': outc['SR'][f'{c}|{k}']['available_mean'],
                                            'match': ok}
    # contrasts (cloud planned_bounds vs view identification lower/upper)
    con_match = con_total = 0
    for c in CONDS:
        for label in ('D', 'Q', 'R'):
            lo, hi = outc['contrasts'][f'{label}|{c}']['planned_bounds']
            idf = view_cons[f'{label}_{c}']['identification']
            ok = close(lo, idf['lower']) and close(hi, idf['upper'])
            con_total += 1; con_match += int(ok)
            cmp['contrasts'][f'{label}|{c}'] = {'cloud': [lo, hi], 'view': [idf['lower'], idf['upper']],
                                                'cloud_available_mean': outc['contrasts'][f'{label}|{c}']['available_mean'],
                                                'match': ok}
    # harm / recovery (definitional difference; report both, no equality requirement)
    for c in CONDS:
        for k in CONFIGS[1:]:
            p = outc['pairs_native_to_edited'][f'{c}|{k}']
            vp = view_rows[(c, k)]['outcomes']['Native_to_edited']
            cmp['harm_recovery'][f'native_to_edited|{c}|{k}'] = {
                'cloud_harm_available_mean': p['harm_available_mean'],
                'cloud_recovery_available_mean': p['recovery_available_mean'],
                'cloud_unknown_pair_mass_planned': p['unknown_pair_mass_planned'],
                'view_known_harm': (vp or {}).get('known_harm'),
                'view_known_recovery': (vp or {}).get('known_recovery'),
                'view_unknown_pair_mass': ((vp or {}).get('task_equal_rates') or {}).get('unknown'),
                'harm_abs_diff': (None if (vp or {}).get('known_harm') is None
                                  else abs(p['harm_available_mean'] - vp['known_harm'])),
                'recovery_abs_diff': (None if (vp or {}).get('known_recovery') is None
                                      else abs(p['recovery_available_mean'] - vp['known_recovery']))}
    for k in CONFIGS:
        for c in CONDS[1:]:
            p = outc['pairs_clean_to_visual'][f'{k}|{c}']
            vp = view_rows[(c, k)]['outcomes']['clean_to_perturbed']
            cmp['harm_recovery'][f'clean_to_perturbed|{k}|{c}'] = {
                'cloud_harm_available_mean': p['harm_available_mean'],
                'cloud_recovery_available_mean': p['recovery_available_mean'],
                'cloud_unknown_pair_mass_planned': p['unknown_pair_mass_planned'],
                'view_known_harm': (vp or {}).get('known_harm'),
                'view_known_recovery': (vp or {}).get('known_recovery'),
                'view_unknown_pair_mass': ((vp or {}).get('task_equal_rates') or {}).get('unknown'),
                'harm_abs_diff': (None if (vp or {}).get('known_harm') is None
                                  else abs(p['harm_available_mean'] - vp['known_harm'])),
                'recovery_abs_diff': (None if (vp or {}).get('known_recovery') is None
                                      else abs(p['recovery_available_mean'] - vp['known_recovery']))}
    max_hr = max((d['harm_abs_diff'] or 0.0) for d in cmp['harm_recovery'].values())
    max_hr = max(max_hr, max((d['recovery_abs_diff'] or 0.0) for d in cmp['harm_recovery'].values()))
    cmp['summary'] = {'signals_match': sig_match, 'signals_total': sig_total,
                      'SR_bounds_match': sr_match, 'SR_bounds_total': sr_total,
                      'contrasts_match': con_match, 'contrasts_total': con_total,
                      'signals_all_match': sig_match == sig_total,
                      'SR_bounds_all_match': sr_match == sr_total,
                      'contrasts_all_match': con_match == con_total,
                      'harm_recovery_max_abs_diff_by_definition': max_hr,
                      'cloud_known_grid_cells': outc['known_grid_cells'],
                      'view_complete_main': view['outcome_main_status_counts'].get('complete')}
    return cmp


# ------- (A) post-hoc, non-preregistered whole-task bootstrap sensitivity for the nine contrasts -------
def _pct_type7(sorted_vals, q):
    if not sorted_vals:
        return None
    n = len(sorted_vals); pos = (n - 1) * q; lo = int(pos); frac = pos - lo
    if lo + 1 < n:
        return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac
    return sorted_vals[lo]


def posthoc_bootstrap(plan, Y, view, draws=2000, seed=20261018):
    """Available-case point on the known outcomes + whole-task bootstrap 95% pointwise intervals.

    The project's frozen component (analysis/formal-readout-v1/readout.py: PRECISION.bootstrap_weights,
    draws=20000, seed=20261014) is a JOINT simultaneous-band weight generator with different parameters
    and objective, so this is a plain stdlib whole-task (cluster) bootstrap honouring the explicit post-hoc
    spec: resample the held-out tasks with replacement, recompute the three-level task-equal mean of the
    available-case contrast inside each draw, percentile 2.5/97.5 (type-7). Not preregistered.
    """
    roots = {r['ordinal']: r for r in plan['roots']}
    tasks = sorted({task_of(r['root_id']) for r in roots.values()})
    by_task = defaultdict(list)
    for r in roots.values():
        by_task[task_of(r['root_id'])].append(r['ordinal'])

    def task_value(coefs, c, t):
        inits = defaultdict(list)
        for o in by_task[t]:
            if all(Y[o][(c, k)] is not None for k in coefs):
                inits[init_of(roots[o]['root_id'])].append(sum(w * float(Y[o][(c, k)]) for k, w in coefs.items()))
        iv = [sum(x) / len(x) for x in inits.values()]
        return (sum(iv) / len(iv)) if iv else None

    names = [f'{label}|{c}' for c in CONDS for label in ('D', 'Q', 'R')]
    pertask = {}
    for nm in names:
        label, c = nm.split('|')
        pertask[nm] = [task_value(CONTRAST_COEFS[label], c, t) for t in tasks]

    def mean_present(vals):
        present = [v for v in vals if v is not None]
        return (sum(present) / len(present)) if present else None

    point = {nm: mean_present(pertask[nm]) for nm in names}
    rng = random.Random(seed); ntask = len(tasks)
    draws_by = {nm: [] for nm in names}
    for _ in range(draws):
        idx = [rng.randrange(ntask) for _ in range(ntask)]  # same task resample across all contrasts
        for nm in names:
            vals = [pertask[nm][i] for i in idx if pertask[nm][i] is not None]
            if vals:
                draws_by[nm].append(sum(vals) / len(vals))
    view_cons = {c['name']: c for c in view['primary_contrasts']}
    out = {'schema': 'nsa-independent-recompute-cloud-posthoc-bootstrap-v1',
           'label': 'post-hoc, not preregistered; identification bounds remain the primary report',
           'method': ('whole-task nonparametric (cluster) bootstrap: resample the 16 held-out tasks with '
                      'replacement; inside each draw recompute the three-level task-equal mean (noise within '
                      'initial state, initial state within task, tasks equal) of the available-case contrast on '
                      'the known outcomes; percentile 2.5/97.5 type-7 linear interpolation. Same task resample '
                      'reused across all nine contrasts per draw. Plain stdlib implementation, NOT the project '
                      'frozen PRECISION.bootstrap_weights (draws=20000, seed=20261014) which is a joint '
                      'simultaneous-band weight generator with a different objective.'),
           'draws': draws, 'seed': seed, 'percentiles': [2.5, 97.5], 'tasks_resampled': ntask,
           'basis': 'available-case point estimates on the known outcomes (view complete main count)',
           'view_complete_main': view['outcome_main_status_counts'].get('complete'),
           'contrasts': {}}
    for nm in names:
        label, c = nm.split('|'); s = sorted(draws_by[nm])
        idf = view_cons[f'{label}_{c}']['identification']
        out['contrasts'][nm] = {
            'available_case_point': point[nm],
            'bootstrap_95_interval': [_pct_type7(s, 0.025), _pct_type7(s, 0.975)],
            'bootstrap_draws_used': len(s),
            'tasks_with_support': sum(1 for v in pertask[nm] if v is not None),
            'frozen_identification_bounds_from_view': [idf['lower'], idf['upper']]}
    return out


# ------- (B) paper ablation value sheet, ported from editorial/ablation_values_view.py -------
def build_ablation_values(view, view_run_id, population):
    rows = [r for r in view['rows'] if r['row_type'] == 'configuration']
    need(len(rows) == 21, f'ablation_sheet_21_rows:{len(rows)}')
    e_key = PAIRS[0] + '.returned_first5.translation_RMS'
    v_key = PAIRS[1] + '.returned_first5.translation_RMS'

    def am(r, m):
        x = r['signals'][m]['available_mean']
        return None if x is None else float(x)

    V = {'source': f'FINAL-PRESENTATION-VIEW-v1.json of {view_run_id}', 'final_run_id': view_run_id,
         'rows': {}, 'contrasts': {}, 'net_change': {}, 'population': population}
    for r in rows:
        c, k = r['condition'], r['configuration']; sr = r['outcomes']['SR']; ne = r['outcomes']['Native_to_edited']
        te, tv = am(r, e_key), am(r, v_key); l18 = am(r, 'L18_response_reduction')
        V['rows'][f'{c}|{k}'] = {
            'SR': None if sr['point'] is None else float(sr['point']) * 100,
            'SR_point_status': 'suppressed_by_frozen_rule' if sr['point'] is None else 'point',
            'SR_bounds': [float(sr['lower']) * 100, float(sr['upper']) * 100],
            'harm': None if ne is None else float(ne['known_harm']) * 100,
            'recovery': None if ne is None else float(ne['known_recovery']) * 100,
            'unknown_pair_mass': None if ne is None else float(ne['unknown_pair_mass']) * 100,
            'edit_l2': am(r, 'actual_L9_edit_l2'),
            'L18_reduction': None if l18 is None else l18 * 100,
            'edit_translation_RMS': te, 'visual_translation_RMS': tv,
            'footprint_pct': None if (te is None or not tv) else te / tv * 100}
        if ne is not None and ne.get('outcome_change_bounds'):
            b = ne['outcome_change_bounds']
            V['net_change'][f'{c}|{k}'] = {'pp': float(b['point']) * 100 if b.get('point') is not None else None,
                                           'bounds': [float(b['lower']) * 100, float(b['upper']) * 100]}
    for pc in view['primary_contrasts']:
        idn = pc['identification']
        V['contrasts'][f"{pc['contrast']}|{pc['condition']}"] = {
            'pp': float(idn['point']) * 100 if idn.get('point') is not None else None,
            'bounds': [float(idn['lower']) * 100, float(idn['upper']) * 100]}
    return V


def main():
    cfg = json.loads(req("RC_CFG_JSON"))
    bucket = cfg["bucket"]; out_prefix = cfg["out_prefix"]; run_id = cfg["run_id"]
    bos = BOS(bucket)
    phase = {"stage": "start"}
    t0 = time.time()
    try:
        # 1. input manifest (holds the 287 signal refs)
        phase["stage"] = "get_manifest"
        man_raw = get_exact(bos, cfg["manifest_ref"], cap=4 * 1024 * 1024)
        manifest = json.loads(man_raw)
        need(manifest["run_id"] == run_id, "manifest_run_id")
        signal_refs = manifest["signals"]
        need(len(signal_refs) == 287, f"signal_ref_count:{len(signal_refs)}")
        # 2. frozen plan
        phase["stage"] = "get_plan"
        plan = json.loads(get_exact(bos, cfg["global_ref"], cap=2 * 1024 * 1024))
        # 3. sealed ledger
        phase["stage"] = "get_sealed_ledger"
        ledger = json.loads(get_exact(bos, cfg["ledger_ref"], cap=64 * 1024 * 1024))
        need('results' in ledger, "ledger_has_results")
        # 4. presentation view
        phase["stage"] = "get_view"
        view = json.loads(get_exact(bos, cfg["view_ref"], cap=16 * 1024 * 1024))
        need(view.get("schema") == "nsa-final-presentation-view-v1", "view_schema")
        # 5. outcome recompute from sealed ledger
        phase["stage"] = "outcomes"
        outc, Y = outcomes_recompute(plan, ledger)
        del ledger
        # 6. signal recompute from 287 per-root signal JSONs
        phase["stage"] = "signals"
        sig = signals_recompute(bos, plan, signal_refs, phase)
        sig["seconds"] = round(time.time() - t0, 1)
        # 7. comparison vs view
        phase["stage"] = "compare"
        cmp = build_comparison(outc, sig, view)
        # 7A. post-hoc whole-task bootstrap sensitivity for the nine contrasts
        phase["stage"] = "posthoc_bootstrap"
        posthoc = posthoc_bootstrap(plan, Y, view)
        # 7B. paper ablation value sheet from the view
        phase["stage"] = "ablation_values"
        ablation = build_ablation_values(view, cfg["view_run_id"], ABLATION_POPULATION)
        # 8. publish outputs + receipt
        phase["stage"] = "write"
        files = {}
        for name, value in (("OUTCOMES-cloud-v1.json", outc),
                            ("SIGNALS-cloud-v1.json", sig),
                            ("COMPARISON-cloud-v1.json", cmp),
                            ("POSTHOC-BOOTSTRAP-cloud-v1.json", posthoc),
                            ("ABLATION-VALUES-cloud-v1.json", ablation)):
            raw = encoded(value)
            bos.put_bytes(out_prefix + name, raw)
            files[name] = ident(raw)
        receipt = {"schema": "nsa-independent-recompute-cloud-receipt-v1", "run_id": run_id,
                   "status": "COMPLETE", "GPU": 0, "epoch": time.time(),
                   "seconds": round(time.time() - t0, 1),
                   "inputs": {"global": cfg["global_ref"], "sealed_ledger": cfg["ledger_ref"],
                              "presentation_view": cfg["view_ref"], "input_manifest": cfg["manifest_ref"],
                              "signal_jsons_read": sig["roots_read"],
                              "signal_bytes_streamed_not_persisted": sig["bytes_streamed_not_persisted"]},
                   "outputs": {out_prefix + k: v for k, v in files.items()},
                   "comparison_summary": cmp["summary"]}
        raw = encoded(receipt)
        bos.put_bytes(out_prefix + "RECEIPT-v1.json", raw)
        print(json.dumps({"status": "COMPLETE", "receipt": out_prefix + "RECEIPT-v1.json",
                          "summary": cmp["summary"]}), flush=True)
        return 0
    except Exception as e:
        body = {"schema": "nsa-independent-recompute-cloud-FAILED-v1", "status": "WORKER_FAILED",
                "run_id": run_id, "phase": phase["stage"], "error_type": type(e).__name__,
                "msg": str(e)[:600], "traceback": traceback.format_exc()[:6000],
                "epoch": time.time(), "GPU": 0}
        try:
            bos.put_bytes(out_prefix + "FAILED-v1.json", encoded(body))
            print(json.dumps({"status": "WORKER_FAILED", "phase": phase["stage"],
                              "error_type": type(e).__name__, "marker": out_prefix + "FAILED-v1.json"}), flush=True)
        except Exception as e2:
            print(json.dumps({"status": "WORKER_FAILED_NO_MARKER", "phase": phase["stage"],
                              "error_type": type(e).__name__, "marker_error": type(e2).__name__}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
