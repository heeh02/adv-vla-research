"""Checks proposed by the senior review of 2026-11-11. Existing records only.

(a) Which early quantity separates lost (S->F) from kept (S->S) successes: motion ratio, raw end-effector distance,
    first-query command-change norm, and the consistent and scatter parts of the command change across the two noise
    draws of a setting. Pooled AUROC and AUROC stratified by family (pairs compared only within their family),
    whole-task bootstrap (2,000 draws, seed 20261111), plus paired differences between predictors.
(b) Restoration test: success of each edited run minus the unedited perturbed run, task-equal, with a pointwise
    whole-task bootstrap interval (post hoc; the whole-L9 run was outside the six prespecified comparisons).
Reads raw/Five_arm_cases_flat.csv and raw/restoration/restoration_case_outcomes.csv.
"""
import collections, csv, json, math, random
import fig5_data as f

SEED, DRAWS = 20261111, 2000


def auroc(pos, neg):
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))


def strat_auroc(rows, key):
    """Compare lost and kept pairs only within the same family; weight families by their number of comparisons."""
    num = den = 0.0
    by = collections.defaultdict(lambda: ([], []))
    for r in rows: by[r['fam']][0 if r['cell'] == 'SF' else 1].append(r[key])
    for pos, neg in by.values():
        if pos and neg:
            n = len(pos) * len(neg); num += auroc(pos, neg) * n; den += n
    return num / den if den else float('nan')


def pairs(cases):
    out = []
    for c in cases:
        q = json.loads(c['q0'])['translation'] if isinstance(c['q0'], str) else c['q0']['translation']
        if q.get('status') != 'qualified': continue
        ref5, ref60 = f._dist(c, 'R0_R1', '5'), f._dist(c, 'R0_R1', '60')
        for n in ('0', '1'):
            cell = c['noise_pair_cells'][n]
            d5, d60 = f._dist(c, 'TR' + n, '5'), f._dist(c, 'TR' + n, '60')
            if cell not in ('SS', 'SF') or None in (d5, d60, ref5, ref60) or ref5 <= 0 or ref60 <= 0: continue
            out.append(dict(cell=cell, task=c['task_id'], fam=c['family'], ratio5=d5 / ref5, raw5=d5, ratio60=d60 / ref60, raw60=d60,
                            cmd=q['norm' + n], cons=math.sqrt(max(q['M'], 0.0)), scat=math.sqrt(max(q['V'], 0.0)),
                            agree=q['U_over_D'] if q.get('ratio_status') == 'defined' else 0.0))
    return out


KEYS = ('ratio5', 'raw5', 'ratio60', 'raw60', 'cmd', 'cons', 'scat', 'agree')


def report(rows, label, stratified):
    by = collections.defaultdict(list)
    for r in rows: by[r['task']].append(r)
    order, rng = sorted(by), random.Random(SEED)
    fn = (lambda rs, k: strat_auroc(rs, k)) if stratified else (lambda rs, k: auroc([r[k] for r in rs if r['cell'] == 'SF'], [r[k] for r in rs if r['cell'] == 'SS']))
    point = {k: fn(rows, k) for k in KEYS}
    draws = {k: [] for k in KEYS}; diffs = {('cons', 'cmd'): [], ('raw5', 'ratio5'): [], ('cmd', 'ratio5'): [], ('cons', 'scat'): []}
    for _ in range(DRAWS):
        d = [q for t in (rng.choice(order) for _ in order) for q in by[t]]
        if not any(q['cell'] == 'SF' for q in d): continue
        v = {k: fn(d, k) for k in KEYS}
        if any(math.isnan(x) for x in v.values()): continue
        for k in KEYS: draws[k].append(v[k])
        for a, b in diffs: diffs[(a, b)].append(v[a] - v[b])
    ci = lambda xs: (sorted(xs)[int(.025 * len(xs))], sorted(xs)[int(.975 * len(xs)) - 1])
    npos = sum(r['cell'] == 'SF' for r in rows)
    print(f'  {label} ({"within family" if stratified else "pooled"}; lost {npos}, kept {len(rows) - npos})')
    for k in KEYS:
        lo, hi = ci(draws[k]); print(f'      {k:8s} {point[k]:.3f} [{lo:.3f}, {hi:.3f}]')
    for (a, b), xs in diffs.items():
        lo, hi = ci(xs); print(f'      {a} - {b}: {point[a] - point[b]:+.3f} [{lo:+.3f}, {hi:+.3f}]')


def restoration():
    rows = list(csv.DictReader(open('raw/restoration/restoration_case_outcomes.csv')))
    for fam in sorted({r['family'] for r in rows}):
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            if r['family'] == fam: by[r['task_id']][r['arm']].append(int(r['success']))
        order, rng = sorted(by), random.Random(SEED)
        rate = lambda tasks, arm: 100 * sum(sum(by[t][arm]) / len(by[t][arm]) for t in tasks) / len(tasks)
        for arm in ('R_rand', 'R_clean', 'R_resp', 'R_full'):
            d = sorted(rate(ts, arm) - rate(ts, 'N_T') for ts in ([rng.choice(order) for _ in order] for _ in range(DRAWS)))
            print(f'  {fam:18s} {arm:8s} minus unedited {rate(order, arm) - rate(order, "N_T"):+6.2f} [{d[49]:+.2f}, {d[1949]:+.2f}]  ({len(order)} tasks)')


if __name__ == '__main__':
    cases = f.load_cases()
    for c in cases: c['q0'] = json.loads(c['q0']) if isinstance(c['q0'], str) else c['q0']
    print('(a) AUROC, lost vs kept successes')
    for pol in ('pi05', 'pi0'):
        rows = pairs([c for c in cases if c['model'] == f.POLICY[pol]])
        report(rows, pol, False); report(rows, pol, True)
        if pol == 'pi0': report([r for r in rows if r['fam'] == f.FAMILY['S']], 'pi0 Sensor Noise', False)
    print('(b) restoration test: success minus the unedited perturbed run (pointwise 95%, post hoc)')
    restoration()
