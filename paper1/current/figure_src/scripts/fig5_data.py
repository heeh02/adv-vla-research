"""Figure 5 inputs, computed from the raw five-arm case table (raw/Five_arm_cases_flat.csv).

Everything drawn in Figure 5 comes from that one table; the two estimate tables next to it are used only as a
cross-check (every family-level point and interval recomputed here must equal the archived value).
Recipe (raw/Five_arm_protocol.md): primary scope = 28 common tasks, 202 settings per policy. Each setting has two
noise-matched clean-perturbed pairs. Values average pairs within setting, settings within task, tasks equally.
Motion ratio = task-equal mean TR distance / task-equal mean R0-R1 distance on settings where TR0, TR1 and R0-R1 are
all measured at that control. A task's ratio divides its own mean TR distance by that same family-mean reference,
as in the earlier version of the figure, so the family ratio is the task-equal mean of the task ratios. Intervals: 2,000 whole-task draws, random.Random(20261011), tasks sorted by their hash,
non-interpolated quantiles. Unrecorded outcomes stay in the planned denominator and are never counted as flips."""
import csv, json, os, random, collections

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'raw')
FAMILY = {'B': 'Background Textures', 'C': 'Camera Viewpoints', 'L': 'Light Conditions', 'S': 'Sensor Noise'}
POLICY = {'pi05': 'pi05_libero', 'pi0': 'pi0_libero'}
SEED, DRAWS = 20261011, 2000


def _read(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f'missing {path}: download the Five_arm_* tables from the project Drive folder into figure_src/raw/')
    csv.field_size_limit(10 ** 9)
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _dist(case, key, clock):
    m = case['motion'].get(key, {}).get(clock, {}).get('position_distance')
    return m['value'] if m and m['status'] == 'measured' and m['value'] is not None else None


def _quantiles(vals, alpha):
    vals = sorted(vals)
    return vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals)) - 1]


def _task_hash(task_id):
    return task_id.split(':', 1)[1]


def load_cases():
    rows = [r for r in _read('Five_arm_cases_flat.csv') if r['common28'] == 'True']
    for r in rows:
        for k in ('outcomes', 'motion', 'noise_pair_cells'):
            r[k] = json.loads(r[k])
    return rows


def family_panel(cases, clock='5'):
    """All Figure 5 quantities for one policy and family."""
    tasks = collections.defaultdict(lambda: dict(tr=[], rr=[], lost=[], gained=[], pairs=0, unknown=0))
    for c in cases:
        t = tasks[c['task_id']]
        a, b, ref = _dist(c, 'TR0', clock), _dist(c, 'TR1', clock), _dist(c, 'R0_R1', clock)
        if None not in (a, b, ref):
            t['tr'].append((a + b) / 2); t['rr'].append(ref)
        cells = [c['noise_pair_cells'][n] for n in ('0', '1')]   # the two pairs of a setting weigh one half each
        t['lost'].append(sum(x == 'SF' for x in cells) / 2); t['gained'].append(sum(x == 'FS' for x in cells) / 2)
        t['pairs'] += 2; t['unknown'] += sum(x is None for x in cells)
    for t in tasks.values():                                    # settings within task
        t['lost_rate'] = sum(t['lost']) / len(t['lost']); t['gained_rate'] = sum(t['gained']) / len(t['gained'])
    order = sorted(tasks, key=_task_hash)
    n_unknown = sum(t['unknown'] for t in tasks.values())
    lost = sum(tasks[t]['lost_rate'] for t in order) / len(order)
    gained = sum(tasks[t]['gained_rate'] for t in order) / len(order)
    out = dict(lost=100 * lost, gained=100 * gained, net=100 * (gained - lost), unknown_pairs=n_unknown,
               planned_pairs=sum(t['pairs'] for t in tasks.values()), net_lo=None, net_hi=None)
    if n_unknown == 0:                                           # sampling interval only when the change is identified
        rng = random.Random(SEED); draws = []
        for _ in range(DRAWS):
            d = [rng.choice(order) for _ in order]
            draws.append(sum(tasks[t]['gained_rate'] - tasks[t]['lost_rate'] for t in d) / len(d))
        lo, hi = _quantiles(draws, 0.0125); out.update(net_lo=100 * lo, net_hi=100 * hi)
    # motion: jointly eligible settings only
    mot = {t: (sum(tasks[t]['tr']) / len(tasks[t]['tr']), sum(tasks[t]['rr']) / len(tasks[t]['rr'])) for t in order if tasks[t]['tr']}
    mt = sorted(mot, key=_task_hash)
    tr = sum(mot[t][0] for t in mt) / len(mt); rr = sum(mot[t][1] for t in mt) / len(mt)
    rng = random.Random(SEED); draws = []
    for _ in range(DRAWS):
        d = [rng.choice(mt) for _ in mt]
        draws.append(sum(mot[t][0] for t in d) / sum(mot[t][1] for t in d))
    lo, hi = _quantiles(draws, 0.05)
    out.update(ratio=tr / rr, ratio_lo=lo, ratio_hi=hi, tr_mm=1000 * tr, rr_mm=1000 * rr,
               motion_settings=sum(len(tasks[t]['tr']) for t in mt), motion_tasks=len(mt),
               # per task: its own separation over the family-mean reference, so the family ratio is the mean of the task values
               task_points=[(mot[t][0] / rr, 100 * (tasks[t]['gained_rate'] - tasks[t]['lost_rate']), tasks[t]['unknown'] > 0) for t in mt])
    return out


def sensor_pairs(cases, clock='5'):
    """Every pi0 Sensor Noise pair: outcome class and its separation over its own setting's clean-resampling distance."""
    out = []
    for c in cases:
        ref = _dist(c, 'R0_R1', clock)
        for n in ('0', '1'):
            out.append(dict(cell=c['noise_pair_cells'][n], ratio=_dist(c, 'TR' + n, clock) / ref, task=c['task_id'], setting=c['variant_id']))
    return out


def _auroc(pos, neg):
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))


def pair_auroc(pairs):
    """Early-separation ratio as a score for lost (S->F) against retained (S->S) successes, with the task bootstrap."""
    point = _auroc([p['ratio'] for p in pairs if p['cell'] == 'SF'], [p['ratio'] for p in pairs if p['cell'] == 'SS'])
    by = collections.defaultdict(list)
    for p in pairs: by[p['task']].append(p)
    order = sorted(by, key=_task_hash); rng = random.Random(SEED); draws = []
    for _ in range(DRAWS):
        d = [q for t in (rng.choice(order) for _ in order) for q in by[t]]
        pos = [q['ratio'] for q in d if q['cell'] == 'SF']; neg = [q['ratio'] for q in d if q['cell'] == 'SS']
        if pos and neg: draws.append(_auroc(pos, neg))
    return (point,) + _quantiles(draws, 0.05)


def _check(panels):
    """Recomputed family values must equal the archived estimate tables."""
    mo = _read('Five_arm_motion_estimates.csv'); oc = _read('Five_arm_outcome_estimates.csv'); worst = 0.0
    def close(a, b):
        nonlocal worst
        worst = max(worst, abs(a - float(b))); assert abs(a - float(b)) < 1e-9, (a, b)
    for key, p in panels.items():
        fam, pol = FAMILY[key[0]], POLICY[key[2:]]
        m = next(e for e in mo if (e['model'], e['family'], e['scope'], e['clock'], e['metric'], e['reference'], e['estimand'])
                 == (pol, fam, 'common28', '5', 'position_distance', 'R0_R1', 'ratio_of_task_means'))
        close(p['ratio'], m['point']); close(p['ratio_lo'], m['ci95_lower']); close(p['ratio_hi'], m['ci95_upper'])
        assert (p['motion_settings'], p['motion_tasks']) == (int(m['joint_variants']), int(m['joint_tasks']))
        o = {e['metric']: e for e in oc if (e['model'], e['family'], e['scope']) == (pol, fam, 'common28')}
        if p['unknown_pairs'] == 0:
            close(p['lost'] / 100, o['harm']['point']); close(p['gained'] / 100, o['recovery']['point']); close(p['net'] / 100, o['risk_difference']['point'])
            close(p['net_lo'] / 100, o['risk_difference']['ci98_75_lower']); close(p['net_hi'] / 100, o['risk_difference']['ci98_75_upper'])
        else:                                                    # recorded flips are the lower ends of the archived bounds
            close(p['lost'] / 100, o['harm']['bound_lower']); close(p['gained'] / 100, o['recovery']['bound_lower'])
    return worst


def build(verbose=True):
    cases = load_cases(); panels = {}
    for f, fam in FAMILY.items():
        for p, pol in POLICY.items():
            panels[f'{f}_{p}'] = family_panel([c for c in cases if c['family'] == fam and c['model'] == pol])
    worst = _check(panels)
    pairs = sensor_pairs([c for c in cases if c['family'] == FAMILY['S'] and c['model'] == POLICY['pi0']])
    auc = pair_auroc(pairs)
    if verbose:
        print(f'fig5: {len(cases)} primary cases; recomputed family values equal the archived tables (max abs diff {worst:.1e})')
        for key, p in panels.items():
            net = f"{p['net']:+.2f} [{p['net_lo']:.2f}, {p['net_hi']:.2f}]" if p['net_lo'] is not None else f"{p['net']:+.2f} (recorded only; {p['unknown_pairs']}/{p['planned_pairs']} pairs unrecorded)"
            print(f"  {key:7s} ratio {p['ratio']:.3f} [{p['ratio_lo']:.3f}, {p['ratio_hi']:.3f}]  net {net}  lost {p['lost']:.2f} gained {p['gained']:.2f}  motion support {p['motion_settings']}/{p['motion_tasks']}")
        for c in ('SS', 'SF', 'FS', 'FF'):
            v = [q for q in pairs if q['cell'] == c]; b = [q for q in v if q['ratio'] < 1]
            print(f"  pi0 Sensor pairs {c}: {len(b)} of {len(v)} below 1x ({len(set(q['task'] for q in b))} tasks)")
        print(f'  pi0 Sensor AUROC, lost vs retained: {auc[0]:.3f} [{auc[1]:.3f}, {auc[2]:.3f}]')
    return dict(panels=panels, pairs=pairs, auroc=auc)


if __name__ == '__main__':
    build()
