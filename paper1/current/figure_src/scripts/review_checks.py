"""Two checks on the execution study that a mock review asked for. Reads figure_src/raw/Five_arm_cases_flat.csv only.

(a) How well the size of the motion change separates lost (S->F) from retained (S->S) successes: pair-level AUROC,
    pooled over families and per family, with a whole-task bootstrap (2,000 draws, seed 20261111).
(b) Whether outcome flips under a visual change exceed those from resampling the policy noise: task-equal flip rates
    of (T_n vs R_n) and (R_0 vs R_1) and their difference, same bootstrap.
Nothing here is used by the figures; the numbers are recorded in latex/candidates/CVPR_mock_rereview2_20261111.md.
"""
import collections, random
import fig5_data as f

SEED, DRAWS = 20261111, 2000


def _auroc(pos, neg):
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg) / (len(pos) * len(neg))


def _pairs(cases, clock, kind):
    out = []
    for c in cases:
        ref = f._dist(c, 'R0_R1', clock)
        for n in ('0', '1'):
            cell, d = c['noise_pair_cells'][n], f._dist(c, 'TR' + n, clock)
            if cell in ('SS', 'SF') and d is not None and (kind == 'abs' or (ref is not None and ref > 0)):
                out.append(dict(cell=cell, x=d if kind == 'abs' else d / ref, task=c['task_id']))
    return out


def _auroc_ci(pairs):
    by = collections.defaultdict(list)
    for p in pairs: by[p['task']].append(p)
    order, rng, draws = sorted(by), random.Random(SEED), []
    for _ in range(DRAWS):
        d = [q for t in (rng.choice(order) for _ in order) for q in by[t]]
        pos = [q['x'] for q in d if q['cell'] == 'SF']; neg = [q['x'] for q in d if q['cell'] == 'SS']
        if pos and neg: draws.append(_auroc(pos, neg))
    draws.sort()
    pos = [p['x'] for p in pairs if p['cell'] == 'SF']; neg = [p['x'] for p in pairs if p['cell'] == 'SS']
    return _auroc(pos, neg), draws[int(.025 * len(draws))], draws[int(.975 * len(draws)) - 1], len(pos), len(neg)


def _flip_rates(cases):
    by = collections.defaultdict(list)
    for c in cases:
        o = c['outcomes']
        if any(o.get(k) is None for k in ('R0', 'R1', 'T0', 'T1')): continue
        by[c['task_id']].append((((o['T0'] != o['R0']) + (o['T1'] != o['R1'])) / 2, float(o['R0'] != o['R1'])))
    return by


def _flip_summary(by, tasks):
    v = sum(sum(a for a, _ in by[t]) / len(by[t]) for t in tasks) / len(tasks)
    r = sum(sum(b for _, b in by[t]) / len(by[t]) for t in tasks) / len(tasks)
    return 100 * v, 100 * r


if __name__ == '__main__':
    cases = f.load_cases()
    print('(a) AUROC of motion size, lost vs retained successes')
    for pol in ('pi05', 'pi0'):
        cs = [c for c in cases if c['model'] == f.POLICY[pol]]
        for clock in ('5', '60'):
            for kind in ('abs', 'ratio'):
                a, lo, hi, npos, nneg = _auroc_ci(_pairs(cs, clock, kind))
                print(f'  {pol:5s} pooled control {clock:>2s} {kind:5s} {a:.3f} [{lo:.3f}, {hi:.3f}]  lost {npos} retained {nneg}')
        for fam in 'BCLS':
            ps = _pairs([c for c in cs if c['family'] == f.FAMILY[fam]], '5', 'ratio')
            if sum(p['cell'] == 'SF' for p in ps) >= 4:
                a, lo, hi, npos, nneg = _auroc_ci(ps)
                print(f'     {pol} {fam} control 5 ratio {a:.3f} [{lo:.3f}, {hi:.3f}]  lost {npos} retained {nneg}')
    print('(b) outcome flips, visual change vs policy-noise resampling (task-equal %)')
    for pol in ('pi05', 'pi0'):
        for fam in 'BCLS':
            by = _flip_rates([c for c in cases if c['model'] == f.POLICY[pol] and c['family'] == f.FAMILY[fam]])
            order, rng = sorted(by), random.Random(SEED)
            v, r = _flip_summary(by, order)
            diff = sorted(a - b for a, b in (_flip_summary(by, [rng.choice(order) for _ in order]) for _ in range(DRAWS)))
            print(f'  {pol:5s} {fam} visual {v:5.2f} resampling {r:5.2f} excess {v - r:+6.2f} [{diff[49]:+.2f}, {diff[1949]:+.2f}]  tasks {len(order)}')
