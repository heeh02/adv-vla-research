"""Further checks on the execution study asked for by the second cold review. Reads raw/Five_arm_cases_flat.csv only.

(a) Successes gained under a perturbation (clean failure -> perturbed success) against successes "gained" by a clean
    rerun (first clean run fails, second succeeds): task-equal rates and their difference, whole-task bootstrap.
(b) Where outcomes are missing on the 28 common tasks, by policy and family.
Nothing here is used by the figures; the numbers are recorded in latex/candidates/R9_claim_calibration.md.
"""
import collections, random
import fig5_data as f

SEED, DRAWS = 20261111, 2000


def gained_rates(cases):
    by = collections.defaultdict(list)
    for c in cases:
        o = c['outcomes']
        if any(o.get(k) is None for k in ('R0', 'R1', 'T0', 'T1')): continue
        pert = ((not o['R0'] and o['T0']) + (not o['R1'] and o['T1'])) / 2      # F -> S under the perturbation
        rerun = ((not o['R0'] and o['R1']) + (not o['R1'] and o['R0'])) / 2      # F -> S between the two clean runs, either order
        lostp = ((o['R0'] and not o['T0']) + (o['R1'] and not o['T1'])) / 2
        by[c['task_id']].append((pert, rerun, lostp))
    return by


def summary(by, tasks):
    m = lambda i: 100 * sum(sum(x[i] for x in by[t]) / len(by[t]) for t in tasks) / len(tasks)
    return m(0), m(1), m(2)


if __name__ == '__main__':
    cases = f.load_cases()
    print('(a) successes gained: perturbation vs clean rerun (task-equal %, rerun counted in either order and halved)')
    for pol in ('pi05', 'pi0'):
        for fam in 'BCLS':
            by = gained_rates([c for c in cases if c['model'] == f.POLICY[pol] and c['family'] == f.FAMILY[fam]])
            order, rng = sorted(by), random.Random(SEED)
            g, r, l = summary(by, order)
            d = sorted(a - b for a, b, _ in (summary(by, [rng.choice(order) for _ in order]) for _ in range(DRAWS)))
            dl = sorted(c - b for a, b, c in (summary(by, [rng.choice(order) for _ in order]) for _ in range(DRAWS)))
            print(f'  {pol:5s} {fam} gained {g:5.2f} rerun {r:5.2f} excess {g - r:+6.2f} [{d[49]:+.2f}, {d[1949]:+.2f}] | lost {l:5.2f} excess {l - r:+6.2f} [{dl[49]:+.2f}, {dl[1949]:+.2f}]  tasks {len(order)}')
    print('(b) cases with a missing outcome on the 28 common tasks')
    for pol in ('pi05', 'pi0'):
        for fam in 'BCLS':
            cs = [c for c in cases if c['model'] == f.POLICY[pol] and c['family'] == f.FAMILY[fam]]
            miss = sum(any(c['outcomes'].get(k) is None for k in ('R0', 'R1', 'T0', 'T1', 'R0prime')) for c in cs)
            print(f'  {pol:5s} {fam} settings {len(cs):3d} with a missing outcome {miss}')
