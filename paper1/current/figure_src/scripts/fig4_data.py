"""Figure 4 inputs, computed from the confirmation-study records in raw/restoration/.

  Figure4_Task_Values.csv        task-mean error of the first five issued translation commands, per arm
  restoration_case_outcomes.csv  binary success of all 1,296 rollouts (216 matched cases x 6 arms)
  Figure4_Plotted_Contrasts.csv  the six prespecified comparisons with their simultaneous 95% intervals
Arms: N_C native clean, N_T unedited, R_resp / R_clean / R_rand rank-16 restorations, R_full all 1,024 channels.
Means average noise labels within setting, settings within task, tasks equally. Points are recomputed here and must
equal the archived contrasts; the intervals come from the archived 10,000-draw whole-task bootstrap and are not
re-drawn. Full L9 is outside the six comparisons and has no interval."""
import csv, os, collections

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'raw', 'restoration')
FAMILY = {'Camera': 'Camera Viewpoints', 'Sensor': 'Sensor Noise'}


def _read(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        raise SystemExit(f'missing {path}: copy the confirmation-study tables into figure_src/raw/restoration/')
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _mean(v):
    v = list(v); return sum(v) / len(v)


def build(verbose=True):
    values, cases, archived = _read('Figure4_Task_Values.csv'), _read('restoration_case_outcomes.csv'), _read('Figure4_Plotted_Contrasts.csv')
    out = {}; worst = 0.0
    for fam, name in FAMILY.items():
        err = collections.defaultdict(dict)
        for r in values:
            if r['family'] == name: err[r['arm']][r['task_id']] = float(r['initial_command_error'])
        tasks = sorted(err['R_resp'])
        nested = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        for r in cases:
            if r['family'] == name: nested[r['arm']][r['task_id']][r['row_id']].append(int(r['success']))
        succ = {arm: {t: _mean(_mean(s) for s in settings.values()) for t, settings in by_task.items()} for arm, by_task in nested.items()}
        arms = {arm: dict(error=_mean(err[arm][t] for t in tasks) if arm in err else 0.0, success=100 * _mean(succ[arm][t] for t in tasks)) for arm in succ}
        rows = {}
        for arm in ('N_T', 'R_rand', 'R_clean', 'R_full'):
            d = [err[arm][t] - err['R_resp'][t] for t in tasks]
            rows[arm] = dict(task_reduction=d, reduction=_mean(d), success_gain=arms['R_resp']['success'] - arms[arm]['success'],
                             reduction_ci=None, success_ci=None, positive=sum(x > 0 for x in d))
            a = [x for x in archived if x['family'] == name and x['comparator'] == arm]
            if a:                                                # one of the six prespecified comparisons
                a = a[0]
                for mine, theirs in ((rows[arm]['reduction'], a['command_reduction']), (rows[arm]['success_gain'], a['success_gain_pp'])):
                    worst = max(worst, abs(mine - float(theirs))); assert abs(mine - float(theirs)) < 1e-9, (fam, arm, mine, theirs)
                rows[arm]['reduction_ci'] = (float(a['command_ci_low']), float(a['command_ci_high']))
                rows[arm]['success_ci'] = (float(a['success_ci_low_pp']), float(a['success_ci_high_pp']))
        out[fam] = dict(tasks=len(tasks), cases=sum(len(s) for by in nested['R_resp'].values() for s in by.values()), arms=arms, rows=rows)
    if verbose:
        print(f'fig4: recomputed contrasts equal the archived table (max abs diff {worst:.1e})')
        for fam, d in out.items():
            print(f"  {fam}: {d['tasks']} tasks, {d['cases']} cases; error " + ', '.join(f"{a} {v['error']:.4f}" for a, v in d['arms'].items() if a != 'N_C')
                  + '; success ' + ', '.join(f"{a} {v['success']:.2f}" for a, v in d['arms'].items()))
            for arm, r in d['rows'].items():
                ci = '' if r['reduction_ci'] is None else f" [{r['reduction_ci'][0]:.4f}, {r['reduction_ci'][1]:.4f}]"
                sci = '' if r['success_ci'] is None else f" [{r['success_ci'][0]:.2f}, {r['success_ci'][1]:.2f}]"
                print(f"    vs {arm:7s} error reduction {r['reduction']:+.4f}{ci} ({r['positive']}/{d['tasks']} tasks > 0); success {r['success_gain']:+.2f}{sci}")
    return out


if __name__ == '__main__':
    build()
