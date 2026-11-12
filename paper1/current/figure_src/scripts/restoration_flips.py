"""Paired outcome changes of every confirmation arm against the unedited perturbed arm N_T.

Reads figure_src/raw/restoration/restoration_case_outcomes.csv (1,296 rows: 216 cases x 6 arms) and prints,
for each family and arm, the raw successes and the cases whose outcome differs from N_T under the same case and
policy-noise label. These are the counts quoted in main Sec. 6 and in the unnumbered table of Supplement Sec. S13.4.
"""
import collections, csv, pathlib

RAW = pathlib.Path(__file__).resolve().parents[1] / 'raw' / 'restoration' / 'restoration_case_outcomes.csv'
ARMS = ('N_T', 'R_rand', 'R_clean', 'R_resp', 'R_full', 'N_C')


def build():
    cases = collections.defaultdict(dict)
    for r in csv.DictReader(open(RAW)):
        cases[(r['family'], r['task_id'], r['row_id'], r['init_index'], r['noise_label'])][r['arm']] = int(r['success'])
    out = {}
    for fam in sorted({k[0] for k in cases}):
        rows = [v for k, v in cases.items() if k[0] == fam]
        assert all(set(v) == set(ARMS) for v in rows)
        out[fam] = {a: dict(cases=len(rows), successes=sum(v[a] for v in rows),
                            gained=sum(1 for v in rows if v['N_T'] == 0 and v[a] == 1),
                            lost=sum(1 for v in rows if v['N_T'] == 1 and v[a] == 0)) for a in ARMS}
    return out


if __name__ == '__main__':
    for fam, arms in build().items():
        print(fam)
        for a, c in arms.items():
            print(f"  {a:8s} successes {c['successes']:3d}/{c['cases']}  gained {c['gained']:2d}  lost {c['lost']:2d}")
