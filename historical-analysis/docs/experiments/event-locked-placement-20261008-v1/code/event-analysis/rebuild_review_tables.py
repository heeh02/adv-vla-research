"""Rebuild the review tables for the event-locked placement analysis.

Input : data/derived-trajectories/shard-*.pkl (robot-side fields streamed from JD OSS,
        see code/extraction/extract_traj.py)
Output: data/review/*.csv

Definitions are identical to the ones used for report v2 (code/event-analysis/fa_*.py);
this script only re-derives them in one place with explicit columns for review.
Run:  /opt/anaconda3/bin/python3 code/event-analysis/rebuild_review_tables.py  (from the package root)
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TRAJ = os.path.join(ROOT, 'data', 'derived-trajectories')
OUT = os.path.join(ROOT, 'data', 'review')
os.makedirs(OUT, exist_ok=True)

LIFT = 0.03          # m, object counts as lifted / moved above this
MOVED = 0.03         # m, median clean-success displacement to call an object "manipulated"
QPOS_START = 10      # physics_flat = [time, 7 arm qpos, 2 finger qpos, free bodies (7 each)...]


def free_body_blocks(phys):
    """Heuristic free-joint detection: 7-vector whose last 4 entries are a unit quaternion and
    whose z is in [0.3, 1.6] m. Not bound to MuJoCo joint names (see README caveats)."""
    out, i = [], QPOS_START
    while i + 7 <= len(phys):
        q = phys[i + 3:i + 7]
        if abs(np.linalg.norm(q) - 1) < 2e-3 and 0.3 < phys[i + 2] < 1.6:
            out.append(i)
            i += 7
        else:
            i += 1
    return out


def gripper_events(actions):
    g = np.sign(actions[:, 6]).astype(float)
    g[g == 0] = -1
    closes = sorted(list(np.where((g[1:] > 0) & (g[:-1] < 0))[0] + 1) + ([0] if g[0] > 0 else []))
    opens = list(np.where((g[1:] < 0) & (g[:-1] > 0))[0] + 1)
    return g, closes, opens


blocks_rows, obj_rows, ep_rows = [], [], []
for f in sorted(glob.glob(os.path.join(TRAJ, 'shard-*.pkl'))):
    o = pickle.load(open(f, 'rb'))
    names = {ep[0]: v.get('task_name') for ep, v in o['task'].items()}
    byroot = {}
    for ep in o['state']:
        byroot.setdefault(ep[0], []).append(ep)
    # episode-level gripper events (all episodes)
    for ep, a in o['exec'].items():
        st = o['state'].get(ep, [])
        res = o['result'].get(ep, {})
        _, closes, opens = gripper_events(a)
        steps = np.array([d['step'] for d in st]) if st else np.array([])
        E = np.stack([d['eef'] for d in st]) if st else np.zeros((0, 3))

        def eef_at(t):
            return [float(np.interp(t, steps, E[:, k])) for k in range(3)] if len(steps) else [np.nan] * 3
        fc = closes[0] if closes else np.nan
        e = eef_at(fc) if closes else [np.nan] * 3
        ep_rows.append(dict(root=ep[0], noise=ep[1], cond=ep[2], attempt=ep[3], task_name=names.get(ep[0]),
                            success=res.get('success'), steps=res.get('steps'), termination=res.get('termination'),
                            n_close=len(closes), close_steps=' '.join(map(str, closes)), open_steps=' '.join(map(str, opens)),
                            first_close=fc, last_open=opens[-1] if opens else np.nan,
                            eef_first_close_x=e[0], eef_first_close_y=e[1], eef_first_close_z=e[2],
                            n_query_states=len(st)))
    for root, eps in byroot.items():
        clean_succ = [e for e in eps if e[2] == 'clean' and o['result'].get(e, {}).get('success')]
        if not clean_succ:
            continue
        p0 = o['state'][clean_succ[0]][0]['phys'].astype(float)
        B = free_body_blocks(p0)
        disp = {b: [] for b in B}
        fin = {b: [] for b in B}
        lif = {b: [] for b in B}
        for e in clean_succ:
            P = np.stack([d['phys'].astype(float) for d in o['state'][e]])
            if P.shape[1] != len(p0):
                continue
            for b in B:
                disp[b].append(np.linalg.norm(P[-1, b:b + 3] - P[0, b:b + 3]))
                fin[b].append(P[-1, b:b + 3])
                lif[b].append((P[:, b + 2] - P[0, b + 2]).max())
        man = [b for b in B if np.median(disp[b]) > MOVED]
        primary = max(man, key=lambda b: np.median(disp[b])) if man else None
        ref = {b: np.median(np.stack(fin[b]), 0) for b in B}
        for b in B:
            blocks_rows.append(dict(root=root, task_name=names.get(root), physics_flat_offset=b,
                                    init_x=p0[b], init_y=p0[b + 1], init_z=p0[b + 2],
                                    clean_median_displacement=float(np.median(disp[b])), clean_median_max_lift=float(np.median(lif[b])),
                                    manipulated=b in man, picked_type=bool(b in man and np.median(lif[b]) > LIFT), primary=b == primary,
                                    ref_final_x=ref[b][0], ref_final_y=ref[b][1], ref_final_z=ref[b][2],
                                    n_clean_success=len(clean_succ), physics_flat_len=len(p0)))
        for e in eps:
            st = o['state'][e]
            a = o['exec'].get(e)
            res = o['result'].get(e, {})
            if a is None or not st:
                continue
            P = np.stack([d['phys'].astype(float) for d in st])
            steps = np.array([d['step'] for d in st])
            if P.shape[1] != len(p0):
                continue
            g, closes, opens = gripper_events(a)
            for b in man:
                z = P[:, b + 2] - P[0, b + 2]
                lift_t = steps[np.argmax(z > LIFT)] if (z > LIFT).any() else np.nan
                d_xy = P[-1, b:b + 2] - ref[b][:2]
                obj_rows.append(dict(root=root, noise=e[1], cond=e[2], task_name=names.get(root), physics_flat_offset=b,
                                     primary=b == primary, picked_type=bool(np.median(lif[b]) > LIFT), success=res.get('success'),
                                     steps=res.get('steps'), n_close=len(closes),
                                     first_close=closes[0] if closes else np.nan, last_open=opens[-1] if opens else np.nan,
                                     init_x=P[0, b], init_y=P[0, b + 1], init_z=P[0, b + 2],
                                     final_x=P[-1, b], final_y=P[-1, b + 1], final_z=P[-1, b + 2],
                                     ref_final_x=ref[b][0], ref_final_y=ref[b][1], ref_final_z=ref[b][2],
                                     dx=d_xy[0], dy=d_xy[1], dz=P[-1, b + 2] - ref[b][2], final_err_xy=float(np.hypot(*d_xy)),
                                     ever_lift=bool((z > LIFT).any()), max_lift=float(z.max()), lift_step=lift_t,
                                     gripper_cmd_last=float(g[-1]), eef_obj_dist_last=float(np.linalg.norm(st[-1]['eef'] - P[-1, b:b + 3]))))

EP = pd.DataFrame(ep_rows)
BL = pd.DataFrame(blocks_rows)
OB = pd.DataFrame(obj_rows)
EP.to_csv(os.path.join(OUT, 'episode_events.csv'), index=False)
BL.to_csv(os.path.join(OUT, 'object_blocks.csv'), index=False)

# clean partner + outcome cell (primary and non-primary objects share the episode outcome)
clean = EP[EP.cond == 'clean'][['root', 'noise', 'success']].rename(columns={'success': 'clean_success'})
OB = OB.merge(clean, on=['root', 'noise'], how='left')
OB['cell'] = np.select([(OB.success == True) & (OB.clean_success == True), (OB.success == False) & (OB.clean_success == True),
                        (OB.success == True) & (OB.clean_success == False)], ['SS', 'harm', 'recovery'], 'FF')
OB.loc[OB.cond == 'clean', 'cell'] = 'clean'
fam = lambda c: ('clean' if c == 'clean' else 'cam±30' if '30' in c else 'cam±15+材质' if ('surface' in c and 'camera' in c)
                 else 'q0_only' if 'q0_only' in c else 'cam±15' if 'camera' in c else 'photometric')
OB['family'] = OB.cond.map(fam)
# per-root success tolerance on the primary object: 95th pct of final xy error among successful episodes (any condition)
tol = OB[OB.primary & (OB.success == True)].groupby('root').final_err_xy.quantile(.95).rename('tol_xy_95_success')
OB = OB.join(tol, on='root')


def mode(r):
    if r.picked_type and not r.ever_lift:
        return 'A_never_lifted'
    if r.gripper_cmd_last > 0 and r.eef_obj_dist_last < 0.06:
        return 'B_still_holding'
    if r.final_err_xy > r.tol_xy_95_success:
        return 'C_outside_success_range'
    return 'D_inside_range_but_failed'


OB['mode_if_failed'] = np.where(OB.success == False, OB.apply(mode, axis=1), '')
OB.to_csv(os.path.join(OUT, 'object_episode.csv'), index=False)
H = OB[OB.primary & (OB.cell == 'harm')].copy()
H['err_over_tol'] = H.final_err_xy / H.tol_xy_95_success
H.to_csv(os.path.join(OUT, 'harm87_primary_object.csv'), index=False)

# systematic placement offset (successful episodes, primary object)
S = OB[OB.primary & (OB.success == True)]
sysrows = []
for (c, r), g in S.groupby(['cond', 'root']):
    sysrows.append(dict(cond=c, root=r, n=len(g), mean_dx=g.dx.mean(), mean_dy=g.dy.mean(),
                        sys_offset=float(np.hypot(g.dx.mean(), g.dy.mean())), median_err=g.final_err_xy.median()))
pd.DataFrame(sysrows).to_csv(os.path.join(OUT, 'placement_offset_by_cond_root.csv'), index=False)

counts = H.mode_if_failed.value_counts().to_dict()
print('episodes', len(EP), '| roots with objects', OB.root.nunique(), '| tasks', OB.root.str.rsplit('-', n=1).str[0].nunique())
print('harm (primary object):', len(H), counts)
expected = {'C_outside_success_range': 61, 'D_inside_range_but_failed': 13, 'A_never_lifted': 8, 'B_still_holding': 5}
print('matches report v2:', counts == expected)
