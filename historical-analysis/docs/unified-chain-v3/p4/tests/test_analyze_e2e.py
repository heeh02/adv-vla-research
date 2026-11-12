"""End-to-end smoke test of analyze.py parts A/B/C on a small synthetic feature set (no real data)."""
import os, sys, time, hashlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud', 'worker'))
import analyze as an
from p4lib import head_geometry, step_terms

an.B_BOOT = 50
rng = np.random.default_rng(3)
W = rng.standard_normal((1024, 32)) * 0.03
Ur, _, _, _ = head_geometry(W[:, :7])
rows = []
tasks = [f'libero_spatial-t{i:02d}' for i in range(10)]
F = {k: [] for k in ['step_R','step_A','step_L','step_V','step_M','step_Rc','step_Ac','step_Vv','qh','closure','x0eq','dt_dev','u','dphys','a0','n0','abar','nbar','vis_hash','rc0_hash','root','noise','cond','q','shard']}
for t in tasks:
    for init in ('i12', 'i13'):
        root = f'{t}-{init}'
        for c in an.PERSIST:
            for qq in ('q000', 'q004', 'q008', 'q012'):
                vis = hashlib.sha1(f'{root}{c}{qq}'.encode()).hexdigest()
                for n in range(3):
                    dr = (rng.standard_normal((10, 5, 1024)) * np.linspace(1, 3, 10)[:, None, None]).astype(np.float32)
                    rc = rng.standard_normal((10, 5, 1024)).astype(np.float32)
                    u = (-0.1 * (dr.astype(np.float64) @ W[:, :7]) * (1 + 0.01 * rng.standard_normal((10, 5, 7)))).astype(np.float32)
                    st = step_terms(dr, rc, u, W[:, :7], Ur, u / -0.1)
                    for k in ('R','A','L','V','M','Rc','Ac','Vv'): F['step_'+k].append(st[k])
                    F['qh'].append([0.003, 0.004, 0.003, 0.004]); F['closure'].append(0.0); F['x0eq'].append(True); F['dt_dev'].append(1e-7)
                    F['u'].append(u); F['dphys'].append(rng.standard_normal((5, 7)).astype(np.float32))
                    r0 = dr[0].astype(np.float64); a0 = r0 @ Ur
                    F['a0'].append(a0.reshape(-1).astype(np.float32)); F['n0'].append((r0 - a0 @ Ur.T).reshape(-1).astype(np.float32))
                    rb = dr.mean(0).astype(np.float64); ab = rb @ Ur
                    F['abar'].append(ab.reshape(-1).astype(np.float32)); F['nbar'].append((rb - ab @ Ur.T).reshape(-1).astype(np.float32))
                    F['vis_hash'].append(vis); F['rc0_hash'].append(f'{root}{c}{qq}{n}')
                    F['root'].append(root); F['noise'].append(n); F['cond'].append(c); F['q'].append(qq); F['shard'].append('s')
F = {k: np.asarray(v) for k, v in F.items()}
cond, root, q_, noise = F['cond'], F['root'], F['q'], F['noise']
task = np.array([x.rsplit('-', 1)[0] for x in root]); suite = np.array([x.split('-')[0] for x in root])
anchor = np.array(['q0' if qq == 'q000' else 'lm1' if qq == an.LM1Q[s] else 'lm2' if qq == an.LM2Q[s] else 'later' for qq, s in zip(q_, suite)])
out = {}
t0 = time.time()
A = an.part_A(F, task, cond, anchor, suite, out); print('A', round(time.time()-t0,1), A['primary']['share'], A['decision']['nu_negligible'])
D = an.build_B(F, Ur, task, cond, anchor)
B = an.part_B(D, out); print('B', round(time.time()-t0,1), {m: round(v['EEF'],3) for m, v in B['metrics'].items()}, B['decision'])
C = an.part_C(F, D, Ur, task, cond, anchor, q_, root, noise, out); print('C', round(time.time()-t0,1), {k: (v.get('qc'), v.get('decision')) for k, v in C.items()})
print('profiles rows', len(out['profiles']))
