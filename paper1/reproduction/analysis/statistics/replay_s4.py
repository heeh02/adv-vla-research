"""Replay bounded S4 scalar estimates from the packaged frozen task moments.

This does not read raw cases, model tensors, or the original private storage.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import internal_link_pure as m

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE / 'S4-SCALAR-RESULT.json').read_text())
    tasks = data['shared_task_ids']
    rng = random.Random(data['bootstrap_seed'])
    draws = [[tasks[rng.randrange(len(tasks))] for _ in tasks]
             for _ in range(data['bootstrap_replicates'])]
    draw_sha = hashlib.sha256(json.dumps(draws, separators=(',', ':'),
                                         ensure_ascii=False).encode()).hexdigest()
    assert draw_sha == data['shared_task_draw_matrix_sha256']
    checked = 0
    for family, table in data['families'].items():
        for signal, endpoints in table['rank_task_moments'].items():
            for endpoint, moments in endpoints.items():
                values = [m.pooled_rho(moments, sample) for sample in draws]
                actual = m.bootstrap_result(m.pooled_rho(moments, tasks), values,
                                            table['rank_eligible_tasks'])
                assert actual == table['association'][signal][endpoint], (family, signal, endpoint)
                checked += 1
        for signal, scores in table['harm_AUC_task_scores'].items():
            values = [m.equal_task_auc(scores, sample) for sample in draws]
            actual = m.bootstrap_result(m.equal_task_auc(scores, tasks), values,
                                        table['auc_eligible_tasks'])
            assert actual == table['harm_AUC_SF_vs_SS'][signal], (family, signal, 'AUC')
            checked += 1
        w = table['rank_task_moments']['W_pre_gate_relative_l2']['issued_translation_first5_delta_rms']
        p = table['rank_task_moments']['P_full_key_TV']['issued_translation_first5_delta_rms']
        values = []
        for sample in draws:
            a, b = m.pooled_rho(w, sample), m.pooled_rho(p, sample)
            values.append(a - b if a is not None and b is not None else None)
        actual = m.bootstrap_result(m.pooled_rho(w, tasks) - m.pooled_rho(p, tasks),
                                    values, table['rank_eligible_tasks'])
        assert actual == table['prelisted_W_minus_P_issued_rho'], (family, 'W-P')
        checked += 1
    assert m.midrank([1, 1, 3, 4]) == [.25, .25, .625, .875]
    assert m.task_moments([{'row_id': str(i // 2), 'x': i, 'y': 0}
                           for i in range(4)], 'x', 'y') == (0.0, .078125, 0.0)
    print(json.dumps({'status': 'scalar_replay_passed', 'S4_exact_checks': checked,
                      'draw_matrix_sha256': draw_sha,
                      'scope': 'saved task moments and fixed bootstrap only'}))


if __name__ == '__main__':
    main()
