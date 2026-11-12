# Original LIBERO scalar analysis replay

This package reproduces the original two-observation association and harm AUC analysis from archived **case-level derived scalars**. It retains all 2,400 planned pairs: 2,344 observed, 56 missing, 40 tasks, 10 conditions and 400 task–condition cells. It contains no neural tensors, model weights or simulator rollouts.

## Run

Tested with Python 3.9.6; dependency versions are in `requirements.txt`.

```bash
python3 -B replay_s10_2.py --out /absolute/new/replay-output
```

The output directory must not already exist. The script uses only files beside itself and performs no network access. It validates input hashes, paired source identities, outcome cells, support and the exact saved 1,000 × 40 task bootstrap matrix. It then recomputes 24 correlations, 33 within-condition AUC rows and 24 interval rows, comparing them to archived expectations. Successful status is `VERIFIED_ARCHIVED_SCALAR_REPLAY`.

Core points: translation–success-change correlation is approximately −.903030 across ten condition means and −.075232 across 400 cells; task/condition-adjusted H–command is .722298 and H–signed outcome is −.000899; within-condition harm AUCs are .535272/.581009/.420696 for H/TV/translation. Exact values and intervals appear in the emitted CSVs. The positive translation–signed-outcome association remains included; scores below .5 are not reversed.

## Definitions and dependence

An episode feature is the arithmetic mean of two scalar measurements at its suite-specific observation clocks, using L18, the first generation call and five action positions. H is relative RMS; main-camera TV is conditional TV; command is planned adapter translation RMS over 15 coordinates. These are not first-query issued-command norms or EEF distances.

Condition means give each observed episode equal weight. The cell statistic gives each of the 400 cells equal weight. Partial ranks are global average-tie ranks followed by additive task/condition demeaning. Harm AUC retains clean-success SF/SS and pools comparisons only within the same condition, weighted by SF × SS pair counts. This differs from the task-equal S4 AUC. No AUC interval is generated.

The whole-task bootstrap uses the original saved draws, verified against PCG64 seed 20261008 and the preserved original task ordering. Each sampled task carries its roots, noises, conditions and shared-clean dependence; ranks are recomputed. Percentile intervals use linear interpolation, without suite stratification or multiplicity correction. These are the original fixed estimands.

## Identity and labels

Episode/task/root IDs are anonymous stable names. Task ordering preserves the original bootstrap order. Original treatment and clean result/state/noise hashes remain; repeated clean-reference IDs have consistent labels and result identities.

The original audit distinguishes a later mixed execution collection: 3,252 executions retained, 25 observed cases rerun, and 83 originally missing slots completed. The 25 reruns have different action/result bytes and do not replace the old neural trajectories' labels. No source-identical result had a corrected label. This replay retains the original source outcomes and missing values.

The input manifest binds the original scalar, registry and estimator hashes. The replay does not reparse raw simulator outcomes or re-extract high-dimensional records. P08, PWH and the five-arm collection are not used to fill missing observations.

## Files and columns

- `data/episode-scalars.csv.gz`: all planned rows, paired outcomes and identity hashes. `slot1_hidden`, `slot2_hidden`, `slot1_main_TV`, `slot2_main_TV`, `slot1_action`, and `slot2_action` are the original six scalar inputs. Empty entries remain missing.
- `pair_status` uses **treatment then clean** order: `11` joint success, `01` harm, `10` recovery, `00` joint failure. Named missing statuses preserve unavailable trajectories. The explicit success columns provide a cross-check.
- `clean_reference_id` identifies the reused standalone clean execution; result/state/noise hashes retain its recorded provenance. `task_id`, `root_id`, `noise_label` and `condition` retain the nesting used by the estimator.
- `data/task-draws.csv.gz`: original 1,000 × 40 draw matrix. Its integers index `task-00` through `task-39` in the original lexicographic task order.
- `original_estimators.py`: original correlation/AUC function ASTs, extracted unchanged without project I/O.
- `data/expected-*.csv`: archived scalar estimates used only as regression expectations.
- `INPUT-MANIFEST.json`: hashes, support and label-version policy.

This is an executable scalar-analysis companion, not a full acquisition replay.
