# Current-dictionary command-readout replay

This entry reproduces the new matched-case statistics in the paper from **archived task-component scores**. It uses Python 3.9 or later and the standard library only.

```bash
python3 -B -m unittest -v test_replay.py
python3 -B replay.py --out replay-output
```

Run the first command from this directory. The second also works from another directory when `replay.py` is given by its full path. The output directory must be new.

Expected result: `PASS`, with 468 readout estimates, 216 paired readout contrasts, 36 reconstruction estimates, and 7,850 scalar/string/support checks. These checks are software comparisons, not independent experiments.

## Data and scope

- The evaluation contains 404 paired cases on 28 common LIBERO-Plus tasks. Background, Camera, Light and Sensor Noise contribute 96, 102, 96 and 110 cases.
- Archived predictions come from sixty original task-and-family held-out splits across L1/L9/L18. Training uses the original 536-case archive. Folds are joined before scoring tasks.
- Every action position is retained in the predictors. The scored targets are the first five returned adapter commands: translation, rotation and continuous gripper separately.
- `data/task_scores.json.gz` contains 13,104 derived readout task rows and 1,008 derived reconstruction task rows. Repeated rows across methods and components do not increase the number of tasks or cases.
- `data/draws.json.gz` supplies the original 2,000-by-28 task-resampling definition explicitly for each family/layer. It was generated from the producer's fixed seed rule and checked by replaying all released intervals. Task identifiers are anonymized in the original sorted order, preserving the paired resampling.
- `data/expected.json` contains the frozen estimates and contrasts used by the figures. Input bytes and hashes are bound by `INPUT-MANIFEST.json`.

The program recomputes task means, paired differences, relative error reductions and linear-interpolation quantiles. It also checks reconstruction retention directly against each task's archived SSE and residual energy. Intervals are pointwise 95%, conditional on the fitted dictionaries and readouts. The analysis is exploratory and does not treat its family/layer/component comparisons as independent replications.

## Names and units

| Code name | Meaning |
|---|---|
| `new_response`, `new_clean`, `new_random` | The current position-centered dictionary comparison and its fixed random control; `new` does not mean new rollouts |
| `old_response`, `old_clean`, `old_random` | Earlier uncentered dictionary comparison and its random control |
| `magnitude` | Ten positionwise hidden-response norms |
| `full` | The complete ten-position, 1024-channel hidden response |
| `flow` | The first native generation-velocity difference, 320 coordinates |
| `flow_plus_new`, `flow_plus_old` | First-flow and the corresponding response-coordinate block together |
| `zero`, `train_mean` | Zero-difference prediction and the training target mean |

Stored layer indices 0/8/17 correspond to paper labels L1/L9/L18. Command MSE is in squared adapter-command coordinates, not metres. Continuous gripper values precede physical sign formatting. Reconstruction retention is measured relative to a saved training position template and is not an estimate of intrinsic dimension.

For a candidate and baseline, relative error reduction is

`1 - mean_task_MSE(candidate) / mean_task_MSE(baseline)`.

Positive values favor the candidate. In the first-flow increment, the candidate includes first-flow plus response coordinates. A zero bootstrap denominator remains undefined. An interval containing zero is not an equivalence result.

## Reproduction boundary

This package replays archived task scores through the reported statistics. It does **not** refit PCA or affine readouts, re-extract activations, run the policy, or reproduce a simulator trajectory. The original full-archive 324-entry fitting check is a producer verification recorded separately; it is not claimed as a task of this replay entry.

The pure fitting functions are included in `producer-reference/` for inspection. They require NumPy and the original input arrays, which are not part of this task-score release. No cloud credentials or author filesystem paths are needed for the two commands above.
