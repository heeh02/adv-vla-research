# D1: case-level derived records → paired outcome and motion estimates

This is an executable, offline **case-level analysis pipeline** for the observed LIBERO-Plus five-arm records. It starts from 536 model–variant derived records, not from raw commands, poses, simulation or a policy checkpoint. It recomputes both models, all four families, the fixed 202-variant primary panel and the 268-variant sensitivity panel. No model, network or cloud operation is in these commands.

## Run in a clean directory

Copy this entire `case-analysis` directory to a fresh temporary directory. Run from that copy using Python 3.9 or later; the statistical pipeline and tests use only the Python standard library.

```sh
python3 -B -m unittest discover -s tests -v
python3 -B pipeline.py --output ./output
python3 -B verify_outputs.py ./output
```

The first command must pass all tests. The second validates exact input/config/plan bytes and produces 16 outcome panels, 128 motion panels and 7,520 panel-specific support records. The third independently compares the regenerated output to 156 original published-result entries, covering 1,448 scalar/interval/support checks. Tolerance is relative 1e-12, absolute 1e-14, allowing summation roundoff. It never feeds saved results back into the estimator.

The optional rendering step requires Matplotlib 3.7.2 and NumPy 1.23.5, listed in `requirements-plot.txt`. Use an interpreter with the listed optional dependencies.

```sh
python3 -B render.py ./output
```

This produces `D1-PRIMARY.svg`, `.png` and `.pdf` plus a figure-source hash receipt. Rendering consumes only `RESULTS.json` and performs no estimation.

## Frozen method and original recipe recovery

`CONFIG.json` and `CONFIG.sha256` were written before this implementation's full result/interval calculation. These are repeated analyses of previously studied data, not an independent preregistration.

The original statistical recipe is recovered in the included implementation: fixed planned support, nested equal weights, whole-task resampling and shared-variable identification bounds. Expected values are verification targets and never inputs to the estimator.

The original frozen slot plan shows that `task_id` in the plot bundle is `suite:original_task_hash`. The original code sorts **the original task hash**, not the suite-prefixed display identifier. `data/PANEL-PLAN.json` is the reduced independent original plan, matching all 536 entries. The recovered bootstrap is `random.Random(20261011)` with `choice(list(task_ids))` once per draw element, 2,000 whole-task draws. No NumPy generator or interpolated quantile is substituted. Endpoints are sorted index `int(alpha/2 * B)` and `int((1-alpha/2) * B)-1`.

Outcome draws resample the entire planned scope task list (28 primary or 40 full), then keep the drawn tasks represented in each family, preserving multiplicity. Motion draws resample the sorted **jointly eligible** tasks for that metric, clock and reference. TR, RR, their difference and their ratio share each draw. Published motion points/intervals and available original outcome points/intervals/bounds are restored; no algorithm was chosen by trying to match rounded endpoints.

## Three different denominators

1. **Planned outcomes:** 202 common-28 variants per model (51 Camera, 55 Sensor Noise, 48 Background, 48 Light). Each variant always contributes two paired noise labels with weights 1/2. Each task averages its planned variants; all planned tasks have equal weight. Unknown arm outcomes stay in this denominator.
2. **Observed outcome cells:** raw SS/SF/FS/FF counts and unknown-pair counts are descriptive counts, not replacement denominators. Different records share tasks and a clean arm; raw counts are not independent binomial samples.
3. **Matched motion support:** require measured TR0, TR1 and the chosen RR reference for the **same case, metric and clock**, including verified pose clocks. Average the two TR noise values first, then variants within eligible task, then eligible tasks. The ratio is `mean_task(TR) / mean_task(RR)`, never a mean of individual ratios. Zero reference mean stays undefined; no epsilon is added. Missing step60 is not interpolated or filled with a terminal pose.

The same-run repeat motion panel additionally requires non-null validated R0prime outcome and no R0prime issues, matching the recovered historical rule. This is a motion-eligibility restriction only; it never changes the outcome denominator.

## Unknown outcomes and uncertainty

Each case has five shared binary outcome variables: R0, T0, R1, T1, R0prime. The pipeline enumerates all assignments to its unknown variables (at most 32) and evaluates every contrast on the **same assignment**. It then averages case-level lower/upper values using the fixed hierarchy. Thus the bounds for differences such as `TR flip − R0/R1 flip` respect the shared unknown R0. Subtracting separately optimized flip bounds would be wrong.

The resulting bounds are sharp **marginal** identification bounds for each metric; extrema of different metrics need not be jointly attainable. They do not express sampling confidence. For example, both the all-success and all-failure cell can have marginal upper bound one, but they cannot both equal one on the same completion.

- Pointwise 95% whole-task intervals: identified outcomes and motion only.
- Family-adjusted 98.75% interval: identified risk difference only, following the original four-family Bonferroni rule.
- Identification bounds: unknown outcomes, no CI substituted.
- A CI is null unless the fixed-panel point is identified and all 2,000 draws are defined. Motion also needs at least two eligible tasks.

## Outputs and status meanings

- `VALIDATION.json`: exact schema/units/key/plan/outcome qualification checks.
- `RESULTS.json`: all point estimates, both interval types, bounds, per-task values and draw-index hashes.
- `OUTCOMES.csv`: all 25 outcome metrics in tidy format (fractions; multiply by 100 for percent or percentage points).
- `MOTION.csv`: 128 panels × four estimands, with physical units and supports.
- `MOTION_SUPPORT.csv`: every candidate case for every motion panel, including exclusions and source reasons.
- `KNOWN-POINT.json`: original pi0 Sensor regression, including ratio 0.5803425149195939, 55 variants and 28 tasks.
- `REGRESSION.json`: independent comparison against existing source result files.
- `RUN.json`: actual command, timestamps, elapsed time, environment, source/config/output hashes and zero model/GPU/episode counters.

`VERIFIED` means reproduced at this **derived-record analysis level**. `PARTIALLY_IDENTIFIED` records unknown outcomes. `UNDEFINED_ZERO_REFERENCE_OR_NO_SUPPORT` remains explicit; it is not replaced by zero. Input run/source hashes are preserved and syntactically validated but are not a new verification of the original simulator or neural execution.

## Expected primary results

| Model | Family | ΔSR (pp) or identification bounds | Step5 motion variants/tasks |
|---|---|---:|---:|
| pi0.5 | Camera | −20.5357 | 51/28 |
| pi0.5 | Sensor Noise | −8.03571 | 55/28 |
| pi0.5 | Background | 0 | 48/28 |
| pi0.5 | Light | [0, 8.03571] (not CI) | 43/26 |
| pi0 | Camera | −20.5357 | 51/28 |
| pi0 | Sensor Noise | −14.2857 | 55/28 |
| pi0 | Background | −3.57143 | 48/28 |
| pi0 | Light | [−2.67857, 5.35714] (not CI) | 44/27 |

Light outcomes retain all 48 planned variants in both models. Source data has 39 unknown arm outcomes across the complete 536-row bundle; no unknown is counted as failure or silently removed. See `expected/PUBLISHED-REFERENCE.json` for exact historical values and hashes.

## Tests and reproducibility scope

Tests cover strict schema/identity, duplicate keys, pairing conflicts, zero denominators, one/multiple/all unknown arms, shared-variable constraints, outcome identities, missing fixed clocks, illegal units, negative distances, distinct planned/known/motion supports, the noise→variant→task hierarchy, bootstrap multiplicities, non-interpolated quantiles and undefined draws. Synthetic fixtures are used only in tests, never appended to scientific data.

Original command vectors and poses are outside this package. It does not provide raw-trajectory or policy-acquisition replay.
