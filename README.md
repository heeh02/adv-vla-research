# Sensitivity Is Not Fragility
### Shared Visual Responses and Task Outcomes in Vision-Language-Action Policies

[Paper (PDF)](paper1/current/latex/main.pdf) · [Supplement (PDF)](paper1/current/latex/supplement.pdf) · [Reproduce the figures and results](reproduce/README.md) · [Download the data](https://github.com/heeh02/adv-vla-research/releases/tag/v1.0.0)

When a robot sees the same task under a different camera view or lighting condition, how much does its response tell us about whether it will succeed? We study frozen π₀.₅ and π₀ policies on LIBERO and LIBERO-Plus. In π₀.₅, 16 shared directions capture **36.5–83.6%** of held-out, mean-centered visual response energy. Restoring clean values along these directions reduces initial translation-command error by **24%** under camera changes and **5%** under sensor noise. Paired executions show why task outcomes also matter. Under sensor noise, π₀'s mean early motion separation is only **0.58×** its clean-resampling reference, yet success drops by **14.3 percentage points**.

![Figure 1 from the paper. The left panel shows the recording sites and the intervention at layer 9. The right panel shows changes in command error and task success.](paper1/current/figure_src/figs/figure1.png)

## Check the reported numbers

From a clean checkout, run the following with Python 3.9 or later. The numerical check uses the standard library and data included in the repository.

```sh
git clone https://github.com/heeh02/adv-vla-research.git
cd adv-vla-research
git checkout paper-v1.1
python3 -B reproduce/paper.py --output paper-results.json
```

| Where in the paper | What you can check |
|---|---|
| **Figure 3 · Shared directions** | Recompute retained energy from held-out task scores for response, clean and random bases. Rebuild all 112 task rows used to compare layer depth and command prediction. |
| **Figures 1b and 4 · Restoration** | Match the six runs at each of 216 starting conditions and noise draws. Recompute translation error from the first five issued commands and check all 1,296 success labels. For Camera, mean error falls from 0.409 to 0.312, while the success change is +1.85 points with a simultaneous 95% interval of [−8.33, 11.11]. |
| **Figure 5 · Motion and outcomes** | Recompute task-weighted motion, lost and gained successes, and intervals from the five-arm case records. Under π₀ Sensor Noise, 12 of 18 lost-success pairs have less early motion separation than their own clean-resampling reference. |

The command writes intermediate estimates and matched-pair summaries to JSON. To inspect an individual case, change a grouping, rerun the broader statistical checks, or rebuild both PDFs, follow the [figure-by-figure reproduction guide](reproduce/README.md). It identifies which inputs are issued commands, episode-level observations, frozen task scores or published estimates. Full original hidden tensors, checkpoints and simulator trajectories are unavailable for a fresh collection.

## Two studies we are releasing for others to continue

We originally planned two companion papers. We did not have enough time to pursue them further. We are releasing the drafts, implementations, retained results and open questions so that others can build on the work.

- **Intervention and prediction.** The draft and reassessment document which controlled interventions were executed, the results of the single-window physical pilot and the remaining questions about task-level effects. [Draft](p2/archive/docs/experiments/paper2-independent-session-20261019-v1/manuscript/rss-revision-v11/latex/main.pdf) · [Reassessment](p2/reassessment.md) · [Code](p2/implementation)
- **Training continuation and behavior.** The final report compares continued training with the parent checkpoints on matched settings. It gives the support for each comparison and where the seeds differ. [Draft](p3/archive/docs/experiments/paper3-training-study-20261019-v1/rss-draft-20261021-v1/manuscript/main.pdf) · [Final report](p3/archive/docs/experiments/paper3-training-study-20261019-v1/final-closeout-report-20261022-v1/REPORT.md) · [Code](p3/training)

[Data inventory for the original release](FILES.csv) · [Software, data and figure licenses](LICENSING.md)
