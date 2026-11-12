# Adv-VLA research collection

Manuscripts, scientific implementations and retained data from three studies of visual changes in vision-language-action policies. The release preserves the conclusions reached by each study, including inconclusive and negative results. The manuscripts are research drafts.

| Study | Read | Build or inspect |
|---|---|---|
| 1. Visual responses, command restoration and task outcomes | [Paper](paper1/current/latex/main.pdf) · [Supplement](paper1/current/latex/supplement.pdf) | [Figure construction](paper1/current/figure_src/scripts) · [Collection and statistical code](paper1/reproduction) |
| 2. Intervention and prediction studies | [RSS v11 paper](p2/archive/docs/experiments/paper2-independent-session-20261019-v1/manuscript/rss-revision-v11/latex/main.pdf) · [Support](p2/archive/docs/experiments/paper2-independent-session-20261019-v1/manuscript/rss-revision-v11/latex/support.pdf) · [Methodological reassessment](p2/reassessment.md) | [Implementations, tests and intermediate candidates](p2/implementation) · [Protocols](p2/protocol) |
| 3. Training continuation and paired behavior | [Manuscript](p3/archive/docs/experiments/paper3-training-study-20261019-v1/rss-draft-20261021-v1/manuscript/main.pdf) · [Final report](p3/archive/docs/experiments/paper3-training-study-20261019-v1/final-closeout-report-20261022-v1/REPORT.md) | [Training implementation](p3/training) · [Endpoint comparisons](p3/comparisons) |

**Data and checksums:** [Download v1.0.0](https://github.com/heeh02/adv-vla-research/releases/tag/v1.0.0). The release assets can be downloaded over HTTPS. Download the archives and SHA256SUMS.txt to the same directory. Run `shasum -a 256 -c SHA256SUMS.txt` to check the downloads. The [file inventory](FILES.csv) records individual paths and content hashes. Each ZIP uses its own repository-relative paths; unpack each in a separate directory.

## Start with the current Paper 1

The November 12 revision uses task-mean command error and paired outcomes for Figure 4, with simultaneous intervals from the archived contrasts. Figure 5 starts from the retained five-arm case table. Earlier, vector-coordinate-based figures and November 2/5 source packages have distinct version identities.

```sh
git clone https://github.com/heeh02/adv-vla-research.git
cd adv-vla-research
git checkout v1.0.0
bash paper1/current/figure_src/build.sh
bash paper1/current/figure_src/build_supplement.sh
```

Use Python 3.9+, NumPy, Pillow, Matplotlib, pdfLaTeX, BibTeX, Poppler, and TeX Live with the manuscript's LaTeX and font packages. The supplement uses the main document labels. The reference build produced 10 pages for the main file and 71 for the supplement. The second study's main and supporting documents compiled to 6 and 5 pages. The third draft compiled to 6 pages. For the latter two projects compile in the directory containing main.tex using pdfLaTeX, BibTeX where applicable, and repeat pdfLaTeX to resolve references.

## Replay the saved statistics

Unpack Analysis_historical_20261029_evidence-public.zip. From its E_Reusable_Scalar_Data directory:

```sh
python3 -B current-basis-replay/replay.py --out ./geometry-output
python3 -B case-analysis/pipeline.py --output ./behavior-output
python3 -B case-analysis/verify_outputs.py ./behavior-output
```

The frozen geometry/readout replay compares 7,850 values. The five-arm replay verifies 1,448 scalars and interval endpoints against 156 historical entries. The matched cohorts and original missing-outcome rules are specific to each study.

## Available evidence

- **Paper 1:** The current manuscript and the earlier source snapshot are both available. Small tables expose paired task scores, command and motion summaries, missingness, and the inputs underlying the retained statistics.
- **Paper 2:** Read the v11 manuscript together with the closeout and reassessment. The completed DEV16 and q11 single-window pilots, planned controls, technical failures and outcomes have different statuses. Source directories include dated experimental candidates. An implementation is not evidence that its campaign ran.
- **Paper 3:** The final analysis covers 4,176 retained endpoint records across model comparisons. Read the final report for changing support, matched denominators and missing endpoints. The separate join-input asset contains the retained supplemental input snapshot. The release does not supply all underlying checkpoints, episodes and training data.
- **Historical analysis:** [Scripts and records](historical-analysis) accompany the historical data supplement. D3 event-alignment and process analyses use their original eligibility rules and proxy outcomes. The historical supplement includes Python pickle files; inspect the data source before loading them with a pickle reader.

The available files support manuscript builds and selected statistical replays. Full original activations, simulator trajectories and model checkpoints from the former employer's archive are unavailable. Historical code can refer to archival runtime layouts. Machine-specific locations, private-service links and operational identifiers were normalized where identified. [Reuse terms](LICENSING.md).
