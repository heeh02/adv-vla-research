# Reproducing Sensitivity Is Not Fragility

Use the [main paper](../paper1/current/latex/main.pdf) and [supplement](../paper1/current/latex/supplement.pdf) in this checkout. The older November 2 manuscript in the data release uses a different figure numbering. Run all commands from the repository root.

## One command for the main claims

```sh
python3 -B reproduce/paper.py --output paper-results.json
```

On success this prints the Figure 3 retention range, the Figure 4 error and success contrasts for each family, and the Figure 5 Sensor Noise comparison. The JSON contains the estimates across all tested families and arms. A mismatched checksum, support size, command-derived task value, outcome label or archived contrast raises an error. The script uses a scratch directory to reconstruct the task table; input files stay intact.

### Figures 1 and 2 and Table 1 · What was measured

Figure 1a locates recording after action-expert blocks L1/L9/L18 and restoration at L9. Figure 2 shows a query, generation updates, the six-arm intervention and the independent five-arm execution study. Table 1 identifies the study populations. Geometry and readout share 404 internal pairs. The six-arm and five-arm populations have distinct starting-case keys. Figure 1b visualizes rounded restoration summaries; use the Figure 4 calculation below to verify them. [Architecture drawing](../paper1/current/figure_src/scripts/fig1.py) · [Protocol diagram](../paper1/current/figure_src/scripts/fig2.py) · [Study definitions](../paper1/current/latex/revised_sections/setup.tex)

### Figure 3 · How much response energy transfers?

Start with the [task score inputs and their provenance](inputs/). For each task, the verifier computes `1 − SSE / residual_energy`, checks the stored retention, then averages tasks equally within family, layer and basis. The test response supplies its own projection coefficients. The 36 groups cover three bases, three layers and four families, each with 28 tasks. The verifier also rebuilds the 112 task-by-family rows from the translation readout and geometry tables and compares the CSV bytes with the manuscript input. The family-mean prediction error ratio divides the mean fitted error by the mean no-change error.

The released task scores start after the original fitting step. To inspect the estimator and original draw matrices, use the larger replay below. To inspect rendering, Figure 3 uses task scores for the response points and bootstrap intervals; some control markers and comparison values in the plotting CSV were recovered from earlier artwork or read from a supplementary table. In the JSON, all three retention ranges come directly from the released task scores. [Task-table builder](../paper1/reproduction/rebuild_task_ordering.py) · [Published task rows](../paper1/current/latex/analysis_archive/section4-task-ordering/three_layer_task_checks.csv) · [Plot-data builder](../paper1/current/figure_src/scripts/build_fig3_data.py) · [Figure 3 renderer](../paper1/current/figure_src/scripts/fig3.py)

### Figure 4 and Table 2 · What happens when clean coordinates are restored?

Open the gzipped [command records](inputs/CASE-BEHAVIOR.jsonl.gz). Each line holds one arm, its initial-state and noise identifiers, success, and the first five seven-dimensional commands. Match all six arms by family, task, setting, initialization and noise. Compare the three translation coordinates at each of the five positions with the native clean command in that same group. Take the Euclidean norm over the resulting 15 differences. Average noise draws within setting, settings within task, then tasks within family.

The verifier compares these task errors with the figure input at absolute tolerance 1e-9, and each success flag with the outcome CSV. It then runs the manuscript's comparison code. The simultaneous intervals for the six planned command and success contrasts are read from the archived contrast table; the quick check recomputes the point estimates. Table 2 also reports edit sizes from the archived intervention study. Those sizes are outside the quick check. Command recovery and success are separate endpoints. [Task, outcome and contrast tables](../paper1/current/figure_src/raw/restoration) · [Figure 4 estimator](../paper1/current/figure_src/scripts/fig4_data.py) · [Intervention implementation](../paper1/reproduction/acquisition/cfr-worker)

### Figure 5 and Table 3 · Which executions change outcome?

The estimator starts from the [five-arm case table](../paper1/current/figure_src/raw/Five_arm_cases_flat.csv) and takes the common 28-task panel, 202 settings per policy. Each setting has two clean–perturbed pairs matched on policy noise. For motion it retains common support for both pairs and the clean–clean reference. The plotted family ratio divides the task-weighted mean perturbed separation by the corresponding mean clean-resampling separation. Panel c uses each setting's own reference.

Success→failure contributes to lost successes; failure→success contributes to gains. Risk difference is gains minus losses. The estimator carries missing Light outcomes in the planned denominator and compares recorded counts with the archived bounds. It uses 2,000 whole-task bootstrap draws and checks the motion and success estimates against their released tables. The JSON exposes the eight policy × family panels and all 110 π₀ Sensor Noise pairs. The quick check covers the motion and paired outcome contrasts used in Table 3. For related quantities, run the authors' scripts:

```sh
python3 -B paper1/current/figure_src/scripts/fig5_data.py
python3 -B paper1/current/figure_src/scripts/review_checks2.py
python3 -B paper1/current/figure_src/scripts/review_checks3.py
```

The last script compares discrimination pooled across families and within families, and reports post-hoc pointwise restoration intervals. The main restoration test uses its own prespecified simultaneous intervals. [Five-arm protocol](../paper1/current/figure_src/raw/Five_arm_protocol.md) · [Estimator](../paper1/current/figure_src/scripts/fig5_data.py) · [Table 3 as printed](../paper1/current/latex/revised_sections/float_outcome_table.tex)

## Redraw the paper and compile both PDFs

Use Python with NumPy, Pillow and Matplotlib, pdfLaTeX/BibTeX, Poppler, and TeX Live with the manuscript's extra LaTeX, font and science packages. The validated rendering used NumPy 1.23.5, Matplotlib 3.7.2, Pillow 11.3.0 and TeX Live 2026. Run in a clean checkout because the figure build writes into the manuscript project.

```sh
bash paper1/current/figure_src/build.sh
bash paper1/current/figure_src/build_supplement.sh
```

Outputs are `paper1/current/latex/main.pdf` (10 pages) and `paper1/current/latex/supplement.pdf` (47 pages). The supplement uses the main document's labels. PDF metadata and font metrics may vary by toolchain.

## Replay the archived statistics

Download [Analysis_historical_20261029_evidence-public.zip](https://github.com/heeh02/adv-vla-research/releases/download/v1.0.0/Analysis_historical_20261029_evidence-public.zip) and compare its SHA-256 with [the release checksums](https://github.com/heeh02/adv-vla-research/releases/download/v1.0.0/SHA256SUMS.txt). Then:

```sh
unzip -q Analysis_historical_20261029_evidence-public.zip -d retained-analysis
cd retained-analysis/docs/exports/paper1-cloud-analysis-20261029/AdvVLA_Paper1_Cloud_Analysis_20261029/03_Experiment_Evidence/E_Reusable_Scalar_Data
python3 -B current-basis-replay/replay.py --out geometry-output
python3 -B case-analysis/pipeline.py --output behavior-output
python3 -B case-analysis/verify_outputs.py behavior-output
```

The archived checks cover 7,850 geometry/readout values and 1,448 behavior scalars or interval endpoints. The underlying recorders, readout implementations and export logic are [available for inspection](../paper1/reproduction). The release contains task scores, issued command prefixes, paired outcomes and derived motion records. Re-fitting from hidden tensors or independently replaying the simulator requires original inputs beyond this release. If you find a discrepancy, include the commit, figure or table, command, input checksum, and observed and expected values in a [repository issue](https://github.com/heeh02/adv-vla-research/issues).
