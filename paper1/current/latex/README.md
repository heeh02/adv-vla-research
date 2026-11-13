# Sensitivity Is Not Fragility — LaTeX project

Source of the November 13 revision.

## Entry points

- `main.tex` → `main.pdf`: the paper (10 pages).
- `supplement.tex` → `supplement.pdf`: the supplement (47 pages).
- Compiler: pdfLaTeX with BibTeX. Keep the directory structure.

## Build

```sh
python3 build_project.py
```

The script compiles `main.tex` first, then `supplement.tex`. The supplement reads `main.aux` for its references to the paper, so rebuild the paper before the supplement whenever paper labels or pages change. No `.aux` or `.bbl` files are tracked; the script creates them.

`VLA_Attention_Supplement_Main_20261109.pdf` is a copy of `main.pdf`. The supplement's clickable links into the paper point to this file name, which is fixed in `supplement.tex`. `build_project.py` refreshes the copy after each paper build.

To redraw the figures before compiling, use `paper1/current/figure_src/build.sh` and `build_supplement.sh`; see the [reproduction guide](../../../reproduce/README.md).

## What is not part of the build

These files are kept from earlier rounds and are not read by either entry point:

- `revised_figures/` files without the `_v3` suffix, `current_figures/figure4.pdf`, and in `figures/` the `v715_*`, `v77_attention_link`, `v77_motion_reference`, `v77_token_structure` and `attention_spatial_full` PDFs.
- `revised_sections/setup_3_1.tex`, `setup_3_2.tex`, `setup_opening.tex` and `structured_opening.tex`.
- `analysis_archive/`: the task-ordering tables. The reproduction check reads `three_layer_task_checks.csv` from it.
- `architecture.md`: the authors' notes on the narrative of the paper's second half (Chinese).

## 中文说明

正文入口为 `main.tex`，补充材料入口为 `supplement.tex`，用 pdfLaTeX + BibTeX 编译，先编正文再编补充。运行 `python3 build_project.py` 即可。上面“What is not part of the build”列出的文件是早先轮次留下的，不参与编译。
