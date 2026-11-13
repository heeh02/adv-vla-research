# Sensitivity Is Not Fragility — LaTeX project

This package contains the current LOCAL-PRESENTATION-POLISH-20261109 manuscript.
Packaging only: scientific text, figures, bibliography and statistics are unchanged.

## Main files

- `main.tex`: main paper (8 scientific pages; LLM statement/references on page 9).
- `supplement.tex`: separate supplementary material (73 pages).
- Compiler: **pdfLaTeX**; bibliography: **BibTeX**.

Import this ZIP as a LaTeX project, preserving its directory structure.
Use `main.tex` as the main document. To compile the supplement, select
`supplement.tex` after compiling `main.tex` in the same project.

The supplement reads `main.aux` for cross-document references. A verified
`main.aux` and both `.bbl` files are included to support the initial import.
Recompile the main document before the supplement after changing main labels
or pagination. Do not delete `main.aux` while compiling the supplement.

`VLA_Attention_Supplement_Main_20261109.pdf` is the matching main-paper PDF
used by the supplement's existing external links. It is included to keep
those links valid without modifying the manuscript source. After editing,
refresh that PDF from the new main build if exporting linked documents.

## Local compilation

Run `python3 build_project.py` in the project directory. Standard TeX Live
with pdfLaTeX and BibTeX is required. The script compiles both documents,
refreshes the main-paper PDF link target, and writes `main.pdf` and
`supplement.pdf`.

## 中文说明

正文入口为 `main.tex`，补充材料入口为 `supplement.tex`。
请选择 pdfLaTeX + BibTeX。先编正文，再编补充；保留目录结构。
本次仅打包当前稿，没有应用评审建议，也没有替换正在其他对话设计的新图。
`AGENTS.md` 保留项目写作约定。未参与编译的章节片段和分析表为原工程保留资料。
本包已在独立解压目录编译核对；未在你的 Prism 账户中实际导入。
