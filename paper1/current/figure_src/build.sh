#!/bin/bash
# Regenerate Figures 1-5, copy them into the LaTeX project, and compile the paper.
# Run from anywhere:  bash paper/cvpr2027/figure_src/build.sh
# Needs: python3 with matplotlib + numpy + pillow, and a TeX installation (pdflatex, bibtex).
set -e
cd "$(dirname "$0")"
mkdir -p figs figs_v3
python3 scripts/fig1.py figs/figure1
python3 scripts/fig2.py figs/figure2
python3 scripts/fig3.py && cp figs_v3/figure3_geometry.* figs/
python3 scripts/fig4.py && cp figs_v3/figure4_restoration.* figs/
python3 scripts/fig5.py figs/figure5_outcomes
for f in figure1 figure2 figure3_geometry figure4_restoration figure5_outcomes; do
  cp figs/$f.pdf ../latex/revised_figures/${f}_v3.pdf
done
cd ../latex
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null; done
pdfinfo main.pdf | grep Pages
echo "warnings: $(grep -c -E 'Overfull|undefined|multiply defined' main.log)"
for p in 8 9; do
  echo "p$p numbered lines: $(pdftotext -layout -f $p -l $p main.pdf - | grep -c -E '^ *[0-9]{3} ')"
done
