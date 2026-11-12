#!/bin/bash
# Compile the supplement. Run after build.sh, because the supplement reads the main paper's labels from main.aux.
# Run from anywhere:  bash paper/cvpr2027/figure_src/build_supplement.sh
set -e
cd "$(dirname "$0")/../latex"
[ -f main.aux ] || { echo "main.aux is missing: run build.sh first"; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex >/dev/null
bibtex supplement >/dev/null
for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error supplement.tex >/dev/null; done
pdfinfo supplement.pdf | grep Pages
echo "warnings: $(grep -c -E 'Overfull|undefined|multiply defined' supplement.log)"
