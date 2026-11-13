# Project page

Source of the project page served by GitHub Pages at <https://heeh02.github.io/adv-vla-research/>.

- `page.html` contains the shared head elements, the marker `<!--BODY-->`, the English body and the shared script. Edit this file for English text, styling and interactive-chart behavior.
- `page.zh-CN.html` contains the Chinese body. Both pages use the same script and figure images; the script reads the document language for interactive labels and copy-button messages.
- The language buttons link between `index.html` and `zh-CN.html`, so language switching also works without JavaScript.
- `build.py` writes both standalone pages and `.nojekyll` into the folder that GitHub Pages serves:

```sh
python3 website/build.py --out docs
```

- The five figure images live in `docs/assets/`. Replace them when the paper's figures change; the two language versions will continue to share them.

## Turning the site on

Repository Settings, Pages, Source "Deploy from a branch", branch `main`, folder `/docs`.

## Before it goes live

- The author line and the BibTeX author field are placeholders in both language versions. Replace them in both language sources, then rebuild.
- The Paper and Supplement buttons link to `paper1/current/latex/main.pdf` and `supplement.pdf` on `main`.

## Where the numbers come from

All numbers are taken from the paper and its supplement. The page draws two charts itself: the opening comparison (π0 under Sensor Noise: 3.5 mm against 6.1 mm, 14.3 points of success lost) and the threshold plot, which uses the 110 paired runs behind Figure 5c (`paper1/current/figure_src/data/fig5d_pairs_recovered.csv`).
