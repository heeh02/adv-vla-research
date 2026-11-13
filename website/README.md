# Project page

Source of the project page served by GitHub Pages at <https://heeh02.github.io/adv-vla-research/>.

- `page.html` is the page: head elements, the marker `<!--BODY-->`, then the body. Edit this file.
- `build.py` writes the standalone `index.html` and `.nojekyll` into the folder that GitHub Pages serves:

```sh
python3 website/build.py --out docs
```

- The five figure images live in `docs/assets/`. Replace them when the paper's figures change.

## Turning the site on

Repository Settings, Pages, Source "Deploy from a branch", branch `main`, folder `/docs`.

## Before it goes live

- The author line and the BibTeX author field are placeholders ("Author list to be added"). Replace both in `page.html`, then rebuild.
- The Paper and Supplement buttons link to `paper1/current/latex/main.pdf` and `supplement.pdf` on `main`.

## Where the numbers come from

All numbers are taken from the paper and its supplement. The page draws two charts itself: the opening comparison (π0 under Sensor Noise: 3.5 mm against 6.1 mm, 14.3 points of success lost) and the threshold plot, which uses the 110 paired runs behind Figure 5c (`paper1/current/figure_src/data/fig5d_pairs_recovered.csv`).
