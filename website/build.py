"""Build index.html, a standalone page for GitHub Pages, from page.html.

page.html holds the head elements, then the marker <!--BODY-->, then the body.

    python3 build.py                      write index.html and .nojekyll next to this file
    python3 build.py --out ../docs        write them into another folder (assets/ is copied if it sits next to this file)
    python3 build.py --site https://<user>.github.io/<repo>/      set the address used for link previews

The site is index.html + assets/ + .nojekyll in the folder that GitHub Pages serves."""
import argparse, pathlib, shutil

SITE_URL = "https://heeh02.github.io/adv-vla-research/"
TITLE = "Sensitivity Is Not Fragility"
DESC = ("How strongly a robot policy reacts to a visual change says little about whether it fails. "
        "Shared visual responses and task outcomes in vision-language-action policies.")

here = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--site", default=SITE_URL)
ap.add_argument("--out", default=str(here))
args = ap.parse_args()
site = args.site.rstrip("/") + "/"
out = pathlib.Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)

head, body = (here / "page.html").read_text(encoding="utf-8").split("<!--BODY-->", 1)
reset = ("<style>:root{color-scheme:light}body{margin:0;font:14px system-ui,sans-serif}"
         "img{max-width:100%}[hidden]{display:none!important}</style>")
icon = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
        "%3Crect width='32' height='32' rx='7' fill='%23003366'/%3E"
        "%3Crect x='7' y='9' width='18' height='5' rx='1.5' fill='%23c9d6e6'/%3E"
        "%3Crect x='7' y='18' width='10.5' height='5' rx='1.5' fill='%23e3b95a'/%3E%3C/svg%3E")
meta = "\n".join([
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
    f'<meta name="description" content="{DESC}">',
    f'<link rel="canonical" href="{site}">',
    f'<link rel="icon" href="{icon}">',
    '<meta property="og:type" content="website">',
    f'<meta property="og:title" content="{TITLE}">',
    f'<meta property="og:description" content="{DESC}">',
    f'<meta property="og:url" content="{site}">',
    f'<meta property="og:image" content="{site}assets/figure1.png">',
    '<meta name="twitter:card" content="summary_large_image">',
])
doc = ('<!doctype html>\n<html lang="en">\n<head>\n' + meta + "\n" + reset + "\n" + head.strip()
       + "\n</head>\n<body>\n" + body.strip() + "\n</body>\n</html>\n")
(out / "index.html").write_text(doc, encoding="utf-8")
(out / ".nojekyll").write_text("", encoding="utf-8")
if out != here and (here / "assets").is_dir():
    shutil.copytree(here / "assets", out / "assets", dirs_exist_ok=True)
print("wrote", out / "index.html", len(doc), "bytes for", site)
