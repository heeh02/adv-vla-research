"""Build the standalone English and Chinese pages for GitHub Pages.

page.html holds the shared head elements, the marker <!--BODY-->, the English
body and the shared script. page.zh-CN.html holds only the Chinese body.

    python3 build.py                      write both pages and .nojekyll next to this file
    python3 build.py --out ../docs        write them into another folder (assets/ is copied if it sits next to this file)
    python3 build.py --site https://<user>.github.io/<repo>/      set the address used for link previews

The site is index.html + zh-CN.html + assets/ + .nojekyll in the folder that
GitHub Pages serves."""
import argparse, pathlib, shutil

SITE_URL = "https://heeh02.github.io/adv-vla-research/"
TITLE = "Sensitivity Is Not Fragility"
DESC = ("How strongly a robot policy reacts to a visual change says little about whether it fails. "
        "Shared visual responses and task outcomes in vision-language-action policies.")
ZH_TITLE = "敏感不等于脆弱 | Sensitivity Is Not Fragility"
ZH_DESC = "机器人策略对视觉变化的反应强弱，并不能说明它是否会失败：视觉—语言—动作策略中的共享视觉响应与任务结果。"

here = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--site", default=SITE_URL)
ap.add_argument("--out", default=str(here))
args = ap.parse_args()
site = args.site.rstrip("/") + "/"
out = pathlib.Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)

head, body = (here / "page.html").read_text(encoding="utf-8").split("<!--BODY-->", 1)
body, script = body.split("<script>", 1)
script = "<script>" + script
zh_body = (here / "page.zh-CN.html").read_text(encoding="utf-8")
reset = ("<style>:root{color-scheme:light}body{margin:0;font:14px system-ui,sans-serif}"
         "img{max-width:100%}[hidden]{display:none!important}</style>")
icon = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
        "%3Crect width='32' height='32' rx='7' fill='%23003366'/%3E"
        "%3Crect x='7' y='9' width='18' height='5' rx='1.5' fill='%23c9d6e6'/%3E"
        "%3Crect x='7' y='18' width='10.5' height='5' rx='1.5' fill='%23e3b95a'/%3E%3C/svg%3E")


def render(body_text, lang, title, desc, url, filename):
    page_head = head.strip().replace(
        "<title>Sensitivity Is Not Fragility</title>", f"<title>{title}</title>", 1)
    meta = "\n".join([
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        f'<meta name="description" content="{desc}">',
        f'<link rel="canonical" href="{url}">',
        f'<link rel="alternate" hreflang="en" href="{site}">',
        f'<link rel="alternate" hreflang="zh-CN" href="{site}zh-CN.html">',
        f'<link rel="icon" href="{icon}">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:title" content="{title}">',
        f'<meta property="og:description" content="{desc}">',
        f'<meta property="og:url" content="{url}">',
        f'<meta property="og:image" content="{site}assets/figure1.png">',
        '<meta name="twitter:card" content="summary_large_image">',
    ])
    doc = (f'<!doctype html>\n<html lang="{lang}">\n<head>\n' + meta + "\n" + reset + "\n" + page_head
           + "\n</head>\n<body>\n" + body_text.strip() + "\n" + script.strip()
           + "\n</body>\n</html>\n")
    (out / filename).write_text(doc, encoding="utf-8")
    return len(doc.encode("utf-8"))


en_size = render(body, "en", TITLE, DESC, site, "index.html")
zh_size = render(zh_body, "zh-CN", ZH_TITLE, ZH_DESC, site + "zh-CN.html", "zh-CN.html")
(out / ".nojekyll").write_text("", encoding="utf-8")
if out != here and (here / "assets").is_dir():
    shutil.copytree(here / "assets", out / "assets", dirs_exist_ok=True)
print("wrote", out / "index.html", en_size, "bytes for", site)
print("wrote", out / "zh-CN.html", zh_size, "bytes for", site + "zh-CN.html")
