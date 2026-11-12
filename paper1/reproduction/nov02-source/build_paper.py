"""Compile both papers, resolve cross-document references, and name deliverables.

Run `python3 build_paper.py`; add --figures to redraw the main figures.
Uses latexmk/pdfLaTeX when available, otherwise Tectonic. Figure rebuilding uses
matplotlib and numpy. All compilation inputs are included in this directory.
"""
from pathlib import Path
import argparse
import re
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--figures', action='store_true')
args = parser.parse_args()
if args.figures:
    subprocess.run([sys.executable, str(root / 'plot_sources/build_figures.py')], check=True)

logs = root / 'build_logs'
logs.mkdir(exist_ok=True)
latexmk_bin = shutil.which('latexmk')
tectonic_bin = shutil.which('tectonic')
if not latexmk_bin and not tectonic_bin:
    raise FileNotFoundError('latexmk or tectonic is required.')
# The main file supplies labels to the supplement; the supplement supplies its
# section/table labels back to the main file. The final calls stabilize both.
for index, name in enumerate(('main', 'supplement', 'main', 'supplement', 'main'), 1):
    with (logs / f'{index:02d}-{name}.log').open('w') as stream:
        cmd = ([latexmk_bin, '-pdf', '-interaction=nonstopmode',
                '-halt-on-error', f'{name}.tex'] if latexmk_bin else
               [tectonic_bin, '-k', '--keep-logs', f'{name}.tex'])
        subprocess.run(cmd, cwd=root,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)

for name, target in [('main', 'Paper1_Chapter_Refined_Main.pdf'),
                     ('supplement', 'Paper1_Chapter_Refined_Supplement.pdf')]:
    log = (root / f'{name}.log').read_text(errors='replace')
    problems = [line for line in log.splitlines()
                if 'undefined' in line or 'multiply defined' in line or 'Overfull' in line]
    if problems:
        raise RuntimeError(f'{name}: unresolved layout/reference checks: ' + '\n'.join(problems))
    shutil.copyfile(root / f'{name}.pdf', root / target)

aux = (root / 'main.aux').read_text()
match = re.search(r'\\newlabel\{page:mainend\}\{\{[^}]*\}\{(\d+)\}', aux)
assert match and int(match.group(1)) <= 8, 'Main research text exceeds eight pages.'
print(f'Build complete: {match.group(1)} research pages; references and LLM statement follow.')
print('Deliverables: Paper1_Chapter_Refined_Main.pdf and Paper1_Chapter_Refined_Supplement.pdf')
