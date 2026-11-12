"""Compile both LaTeX entry points using standard TeX Live."""
from pathlib import Path
import hashlib
import shutil
import subprocess
ROOT = Path(__file__).resolve().parent

def run(args):
    subprocess.run(args, cwd=ROOT, check=True)

def state(name):
    return tuple(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
                 for p in (ROOT / (name + ext) for ext in ('.aux', '.out', '.brf')))

for name in ('main', 'supplement'):
    args = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex']
    run(args)
    run(['bibtex', name])
    for _ in range(6):
        previous = state(name)
        run(args)
        if state(name) == previous:
            break
    else:
        raise RuntimeError(name + ': references did not stabilize')
    if name == 'main':
        shutil.copy2(ROOT / 'main.pdf', ROOT / 'VLA_Attention_Supplement_Main_20261109.pdf')
