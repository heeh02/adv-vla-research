"""Render the seven main figures from frozen bundled inputs.

This editorial build deliberately does not regenerate estimates, table inputs,
bootstrap intervals, command audits or task-mean profiles.
"""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
for name in ("build_overview.py", "build_r11_results.py"):
    subprocess.run([sys.executable, str(root/name)], check=True)
