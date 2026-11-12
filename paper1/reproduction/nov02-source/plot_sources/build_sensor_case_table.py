"""Render all existing pi0 Sensor pairs by outcome and own motion reference."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "behavior_case_audit"
subprocess.run([sys.executable, str(DATA / "build_case_audit.py")], check=True)
summary = json.loads((DATA / "SUMMARY.json").read_text())
assert summary["selection"]["variant_count"] == 55
assert summary["selection"]["task_count"] == 28
assert summary["selection"]["visual_pair_count"] == 110
cells = summary["all_110_pairs_relative_to_clean_resampling"]
labels = {"SS": "SS: both succeed", "SF": "SF: harm",
          "FS": "FS: recovery", "FF": "FF: both fail"}
totals = [0, 0, 0, 0]
out = [r"\par\medskip\noindent\begin{minipage}{\columnwidth}", r"\centering\small",
       r"\begin{tabular}{lrrrr}\toprule",
       r"Outcome & $d<r$ & $d=r$ & $d>r$ & Total\\\midrule"]
for group in ("SS", "SF", "FS", "FF"):
    cell = cells[group]
    assert cell["relations"]["missing"]["pair_count"] == 0
    values = [cell["relations"][rel]["pair_count"]
              for rel in ("below", "equal", "above")]
    values.append(cell["pair_count"])
    assert sum(values[:3]) == values[3]
    totals = [a + b for a, b in zip(totals, values)]
    out.append(labels[group] + " & " + " & ".join(map(str, values)) + r"\\")
assert totals == [83, 0, 27, 110]
out += [r"\midrule", "All pairs & " + " & ".join(map(str, totals)) + r"\\",
        r"\bottomrule\end{tabular}",
        r"\captionof{table}{\textbf{Early separation and paired outcomes on the same executions.} "
        r"Raw counts for all 110 $\pi_0$ Sensor visual pairs: 55 variants on 28 tasks. "
        r"Each pair's EEF-position separation $d_{vn}$ at returned control 5 is "
        r"compared with its own variant's clean-resampling distance $r_v$. "
        r"The two noise-paired executions in each variant share $r_v$. "
        r"All distances and outcomes are available; all 55 reference distances are positive.}",
        r"\label{tab:sensor-case-reference}", r"\end{minipage}\par\medskip"]
(ROOT.parent / "sections/sensor_case_reference_table.tex").write_text("\n".join(out) + "\n")
