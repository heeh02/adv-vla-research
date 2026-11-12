"""Render the verified development controls as descriptive plan comparisons."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dev_controls"
subprocess.run([sys.executable, str(DATA / "build_dev_controls.py")], check=True)
summary = json.loads((DATA / "dev_controls_summary.json").read_text())
families = ["Camera Viewpoints", "Sensor Noise"]
lookup = {
    family: {(r["arm"], r["schedule_role"]): r
             for r in summary["families"][family]["condition_means"]}
    for family in families
}
rows = [
    ("Native clean", "N_C", "all_calls"),
    ("Native perturbed", "N_T", "all_calls"),
    ("Response, first", "R_resp", "first_call"),
    ("Response, all", "R_resp", "all_calls"),
    ("Clean, all", "R_clean", "all_calls"),
    ("Random, all", "R_rand", "all_calls"),
    ("Full-$H$, all", "R_full", "all_calls"),
    ("Common: response", "eq_B", "all_calls"),
    ("Common: clean", "eq_C", "all_calls"),
    ("Common: random", "eq_G", "all_calls"),
]
out = [r"\begin{table}[t]", r"\centering\small",
       r"\begin{tabular}{lrr}\toprule",
       r"Rule & Camera / $S_f$ & Sensor / $S_f$\\\midrule"]
for i, (label, arm, schedule) in enumerate(rows):
    values = [lookup[f][(arm, schedule)]["mean_normalized_translation_error_improvement"]
              for f in families]
    out.append(label + " & " + " & ".join(f"{v:.5f}" for v in values) + r"\\")
    if i in (1, 6):
        out.append(r"\midrule")
out += [r"\bottomrule\end{tabular}",
        r"\caption{\textbf{Development plan gains depend on schedule and write rule.} "
        r"Descriptive task-equal means of Eq.~\ref{eq:dev-plan-gain} on all 64 Camera "
        r"and 136 Sensor anchors: 8/17 variants on 5/9 tasks, two noise samples and "
        r"saved queries $q=0,2,5,10$. First/all denote $j=0$ and $j=0,\ldots,9$. "
        r"Common uses Eq.~\ref{eq:dev-common-target} at each arm's current state. "
        r"Native clean supplies the target; native perturbed is the zero-gain reference.}",
        r"\label{tab:dev-restoration-controls}", r"\end{table}"]
(ROOT.parent / "sections/dev_restoration_controls_table.tex").write_text("\n".join(out) + "\n")
