"""Render the complete recorded CONF command decomposition without new inference."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "command_direction_audit"
subprocess.run([sys.executable, str(DATA / "build_command_direction.py")], check=True)
summary = json.loads((DATA / "SUMMARY.json").read_text())
conf = summary["CONF"]
assert conf["anchors"] == 216 and conf["original_records"] == 1296
rows = {(r["family"], r["arm"]): r for r in conf["translation_family_arm_means"]}
families = ("Camera Viewpoints", "Sensor Noise")
arms = (("Native clean", "N_C"), ("Native perturbed", "N_T"),
        ("Response", "R_resp"), ("Clean basis", "R_clean"),
        ("Random", "R_rand"), ("Full-$H$", "R_full"))
out = [r"\begin{table*}[t]", r"\centering\small",
       r"\begin{tabular}{lrrrrrr}\toprule",
       r"& \multicolumn{3}{c}{Camera Viewpoints} & \multicolumn{3}{c}{Sensor Noise}\\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
       r"Arm & Along $d_\parallel$ & Orthogonal $d_\perp$ & Error $e$"
       r" & Along $d_\parallel$ & Orthogonal $d_\perp$ & Error $e$\\\midrule"]
for i, (label, arm) in enumerate(arms):
    values = []
    for family in families:
        metrics = rows[(family, arm)]["metrics"]
        for metric in ("signed_along_clean", "orthogonal_norm", "edited_error"):
            assert metrics[metric]["status"] == "complete"
            values.append(metrics[metric]["value"])
    out.append(label + " & " + " & ".join(f"{v:.5f}" for v in values) + r"\\")
    if i == 1:
        out.append(r"\midrule")
out += [r"\bottomrule\end{tabular}",
        r"\caption{\textbf{Direction and residual error of the recorded issued-command changes.} "
        r"Descriptive task-equal means on the complete native-edit confirmation support: "
        r"96 Camera anchors (48 settings, 27 tasks) and 120 Sensor anchors "
        r"(60 settings, 30 tasks), with two noise samples per setting and all six arms. "
        r"Each scalar is computed from the first five returned translation commands, "
        r"then averaged over noises, settings within tasks, and tasks equally. "
        r"$d_\parallel,d_\perp$ are defined in Eq.~\ref{eq:issued-command-direction}; "
        r"$e$ is the distance to the paired native-clean commands. "
        r"Units are adapter translation-command units; entries are point means. "
        r"The geometric identity applies to individual records before averaging.}",
        r"\label{tab:issued-command-direction}", r"\end{table*}"]
(ROOT.parent / "sections/issued_command_direction_table.tex").write_text("\n".join(out) + "\n")
