"""Paper overview and transfer plots; real numerical rendering is cloud CPU only."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
EXTRA_INPUTS = ("FIGURE-VALUES.json", "QA.json", "signed-figures-result.json")
FAMILIES = ("c9", "norm4", "pooled4", "current4", "pooled_full",
            "current_full", "headpos_signed", "headpos_norms", "headpos_signed_sum")
PRIMARY = ("current4", "current_full", "headpos_signed")
LABELS = ("Four", "Full", "HeadPos")
DIMS = ("Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Grip")
ROOTS = tuple(f"T{task:02d}/{parent}" for task in range(6, 10) for parent in ("C", "T"))
MODES = ("lambda1", "edf_match")
COLORS = ("#C97825", "#216A91")
STYLE = {"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
         "savefig.facecolor": "white", "figure.facecolor": "white"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def load(stage, *, synthetic=False, manifest=None):
    need(synthetic or sys.platform.startswith("linux"), "actual_panel_data_only_on_cloud_Linux")
    doc = manifest if manifest is not None else json.loads((HERE / "INPUTS.json").read_bytes())
    refs = {r["path"]: r for r in doc["input_objects"]}
    result, observed = {}, {}
    for name in EXTRA_INPUTS:
        path = Path(stage) / name
        need(path.is_file() and not path.is_symlink(), "regular_cached_JSON")
        need(path.stat().st_size <= 2 * 1024**2, "small_cached_JSON_only")
        raw = path.read_bytes()
        pin = {k: refs[name][k] for k in ("bytes", "sha256")}
        need(identity(raw) == pin, "cached_panel_input_SHA")
        result[name], observed[name] = json.loads(raw), pin
    values, qa, producer = (result[n] for n in EXTRA_INPUTS)
    need(values["synthetic"] is synthetic and qa["synthetic"] is synthetic
         and producer["synthetic"] is synthetic, "same_declared_cache_scope")
    need(qa["passed"] is True and qa["all37_predictors"] is True
         and qa["CONF_KT_rows"] == 96 and qa["DEV_rows"] == 48
         and qa["fits_retrained"] == 0, "previous_verified_numeric_cache")
    need(producer["analysis_complete"] is True and producer["render_complete"] is True
         and producer["input_qualified"] is True
         and all(producer[k] == 0 for k in ("new_model_calls", "new_env_steps", "new_fits",
                                          "training_updates", "JPFS_new_bytes")),
         "actual_completed_no_fit_producer")
    outputs = {r["path"]: r for r in producer["outputs"]}
    for name in EXTRA_INPUTS[:2]:
        need(name in outputs and {k: outputs[name][k] for k in ("bytes", "sha256")} == observed[name],
             "producer_binds_exact_scalar_cache")
    need(len(values["dimensions"]) == 7 and len(values["model_names"]) == 37,
         "seven_dimensions_thirtyseven_models")
    f1, f3 = values["main"]["F1-CAPACITY-TRANSFER"], values["main"]["F3-ROOT-COUNTEREXAMPLES"]
    for mode in MODES:
        for parent in ("C", "T"):
            arr = np.asarray(f1[f"{parent}_{mode}"], dtype=float)
            need(arr.shape == (9, 7) and not np.isinf(arr).any() and np.all(arr[np.isfinite(arr)] >= 0),
                 "complete_fixed_population_cells")
        for family in FAMILIES:
            row = f1["DEV_" + mode][family]
            arr = np.asarray(row["ratios"], dtype=float)
            need(arr.shape == (7,) and not np.isinf(arr).any() and np.all(arr[np.isfinite(arr)] >= 0),
                 "complete_train_coordinate_cells")
    for window in ("First 5", "Full 10"):
        arr = np.asarray(f3[window], dtype=float)
        need(arr.shape == (8, 7) and not np.isinf(arr).any() and np.all(arr[np.isfinite(arr)] >= 0),
             "all_eight_parent_trajectories_and_dimensions")
    return values, observed


def box(ax, x, y, w, h, text, color, fontsize=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                               facecolor=color, edgecolor="#435465", linewidth=0.85))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#183247", linespacing=1.32)


def arrow(ax, a, b, *, dashed=False, color="#435465"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                linewidth=1.0, color=color,
                                linestyle=(0, (3, 2)) if dashed else "-"))


def overview(*, synthetic=False):
    fig = plt.figure(figsize=(180 / 25.4, 55 / 25.4))
    ax = fig.add_axes([0.015, 0.035, 0.97, 0.92])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.01, 0.97, ("SYNTHETIC " if synthetic else "") + "Predict the effect, not the action",
            fontsize=10, weight="bold", ha="left", va="top", color="#183247")
    box(ax, .025, .35, .155, .33, "Recorded input s\nFixed noise ξ\nPaired reference", "#EDF2F4", 8.2)
    box(ax, .235, .35, .18, .33, "Baseline at j5\nP, gate and v5\nContent increment E", "#E7F0F7", 8.2)
    box(ax, .485, .65, .215, .23, "Reading features\nSigned / head-position\nor compressed", "#DCEBF5", 8)
    box(ax, .785, .65, .18, .23, "Predictor → ŷ\nTrained on DEV", "#DCEBF5", 8.4)
    box(ax, .485, .14, .215, .25, "Controlled continuation\nSame s and ξ\nNone vs. edited", "#FAEDDF", 8)
    box(ax, .785, .12, .18, .29, "Candidate chunks\nA(E) and A(0)\ny = A(E) − A(0)", "#FAEDDF", 8.2)
    arrow(ax, (.185, .515), (.22, .515))
    arrow(ax, (.425, .59), (.473, .735))
    arrow(ax, (.425, .44), (.473, .27))
    arrow(ax, (.713, .765), (.773, .765))
    arrow(ax, (.713, .265), (.773, .265))
    arrow(ax, (.88, .635), (.88, .425), dashed=True)
    ax.text(.898, .53, "Compare", ha="left", va="center", fontsize=7.5)
    ax.plot([.47, .98], [.51, .51], color="#AEBBC4", linewidth=.8, linestyle=(0, (2, 3)))
    ax.text(.49, .55, "Before the edited output", ha="left", va="bottom", fontsize=7.5, color="#216A91")
    ax.text(.49, .47, "Later outputs are targets, not features", ha="left", va="top",
            fontsize=7.5, color="#9F5A24")
    ax.text(.025, .035, "Signed 10 × 7 candidate-command change; executed motion is a separate test.",
            fontsize=7.5, color="#435465", ha="left", va="bottom")
    return fig


def scatter_panel(ax, f1, *, train, records):
    vals = []
    for mode in MODES:
        for family in PRIMARY:
            if train:
                vals.extend(f1["DEV_" + mode][family]["ratios"])
            else:
                for parent in ("C", "T"):
                    vals.extend(f1[parent + "_" + mode][FAMILIES.index(family)])
    positive = [float(x) for x in vals if x is not None and np.isfinite(x) and x > 0]
    lower = 10 ** np.floor(np.log10(min(positive)) - .12) if positive else .1
    upper = 10 ** np.ceil(np.log10(max(max(positive), 1)) + .05) if positive else 10
    # Zero is explicitly marked below the positive log range, never made epsilon.
    zero_y = lower / 2
    for mi, mode in enumerate(MODES):
        shift = -.17 if mi == 0 else .17
        for fi, family in enumerate(PRIMARY):
            parents = (None,) if train else ("C", "T")
            for pi, parent in enumerate(parents):
                raw = f1["DEV_" + mode][family]["ratios"] if train else f1[parent + "_" + mode][FAMILIES.index(family)]
                for di, value in enumerate(raw):
                    jitter = (di - 3) * .027 + ((pi - .5) * .07 if not train else 0)
                    defined = value is not None and np.isfinite(value)
                    y = float(value) if defined and value > 0 else zero_y
                    if defined:
                        ax.scatter(fi + shift + jitter, y, s=13, linewidths=.7,
                                   marker="o" if value > 0 else "v", zorder=3,
                                   facecolors="none" if mi == 0 else COLORS[mi],
                                   edgecolors=COLORS[mi], alpha=.9)
                    else:
                        ax.scatter(fi + shift + jitter, y, s=13, linewidths=.7,
                                   marker="x", color="#777777", zorder=3)
                    records.append({"split": "DEV" if train else "CONF_KT",
                                    "family": family, "mode": mode, "parent": parent,
                                    "command_dimension": di, "relative_MSE": float(value) if defined else None,
                                    "zero_marker": bool(defined and value == 0),
                                    "undefined_marker": not defined})
    ax.set_yscale("log")
    ax.set_ylim(zero_y * .7 if any(x is None or not np.isfinite(x) or x == 0 for x in vals) else lower, upper)
    ylo, yhi = ax.get_ylim()
    powers = range(int(np.ceil(np.log10(ylo))), int(np.floor(np.log10(yhi))) + 1)
    ticks = [10.**power for power in powers]
    ax.set_yticks(ticks, labels=[f"{value:g}" for value in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1, color="#515D63", linewidth=.8, linestyle="--")
    ax.set_xticks(range(3), LABELS)
    ax.tick_params(axis="both", labelsize=7.5, length=2)
    ax.set_xlim(-.55, 2.55)
    ax.grid(axis="y", which="major", color="#DFE5E8", linewidth=.55, zorder=0)
    ax.set_ylabel("MSE / zero-effect MSE", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return {"points": len(vals), "zero_values": sum(x == 0 for x in vals),
            "undefined_values": sum(x is None or not np.isfinite(x) for x in vals),
            "log_axis_limits": list(ax.get_ylim()), "cross_coordinate_mean_computed": False}


def transfer(values, *, synthetic=False):
    fig = plt.figure(figsize=(180 / 25.4, 86 / 25.4))
    left = fig.add_axes([.075, .24, .205, .58])
    middle = fig.add_axes([.355, .24, .205, .58])
    grid = fig.add_axes([.69, .30, .275, .46])
    records = []
    f1, f3 = values["main"]["F1-CAPACITY-TRANSFER"], values["main"]["F3-ROOT-COUNTEREXAMPLES"]
    a = scatter_panel(left, f1, train=True, records=records)
    b = scatter_panel(middle, f1, train=False, records=records)
    if synthetic:
        fig.text(.075, .98, "SYNTHETIC", fontsize=7.5, color="#A54628", va="top")
    fig.text(.075, .9, "A  Development fit", fontsize=8.5, weight="bold")
    fig.text(.355, .9, "B  Across-task evaluation", fontsize=8.5, weight="bold")
    fig.text(.66, .9, "C  All parent trajectories", fontsize=8.5, weight="bold")
    matrix = np.concatenate([np.asarray(f3[w], dtype=float) for w in ("First 5", "Full 10")], axis=1)
    positive = matrix[matrix > 0]
    lo = min(-1., float(np.floor(np.log2(positive.min())))) if positive.size else -1.
    hi = max(1., float(np.ceil(np.log2(positive.max())))) if positive.size else 1.
    plotted = np.where(matrix > 0, np.log2(np.maximum(matrix, np.finfo(float).tiny)), lo)
    plotted[~np.isfinite(matrix)] = np.nan
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#DDDDDD")
    im = grid.imshow(np.ma.masked_invalid(plotted), aspect="auto", interpolation="nearest", cmap=cmap,
                     norm=TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi))
    grid.set_xticks(range(14), DIMS + DIMS, rotation=90)
    grid.set_yticks(range(8), ROOTS)
    grid.tick_params(axis="both", labelsize=7.5, length=0, pad=2)
    grid.axvline(6.5, color="white", linewidth=1.5)
    grid.set_xticks(np.arange(-.5, 14, 1), minor=True)
    grid.set_yticks(np.arange(-.5, 8, 1), minor=True)
    grid.grid(which="minor", color="white", linewidth=.35)
    grid.tick_params(which="minor", length=0)
    grid.text(.25, 1.04, "First 5", transform=grid.transAxes, ha="center", fontsize=7.5)
    grid.text(.75, 1.04, "Full 10", transform=grid.transAxes, ha="center", fontsize=7.5)
    for i, j in np.argwhere(matrix == 0):
        grid.text(j, i, "0", ha="center", va="center", fontsize=7.5)
    for i, j in np.argwhere(~np.isfinite(matrix)):
        grid.text(j, i, "NA", ha="center", va="center", fontsize=7.5)
    bar_ax = fig.add_axes([.69, .13, .275, .03])
    bar = fig.colorbar(im, cax=bar_ax, orientation="horizontal")
    ticks = np.arange(lo, hi + 1)
    bar.set_ticks(ticks, labels=[f"{2**t:g}" for t in ticks])
    bar.ax.tick_params(labelsize=7.5, length=2)
    bar.set_label("Full, matched EDF: MSE / zero MSE", fontsize=7.5, labelpad=2)
    handles = [
        plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=COLORS[0],
                   linestyle="none", markersize=4, label="Common penalty = 1"),
        plt.Line2D([], [], marker="o", markerfacecolor=COLORS[1], markeredgecolor=COLORS[1],
                   linestyle="none", markersize=4, label="Matched training EDF"),
    ]
    if any(row["undefined_marker"] for row in records):
        handles.append(plt.Line2D([], [], marker="x", color="#777777", linestyle="none",
                                  markersize=4, label="Undefined (below log range)"))
    if any(row["zero_marker"] for row in records):
        handles.append(plt.Line2D([], [], marker="v", color="#777777", linestyle="none",
                                  markersize=4, label="Zero (below log range)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.065, .015),
               frameon=False, fontsize=7.5, handletextpad=.4, labelspacing=.5,
               ncol=2, columnspacing=1.)
    return fig, {"scatter_points": records, "training_display": a, "evaluation_display": b,
                 "root_matrix": [[float(x) if np.isfinite(x) else None for x in row] for row in matrix],
                 "root_order": ROOTS, "root_undefined_count": int(np.count_nonzero(~np.isfinite(matrix))),
                 "matrix_zero_count": int(np.count_nonzero(matrix == 0)),
                 "root_cells": int(matrix.size), "root_color_log2_limits": [lo, hi],
                 "not_independent_tests": True}


def save(fig, out, name):
    fig.canvas.draw()
    renderer, bounds = fig.canvas.get_renderer(), fig.bbox
    problems, smallest = [], 100.
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        smallest = min(smallest, float(text.get_fontsize()))
        bb = text.get_window_extent(renderer)
        if bb.x0 < bounds.x0 - 1 or bb.y0 < bounds.y0 - 1 or bb.x1 > bounds.x1 + 1 or bb.y1 > bounds.y1 + 1:
            problems.append(text.get_text())
    need(smallest >= 7.5 and not problems, "paper_panel_text_outside_or_too_small:" + str(problems))
    names = []
    for extension in ("pdf", "svg", "png"):
        filename = name + "." + extension
        fig.savefig(Path(out) / filename, dpi=300)
        names.append(filename)
    result = {"name": name, "width_mm": float(fig.get_figwidth() * 25.4),
              "height_mm": float(fig.get_figheight() * 25.4), "minimum_font_points": smallest,
              "text_bbox_check": "PASS", "PNG_dpi": 300, "outputs": names}
    plt.close(fig)
    return result


def render(stage, out, check, *, synthetic=False, manifest=None):
    values, refs = load(stage, synthetic=synthetic, manifest=manifest)
    check()
    with plt.rc_context(STYLE):
        layouts = [save(overview(synthetic=synthetic), out, "F1-OVERVIEW")]
        fig, data = transfer(values, synthetic=synthetic)
        layouts.append(save(fig, out, "F2-CAPACITY-TRANSFER"))
    need(len(data["scatter_points"]) == 126 and data["root_cells"] == 112, "all_declared_paper_points")
    report = {"schema": "paper2-paper-panel-values-v1", "synthetic": synthetic,
              "source_cache_refs": refs, "F1": {"conceptual_only": True, "data_or_model_calls": 0},
              "F2": data, "unselected_families_preserved_in_original_cache": list(FAMILIES),
              "F2_train_and_population_window": "full10",
              "F2_all_panels_scale": "frozen_DEV_per_position_sigma_before_MSE",
              "F2_shift_scope": "archived_task_Camera_and_visited_state_joint_change",
              "old_all37_all54_support_preserved": True, "new_fits": 0}
    qa = {"schema": "paper2-paper-panel-QA-v1", "synthetic": synthetic,
          "passed": True, "source_cache_refs": refs, "scatter_points": 126,
          "all_eight_roots_both_windows_coordinates": 112,
          "F2_train_and_population_window": "full10",
          "F2_all_panels_scale": "frozen_DEV_per_position_sigma_before_MSE",
          "independent_original_array_recomputation": False,
          "previous_score_QA_inherited_not_rerun": True, "new_model_calls": 0,
          "new_env_steps": 0, "new_fits": 0, "layouts": layouts,
          "main_RSS_uses_300dpi_PNG": True,
          "vector_PDF_uses_embedded_Type42_and_is_separate_editable_artifact": True}
    for name, value in (("PAPER-PANEL-VALUES.json", report), ("PAPER-PANEL-QA.json", qa)):
        with (Path(out) / name).open("xb") as stream:
            stream.write(encoded(value))
    check()
    return {"outputs": [n for row in layouts for n in row["outputs"]]
            + ["PAPER-PANEL-VALUES.json", "PAPER-PANEL-QA.json"],
            "layouts": layouts, "numerical_QA": qa}
