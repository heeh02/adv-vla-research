"""Predeclared mass-granularity figure from a verified small result review.

No ndarray/NPZ reader, research aggregation, fit, network or cloud creation.
Actual data rendering is restricted to an explicitly identified Linux cloud job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASELINES = ("global_window_prefix_mass", "per_position_prefix_mass",
             "per_head_position_prefix_mass", "modality_mass")
LABELS = ("Window mass", "Position mass", "Head-position mass", "Modality mass")
COLORS = ("#496899", "#29A1A8", "#D78B25", "#BD4460")
MARKERS = ("o", "s", "D", "^")
WINDOWS = ("full10", "first5", "tail5")
WINDOW_LABELS = {"full10": "All 10 candidate positions", "first5": "First 5 candidate positions",
                 "tail5": "Last 5 candidate positions"}
VIEWS = ("full", "response_B3", "response_residual", "clean_B3", "clean_residual")
TASK_COLORS = ("#3F658D", "#846497", "#497C64", "#9E7044")
MAX_REVIEW_BYTES = 512 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def need(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def finite_nonnegative(value, label):
    need(value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                          and math.isfinite(value) and value >= 0), label)


def ratio(sse, target):
    """Display division of certified absolute values, never re-aggregation."""
    if sse is None or target is None or target <= 0:
        return None
    value = sse / target
    need(math.isfinite(value), "finite_display_ratio_without_clipping_positive_target")
    return value


def _check_row(row):
    need(row["view"] in VIEWS and row["window"] in WINDOWS, "fixed_view_window")
    for field in ("target_delta_energy", "full_target_delta_energy"):
        finite_nonnegative(row[field], "absolute_energy_nonnegative_or_explicit_null")
    need(set(row["baselines"]) == set(BASELINES), "four_predeclared_mass_baselines")
    for baseline in BASELINES:
        value = row["baselines"][baseline]
        finite_nonnegative(value["SSE"], "SSE_nonnegative_or_explicit_null")
        need(isinstance(value["status"], str) and value["status"], "baseline_status_required")


def validate_review(review, *, allow_real_cloud=False, cloud_job_id=None):
    """Fail before numerical use if actual input is submitted on a Mac."""
    need(review.get("schema") == "PAPER2-result-review-v1", "review_schema")
    synthetic = review.get("synthetic")
    need(type(synthetic) is bool, "explicit_synthetic_identity")
    if not synthetic:
        need(sys.platform.startswith("linux") and allow_real_cloud and
             isinstance(cloud_job_id, str) and bool(cloud_job_id.strip()),
             "actual_figures_require_explicit_Linux_cloud_job_never_Mac")
    need(review.get("accepted") is True and review.get("status") == "ACCEPTED_COMPLETE"
         and review.get("failures") == [], "accepted_complete_review_before_render")
    source_files = review.get("source_files")
    need(isinstance(source_files, list) and len(source_files) > 0, "verified_source_file_identities")
    for ref in source_files:
        need(isinstance(ref.get("path"), str) and ref["path"] and
             isinstance(ref.get("bytes"), int) and ref["bytes"] > 0 and
             isinstance(ref.get("sha256"), str) and len(ref["sha256"]) == 64 and
             all(c in "0123456789abcdef" for c in ref["sha256"]), "source_SHA256_bytes_identity")
    data = review.get("figure_data")
    need(isinstance(data, dict) and data.get("schema") == "PAPER2-mass-baseline-figure-data-v1"
         and data.get("synthetic") is synthetic, "figure_data_schema_synthetic_agreement")
    roots = data["planned_roots"]
    need(len(roots) == 8 and len({r["root_id"] for r in roots}) == 8, "all_eight_roots_fixed_order")
    tasks = list(dict.fromkeys(r["task_id"] for r in roots))
    need(len(tasks) == 4 and all(sum(r["task_id"] == t for r in roots) == 2 for t in tasks), "four_tasks_two_roots_each")
    for r in roots:
        need(r["query_indices"] == [11, 16, 22] and r["parent_condition"] in ("C", "T", "PC", "PT"),
             "three_original_queries_and_parent_identity")
    root_by_id = {r["root_id"]: r for r in roots}
    indexes = {}
    for kind, identities in (("root", list(root_by_id)), ("task", tasks), ("population", ["all"])):
        rows = data[kind + "_rows"]
        expected = {(i, view, window) for i in identities for view in VIEWS for window in WINDOWS}
        index = {}
        for r in rows:
            _check_row(r)
            key = (r["aggregation_id"], r["view"], r["window"])
            need(key not in index and key in expected, "unique_planned_" + kind + "_row")
            if kind == "root":
                need(r["task_id"] == root_by_id[key[0]]["task_id"], "root_task_identity")
            index[key] = r
        need(set(index) == expected, "complete_" + kind + "_slots_including_null_rows")
        for entity in identities:
            for window in WINDOWS:
                full = index[(entity, "full", window)]
                target = full["target_delta_energy"]
                for view in VIEWS:
                    current = index[(entity, view, window)]
                    need(current["full_target_delta_energy"] == target, "same_full_denominator_across_views")
                for basis in ("response", "clean"):
                    parts = [index[(entity, basis + "_" + p, window)] for p in ("B3", "residual")]
                    for baseline in BASELINES:
                        values = [r["baselines"][baseline]["SSE"] for r in [full] + parts]
                        if all(v is not None for v in values):
                            error = values[0] - values[1] - values[2]
                            need(abs(error) <= 1e-10 * max(1., sum(abs(v) for v in values)),
                                 "same_full_denominator_B3_residual_SSE_closure")
        indexes[kind] = index
    expected_queries = {(r["root_id"], q) for r in roots for q in r["query_indices"]}
    query_index = {}
    for row in data["query_rows"]:
        _check_row(row)
        key = (row["root_id"], row["query_index"])
        need(key in expected_queries and key not in query_index and row["view"] == "full"
             and row["window"] == "full10", "all_24_primary_query_slots")
        need(row["task_id"] == root_by_id[key[0]]["task_id"], "query_task_identity")
        need(row["target_delta_energy"] == row["full_target_delta_energy"], "query_full_denominator")
        query_index[key] = row
    need(set(query_index) == expected_queries, "all_24_queries_present_or_explicit_null")
    indexes["query"] = query_index
    return data, indexes, tasks


def display_values(data, indexes):
    """Explicit numerical drawing table; retains missing/zero and absolute values."""
    points, differences, parts, queries, unavailable = [], [], [], [], []
    for window in WINDOWS:
        for ri, root in enumerate(data["planned_roots"]):
            r = indexes["root"][(root["root_id"], "full", window)]
            target = r["full_target_delta_energy"]
            for name in BASELINES:
                b = r["baselines"][name]
                value = ratio(b["SSE"], target)
                rec = {"root_id": root["root_id"], "task_id": root["task_id"], "root_index": ri,
                       "window": window, "baseline": name, "SSE": b["SSE"], "target": target,
                       "ratio": value, "source_status": b["status"]}
                points.append(rec)
                if value is None:
                    unavailable.append({"panel": "A", **rec,
                                        "reason": "ZERO_TARGET" if target == 0 else "MISSING_TARGET" if target is None else "MISSING_SSE"})
            h, m = (r["baselines"][n]["SSE"] for n in BASELINES[2:])
            diff = None if h is None or m is None else h-m
            differences.append({"root_id": root["root_id"], "task_id": root["task_id"], "root_index": ri,
                                "window": window, "head_SSE": h, "modality_SSE": m,
                                "absolute_difference": diff, "target": target,
                                "normalized_difference": ratio(diff, target)})
    for basis in ("response", "clean"):
        for baseline in BASELINES[2:]:
            full = indexes["population"][("all", "full", "full10")]
            record = {"basis": basis, "baseline": baseline, "window": "full10",
                      "target": full["full_target_delta_energy"], "full_SSE": full["baselines"][baseline]["SSE"]}
            for part in ("B3", "residual"):
                row = indexes["population"][("all", basis+"_"+part, "full10")]
                record[part+"_SSE"] = row["baselines"][baseline]["SSE"]
                record[part+"_ratio"] = ratio(record[part+"_SSE"], record["target"])
                record[part+"_status"] = row["baselines"][baseline]["status"]
            parts.append(record)
    for root in data["planned_roots"]:
        for q in root["query_indices"]:
            row = indexes["query"][(root["root_id"], q)]
            h, m = (row["baselines"][name]["SSE"] for name in BASELINES[2:])
            delta = None if h is None or m is None else h-m
            queries.append({"root_id": root["root_id"], "task_id": root["task_id"], "query_index": q,
                            "target": row["full_target_delta_energy"],
                            "absolute_difference": delta,
                            "normalized_difference": ratio(delta, row["full_target_delta_energy"]),
                            "baselines": {name: {**row["baselines"][name],
                                                 "ratio": ratio(row["baselines"][name]["SSE"], row["full_target_delta_energy"])}
                                          for name in BASELINES}})
    return {"schema": "PAPER2-mass-figure-drawing-values-v1", "synthetic": data["synthetic"],
            "panel_A": points, "panel_B": differences, "panel_C": parts,
            "queries": queries, "unavailable_root_comparisons": unavailable,
            "source_order_preserved": True, "new_aggregation": False, "jitter": False,
            "normalization": "all ratios use the same full target for that entity and window"}


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})


def task_label(task_id):
    """Short display only; all tables retain the full original task identity."""
    name = str(task_id)
    return name.removeprefix("CCONF-") if name.startswith("CCONF-T") else name


def _root_axis(ax, roots, tasks):
    ax.set_xlim(-.5, 7.5)
    ax.set_xticks(range(8))
    ax.set_xticklabels([task_label(r["task_id"]) + "\n" + r["parent_condition"] for r in roots])
    for x, root in enumerate(roots):
        ti = tasks.index(root["task_id"])
        if ti % 2 == 0:
            ax.axvspan(x-.5, x+.5, color="#F2F4F7", zorder=0)
        ax.get_xticklabels()[x].set_color(TASK_COLORS[ti])
    ax.grid(axis="y", color="#E5E8ED", linewidth=.5, zorder=0)


def _tag(ax, x, reason):
    ax.text(x, 1.015, "0E" if reason == "ZERO_TARGET" else "NA", ha="center", va="bottom",
            fontsize=6.5, color="#92333B", transform=ax.get_xaxis_transform())


def _finish(fig, synthetic, subtitle):
    if synthetic:
        fig.suptitle("SYNTHETIC SOFTWARE PREVIEW - NOT RESEARCH RESULTS", x=.06, y=.99,
                     ha="left", fontsize=13, weight="bold", color="#9D2D44")
        fig.text(.55, .53, "SYNTHETIC", fontsize=56, rotation=22, color="#9D2D44", alpha=.075,
                 ha="center", va="center", zorder=0)
    else:
        fig.suptitle("How much attention-mass detail is needed to reconstruct the read?",
                     x=.06, y=.99, ha="left", fontsize=13, weight="bold")
    fig.text(.06, .952, subtitle, fontsize=9, color="#374151")
    fig.text(.06, .026,
             "Late mass is observed: these are conditional reconstructions with different information / degrees of freedom.\n"
             "Post-hoc seen CONF; 8 roots / 4 tasks / 24 queries. No prediction, native W, action, or behavior claim.\n"
             "0E = zero full target; NA = missing value. Slots remain; absolute values and reasons are in FIGURE-VALUES.json.",
             ha="left", va="bottom", fontsize=8, color="#4B5563", linespacing=1.35)


def build_main(data, indexes, values, tasks):
    _style()
    fig = plt.figure(figsize=(13.6, 9.1))
    grid = fig.add_gridspec(2, 3, left=.065, right=.975, top=.865, bottom=.18,
                            hspace=.56, wspace=.25, height_ratios=(1.0, 1.05))
    finite_a = [p["ratio"] for p in values["panel_A"] if p["ratio"] is not None]
    ymax = 1.14 * max([1.] + finite_a)
    roots = data["planned_roots"]
    for wi, window in enumerate(WINDOWS):
        ax = fig.add_subplot(grid[0, wi])
        _root_axis(ax, roots, tasks)
        ax.set_ylim(0, ymax)
        ax.axhline(1., color="#4B5563", linestyle=(0, (3, 3)), linewidth=.8, zorder=1)
        ax.set_title(("A  " if wi == 0 else "") + WINDOW_LABELS[window], loc="left", pad=17)
        for bi, name in enumerate(BASELINES):
            points = [p for p in values["panel_A"] if p["window"] == window and p["baseline"] == name
                      and p["ratio"] is not None]
            ax.scatter([p["root_index"] for p in points], [p["ratio"] for p in points],
                       s=31, marker=MARKERS[bi], edgecolors=COLORS[bi], facecolors="none", linewidths=1.2, zorder=3+bi)
        for ri in range(8):
            missing = [p for p in values["unavailable_root_comparisons"] if p["window"] == window and p["root_index"] == ri]
            if missing:
                _tag(ax, ri, missing[0]["reason"])
        if wi == 0:
            ax.set_ylabel("SSE / full change energy")
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("Task / parent root")
    handles = [Line2D([], [], marker=m, linestyle="none", markerfacecolor="none", markeredgecolor=c,
                      markersize=6, label=label) for m, c, label in zip(MARKERS, COLORS, LABELS)]
    handles.append(Line2D([], [], color="#4B5563", linestyle="--", lw=.8, label="No change = 1"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .932), ncol=5, frameon=False,
               handletextpad=.5, columnspacing=1.6)

    ax = fig.add_subplot(grid[1, :2])
    _root_axis(ax, roots, tasks)
    ax.axhline(0, color="#4B5563", linewidth=.8)
    finite_differences = [abs(p["normalized_difference"]) for p in values["panel_B"]
                          if p["normalized_difference"] is not None]
    maxabs = max(finite_differences, default=0.) or .01
    ax.set_ylim(-1.5*maxabs, 1.5*maxabs)
    ax.set_title("B  Head-total versus modality-specific error", loc="left", pad=17)
    for window, mark, color, label in zip(WINDOWS, ("o", "s", "^"), ("#303D55", "#7789A8", "#B58758"),
                                          ("Full 10", "First 5", "Last 5")):
        points = [p for p in values["panel_B"] if p["window"] == window and p["normalized_difference"] is not None]
        ax.scatter([p["root_index"] for p in points], [p["normalized_difference"] for p in points],
                   s=37, marker=mark, facecolors="none", edgecolors=color, linewidths=1.3, label=label, zorder=4)
        for p in values["panel_B"]:
            if p["window"] == window and p["normalized_difference"] is None:
                _tag(ax, p["root_index"], "ZERO_TARGET" if p["target"] == 0 else "MISSING")
    ax.set_ylabel("(Head-position SSE - modality SSE)\n/ the same full change energy")
    ax.set_xlabel("Fixed root order; positive = smaller error with modality masses")
    ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=8)

    ax = fig.add_subplot(grid[1, 2])
    ax.set_title("C  Where is the remaining error?", loc="left", pad=17)
    ax.grid(axis="y", color="#E5E8ED", linewidth=.5, zorder=0)
    heights = []
    for x, rec in enumerate(values["panel_C"]):
        a, b = rec["B3_ratio"], rec["residual_ratio"]
        color = COLORS[2] if rec["baseline"] == BASELINES[2] else COLORS[3]
        if a is None or b is None:
            _tag(ax, x, "ZERO_TARGET" if rec["target"] == 0 else "MISSING")
            continue
        ax.bar(x, a, width=.56, color=color, alpha=.38, edgecolor=color, linewidth=.8, zorder=3)
        ax.bar(x, b, bottom=a, width=.56, color="white", edgecolor=color, linewidth=1.0, hatch="////", zorder=3)
        if a+b == 0:
            ax.scatter([x], [0.], marker="o", s=18, color=color, zorder=4)
        heights.append(a+b)
    ax.set_xlim(-.6, 3.6)
    ax.set_ylim(0, (max(heights, default=0.) or .03)*1.35)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Response\nHead", "Response\nModality", "Clean\nHead", "Clean\nModality"])
    ax.set_ylabel("SSE / the same full change energy")
    ax.set_xlabel("Full 10; equal-root absolute means")
    ax.legend(handles=[Patch(facecolor="#ABB4C2", label="Inside B3"),
                       Patch(facecolor="white", edgecolor="#666666", hatch="////", label="Residual")],
              frameon=False, fontsize=8, loc="upper right")
    _finish(fig, data["synthetic"], "Primary: P_C (clean attention), L18, flow 0 to 9. PC / PT label the two visiting parent roots; no jitter or selected winners.")
    return fig


def build_queries(data, values, tasks):
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(13.6, 8.5), sharex=True)
    fig.subplots_adjust(left=.075, right=.975, top=.82, bottom=.23, hspace=.30)
    ax, bx = axes
    maxval = max([1.] + [b["ratio"] for q in values["queries"] for b in q["baselines"].values() if b["ratio"] is not None])
    ax.set_ylim(0, 1.18*maxval)
    ax.axhline(1, color="#4B5563", linestyle="--", lw=.8)
    for bi, name in enumerate(BASELINES):
        points = [(i, q["baselines"][name]["ratio"]) for i, q in enumerate(values["queries"])
                  if q["baselines"][name]["ratio"] is not None]
        ax.scatter([p[0] for p in points], [p[1] for p in points], marker=MARKERS[bi], s=31,
                   facecolors="none", edgecolors=COLORS[bi], linewidths=1.1, label=LABELS[bi], zorder=4+bi)
    ax.set_ylabel("SSE / query full change energy")
    ax.set_title("All 24 query slots - no counterexample removed", loc="left")
    ax.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    maxdelta = max([abs(q["normalized_difference"]) for q in values["queries"]
                    if q["normalized_difference"] is not None], default=0.) or .01
    bx.set_ylim(-1.18*maxdelta, 1.18*maxdelta)
    bx.axhline(0, color="#4B5563", lw=.8)
    for i, q in enumerate(values["queries"]):
        val = q["normalized_difference"]
        if val is None:
            _tag(bx, i, "ZERO_TARGET" if q["target"] == 0 else "MISSING")
        else:
            bx.scatter(i, val, color=TASK_COLORS[tasks.index(q["task_id"])], s=24,
                       marker="v" if val < 0 else "o", zorder=4)
        if any(b["ratio"] is None for b in q["baselines"].values()):
            _tag(ax, i, "ZERO_TARGET" if q["target"] == 0 else "MISSING")
    bx.set_ylabel("(Head SSE - modality SSE)\n/ query full change energy")
    bx.set_xlabel("q11 / q16 / q22 within each planned root; negative points are retained counterexamples")
    bx.set_xticks(range(24))
    bx.set_xticklabels([f"{task_label(q['task_id'])} {data['planned_roots'][i//3]['parent_condition']}\nq{q['query_index']}"
                        for i, q in enumerate(values["queries"])], rotation=60, ha="right", fontsize=7)
    for a in axes:
        a.set_xlim(-.6, 23.6)
        a.grid(axis="y", color="#E5E8ED", lw=.5)
        for ri, root in enumerate(data["planned_roots"]):
            if tasks.index(root["task_id"]) % 2 == 0:
                a.axvspan(3*ri-.5, 3*ri+2.5, color="#F2F4F7", zorder=0)
            if ri:
                a.axvline(3*ri-.5, color="#C9CFD8", lw=.55, zorder=0)
    _finish(fig, data["synthetic"], "Query diagnostics: P_C, L18, flow 0 to 9, full 10; PC / PT = visiting parent roots. Query ratios differ from ratios of root means.")
    return fig


def render_review(review, output_dir, *, allow_real_cloud=False, cloud_job_id=None,
                  input_identity=None, dpi=180):
    data, indexes, tasks = validate_review(review, allow_real_cloud=allow_real_cloud, cloud_job_id=cloud_job_id)
    need(isinstance(dpi, int) and 100 <= dpi <= 300, "bounded_export_dpi")
    output = Path(output_dir)
    need(not output.exists(), "new_output_directory_no_overwrite")
    values = display_values(data, indexes)
    output.mkdir(parents=True)
    written = []
    synthetic_prefix = "SYNTHETIC-" if data["synthetic"] else ""
    for name, make in (("MASS-BASELINE", lambda: build_main(data, indexes, values, tasks)),
                       ("QUERY-DIAGNOSTICS", lambda: build_queries(data, values, tasks))):
        fig = make()
        for suffix in ("svg", "png", "pdf"):
            path = output / f"{synthetic_prefix}{name}.{suffix}"
            metadata = {"Creator": "paper2 result_figures_v1", "Title": synthetic_prefix+name} if suffix == "pdf" else None
            fig.savefig(path, dpi=dpi, metadata=metadata)
            written.append({"path": path.name, **identity(path.read_bytes())})
        plt.close(fig)
    path = output / "FIGURE-VALUES.json"
    raw = encode(values)
    path.write_bytes(raw)
    written.append({"path": path.name, **identity(raw)})
    need(sum(r["bytes"] for r in written) <= MAX_OUTPUT_BYTES, "bounded_figure_output_bytes")
    source_path = Path(__file__).resolve()
    manifest = {"schema": "PAPER2-mass-figure-render-v1", "synthetic": data["synthetic"],
                "render_complete": True, "visual_review_complete": False,
                "input_review": input_identity or identity(encode(review)),
                "source_files": review["source_files"],
                "renderer_source": {"path": source_path.name, **identity(source_path.read_bytes())},
                "cloud_job_id": cloud_job_id if not data["synthetic"] else None,
                "planned_support": {"roots": 8, "tasks": 4, "queries": 24},
                "outputs": written, "output_bytes": sum(r["bytes"] for r in written),
                "scientific_scope": "post_hoc_conditional_mass_reconstruction_no_W_action_behavior_claim",
                "normalization": "common_full_target_per_entity_window", "no_original_array_reads": True}
    (output / "RENDER-MANIFEST.json").write_bytes(encode(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-real-cloud", action="store_true")
    parser.add_argument("--cloud-job-id")
    args = parser.parse_args()
    need(args.review.is_file() and not args.review.is_symlink() and
         args.review.stat().st_size <= MAX_REVIEW_BYTES, "bounded_regular_review_file")
    raw = args.review.read_bytes()
    ref = {"path": str(args.review), **identity(raw)}
    need(ref["sha256"] == args.review_sha256, "review_whole_file_SHA256")
    manifest = render_review(json.loads(raw), args.output, allow_real_cloud=args.allow_real_cloud,
                             cloud_job_id=args.cloud_job_id, input_identity=ref)
    print(json.dumps({"render_complete": True, "synthetic": manifest["synthetic"],
                      "output_bytes": manifest["output_bytes"]}))


if __name__ == "__main__":
    main()
