"""Explicitly artificial rendering fixture, never a research data substitute."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import renderer as R


def _mean(values):
    return None if any(v is None for v in values) else sum(values)/len(values)


def make_review():
    roots = [{"root_id": f"SYNTHETIC-T{task:02d}-P{parent}", "task_id": f"T{task:02d}",
              "parent_condition": "P"+parent, "query_indices": [11, 16, 22]}
             for task in (6, 7, 8, 9) for parent in ("C", "T")]
    roots_rows, query_rows, task_rows, pop_rows = [], [], [], []
    for ri, root in enumerate(roots):
        for wi, window in enumerate(R.WINDOWS):
            qrows = []
            for qi, q in enumerate((11, 16, 22)):
                target = 0. if ri == 6 else (15.+4*ri+qi*3) * (1., .55, .45)[wi]
                ratios = [.66+.025*ri, .39+.014*qi, .17+.018*ri, .10+.01*qi]
                if ri in (2, 5):
                    ratios[3] = ratios[2]+.08+.025*qi  # Deliberate counterexamples.
                if wi == 1:
                    ratios[1] *= 1.15
                    ratios[3] *= .8
                if wi == 2:
                    ratios[0] *= 1.5
                    ratios[2] *= .7
                baselines = {}
                for bi, name in enumerate(R.BASELINES):
                    sse = ratios[bi] * target
                    if target == 0 and bi >= 2:
                        sse = .02*(bi+qi)  # Zero target does not erase absolute error.
                    missing = (ri == 7 and qi == 1 and ((wi == 1 and bi == 2) or (wi == 0 and bi == 0)))
                    baselines[name] = {"SSE": None if missing else sse,
                                       "status": "UNAVAILABLE_ZERO_MASS_ALLOCATION" if missing else
                                                 "ZERO_TARGET" if target == 0 else "DEFINED"}
                row = {"root_id": root["root_id"], "task_id": root["task_id"], "query_index": q,
                       "view": "full", "window": window, "target_delta_energy": target,
                       "full_target_delta_energy": target, "baselines": baselines}
                qrows.append(row)
                if wi == 0:
                    query_rows.append(copy.deepcopy(row))
            full = {"aggregation_id": root["root_id"], "task_id": root["task_id"], "view": "full", "window": window,
                    "target_delta_energy": _mean([r["target_delta_energy"] for r in qrows]),
                    "full_target_delta_energy": _mean([r["full_target_delta_energy"] for r in qrows]),
                    "baselines": {n: {"SSE": _mean([r["baselines"][n]["SSE"] for r in qrows]),
                                      "status": "UNAVAILABLE" if any(r["baselines"][n]["SSE"] is None for r in qrows)
                                                else "ZERO_TARGET" if ri == 6 else "DEFINED"} for n in R.BASELINES}}
            roots_rows.append(full)
            for basis, fraction in (("response", .22+.02*ri), ("clean", .55-.02*ri)):
                for part in ("B3", "residual"):
                    r = copy.deepcopy(full)
                    r["view"] = basis+"_"+part
                    target_fraction = .35 if basis == "response" else .6
                    r["target_delta_energy"] *= target_fraction if part == "B3" else 1.-target_fraction
                    for bi, name in enumerate(R.BASELINES):
                        f = fraction + .025*bi
                        if r["baselines"][name]["SSE"] is not None:
                            r["baselines"][name]["SSE"] *= f if part == "B3" else 1.-f
                    roots_rows.append(r)

    def aggregate(rows, label):
        result = {"aggregation_id": label, "view": rows[0]["view"], "window": rows[0]["window"],
                  "target_delta_energy": _mean([r["target_delta_energy"] for r in rows]),
                  "full_target_delta_energy": _mean([r["full_target_delta_energy"] for r in rows]), "baselines": {}}
        for name in R.BASELINES:
            values = [r["baselines"][name]["SSE"] for r in rows]
            sse = _mean(values)
            result["baselines"][name] = {"SSE": sse, "status": "UNAVAILABLE" if sse is None else "DEFINED"}
        return result

    for view in R.VIEWS:
        for window in R.WINDOWS:
            selected = [r for r in roots_rows if r["view"] == view and r["window"] == window]
            pop_rows.append(aggregate(selected, "all"))
            for task in ("T06", "T07", "T08", "T09"):
                task_rows.append(aggregate([r for r in selected if r["task_id"] == task], task))
    fake = b"SYNTHETIC fixture identity; this is not a research source file.\n"
    return {"schema": "PAPER2-result-review-v1", "status": "ACCEPTED_COMPLETE", "accepted": True,
            "synthetic": True, "failures": [],
            "source_files": [{"role": "synthetic_fixture", "path": "SYNTHETIC-NOT-RESEARCH.txt", **R.identity(fake)}],
            "figure_data": {"schema": "PAPER2-mass-baseline-figure-data-v1", "synthetic": True,
                            "planned_roots": roots, "root_rows": roots_rows, "task_rows": task_rows,
                            "population_rows": pop_rows, "query_rows": query_rows}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    R.need(not args.output.exists(), "synthetic_fixture_no_overwrite")
    raw = R.encode(make_review())
    args.output.write_bytes(raw)
    print(R.identity(raw))


if __name__ == "__main__":
    main()
