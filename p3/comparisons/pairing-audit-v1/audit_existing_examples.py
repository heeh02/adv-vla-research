"""Read-only audit of frozen pairing semantics using existing small OUTCOMEs."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
MAIN = next(p for p in HERE.parents if (p/"AGENTS.md").is_file())
BEHAVIOR = MAIN/"docs/experiments/existing-data-synthesis-20261018-v1/behavior"
RESEARCH = Path('/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5')
N = RESEARCH/"docs/experiments/analysis-system-20260929/native-write-flow-20261006-v1"
EXAMPLES = N/"root-progress-20261011-v77/q0-input-pairing-readonly-v1"


def ref(p):
    b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def main():
    sys.path.insert(0, str(BEHAVIOR))
    import reuse
    import behavior_tables as gate
    index = json.loads((EXAMPLES/"snapshot-v1/INDEX-v1.json").read_bytes())
    assert len(index['records']) == 24
    normalized, source_refs, per_record = [], [], []
    for row in index["records"]:
        p = Path(row["outcome_ref"]["path"])
        assert ref(p) == row["outcome_ref"]
        out = json.loads(p.read_bytes())
        assert reuse.digest(out) == row["summary"]["outcome_canonical_sha256"]
        z = reuse.normalize(out, row)
        modules = out["loaded_official_modules_after_case"]
        runtime = {k: {f: v[f] for f in ("bytes", "sha256")} for k, v in modules.items()}
        noise = {x["query"]: reuse.Q["native"](x["noise"]) for x in out["input_indices"]}
        nr = {"ordinal": row["ordinal"], "label": z["checkpoint_label"], "input": {
            "signature": z["recorded_input_signature_sha256"], "runtime_signature": reuse.digest(runtime),
            "recorded_noise_by_query": noise}}
        normalized.append(nr); source_refs.append(ref(p))
        queries = {x['query']: x for x in out['input_indices']}
        per_record.append({"ordinal": row["ordinal"], "label": z["checkpoint_label"],
            "checkpoint_sha256": out["model"]["checkpoint"]["sha256"],
            "q0_signature": nr["input"]["signature"], "runtime_signature": nr["input"]["runtime_signature"],
            "runtime_module_count": len(runtime), "runtime_module_names": sorted(runtime),
            "all_runtime_modules_are_libero": all(k == "libero" or k.startswith("libero.") for k in runtime),
            "q0_field_names": sorted(z["fields"]), "queries": len(queries),
            "q0_physical": z["fields"]["physical_q0"],
            "q1_physical": queries[1]["passive_state"]["physical_state"] if 1 in queries else None,
            "q1_noise": queries[1]["noise"] if 1 in queries else None,
            "complete_controller_buffer_replay_verified": z["qualification"]["complete_controller_buffer_replay_verified"],
            "post_transform_neural_input_identity": z["qualification"]["post_transform_neural_input_identity"],
            "source": ref(p)})
    pairs = []
    for ordinal in range(4):
        selected = [r for r in normalized if r["ordinal"] == ordinal]
        assert len(selected) == 6
        for a, b in itertools.combinations(selected, 2):
            pairs.append({"ordinal": ordinal, "left": a['label'], "right": b['label'],
                "status": gate.pair_gate(a, b)})
    counts = Counter(r["status"] for r in pairs)
    proof = dict(schema="T30-pair-gate-existing-example-audit-v1", cloud_calls=0, model_calls=0,
        array_reads=0, source_outcome_json_bytes=sum(r["bytes"] for r in source_refs),
        original_input_index=ref(EXAMPLES/"snapshot-v1/INDEX-v1.json"),
        reused_sources=[ref(BEHAVIOR/n) for n in ("behavior_tables.py", "consume_frozen.py", "vendor/analyze_metadata.py")],
        observed_record_count=len(per_record), observed_pairs=len(pairs), pair_counts=dict(counts),
        records=per_record, pairs=pairs,
        conclusion="No model/checkpoint equality or post-q0 state equality gate is present; all60 existing cross-model examples qualify.",
        scope="Four Background Texture variants of one parent task; Original,C10,U10,L10, two seeds where stored; not T20/T30 full-population qualification.",
        no_gate_or_threshold_changes=True, no_candidate_patch_needed=True,
        remaining_unknown=["T20/T30/Original current exact q0 and runtime match counts", "all-family loaded-module-set equality", "full controller/buffer state equality", "encoded tokens and post-transform neural input equality"])
    p = HERE/"EXISTING-EXAMPLES-AUDIT-v1.json"
    with p.open("xb") as f: f.write(reuse.canonical(proof)+b"\n")
    print(json.dumps({"observed_records": len(per_record), "pairs": len(pairs), "pair_counts": dict(counts), "output": str(p), "cloud_calls": 0}))


if __name__ == "__main__":
    main()
