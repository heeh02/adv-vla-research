"""New candidate contract. Neither old GPU grants nor CPU data audits authorize it."""
import hashlib
import json
import math
from pathlib import Path
from campaign_core import ROOT, need

RUN = "jinwen-paper2-q11-cached-physical-20261021-v1"
RESOURCES = {"CPU": 8, "GPU": 0, "memory_GiB": 32}
IMAGE = "onevla-model-cn-north-1.jcr.service.jdcloud.com/one-vla/adv-vla-light@sha256:8a0b1fb7132dcef3b799acb883cc1b6e6e08487eb220a580b7051cb5db4b2fb4"
SCIENCE_SECONDS, ARCHIVE_SECONDS, ALLOCATION_SECONDS = 1500, 1680, 1800
INPUT_BYTES, INPUT_COUNT = 796390625, 243
SCRATCH_BYTES = 4*1024**3
SCOPE = "q11_cached_command_physical_pilot"


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def ident(value):
    raw = Path(value).read_bytes() if isinstance(value, Path) else value
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def authority_gate(authority, manifest, inputs, *, t0, deadline):
    need(authority.get("direct_human_approved") is True, "new_direct_environment_authority_required")
    expected = {"scope": SCOPE, "run_id": RUN, "root_id": ROOT, "query_index": 11,
                "maximum_explicit_env_steps": 350, "new_model_calls": 0, "GPU": 0,
                "resources": RESOURCES, "source_manifest": manifest, "inputs_identity": inputs,
                "science_seconds": SCIENCE_SECONDS, "archive_seconds": ARCHIVE_SECONDS,
                "allocation_seconds": ALLOCATION_SECONDS, "automatic_retries": 0}
    need(all(authority.get(k) == v for k, v in expected.items()), "exact_new_pilot_authority_binding")
    need(all(type(v) in (int, float) and math.isfinite(v) for v in (t0, deadline)) and
         deadline == t0+SCIENCE_SECONDS, "creation_anchored_pilot_deadline")
    evidence = authority.get("direct_user_evidence", {})
    need(type(evidence.get("bytes")) is int and evidence["bytes"] > 0 and
         isinstance(evidence.get("sha256"), str) and len(evidence["sha256"]) == 64,
         "bound_direct_user_evidence_required")


def verify_sources(package, expected):
    package = Path(package)
    raw = (package/"SOURCE-MANIFEST.json").read_bytes()
    need(ident(raw) == expected, "sealed_pilot_source_manifest")
    document = json.loads(raw)
    need(document["schema"] == "paper2-q11-physical-source-v1", "pilot_source_schema")
    rows = document["files"]
    need(0 < len(rows) <= 512 and len({r["path"] for r in rows}) == len(rows),
         "bounded_unique_pilot_sources")
    for row in rows:
        path = package/row["path"]
        need(path.resolve().is_relative_to(package.resolve()) and not path.is_symlink()
             and path.is_file() and ident(path) == {k: row[k] for k in ("bytes", "sha256")},
             "sealed_pilot_source_member")
    return document
