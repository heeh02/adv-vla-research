"""Freeze this owned candidate; no credentials, cloud or native imports."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def freeze(out):
    names = sorted([p.name for p in HERE.glob("*.py")] +
                   ["INPUTS.json", "IMAGE-SOURCE-PINS.json", "PROTOCOL.json", "README.md", "RECOVERY-BUDGET.json"] +
                   [p.relative_to(HERE).as_posix() for directory in ("retained_cpu3", "retained_reader")
                    for p in (HERE/directory).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"])
    assert len(names) == len(set(names)) <= 512
    bodies = {name: (HERE/name).read_bytes() for name in names}
    manifest = {"schema": "paper2-q11-physical-source-v1",
                "files": [{"path": name, **identity(raw)} for name, raw in bodies.items()],
                "runtime_execution": "Requires a new source-bound direct human authorization.",
                "retained_sources_unchanged": True, "new_model_payloads": 0}
    raw_manifest = (json.dumps(manifest, sort_keys=True, separators=(",", ":"))+"\n").encode()
    bodies["SOURCE-MANIFEST.json"] = raw_manifest
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(bodies.items()):
            member = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100440 << 16
            archive.writestr(member, raw)
    out = Path(out)
    out.mkdir(exist_ok=False)
    (out/"SOURCE-PAYLOAD-v1.zip").write_bytes(blob.getvalue())
    (out/"SOURCE-MANIFEST.json").write_bytes(raw_manifest)
    receipt = {"source_zip": identity(blob.getvalue()), "source_manifest": identity(raw_manifest),
               "input_manifest": identity(bodies["INPUTS.json"]), "files": len(names),
               "expanded_bytes": sum(map(len, bodies.values())), "cloud_calls": 0,
               "environment_calls": 0, "direct_human_approved": False}
    (out/"FREEZE.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(freeze(parser.parse_args().out), sort_keys=True))
