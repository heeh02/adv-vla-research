"""Freeze only this small existing-data reader. No credentials or cloud activity."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
NAMES = ("adapter.py", "inventory.py", "extract.py", "archive_reader.py", "INPUTS.json",
         "LIBRARY-ROSTER.json", "REFERENCE-ROOTS.json", "PROTOCOL.json", "test_inventory.py", "freeze.py")


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def freeze(out):
    bodies = {n: (HERE/n).read_bytes() for n in NAMES}
    manifest = {"schema": "paper2-init-library-inventory-source-v1",
                "files": [{"path": n, **identity(raw)} for n, raw in bodies.items()]}
    mr = (json.dumps(manifest, sort_keys=True, separators=(",", ":"))+"\n").encode()
    bodies["SOURCE-MANIFEST.json"] = mr
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for n, raw in sorted(bodies.items()):
            member = zipfile.ZipInfo(n, (2026, 10, 21, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100440 << 16
            z.writestr(member, raw)
    out = Path(out)
    out.mkdir(exist_ok=False)
    (out/"SOURCE-PAYLOAD-v1.zip").write_bytes(stream.getvalue())
    (out/"SOURCE-MANIFEST.json").write_bytes(mr)
    receipt = {"source_zip": identity(stream.getvalue()), "source_manifest": identity(mr),
               "inputs": identity(bodies["INPUTS.json"]), "files": len(NAMES),
               "cloud_calls": 0, "environment_calls": 0}
    (out/"FREEZE.json").write_text(json.dumps(receipt, sort_keys=True, indent=2)+"\n")
    return receipt


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    print(json.dumps(freeze(p.parse_args().out)))
