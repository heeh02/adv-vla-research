"""Seal only this reviewed extraction source, not the runtime archives."""
import argparse
from pathlib import Path
import zipfile
import adapter as A

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.parse_args()
    manifest = {"schema": "paper2-cpu-replay-source-closure-source-v1",
                "files": [{"path": name, **A.fid((HERE / name).read_bytes())}
                          for name in A.SOURCE_FILES]}
    raw = A.encoded(manifest)
    with (HERE / "SOURCE-MANIFEST.json").open("xb") as stream:
        stream.write(raw)
    with zipfile.ZipFile(HERE / "SCIENCE-v1.zip", "x", zipfile.ZIP_DEFLATED) as output:
        for name in (*A.SOURCE_FILES, "SOURCE-MANIFEST.json"):
            info = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, (HERE / name).read_bytes())
    print(A.encoded({"manifest": A.fid(raw),
                     "package": A.fid((HERE / "SCIENCE-v1.zip").read_bytes())}).decode())


if __name__ == "__main__":
    main()
