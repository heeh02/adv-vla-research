"""Local immutable source seal. Never reads actual producer arrays."""
import argparse
from pathlib import Path
import zipfile
import adapter as A

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", required=True)
    parser.parse_args()
    manifest = {"schema": "paper2-dev16-feature-bridge-source-v1",
                "files": [{"path": name, **A.fid((HERE / name).read_bytes())} for name in A.SOURCE_FILES]}
    raw = A.encoded(manifest)
    with (HERE / "SOURCE-MANIFEST.json").open("xb") as f:
        f.write(raw)
    with zipfile.ZipFile(HERE / "SCIENCE-v1.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in (*A.SOURCE_FILES, "SOURCE-MANIFEST.json"):
            info = zipfile.ZipInfo(name, (2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (HERE / name).read_bytes())
    print(A.encoded({"manifest": A.fid(raw), "package": A.fid((HERE / "SCIENCE-v1.zip").read_bytes())}).decode())


if __name__ == "__main__":
    main()
