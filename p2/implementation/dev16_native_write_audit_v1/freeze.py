"""Root-owned immutable source seal; no real array read or cloud action."""
import argparse
from pathlib import Path
import zipfile
import adapter as A

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", action="store_true", required=True)
    p.parse_args()
    manifest = {"schema": "paper2-dev16-native-write-audit-source-v1",
                "files": [{"path": n, **A.fid((HERE / n).read_bytes())} for n in A.SOURCE_FILES]}
    raw = A.encoded(manifest)
    with (HERE / "SOURCE-MANIFEST.json").open("xb") as f:
        f.write(raw)
    with zipfile.ZipFile(HERE / "SCIENCE-v1.zip", "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name in (*A.SOURCE_FILES, "SOURCE-MANIFEST.json"):
            info = zipfile.ZipInfo(name, (2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (HERE / name).read_bytes())
    print(A.encoded({"manifest": A.fid(raw), "package": A.fid((HERE / "SCIENCE-v1.zip").read_bytes())}).decode())


if __name__ == "__main__":
    main()
