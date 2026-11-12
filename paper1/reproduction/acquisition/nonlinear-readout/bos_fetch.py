"""Read-only BOS fetch for the E5 bounded local-fetch exception (owner-granted).

Run with the one_vla venv python (has baidubce). Reads operator ak/sk into memory
only (never printed/written/returned). Downloads named objects to a local dir and
verifies bytes+SHA. Supports listing a prefix to discover the centered-basis bundles.
No writes to BOS, no AIHC, no GPU.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient

CREDS = Path("/Users/researcher/.config/one_vla/baidu/credentials.json")
DEFAULT_ENDPOINT = "bj.bcebos.com"


def make_client(endpoint=DEFAULT_ENDPOINT):
    d = json.loads(CREDS.read_bytes())
    ak, sk = d["access_key_id"], d["secret_access_key"]
    d.clear()
    conf = BceClientConfiguration(credentials=BceCredentials(ak, sk), endpoint=endpoint)
    ak = sk = None
    return BosClient(conf)


def parse(uri):
    u = urlsplit(uri)
    assert u.scheme == "bos", uri
    return u.netloc, u.path.lstrip("/")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head(client, uri):
    bucket, key = parse(uri)
    return int(client.get_object_meta_data(bucket, key).metadata.content_length)


def get_small(client, uri, cap=4 * 1024 * 1024):
    bucket, key = parse(uri)
    size = int(client.get_object_meta_data(bucket, key).metadata.content_length)
    assert size <= cap, f"too_big {size}"
    raw = client.get_object_as_string(bucket, key)
    return raw


def get_to_file(client, uri, dest, expect_bytes=None, expect_sha=None):
    bucket, key = parse(uri)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.get_object_to_file(bucket, key, str(dest))
    n = dest.stat().st_size
    s = sha256_file(dest)
    ok = (expect_bytes is None or n == expect_bytes) and (expect_sha is None or s == expect_sha)
    return {"uri": uri, "bytes": n, "sha256": s,
            "bytes_ok": expect_bytes is None or n == expect_bytes,
            "sha_ok": expect_sha is None or s == expect_sha, "ok": ok}


def list_common_prefixes(client, bucket, prefix):
    out, marker = [], None
    while True:
        resp = client.list_objects(bucket, prefix=prefix, delimiter="/", marker=marker, max_keys=1000)
        for p in getattr(resp, "common_prefixes", []) or []:
            out.append(p.prefix)
        if not getattr(resp, "is_truncated", False):
            return sorted(set(out))
        marker = getattr(resp, "next_marker", None)
        if marker is None:
            return sorted(set(out))


def list_prefix(client, bucket, prefix, limit=4000):
    out, marker = [], None
    while True:
        resp = client.list_objects(bucket, prefix=prefix, marker=marker, max_keys=1000)
        for row in getattr(resp, "contents", []) or []:
            out.append({"key": row.key, "bytes": int(row.size)})
            if len(out) >= limit:
                return out
        if not getattr(resp, "is_truncated", False):
            return out
        marker = getattr(resp, "next_marker", None) or (out[-1]["key"] if out else None)
        if marker is None:
            return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe"); p.add_argument("uri"); p.add_argument("--bytes", type=int); p.add_argument("--sha")
    p = sub.add_parser("head"); p.add_argument("uri")
    p = sub.add_parser("getjson"); p.add_argument("uri")
    p = sub.add_parser("list"); p.add_argument("bucket"); p.add_argument("prefix")
    p = sub.add_parser("prefixes"); p.add_argument("bucket"); p.add_argument("prefix")
    p = sub.add_parser("fetch"); p.add_argument("manifest"); p.add_argument("destdir")
    args = ap.parse_args()
    client = make_client(args.endpoint)
    if args.cmd == "probe":
        size = head(client, args.uri)
        raw = get_small(client, args.uri)
        print(json.dumps({"uri": args.uri, "head_bytes": size, "got_bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest(),
                          "bytes_ok": args.bytes is None or len(raw) == args.bytes,
                          "sha_ok": args.sha is None or hashlib.sha256(raw).hexdigest() == args.sha}))
    elif args.cmd == "head":
        print(json.dumps({"uri": args.uri, "bytes": head(client, args.uri)}))
    elif args.cmd == "getjson":
        print(get_small(client, args.uri).decode())
    elif args.cmd == "list":
        rows = list_prefix(client, args.bucket, args.prefix)
        print(json.dumps({"prefix": args.prefix, "count": len(rows), "objects": rows}, indent=2))
    elif args.cmd == "prefixes":
        pref = list_common_prefixes(client, args.bucket, args.prefix)
        print(json.dumps({"prefix": args.prefix, "count": len(pref), "common_prefixes": pref}, indent=2))
    elif args.cmd == "fetch":
        spec = json.loads(Path(args.manifest).read_text())
        dest = Path(args.destdir)
        results = []
        for item in spec["objects"]:
            r = get_to_file(client, item["uri"], dest / item["local"], item.get("bytes"), item.get("sha256"))
            results.append({**r, "local": item["local"]})
        print(json.dumps({"destdir": str(dest), "results": results,
                          "all_ok": all(r["ok"] for r in results)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
