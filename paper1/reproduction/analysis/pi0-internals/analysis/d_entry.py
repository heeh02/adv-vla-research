"""E1 Phase D cloud-CPU entry: run the horizon-50 readout over the C3 recordings.

Runs inside the D bundle on a CPU worker (GPU=0). Downloads the C3 admitted per-arm
science npz (presigned URLs in the jobspec) to instance temp, loads the staged coverage
map, runs d_readout, and writes D-RESULTS to the E1 BOS prefix. Archived actions never
enter (only the C3 within-run recordings). Credentials from ONE_VLA_BOS_STS_* only.
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent
sys.path[:0] = [str(BUNDLE / "analysis")]
sys.path.append(str(BUNDLE))

from bos_stdlib import BOSWriter  # noqa: E402
import d_readout  # noqa: E402

RECV = os.environ["E1_RECEIPT_URI"]
DEADLINE = float(os.environ["E1_DEADLINE"])
TMP = Path(os.environ["E1_TMP"])


def sink(rel, obj):
    raw = (json.dumps(obj, sort_keys=True, default=str) + "\n").encode()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"; p.write_bytes(raw)
        return BOSWriter(RECV, DEADLINE).put(p, rel)


def dl(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(); n = 0
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        if r.status != 200:
            raise RuntimeError("http_" + str(r.status))
        while True:
            chunk = r.read1(1 << 20)
            if not chunk:
                break
            f.write(chunk); h.update(chunk); n += len(chunk)
    return h.hexdigest(), n


def dl_bytes(url):
    buf = bytearray()
    with urllib.request.urlopen(url, timeout=600) as r:
        while True:
            c = r.read1(1 << 20)
            if not c:
                break
            buf += c
    return bytes(buf)


def main():
    phase = "startup"
    try:
        js = json.loads(dl_bytes(os.environ["E1_JOBSPEC_URL"]).decode())
        coverage = json.loads((BUNDLE / "COVERAGE-MAP-v1.json").read_text())["map"]
        sink("D-STARTUP-v1.json", {"schema": "e1-d-startup", "epoch": time.time(),
             "n_science": len(js["science"]), "coverage_entries": len(coverage)})
        phase = "download"
        sdir = TMP / "science"
        import urllib.error
        got_n = 0; missing = []; sha_fail = []
        for s in js["science"]:
            dest = sdir / s["row_id"] / s["arm"] / "recording-v1.npz"
            try:
                got, n = dl(s["url"], dest)
            except urllib.error.HTTPError as e:
                # A claimed-admitted arm whose science object is absent (rare put failure)
                # drops its pair; it is skipped and listed, never aborts Phase D.
                missing.append({"row_id": s["row_id"], "arm": s["arm"], "http": int(e.code)})
                continue
            got_n += 1
            if s.get("sha256") and got != s["sha256"]:
                sha_fail.append({"row_id": s["row_id"], "arm": s["arm"]})
        if sha_fail:
            raise RuntimeError("science_sha_mismatch:" + str(len(sha_fail)))
        sink("D-DOWNLOADED-v1.json", {"n_requested": len(js["science"]), "n_downloaded": got_n,
             "n_missing_science": len(missing), "missing_science": missing[:60], "epoch": time.time()})
        phase = "assemble"
        recs = d_readout.load_recordings(sdir)
        features, rows, dropped = d_readout.assemble(recs, coverage)
        sink("D-ASSEMBLED-v1.json", {"n_pairs": len(rows), "n_dropped": len(dropped), "dropped": dropped[:50],
             "hidden_delta_shape": list(features["hidden_delta"].shape),
             "target_shape": list(features["target"].shape)})
        phase = "readout"
        out = d_readout.run(features, rows)
        out.update(schema="e1-d-results-v1", epoch=time.time(), protocol="PROTOCOL-v2", model="pi0_libero",
                   horizon=50, dropped_pairs=dropped, n_dropped_pairs=len(dropped))
        sink("D-RESULTS-v1.json", out)
        # compact headline for the host
        head = {m: {} for m in ("new_response",)}
        sink("D-DONE-v1.json", {"n_pairs": out["n_pairs"], "n_cells_realized": out["n_cells_realized"],
             "n_estimates": len(out["estimates"]), "n_contrasts": len(out["contrasts"]),
             "n_retention": len(out["retention"]), "n_dropped": len(dropped)})
        print(json.dumps({"d_done": True, "n_pairs": out["n_pairs"], "cells": out["n_cells_realized"]}))
    except Exception as e:
        try:
            sink("D-FAILURE-v1.json", {"phase": phase, "type": type(e).__name__, "msg": str(e)[:400],
                 "tb": traceback.format_exc()[-2000:]})
        except Exception as e2:
            print(json.dumps({"double_fault": str(e2)[:160], "phase": phase, "orig": str(e)[:200]}))
        raise


if __name__ == "__main__":
    main()
