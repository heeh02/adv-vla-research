"""I-3 JD job output writer (stage1a/INTERFACES-v1.md, I-3).

Layout under <results>/<job_id>/:
  MANIFEST-v1.json   path, bytes, sha256 for every other file (written last)
  RESULT-v1.json     status, tests, pass/fail with measured errors and tolerances, failures, timing
  TESTS-v1.json      per test: inputs, sample ids, measured values (assembled at the end)
  CALL-LOG-v1.jsonl.gz
  SHARDS/*.npz       small BF16 (uint16 bit views) H_T/H_C/H' (+L10/L18) for qualification observations
  EPISODES/*.json    per episode: arm, unit, label, queries, issued commands, outcome (outcome sealed)
  THROUGHPUT-v1.json
Additions (documented in JD-WORKER-NOTES-v1.md): TESTS/<seq>-<id>.json written as each test
closes (crash safety), SEALED/* for fields 1A must not analyse (T7/T8 edited plans, T9
outcomes), SEALED-FIELDS-v1.json naming every sealed path/key for the Baidu readout whitelist.
Byte cap <= 2 GiB per job; reaching it is a graceful stop, never a silent truncation.
"""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path

import numpy as np

from .common import canonical, file_identity, jsonable, write_json

DEFAULT_BYTE_CAP = 2 * 1024 ** 3


class BudgetReached(RuntimeError):
    pass


class JobOutput:
    def __init__(self, results_root, job_id, *, byte_cap=DEFAULT_BYTE_CAP):
        if not (0 < byte_cap <= DEFAULT_BYTE_CAP):
            raise ValueError("I3_job_output_cap_at_most_2GiB")
        self.root = Path(results_root) / job_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.job_id = job_id
        self.cap = byte_cap
        self.used = 0
        self.test_entries = []
        self.sealed = []
        self.closed = False
        handle = (self.root / "CALL-LOG-v1.jsonl.gz").open("xb")
        self._log_raw = handle
        self._log = gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0)
        self.call_lines = 0
        self.call_bytes_uncompressed = 0

    def _reserve(self, nbytes):
        if self.used + nbytes + 4 * 1024 * 1024 > self.cap:
            raise BudgetReached("job_output_byte_cap")
        self.used += nbytes

    def _json(self, relative, value):
        raw = canonical(jsonable(value)) + b"\n"
        self._reserve(len(raw))
        write_json(self.root / relative, jsonable(value))
        return {"relative": relative, "bytes": len(raw)}

    def log_call(self, record):
        line = canonical(record) + b"\n"
        self._reserve(len(line))
        self._log.write(line)
        self.call_lines += 1
        self.call_bytes_uncompressed += len(line)

    def shard(self, name, arrays):
        relative = f"SHARDS/{name}.npz"
        packed = {}
        for key, value in arrays.items():
            a = np.asarray(value)
            packed[key + ("__bf16_bits" if a.dtype.name == "bfloat16" else "")] = (
                np.ascontiguousarray(a).view(np.uint16) if a.dtype.name == "bfloat16" else np.ascontiguousarray(a))
        buffer = io.BytesIO()
        np.savez(buffer, **packed)
        raw = buffer.getvalue()
        self._reserve(len(raw))
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        return {"relative": relative, "bytes": len(raw)}

    def episode(self, name, document):
        return self._json(f"EPISODES/{name}.json", document)

    def sealed_json(self, name, document, *, reason):
        ref = self._json(f"SEALED/{name}.json", document)
        self.sealed.append({"path": ref["relative"], "reason": reason})
        return ref

    def sealed_arrays(self, name, arrays, *, reason):
        relative = f"SEALED/{name}.npz"
        buffer = io.BytesIO()
        np.savez(buffer, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
        raw = buffer.getvalue()
        self._reserve(len(raw))
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)
        self.sealed.append({"path": relative, "reason": reason})
        return {"relative": relative, "bytes": len(raw)}

    def test(self, entry):
        sequence = len(self.test_entries)
        ref = self._json(f"TESTS/{sequence:03d}-{entry['id']}.json", entry)
        self.test_entries.append(entry)
        return ref

    def finish(self, result, throughput):
        if self.closed:
            return
        self.closed = True
        self._log.close()
        self._log_raw.close()
        write_json(self.root / "TESTS-v1.json", jsonable({"schema": "cf-repair-1a-tests-v1", "job_id": self.job_id,
                                                          "tests": self.test_entries}))
        write_json(self.root / "THROUGHPUT-v1.json", jsonable(throughput))
        write_json(self.root / "SEALED-FIELDS-v1.json", {"schema": "cf-repair-1a-sealed-fields-v1", "job_id": self.job_id,
                                                         "sealed_paths": self.sealed,
                                                         "sealed_json_keys": ["sealed"],
                                                         "rule": "TECH-CONTRACT-v1 7.0: the Baidu 1A readout reads none of these"})
        result = {**result, "call_log": {"lines": self.call_lines, "uncompressed_bytes": self.call_bytes_uncompressed},
                  "sealed_paths": self.sealed, "output_bytes_reserved": self.used, "output_byte_cap": self.cap}
        write_json(self.root / "RESULT-v1.json", jsonable(result))
        self.write_manifest()

    def write_manifest(self):
        rows = []
        for path in sorted(p for p in self.root.rglob("*") if p.is_file()):
            relative = str(path.relative_to(self.root))
            if relative == "MANIFEST-v1.json" or relative.endswith(".part"):
                continue
            rows.append({"path": relative, **file_identity(path), "sealed": relative.startswith("SEALED/"),
                         "sealed_fields": ["sealed"] if relative.startswith("EPISODES/") else []})
        write_json(self.root / "MANIFEST-v1.json", {"schema": "cf-repair-1a-results-manifest-v1", "job_id": self.job_id,
                                                    "files": rows, "file_count": len(rows),
                                                    "total_bytes": sum(r["bytes"] for r in rows)})
        return rows
