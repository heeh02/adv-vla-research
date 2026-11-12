"""Credential-free query IPC and parent-only exact streaming archive.

No cloud SDK, environment loading, model calls, tar creation, or implicit retry.
The parent injects authenticated transport; the child only reads/writes local IPC.
"""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import time
import uuid

MAX_MANIFEST = 4 * 1024**2
INFLIGHT = 512 * 1024**2
MEMBER_LIMIT = 64 * 1024**2  # existing audited BOSWriter.put limit
ROOT_OUTPUT = 17 * 1024**3 // 2
BINDING_KEYS = {"run_id", "root_id", "source_manifest_sha256", "source_package_sha256", "request_sha256", "plan_sha256"}


class BrokerError(RuntimeError):
    pass


def need(ok, code):
    if not ok:
        raise BrokerError(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_binding(binding):
    need(set(binding) == BINDING_KEYS, "exact_IPC_binding_fields")
    for k, v in binding.items():
        need(type(v) is str, "string_binding")
        need(bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,100}" if k.endswith("_id") else r"[0-9a-f]{64}", v)),
             "safe_bound_identity")


def relative(value):
    need(type(value) is str and value and "\\" not in value, "relative_member_string")
    p = PurePosixPath(value)
    need(not p.is_absolute() and str(p) == value and all(x not in ("", ".", "..") for x in value.split("/")),
         "canonical_relative_member")
    return value


def owned(root, rel):
    relative(rel)
    p = root
    for part in PurePosixPath(rel).parts:
        p = p / part
        need(not p.is_symlink(), "symlink_member_or_parent_forbidden")
    need(p.resolve().is_relative_to(root) and p != root, "member_escaped_owned_science")
    return p


def file_identity(path):
    before = path.stat()
    need(path.is_file() and not path.is_symlink() and before.st_nlink == 1, "regular_single_link_science_member")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024**2), b""):
            h.update(part)
    after = path.stat()
    def token(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(token(before) == token(after), "member_changed_during_hash")
    return {"bytes": after.st_size, "sha256": h.hexdigest()}, token(after)


def read_small(path):
    need(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= MAX_MANIFEST, "bounded_regular_IPC_file")
    with path.open("rb") as f:
        raw = f.read(MAX_MANIFEST + 1)
    need(0 < len(raw) <= MAX_MANIFEST, "IPC_read_cap")
    return json.loads(raw), raw


def write_once(path, value):
    """Expose only complete fsynced JSON, never replace an existing receipt."""
    raw = encoded(value)
    need(len(raw) <= MAX_MANIFEST, "IPC_write_cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(read_small(path)[1] == raw, "immutable_IPC_receipt_changed")
        return raw
    tmp = path.parent / (".writing-" + uuid.uuid4().hex)
    try:
        with tmp.open("xb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(tmp, path)  # atomic exclusive publication; no replacement race
    finally:
        if tmp.exists():
            tmp.unlink()
    return raw


def validate_members(members, limit):
    need(isinstance(members, list) and 0 < len(members) <= 4096, "bounded_nonempty_member_inventory")
    seen = set()
    for row in members:
        need(set(row) == {"path", "bytes", "sha256"}, "exact_member_fields")
        relative(row["path"])
        need(row["path"] not in seen, "duplicate_member")
        seen.add(row["path"])
        need(type(row["bytes"]) is int and 0 <= row["bytes"] <= min(limit, MEMBER_LIMIT)
             and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "member_size_SHA")
    need(sum(r["bytes"] for r in members) <= limit, "whole_query_inflight_cap")


def local_inventory(root, members):
    tokens = {}
    for row in members:
        actual, token = file_identity(owned(root, row["path"]))
        need(actual == {k: row[k] for k in ("bytes", "sha256")}, "local_member_identity_changed")
        tokens[row["path"]] = token
    return tokens


def verify_ack(ack, ready, ready_sha):
    need(ack.get("schema") == "continuous-query-ack-v1" and ack.get("status") == "VERIFIED",
         "complete_verified_ACK_required")
    need(ack.get("binding") == ready["binding"] and ack.get("ready_sha256") == ready_sha
         and ack.get("nonce") == ready["nonce"] and ack.get("sequence") == ready["sequence"]
         and ack.get("query_index") == ready["query_index"] and ack.get("role") == ready["role"], "ACK_exact_manifest_binding")
    rows = ack.get("members", [])
    need(isinstance(rows, list) and len(rows) == len(ready["members"]), "ACK_complete_inventory")
    qi = ready["query_index"]
    keyroot = "queries/" + ready["binding"]["root_id"] + "/" + ("initialization" if qi is None else f"query-{qi:04d}") + "/" + ready["nonce"]
    for expected, row in zip(ready["members"], rows):
        need(all(row.get(k) == expected[k] for k in ("path", "bytes", "sha256")), "ACK_exact_member_identity")
        need(row.get("object_key") == keyroot + "/" + expected["path"], "ACK_exact_member_key")
        need(row.get("verification") == {"method": "fresh_GET", "bytes": expected["bytes"], "sha256": expected["sha256"]},
             "ACK_requires_full_fresh_GET_SHA")
    need(ack.get("manifest_verification") == {"method": "fresh_GET", "bytes": len(encoded(ready)), "sha256": ready_sha},
         "ACK_remote_manifest_verified")
    need(ack.get("manifest_key") == keyroot + "/READY-v1.json", "ACK_exact_manifest_key")
    body = {k: v for k, v in ack.items() if k not in ("durable_ack_object_key", "durable_ack_verification")}
    need(ack.get("durable_ack_object_key") == keyroot + "/ACK-v1.json"
         and ack.get("durable_ack_verification") == {"method": "fresh_GET", "bytes": len(encoded(body)), "sha256": digest(encoded(body))},
         "ACK_remotely_durable_verified_receipt")
    return [{k: row[k] for k in ("path", "object_key", "bytes", "sha256")} for row in rows]


class ChildQueryClient:
    """Returns receipts only; existing QueryArchive rechecks tokens then unlinks.

    One synchronous query at a time bounds hot data. A failed ACK/timeout keeps
    every source member. Deletion is never inferred from PUT or object existence.
    """
    def __init__(self, owned_science_root, spool_root, *, binding, deadline_epoch,
                 inflight_limit_bytes=INFLIGHT, now=time.time, sleeper=time.sleep):
        validate_binding(binding)
        self.root = Path(owned_science_root).resolve(strict=True)
        self.spool = Path(spool_root).resolve()
        need(not self.spool.is_relative_to(self.root), "IPC_outside_science_members")
        self.spool.mkdir(parents=True, exist_ok=True)
        need(not any(self.spool.glob("*/READY-v1.json")), "fresh_child_IPC_owner_required")
        need(inflight_limit_bytes == INFLIGHT, "frozen_512MiB_inflight")
        self.binding, self.deadline, self.limit = dict(binding), deadline_epoch, inflight_limit_bytes
        self.now, self.sleep, self.sequence = now, sleeper, 0
        self.seen_queries = set()

    def commit_query(self, query_index, members, *, role="complete"):
        need(role in ("complete", "partial") and (type(query_index) is int and query_index >= 0
             or role == "partial" and query_index is None), "explicit_query_or_partial_initialization")
        need(role != "complete" or query_index not in self.seen_queries, "one_complete_query_commit")
        need(self.now() < self.deadline, "archive_deadline_before_READY")
        members = [{k: row[k] for k in ("path", "bytes", "sha256")} for row in members]
        validate_members(members, self.limit)
        before = local_inventory(self.root, members)
        nonce = uuid.uuid4().hex
        directory = self.spool / (f"{self.sequence:06d}-" + nonce)
        ready = {"schema": "continuous-query-ready-v1", "binding": self.binding,
                 "sequence": self.sequence, "nonce": nonce, "query_index": query_index, "role": role,
                 "members": members, "total_bytes": sum(r["bytes"] for r in members)}
        self.sequence += 1
        raw = write_once(directory / "READY-v1.json", ready)
        while self.now() < self.deadline:
            if (directory / "ERROR-v1.json").exists():
                error, _ = read_small(directory / "ERROR-v1.json")
                need(error.get("ready_sha256") == digest(raw), "error_manifest_binding")
                raise BrokerError("parent_archive_failed_preserve_local")
            if (directory / "ACK-v1.json").exists():
                ack, _ = read_small(directory / "ACK-v1.json")
                result = verify_ack(ack, ready, digest(raw))
                need(local_inventory(self.root, members) == before, "source_changed_before_ACK_consumption")
                if role == "complete":
                    self.seen_queries.add(query_index)
                return result
            self.sleep(min(.1, max(0., self.deadline - self.now())))
        raise BrokerError("archive_ACK_deadline_preserve_local")


class ParentQueryBroker:
    """Parent alone calls transport.put_file and fresh readback_sha256.

    Transport is scoped to the one run prefix. No bucket LIST, tar, hidden data
    retries or science callbacks. Reservations precede PUT, including failures.
    """
    def __init__(self, owned_science_root, spool_root, *, binding, transport, object_prefix,
                 deadline_epoch, output_limit_bytes=ROOT_OUTPUT, inflight_limit_bytes=INFLIGHT, now=time.time):
        validate_binding(binding)
        self.root = Path(owned_science_root).resolve(strict=True)
        self.spool = Path(spool_root).resolve()
        self.spool.mkdir(parents=True, exist_ok=True)
        need(not self.spool.is_relative_to(self.root), "IPC_outside_science_members")
        relative(object_prefix)
        need(object_prefix == "queries/" + binding["root_id"], "one_exact_root_archive_namespace")
        need(output_limit_bytes <= ROOT_OUTPUT and 0 < output_limit_bytes
             and inflight_limit_bytes == INFLIGHT, "root_output_and_inflight_caps")
        self.binding, self.io, self.prefix = dict(binding), transport, object_prefix
        self.deadline, self.output_limit, self.limit, self.now = deadline_epoch, output_limit_bytes, inflight_limit_bytes, now
        self.reserved, self.processed = 0, set()
        self.verified = []
        self.failed = False
        reservations = sorted(self.spool.glob("*/RESERVED-v1.json"))
        need(len(reservations) <= 4096, "bounded_broker_reservation_history")
        for path in reservations:
            row, _ = read_small(path)
            need(row.get("binding") == self.binding and type(row.get("reserved_bytes")) is int
                 and row["reserved_bytes"] > 0, "prior_reservation_binding")
            self.reserved += row["reserved_bytes"]
            self.processed.add(row["ready_sha256"])
            need((path.parent / "ACK-v1.json").exists() or (path.parent / "ERROR-v1.json").exists(),
                 "unfinished_parent_attempt_no_implicit_retry")
        need(self.reserved <= self.output_limit, "prior_reservations_exceed_root_cap")

    def _check(self):
        need(self.now() < self.deadline, "parent_archive_absolute_deadline")

    def process_ready(self, directory):
        self._check()
        directory = Path(directory)
        need(directory.parent == self.spool and directory.is_dir() and not directory.is_symlink(), "exact_IPC_request_directory")
        ready, raw = read_small(directory / "READY-v1.json")
        sha = digest(raw)
        need(raw == encoded(ready), "canonical_READY_bytes")
        need(ready.get("schema") == "continuous-query-ready-v1" and ready.get("binding") == self.binding,
             "READY_exact_run_root_source_binding")
        seq, nonce = ready.get("sequence"), ready.get("nonce")
        need(type(seq) is int and seq >= 0 and type(nonce) is str and re.fullmatch(r"[0-9a-f]{32}", nonce)
             and directory.name == f"{seq:06d}-{nonce}", "READY_sequence_nonce_directory")
        need(sha not in self.processed and not (directory / "ACK-v1.json").exists()
             and not (directory / "ERROR-v1.json").exists(), "one_archive_attempt_per_READY")
        self.processed.add(sha)
        verified = []
        try:
            members = ready["members"]
            validate_members(members, self.limit)
            need(ready["total_bytes"] == sum(r["bytes"] for r in members), "READY_total_bytes")
            qi, role = ready["query_index"], ready["role"]
            need(role in ("complete", "partial") and (type(qi) is int and qi >= 0 or role == "partial" and qi is None),
                 "READY_role_and_query")
            keyroot = self.prefix + "/" + ("initialization" if qi is None else f"query-{qi:04d}") + "/" + nonce
            expected_ack = {"schema": "continuous-query-ack-v1", "status": "VERIFIED", "binding": self.binding,
                "ready_sha256": sha, "sequence": seq, "nonce": nonce, "query_index": qi, "role": role,
                "members": [{**row, "object_key": keyroot + "/" + row["path"],
                    "verification": {"method": "fresh_GET", **{k: row[k] for k in ("bytes", "sha256")}}} for row in members],
                "manifest_key": keyroot + "/READY-v1.json",
                "manifest_verification": {"method": "fresh_GET", "bytes": len(raw), "sha256": sha}}
            ack_bytes = encoded(expected_ack)
            need(len(ack_bytes) <= MAX_MANIFEST, "bounded_complete_ACK_body")
            charge = ready["total_bytes"] + len(raw) + len(ack_bytes)
            need(self.reserved + charge <= self.output_limit, "cumulative_root_archive_cap")
            self.reserved += charge
            write_once(directory / "RESERVED-v1.json", {"binding": self.binding, "ready_sha256": sha,
                "reserved_bytes": charge, "cumulative_reserved_bytes": self.reserved, "refunded_on_failure": False})
            before = local_inventory(self.root, members)
            for row in members:
                self._check()
                path = owned(self.root, row["path"])
                need(file_identity(path)[1] == before[row["path"]], "member_changed_before_PUT")
                key = keyroot + "/" + row["path"]
                self.io.put_file(path, key)
                self._check()
                remote = self.io.readback_sha256(key, expected_bytes=row["bytes"])
                need(remote == {k: row[k] for k in ("bytes", "sha256")}, "fresh_GET_member_SHA_mismatch")
                need(file_identity(path)[1] == before[row["path"]], "member_changed_during_archive")
                verified.append({**row, "object_key": key, "verification": {"method": "fresh_GET", **remote}})
            self._check()
            manifest_key = keyroot + "/READY-v1.json"
            self.io.put_file(directory / "READY-v1.json", manifest_key)
            remote = self.io.readback_sha256(manifest_key, expected_bytes=len(raw))
            need(remote == {"bytes": len(raw), "sha256": sha}, "fresh_GET_manifest_SHA_mismatch")
            need(local_inventory(self.root, members) == before, "query_source_changed_before_ACK")
            self._check()
            ack = {"schema": "continuous-query-ack-v1", "status": "VERIFIED", "binding": self.binding,
                   "ready_sha256": sha, "sequence": seq, "nonce": nonce, "query_index": qi, "role": role,
                   "members": verified, "manifest_key": manifest_key,
                   "manifest_verification": {"method": "fresh_GET", **remote}}
            need(ack == expected_ack, "reserved_ACK_shape_matches_actual_verified_receipt")
            ack_body = write_once(directory / "ACK-BODY-v1.json", ack)
            ack_key = keyroot + "/ACK-v1.json"
            self.io.put_file(directory / "ACK-BODY-v1.json", ack_key)
            durable = self.io.readback_sha256(ack_key, expected_bytes=len(ack_body))
            need(durable == {"bytes": len(ack_body), "sha256": digest(ack_body)}, "fresh_GET_ACK_receipt_mismatch")
            self._check()
            need(local_inventory(self.root, members) == before, "source_changed_before_durable_ACK")
            ack.update(durable_ack_object_key=ack_key, durable_ack_verification={"method": "fresh_GET", **durable})
            verify_ack(ack, ready, sha)
            write_once(directory / "ACK-v1.json", ack)
            self.verified.append({"ready_sha256": sha, "query_index": qi, "role": role,
                                  "total_bytes": ready["total_bytes"], "manifest_key": manifest_key})
            return ack
        except BaseException as error:
            self.failed = True
            write_once(directory / "ERROR-v1.json", {"schema": "continuous-query-archive-error-v1",
                "binding": self.binding, "ready_sha256": sha, "status": "FAILED_PRESERVED",
                "code": str(error) if isinstance(error, BrokerError) else "transport_or_writer_failure",
                "exception_type": type(error).__name__, "verified_members": verified,
                "ACK_issued": False, "local_files_deleted": False, "automatic_retry": False})
            raise

    def poll_once(self):
        self._check()
        done = []
        for ready in sorted(self.spool.glob("*/READY-v1.json")):
            if (ready.parent / "ACK-v1.json").exists() or (ready.parent / "ERROR-v1.json").exists():
                continue
            done.append(self.process_ready(ready.parent))
        return done
