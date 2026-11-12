"""One reviewed package's bounded CPU0 contract for the borrowed root host."""

import hashlib
import json
from pathlib import Path
import re

RUN = OUTPUT = IMAGE = None
QUEUE = "redacted-68f1c47a0c15"
PARENT = "redacted-6e77221ca60e"
POOL = "aihc-serverless"
OWNER = "3a59a0c408234b37a90a806c3ddc7a5a"
RESOURCES = {"GPU": 0, "CPU": 2, "memory_GiB": 8}
PHASES = {"science_runtime_seconds": 3000, "archive_target_seconds": 3480, "max_runtime_seconds": 3600}
LIMITS = {
    "input_limit_bytes": 400 * 1024**2,
    "transfer_limit_bytes": 512 * 1024**2,
    "output_limit_bytes": 32 * 1024**2,
    "private_tmp_limit_bytes": 1024**3,
    "mac_readback_limit_bytes": 8 * 1024**2,
}
SMALL = ("DELIVERY-v1.json", "RESULT-v1.json", "INPUT-RECEIPT-v1.json")
THREAD_SCOPE_INSTALLED = False


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def ident(path):
    return identity(Path(path).read_bytes())


def exact(path, ref):
    path = Path(path)
    need(
        path.is_file() and not path.is_symlink() and ident(path) == {k: ref[k] for k in ("bytes", "sha256")},
        "exact_controller_source_SHA",
    )


def read(path):
    path = Path(path)
    need(path.is_file() and path.stat().st_size <= 8 * 1024**2, "bounded_control_JSON")
    return json.loads(path.read_bytes())


def safe_error(error):
    message = str(error)
    return {
        "type": type(error).__name__,
        "code": message if re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", message) else "redacted",
    }


def configure(q):
    global RUN, OUTPUT, IMAGE
    need(
        q["schema"] == "paper3-data-driven-CPU0-request-v1"
        and q["queue_id"] == QUEUE
        and q["physical_queue_id"] == PARENT
        and q["pool"] == POOL
        and q["resources"] == RESOURCES
        and q["phases"] == PHASES
        and q["limits"] == LIMITS
        and q["JPFS_new_bytes"] == 0
        and q["new_model_calls"] == q["new_episodes"] == 0,
        "exact_reviewed_CPU0_scope",
    )
    need(
        q["credential_scope_wider_than_runtime_readset"] is True
        and q["input_permission"] == ["READ"]
        and len(q["input_credential_resources"]) == 6,
        "exact_two_baseline_four_run_ACL_scopes",
    )
    RUN, OUTPUT, IMAGE = q["run_id"], q["output_prefix"], q["image"]
    need(
        RUN.startswith("jinwen-paper3-inc-")
        and OUTPUT.startswith("redacted-9dff0b6ceaff"),
        "owned_run_output_namespace",
    )
    return q


def account_gate(*_args, **_kwargs):
    raise ValueError("mandatory_thread_scope_CPU0_adapter_not_installed")
