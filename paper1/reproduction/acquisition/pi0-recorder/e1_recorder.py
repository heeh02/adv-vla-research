"""E1 pi0 whitebox recorder (cloud GPU worker entrypoint) — PROTOCOL-v2 tolerance gate.

Loads the stock pi0_libero policy via pi0_stock_adapter.load_stock_policy (patched
models/pi0.py exposes sample_actions_with_trajectory). Streams each case (download ->
verify archive SHA -> extract arms -> free). Per arm:
  * rebuild the policy observation from the archived q0 obs + prompt,
  * regenerate native raw + adapter actions (canonical sample_actions, num_steps=10),
  * BRIDGE admission gate vs archived query-0000-actions.npz under PROTOCOL-v2 tolerances
    (technical reproduction gate, anchored to Notion A-20261012; NOT a scientific gate),
  * internal-consistency check: module_jit(sample_actions_with_trajectory) actions match
    native within the same tolerance (so recorded H matches the recorded plan),
  * on ADMIT, record H [10,3,1,50,1024], velocity(flow), x_t, time, plan=NEW native raw+
    adapter, noise -> BOS. Archived actions are the bridge reference ONLY; the recorded
    target/plan is this run's own regenerated output (PI condition 2).
Every arm's per-component bridge difference is recorded; over-tolerance arms are excluded
and listed; no retry-until-pass. Credentials from ONE_VLA_BOS_STS_* env only.
"""
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parent
sys.path[:0] = [str(BUNDLE / "repo/src"), str(BUNDLE / "repo/packages/openpi-client/src"), str(BUNDLE / "adapter")]
sys.path.append(str(BUNDLE))

from bos_stdlib import BOSWriter  # noqa: E402

RECV = os.environ["E1_RECEIPT_URI"]
DEADLINE = float(os.environ["E1_DEADLINE"])
TMP = Path(os.environ["E1_TMP"])

# PROTOCOL-v2 tolerances (anchored to Notion A-20261012; 2x archived cross-worker maxima).
TOL_RAW_MAXABS = 0.018
ADAPTER_TOL = {"translation": (slice(0, 3), 3.1e-3, "rms"),
               "rotation": (slice(3, 6), 2.1e-3, "rms"),
               "gripper": (slice(6, 7), 3.9e-3, "max")}
TOL_TRAJ_MAXABS = 0.018
EXEC_PREFIX = 5


def sink(rel, obj):
    raw = (json.dumps(obj, sort_keys=True, default=str) + "\n").encode()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.json"; p.write_bytes(raw)
        return BOSWriter(RECV, DEADLINE).put(p, rel)


def put_npz(rel, arrays):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.npz"
        with open(p, "wb") as f:
            np.savez(f, **arrays)
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
        if r.status != 200:
            raise RuntimeError("http_" + str(r.status))
        while True:
            chunk = r.read1(1 << 20)
            if not chunk:
                break
            buf += chunk
    return bytes(buf)


def guard():
    if time.time() > DEADLINE - 120:
        raise RuntimeError("absolute_deadline_exceeded")


def maxabs(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def typed(a):
    a = np.asarray(a)
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes(order="C")).hexdigest()}


def bridge_gate(native_raw, native_adapter, arch_raw, arch_adapter):
    """PROTOCOL-v2 technical admission gate. native_* = this run; arch_* = bridge reference.

    native_raw (50,32) f32; native_adapter (50,7); arch_raw (50,32); arch_adapter (50,7).
    """
    nr, na = np.asarray(native_raw, np.float64), np.asarray(native_adapter, np.float64)
    ar, aa = np.asarray(arch_raw, np.float64), np.asarray(arch_adapter, np.float64)
    no_nan = bool(np.isfinite(nr).all() and np.isfinite(na).all())
    raw_maxabs = maxabs(nr, ar)
    raw_ok = (raw_maxabs is not None and raw_maxabs <= TOL_RAW_MAXABS)
    comps = {}; adapter_ok = True; sign_ok = True
    for name, (dims, thr, kind) in ADAPTER_TOL.items():
        a5 = aa[:EXEC_PREFIX, dims]; n5 = na[:EXEC_PREFIX, dims]; d = n5 - a5
        val = float(np.sqrt(np.mean(d ** 2))) if kind == "rms" else float(np.max(np.abs(d)))
        ok = bool(val <= thr)
        mask = np.abs(a5) >= 3 * thr
        checked = int(mask.sum())
        if checked:
            agree = (np.sign(n5[mask]) == np.sign(a5[mask]))
            sign_agreement = float(np.mean(agree)); c_sign_ok = bool(agree.all())
        else:
            sign_agreement = None; c_sign_ok = True
        comps[name] = {"value": val, "threshold": thr, "kind": kind, "within_tol": ok,
                       "sign_checked_entries": checked, "sign_agreement": sign_agreement, "sign_ok": c_sign_ok}
        adapter_ok = adapter_ok and ok; sign_ok = sign_ok and c_sign_ok
    admit = bool(no_nan and raw_ok and adapter_ok and sign_ok)
    return admit, {"admit": admit, "no_nan": no_nan, "raw_maxabs": raw_maxabs, "raw_within_tol": raw_ok,
                   "raw_threshold": TOL_RAW_MAXABS, "adapter_within_tol": adapter_ok, "sign_ok": sign_ok,
                   "adapter_components": comps}


def main():
    phase = "startup"
    admitted, excluded, bridge_rows = [], [], []
    try:
        js = json.loads(dl_bytes(os.environ["E1_JOBSPEC_URL"]).decode())
        unit = js.get("unit"); write_science = bool(js.get("write_science", True))
        sink("STARTUP-v1.json", {"schema": "e1-recorder-startup-v2", "epoch": time.time(), "unit": unit,
             "protocol": "PROTOCOL-v2", "n_cases": len(js["cases"]),
             "tolerances": {"raw_maxabs": TOL_RAW_MAXABS, "adapter": {k: [v[1], v[2]] for k, v in ADAPTER_TOL.items()},
                            "traj_maxabs": TOL_TRAJ_MAXABS}})

        # 1. checkpoint (whole-file SHA-verified vs binding)
        phase = "checkpoint"
        ck = js["checkpoint"]; root = TMP / "ckpt"
        member_report = []
        for m in ck["members"]:
            got, n = dl(m["url"], root / m["relative"])
            ok = (got == m["sha256"] and n == m["bytes"])
            member_report.append({"relative": m["relative"], "sha_match": ok})
            if not ok:
                raise RuntimeError("checkpoint_member_sha_mismatch:" + m["relative"])
        tok = TMP / "tokenizer" / "paligemma_tokenizer.model"
        tgot, tn = dl(ck["tokenizer_url"], tok)
        if not (tgot == ck["tokenizer_sha256"] and tn == ck["tokenizer_bytes"]):
            raise RuntimeError("tokenizer_sha_mismatch")
        sink("CHECKPOINT-v1.json", {"schema": "e1-recorder-checkpoint-v2", "n_members": len(member_report),
             "all_member_sha_match": all(r["sha_match"] for r in member_report), "tokenizer_sha_match": True})

        # 2. load model once
        phase = "load_model"
        guard()
        import jax
        import jax.numpy as jnp
        import pi0_stock_adapter as base
        from openpi.models import model as _model
        from openpi.shared import nnx_utils
        stock, load_receipt = base.load_stock_policy(
            str(root), str(tok), checkpoint_identity="sha256:" + ck["keeper_sha"], num_steps=10)
        model = stock._model; model.eval()
        sample = nnx_utils.module_jit(model.sample_actions)
        sample_traj = nnx_utils.module_jit(model.sample_actions_with_trajectory)
        sink("LOAD-v1.json", {"schema": "e1-recorder-load-v2", "load_receipt": load_receipt,
             "jax_devices": [str(d) for d in jax.devices()]})

        # 3. stream cases
        for ci, case in enumerate(js["cases"]):
            guard()
            cid = case["case_id"]
            phase = f"download:{cid}"
            buf = bytearray()
            for u in case["part_urls"]:
                buf += dl_bytes(u)
            if hashlib.sha256(bytes(buf)).hexdigest() != case["archive_sha"]:
                raise RuntimeError("archive_sha_mismatch:" + cid)
            tf = tarfile.open(fileobj=io.BytesIO(bytes(buf)), mode="r:")
            names = tf.getnames()
            for arm in case["arms"]:
                phase = f"record:{cid}:{arm}"
                try:
                    def member(kind, ext):
                        suf = f"arms/{arm}/Y/query-0000-{kind}.{ext}"
                        hit = [x for x in names if x.endswith(suf)]
                        if not hit:
                            raise RuntimeError("missing_member:" + suf)
                        return tf.extractfile(hit[0]).read()
                    with np.load(io.BytesIO(member("input", "npz")), allow_pickle=False) as zin:
                        raw_obs = {k: zin[k] for k in zin.files}
                    noise = np.asarray(raw_obs["noise"])
                    noise2 = np.ascontiguousarray(noise[0])
                    with np.load(io.BytesIO(member("actions", "npz")), allow_pickle=False) as zac:
                        arch_raw = np.asarray(zac["raw_actions"]); arch_act = np.asarray(zac["actions"])
                    req = json.loads(member("request", "json").decode())
                    prompt = req["actual_prompt"]
                    prompt_sha_ok = (hashlib.sha256(prompt.encode()).hexdigest() == req.get("actual_prompt_sha256"))

                    obs_dict = base.make_policy_observation(raw_obs, prompt)
                    inputs = stock._input_transform({k: (v.copy() if isinstance(v, np.ndarray) else v)
                                                     for k, v in obs_dict.items()})
                    inputs_b = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
                    observation = _model.Observation.from_dict(inputs_b)
                    noise_j = jnp.asarray(noise2)[None, ...]
                    raw = sample(jax.random.key(0), observation, num_steps=10, noise=noise_j)
                    raw_np = np.asarray(raw)                                   # (1,50,32) new native
                    out = jax.tree.map(lambda x: np.asarray(x[0, ...]), {"state": inputs_b["state"], "actions": raw})
                    out = stock._output_transform(out)
                    native_adapter = np.asarray(out["actions"])               # (50,7) new native adapter
                    native_raw = raw_np[0]                                    # (50,32)

                    traj, path = sample_traj(observation, noise_j)
                    traj_np = np.asarray(traj)
                    traj_consistency = maxabs(traj_np, raw_np)
                    traj_ok = (traj_consistency is not None and traj_consistency <= TOL_TRAJ_MAXABS)

                    admit, bridge = bridge_gate(native_raw, native_adapter, arch_raw[0], arch_act[0])
                    admit = bool(admit and prompt_sha_ok and traj_ok)
                    row = {"case_id": cid, "arm": arm, "admit": admit, "prompt_sha_ok": prompt_sha_ok,
                           "traj_consistency_maxabs": traj_consistency, "traj_within_tol": traj_ok, **bridge}
                    bridge_rows.append(row)
                    sink(f"bridge/BRIDGE-{cid}-{arm}-v1.json",
                         {**row, "native_raw": typed(native_raw), "archive_raw": typed(arch_raw),
                          "native_adapter": typed(native_adapter), "archive_adapter": typed(arch_act)})
                    (admitted if admit else excluded).append({"case_id": cid, "arm": arm})

                    if admit and write_science:
                        H = np.asarray(path["hidden"])         # [10,3,1,51,1024] bf16 = 1 state + 50 action tokens
                        vel = np.asarray(path["velocity"]); xt = np.asarray(path["x_t"]); tt = np.asarray(path["time"])
                        arm_noise = 0 if arm in ("R0", "T0", "R0prime") else 1  # R1/T1 are noise 1
                        arm_role = "clean" if arm in ("R0", "R1", "R0prime") else "perturbed"
                        rec = {"hidden": H, "velocity": vel, "x_t": xt, "time": tt,
                               "raw_actions": raw_np.astype(np.float32)[0], "actions_adapter": native_adapter,
                               "noise": noise2, "arm": np.asarray(arm), "arm_role": np.asarray(arm_role),
                               "noise_label": np.asarray(arm_noise),
                               "suite": np.asarray(case.get("suite", req.get("suite", ""))),
                               "family": np.asarray(case.get("family", "")), "row_id": np.asarray(case.get("row_id", cid)),
                               "task_id": np.asarray(case.get("task_id", "")),
                               "condition": np.asarray(case.get("condition", ""))}
                        proof = put_npz(f"science/{cid}/{arm}/recording-v1.npz", rec)
                        sink(f"science/{cid}/{arm}/RECORDING-SHA-v1.json",
                             {"case_id": cid, "arm": arm, "put": proof, "arrays": {k: typed(v) for k, v in rec.items()},
                              "note": "H=AE post-block hidden [10 calls,3 layers=L1/L9/L18,1,51 pos=1 state+50 action,1024]; "
                                      "Phase D uses the 50 action positions H[...,-50:,:]. velocity/x_t are [10,1,50,32]. "
                                      "plan=THIS RUN's native raw+adapter (archived used only as bridge reference, never in analysis); "
                                      "prefix V not captured by sample_actions_with_trajectory (needs a gemma KV tap), omitted; not consumed by the readout."})
                except Exception as ae:
                    # PROTOCOL-v2 §2: missing arm / non-finite / per-arm failure -> technical-missing,
                    # excluded and listed; never zero-filled; never aborts the batch. (sink failure still raises.)
                    excluded.append({"case_id": cid, "arm": arm, "technical_missing": True,
                                     "reason": type(ae).__name__ + ":" + str(ae)[:160]})
                    bridge_rows.append({"case_id": cid, "arm": arm, "admit": False, "technical_missing": True,
                                        "reason": type(ae).__name__ + ":" + str(ae)[:160]})
                    sink(f"bridge/BRIDGE-{cid}-{arm}-v1.json", {"case_id": cid, "arm": arm, "admit": False,
                         "technical_missing": True, "reason": type(ae).__name__ + ":" + str(ae)[:200]})
            del buf, tf
            if ci % 10 == 0:
                sink(f"progress/PROGRESS-{ci:04d}.json", {"cases_done": ci + 1, "admitted": len(admitted),
                     "excluded": len(excluded), "epoch": time.time()})
            # free per-case ckpt-independent temp (capsule held only in memory, already freed)

        phase = "finalize"
        receipt = {"schema": "e1-recorder-receipt-v2", "epoch": time.time(), "unit": unit, "protocol": "PROTOCOL-v2",
                   "n_arms": len(bridge_rows), "n_admitted": len(admitted), "n_excluded": len(excluded),
                   "excluded": excluded, "all_admitted": len(excluded) == 0,
                   "bridge_rows": bridge_rows}
        sink(("C2-RECEIPT-v1.json" if unit == "C2" else "C3-RECEIPT-v1.json"), receipt)
        print(json.dumps({"unit": unit, "n_arms": len(bridge_rows), "admitted": len(admitted),
                          "excluded": len(excluded)}, sort_keys=True))
    except Exception as e:
        try:
            sink("FAILURE-v1.json", {"schema": "e1-recorder-failure-v2", "phase": phase,
                 "type": type(e).__name__, "msg": str(e)[:400], "tb": traceback.format_exc()[-2000:],
                 "admitted_so_far": len(admitted), "excluded_so_far": len(excluded)})
        except Exception as e2:
            print(json.dumps({"double_fault": type(e2).__name__ + ":" + str(e2)[:160], "phase": phase,
                              "orig": type(e).__name__ + ":" + str(e)[:200]}))
        raise


if __name__ == "__main__":
    main()
