#!/usr/bin/env python3
"""Synthetic integrity tests only; no cloud calls or original neural reads."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time
import unittest
import zipfile
import numpy as np
import p08_reader as R

HERE = Path(__file__).resolve().parent


def fixture():
    config, rows = R.load_inputs(HERE / "INPUTS-v2.json")
    row = dict(next(r for r in rows if r["original_present"] == "True"))
    arrays, schema, receipts = {}, {}, {}
    for name, spec in config["fields"].items():
        a = np.zeros(spec["shape"], dtype=spec["stored_dtype"])
        if spec["logical_dtype"] == "bfloat16":
            a.flat[:4] = [0x3f80, 0xc000, 0x0000, 0x8000]
        elif name == "compact/times":
            a[:] = np.arange(10, 0, -1, dtype=np.float32) / 10
        else:
            a[:] = True
        arrays[name] = a
        schema[name] = {"dtype": spec["logical_dtype"], "dtype_str": "<V2" if spec["logical_dtype"] == "bfloat16" else a.dtype.str,
                        "shape": list(a.shape), "stored_dtype": a.dtype.str}
        receipts[name] = {"slice_bytes": a.nbytes, "slice_SHA256": R.sha(a.tobytes()), "slice_shape": list(a.shape),
                          "native_dtype": spec["logical_dtype"], "native_axes": R.NATIVE_AXES[name],
                          "selection": "tau0/first5; times full10", "member_SHA256": "0" * 64}
    meta = {"schema": R.SCHEMA, "index_sha256": config["producer_identity"]["index_sha256"],
            "contract_sha256": config["producer_identity"]["contract_sha256"],
            "program_sha256": config["producer_identity"]["program_sha256"],
            "owner": {k: row[k] for k in config["embedded_owner_keys"]},
            "selection": {"tau": 0, "layers": [1, 9, 18], "queries": [0, 1, 2, 3, 4]},
            "source_whole_SHA_recomputed": True, "persisted_logical_bytes": 111560,
            "source_cache_sha256": row["cache_sha256"], "source_trace_sha256": row["trace_sha256"],
            "selected_members": receipts}
    return config, row, arrays, schema, meta


def pack(row, arrays, schema, meta):
    packed = dict(arrays)
    packed["__selected_meta__"] = np.asarray(json.dumps(meta, sort_keys=True))
    schema = copy.deepcopy(schema)
    schema["__selected_meta__"] = {"dtype": packed["__selected_meta__"].dtype.name,
                                 "dtype_str": packed["__selected_meta__"].dtype.str,
                                 "shape": [], "stored_dtype": packed["__selected_meta__"].dtype.str}
    packed["__schema__"] = np.asarray(json.dumps(schema, sort_keys=True))
    stream = io.BytesIO()
    np.savez(stream, **packed)
    raw = stream.getvalue()
    row = dict(row, selected_file_bytes=str(len(raw)), selected_file_sha256=R.sha(raw))
    return raw, row


class ReaderTests(unittest.TestCase):
    def test_frozen_small_inputs_and_missing(self):
        config, rows = R.load_inputs(HERE / "INPUTS-v2.json")
        self.assertEqual(len(rows), 6720)
        self.assertEqual(sum(r["original_present"] == "False" for r in rows), 166)
        missing = next(r for r in rows if r["original_present"] == "False")
        self.assertIsNone(R.metadata(missing)["owner"]["original_success"])
        self.assertEqual(config["archive_tar_bytes"], 1095936000)

    def test_BF16_bits_and_owner(self):
        config, row, arrays, schema, meta = fixture()
        raw, row = pack(row, arrays, schema, meta)
        value = R.decode_selected(raw, row, config)
        self.assertEqual(value["Hclean"].shape, (3, 5, 1024))
        self.assertEqual(value["Aactual"].shape, (3, 8, 2, 16, 16))
        self.assertEqual(value["flow_delta"].shape, (5, 32))
        self.assertEqual(value["Hclean"].dtype, np.float32)
        np.testing.assert_array_equal(value["Hclean"].ravel()[:4], [1, -2, 0, -0.0])
        self.assertTrue(np.signbit(value["Hclean"].ravel()[3]))
        self.assertFalse(value["Hclean"].flags.writeable)
        self.assertEqual(value["validity"]["clean_source_mass"].dtype, np.bool_)
        self.assertEqual(value["owner"]["actual_control_step"], 5 * value["owner"]["query_index"])

    def test_wrong_owner_rejected(self):
        config, row, arrays, schema, meta = fixture()
        meta["owner"]["query_index"] = "999"
        raw, row = pack(row, arrays, schema, meta)
        with self.assertRaisesRegex(ValueError, "embedded_owner_full_join"):
            R.decode_selected(raw, row, config)

    def test_wrong_slice_hash_rejected_even_with_new_NPZ_hash(self):
        config, row, arrays, schema, meta = fixture()
        arrays["full/free/hidden_delta"].flat[0] = 0x4000
        raw, row = pack(row, arrays, schema, meta)
        with self.assertRaisesRegex(ValueError, "original_slice_bits_SHA"):
            R.decode_selected(raw, row, config)

    def test_nonfinite_rejected_even_with_consistent_hashes(self):
        config, row, arrays, schema, meta = fixture()
        name = "full/free/hidden_delta"
        arrays[name].flat[0] = 0x7f80
        meta["selected_members"][name]["slice_SHA256"] = R.sha(arrays[name].tobytes())
        raw, row = pack(row, arrays, schema, meta)
        with self.assertRaisesRegex(ValueError, "finite_selected_values"):
            R.decode_selected(raw, row, config)

    def test_shape_not_accepted_from_hash_alone(self):
        config, row, arrays, schema, meta = fixture()
        arrays["full/free/hidden_delta"] = arrays["full/free/hidden_delta"].reshape(3, 1024, 5)
        raw, row = pack(row, arrays, schema, meta)
        with self.assertRaisesRegex(ValueError, "actual_field_shape_dtype"):
            R.decode_selected(raw, row, config)

    def test_whole_NPZ_SHA_rejected(self):
        config, row, arrays, schema, meta = fixture()
        raw, row = pack(row, arrays, schema, meta)
        with self.assertRaisesRegex(ValueError, "selected_NPZ_bytes_SHA"):
            R.decode_selected(raw[:-1] + bytes([raw[-1] ^ 1]), row, config)

    def tar_fixture(self, directory, name="results/shards/s00/selected/synthetic.npz"):
        content = b"synthetic-payload"
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tf:
            member = tarfile.TarInfo(name); member.size = len(content)
            tf.addfile(member, io.BytesIO(content))
        blob = stream.getvalue()
        manifest = {"part": "s00", "tar_sha256": R.sha(blob), "tar_bytes": len(blob),
                    "members": [{"path": name, "bytes": len(content), "sha256": R.sha(content)}]}
        manifest_raw = json.dumps(manifest).encode()
        (directory / "members.tar").write_bytes(blob)
        (directory / "MEMBER_MANIFEST.json").write_bytes(manifest_raw)
        part = {"part": "s00", "tar_sha256": R.sha(blob), "tar_bytes": len(blob),
                "manifest_sha256": R.sha(manifest_raw), "member_count": 1, "source_member_bytes": len(content),
                "artifact_id": "sha256:" + "a" * 64, "artifact_uri": R.PREFIX + "s00/sha256-" + "a" * 64 + "/"}
        return part, content, name

    def test_tar_identity_and_header(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); part, content, name = self.tar_fixture(p)
            v = R.VerifiedPart(p, part)
            self.assertEqual(v.read(name), content)
            v.close()
            raw = (p / "members.tar").read_bytes()
            (p / "members.tar").write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            with self.assertRaisesRegex(ValueError, "whole_tar_bytes_SHA"):
                R.VerifiedPart(p, part)

    def test_path_traversal_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); part, _, _ = self.tar_fixture(p, "../escape")
            with self.assertRaisesRegex(ValueError, "actual_tar_header_name_type_size"):
                R.VerifiedPart(p, part)

    def test_callback_exact_objects_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); source = p / "source"; source.mkdir()
            part, content, name = self.tar_fixture(source)
            calls = []
            def download(uri, expected, dest):
                calls.append(uri)
                dest.write_bytes((source / uri.rsplit("/", 1)[1]).read_bytes())
            v = R.restore_part(download, part, p / "destination", time.time() + 60)
            self.assertEqual(v.read(name), content); v.close()
            self.assertEqual(calls, [part["artifact_uri"] + "payload/MEMBER_MANIFEST.json",
                                     part["artifact_uri"] + "payload/members.tar"])
            v = R.restore_part(download, part, p / "destination", time.time() + 60); v.close()
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(ValueError, "read_deadline"):
                R.restore_part(download, part, p / "late", time.time() - 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

