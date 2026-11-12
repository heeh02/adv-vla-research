"""Synthetic archives only; never open real runtime arrays or construct envs."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import adapter as A


def fixture(root, *, change=None, missing=None, extra=None, long_name=False):
    entries = [
        ("official/libero/libero/envs/env_wrapper.py", b"# wrapper\n"),
        ("python/site-packages/robosuite/environments/base.py", b"# reset\n"),
        ("python/site-packages/robosuite/utils/observables.py", b"# observable\n"),
        ("python/site-packages/robosuite/utils/binding_utils.py", b"# sim\n"),
        ("python/site-packages/robosuite/utils/control_utils.py", b"# control\n"),
        ("python/site-packages/robosuite-1.4.1.dist-info/METADATA", b"Version: 1.4.1\n"),
        ("python/ADV_VLA_ENVIRONMENT-v1.json", b'{"runtime":"synthetic"}\n'),
        ("assets/no-execute.npz", b"\x00not really an NPZ\xff"),
    ]
    if long_name:
        entries.append(("python/site-packages/robosuite/" + "a" * 120 + ".py",
                        b"raise RuntimeError('THIS SOURCE MUST NEVER RUN')\n"))
    if missing:
        entries = [(n, d) for n, d in entries if missing not in n]
    if extra:
        entries.append(extra)
    if change:
        entries = change(entries)
    tar_bytes = io.BytesIO()
    rows = []
    with tarfile.open(fileobj=tar_bytes, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, raw in entries:
            info = tarfile.TarInfo(name)
            info.mode, info.size = 0o644, len(raw)
            tar.addfile(info, io.BytesIO(raw))
            rows.append({"path": name, "mode": info.mode, **A.fid(raw)})
    blob = tar_bytes.getvalue()
    manifest = b"".join(A.encoded(r) for r in rows)
    (root / "manifest.jsonl").write_bytes(manifest)
    refs = [{"path": "manifest.jsonl", "uri": "redacted-ebcd863ccab3", **A.fid(manifest)}]
    part_rows, offset = [], 0
    # Deliberately split through TAR headers and member payloads.
    for i, chunk in enumerate((blob[:137], blob[137:877], blob[877:])):
        name = f"part-{i}.bin"
        (root / name).write_bytes(chunk)
        ref = {"path": name, "uri": "redacted-e71965047044" + name, **A.fid(chunk)}
        refs.append(ref)
        part_rows.append({"stage_path": name, "offset": offset, **A.fid(chunk)})
        offset += len(chunk)
    inputs = {"schema": "paper2-cpu-replay-source-closure-inputs-v1",
              "input_object_count": len(refs), "input_body_bytes": sum(r["bytes"] for r in refs),
              "input_objects": refs, "archives": [{
                  "name": "base", "archive": A.fid(blob), "members": len(rows),
                  "logical_file_bytes": sum(r["bytes"] for r in rows),
                  "manifest_paths": ["manifest.jsonl"], "parts": part_rows,
              }]}
    return inputs


def rewrite_manifest(root, inputs, edit):
    p = root / "manifest.jsonl"
    rows = [json.loads(line) for line in p.read_bytes().splitlines()]
    rows = edit(rows)
    raw = b"".join(A.encoded(r) for r in rows)
    old = inputs["input_objects"][0]["bytes"]
    p.write_bytes(raw)
    inputs["input_objects"][0].update(A.fid(raw))
    inputs["input_body_bytes"] += len(raw) - old
    return rows


def rewrite_tar(root, inputs, edit):
    old = b"".join((root / p["stage_path"]).read_bytes()
                   for p in inputs["archives"][0]["parts"])
    dst = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(old)) as source, tarfile.open(
            fileobj=dst, mode="w", format=tarfile.PAX_FORMAT) as output:
        for i, member in enumerate(source):
            body = source.extractfile(member).read()
            edit(i, member)
            output.addfile(member, io.BytesIO(body) if member.isfile() else None)
    blob = dst.getvalue()
    offset = 0
    chunks = (blob[:137], blob[137:877], blob[877:])
    for ref, part, chunk in zip(inputs["input_objects"][1:],
                               inputs["archives"][0]["parts"], chunks):
        (root / ref["path"]).write_bytes(chunk)
        ref.update(A.fid(chunk))
        part.update(A.fid(chunk), offset=offset)
        offset += len(chunk)
    inputs["archives"][0]["archive"] = A.fid(blob)
    inputs["input_body_bytes"] = sum(r["bytes"] for r in inputs["input_objects"])


class SourceExtraction(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_ok(self, spec):
        return A.execute(self.root, spec, lambda: None, synthetic=True)

    def test_split_header_payload_PAX_and_binary_exclusion(self):
        spec = fixture(self.root, long_name=True)
        blob, rows, summary, counts = self.run_ok(spec)
        self.assertEqual(summary["members_verified"], 9)
        self.assertEqual(counts, {"archives": 1, "input_objects": 4})
        self.assertEqual(summary["selected_source_files"], 8)
        self.assertTrue(summary["archive_proofs"][0]["all_part_archive_member_SHA_verified"])
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            self.assertEqual(set(z.namelist()), {r["bundle_path"] for r in rows})
            self.assertFalse(any(n.endswith(".npz") for n in z.namelist()))
            self.assertTrue(any(b"THIS SOURCE MUST NEVER RUN" in z.read(n) for n in z.namelist()))
            for r in rows:
                self.assertEqual(A.fid(z.read(r["bundle_path"])),
                                 {k: r[k] for k in ("bytes", "sha256")})
        self.assertEqual({p.name for p in self.root.iterdir()},
                         {r["path"] for r in spec["input_objects"]})

    def test_corrupted_part_rejected(self):
        spec = fixture(self.root)
        p = self.root / "part-2.bin"
        raw = p.read_bytes()
        p.write_bytes(raw[:-1] + b"\x01")
        with self.assertRaisesRegex(A.ExtractionError, "part_SHA"):
            self.run_ok(spec)

    def test_wrong_whole_archive_rejected(self):
        spec = fixture(self.root)
        spec["archives"][0]["archive"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(A.ExtractionError, "whole_archive_SHA"):
            self.run_ok(spec)

    def test_wrong_member_digest_rejected(self):
        spec = fixture(self.root)
        def edit(rows):
            rows[0]["sha256"] = "0" * 64
            return rows
        rewrite_manifest(self.root, spec, edit)
        with self.assertRaisesRegex(A.ExtractionError, "member_SHA"):
            self.run_ok(spec)

    def test_wrong_manifest_digest_rejected(self):
        spec = fixture(self.root)
        spec["input_objects"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(A.ExtractionError, "input_SHA"):
            self.run_ok(spec)

    def test_duplicate_manifest_member_rejected(self):
        spec = fixture(self.root)
        def edit(rows):
            rows[1]["path"] = rows[0]["path"]
            return rows
        rewrite_manifest(self.root, spec, edit)
        with self.assertRaisesRegex(A.ExtractionError, "duplicate_manifest_member"):
            self.run_ok(spec)

    def test_archive_member_order_bound(self):
        spec = fixture(self.root)
        rewrite_manifest(self.root, spec, lambda rows: [rows[1], rows[0], *rows[2:]])
        with self.assertRaisesRegex(A.ExtractionError, "actual_tar_member_header"):
            self.run_ok(spec)

    def test_unsafe_path_rejected_before_materialization(self):
        spec = fixture(self.root, extra=("../escape.py", b"unsafe\n"))
        with self.assertRaisesRegex(A.ExtractionError, "unsafe_relative_path"):
            self.run_ok(spec)
        self.assertFalse((self.root.parent / "escape.py").exists())

    def test_symlink_stage_rejected(self):
        spec = fixture(self.root)
        path = self.root / "part-0.bin"
        raw = path.read_bytes()
        path.unlink()
        outside = self.root.parent / ("tmp-extract-" + self.root.name)
        outside.write_bytes(raw)
        try:
            path.symlink_to(outside)
            with self.assertRaisesRegex(A.ExtractionError, "regular_input_required"):
                self.run_ok(spec)
        finally:
            outside.unlink()

    def test_symlink_tar_member_rejected(self):
        spec = fixture(self.root)
        def link(i, member):
            if i == 0:
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/forbidden"
                member.size = 0
        rewrite_tar(self.root, spec, link)
        with self.assertRaisesRegex(A.ExtractionError, "actual_tar_member_header"):
            self.run_ok(spec)

    def test_hardlink_tar_member_rejected(self):
        spec = fixture(self.root)
        def link(i, member):
            if i == 0:
                member.type = tarfile.LNKTYPE
                member.linkname = "other.py"
                member.size = 0
        rewrite_tar(self.root, spec, link)
        with self.assertRaisesRegex(A.ExtractionError, "actual_tar_member_header"):
            self.run_ok(spec)

    def test_member_permission_mismatch_rejected(self):
        spec = fixture(self.root)
        def mode(i, member):
            if i == 0:
                member.mode = 0o755
        rewrite_tar(self.root, spec, mode)
        with self.assertRaisesRegex(A.ExtractionError, "actual_tar_member_mode"):
            self.run_ok(spec)

    def test_unrostered_stage_object_rejected(self):
        spec = fixture(self.root)
        (self.root / "extra.bin").write_bytes(b"")
        with self.assertRaisesRegex(A.ExtractionError, "exact_stage_roster"):
            self.run_ok(spec)

    def test_missing_required_source_rejected(self):
        spec = fixture(self.root, missing="binding_utils")
        blob, rows, summary, _ = self.run_ok(spec)
        self.assertEqual(summary["missing_source_interfaces"], ["robosuite_binding_utils"])
        self.assertGreater(len(blob), 0)
        self.assertTrue(rows)

    def test_selected_binary_rejected(self):
        spec = fixture(self.root, extra=("python/robosuite/binary.py", b"\x00\xff"))
        with self.assertRaisesRegex(A.ExtractionError, "selected_binary_NUL"):
            self.run_ok(spec)

    def test_unknown_encoding_rejected(self):
        spec = fixture(self.root, extra=("python/robosuite/binary.py", b"\xff"))
        with self.assertRaisesRegex(A.ExtractionError, "selected_not_UTF8"):
            self.run_ok(spec)

    def test_deadline_checked(self):
        spec = fixture(self.root)
        def expired():
            raise A.ExtractionError("science_absolute_deadline")
        with self.assertRaisesRegex(A.ExtractionError, "science_absolute_deadline"):
            A.execute(self.root, spec, expired, synthetic=True)

    def test_selected_size_cap(self):
        spec = fixture(self.root, extra=("python/robosuite/huge.py", b"#" * (2 * A.MIB + 1)))
        with self.assertRaisesRegex(A.ExtractionError, "selected_member_size_cap"):
            self.run_ok(spec)

    def test_offset_mismatch_rejected(self):
        spec = fixture(self.root)
        spec["archives"][0]["parts"][1]["offset"] += 1
        with self.assertRaisesRegex(A.ExtractionError, "part_roster_identity"):
            self.run_ok(spec)

    def test_missing_archive_member_rejected(self):
        spec = fixture(self.root)
        spec["archives"][0]["members"] += 1
        with self.assertRaisesRegex(A.ExtractionError, "manifest_population"):
            self.run_ok(spec)

    def test_JSON_duplicate_key_rejected(self):
        with self.assertRaisesRegex(A.ExtractionError, "duplicate_JSON_key"):
            A.json_read(b'{"a":1,"a":2}')


class ImageSourceExtraction(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.stage = self.root / "stage"
        self.stage.mkdir()
        spec = fixture(self.stage, missing="robosuite")
        self.archive = A.execute(self.stage, spec, lambda: None, synthetic=True)
        self.site = self.root / "site"
        self.package = self.site / "robosuite"
        self.package.mkdir(parents=True)
        for key, suffixes in A.REQUIRED.items():
            if not key.startswith("robosuite_"):
                continue
            path = self.site / suffixes[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("raise RuntimeError('never execute this module')\n")
        (self.package / "__init__.py").write_text(
            "__version__ = '1.4.1'\nraise RuntimeError('never import robosuite')\n")
        self.dist = self.site / "robosuite-1.4.1.dist-info"
        self.dist.mkdir()
        (self.dist / "METADATA").write_text("Name: robosuite\nVersion: 1.4.1\n")
        (self.dist / "WHEEL").write_text("Wheel-Version: 1.0\n")
        (self.dist / "RECORD").write_text("")
        self.provenance = {
            "package": "robosuite", "package_version": "1.4.1",
            "fixed_site": str(self.site),
            "image_sha256": "8a0b1fb7132dcef3b799acb883cc1b6e6e08487eb220a580b7051cb5db4b2fb4",
            "historical_module": {"imported": True, "origin":
                "/opt/capsule/.venv/lib/python3.12/site-packages/robosuite/__init__.py"},
            "allowed_metadata_basenames": ["METADATA", "WHEEL", "RECORD", "top_level.txt"],
        }

    def tearDown(self):
        self.temp.cleanup()

    def run_image(self):
        blob, rows, summary, _ = self.archive
        return A.append_image_sources(blob, rows, summary, self.provenance, lambda: None,
                                      synthetic=True)

    def test_image_text_and_distinct_source_domains_without_import(self):
        blob, rows, summary = self.run_image()
        self.assertFalse(summary["missing_source_interfaces"])
        self.assertEqual(summary["image_source_proof"]["initializer_literal_version"], "1.4.1")
        self.assertFalse(summary["image_source_proof"]["self_observed_image_digest"])
        self.assertEqual(summary["image_source_proof"]["package_modules_imported"], 0)
        self.assertEqual({r["source_domain"] for r in rows},
                         {"capsule_member_manifest", "immutable_image_digest"})
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            self.assertIn("image/robosuite/__init__.py", archive.namelist())
            self.assertIn(b"never import", archive.read("image/robosuite/__init__.py"))

    def test_image_init_version_mismatch(self):
        (self.package / "__init__.py").write_text("__version__ = '1.5.0'\n")
        with self.assertRaisesRegex(A.ExtractionError, "image_init_metadata_version_mismatch"):
            self.run_image()

    def test_image_metadata_version_mismatch(self):
        (self.dist / "METADATA").write_text("Name: robosuite\nVersion: 1.5.0\n")
        with self.assertRaisesRegex(A.ExtractionError, "image_metadata_version"):
            self.run_image()

    def test_image_dynamic_version_not_executed(self):
        (self.package / "__init__.py").write_text("__version__ = dangerous_call()\n")
        with self.assertRaisesRegex(A.ExtractionError, "image_init_version_not_literal"):
            self.run_image()

    def test_image_root_link_rejected(self):
        old = self.root / "linked-package"
        self.package.rename(old)
        self.package.symlink_to(old, target_is_directory=True)
        with self.assertRaisesRegex(A.ExtractionError, "image_symlink_ancestor"):
            self.run_image()

    def test_image_source_link_rejected(self):
        (self.package / "linked.py").symlink_to(self.package / "__init__.py")
        with self.assertRaisesRegex(A.ExtractionError, "image_package_symlink"):
            self.run_image()

    def test_image_metadata_link_rejected(self):
        p = self.dist / "METADATA"
        p.unlink()
        p.symlink_to(self.package / "__init__.py")
        with self.assertRaisesRegex(A.ExtractionError, "regular_input_required"):
            self.run_image()

    def test_image_duplicate_distributions_rejected(self):
        (self.site / "robosuite-1.5.0.dist-info").mkdir()
        with self.assertRaisesRegex(A.ExtractionError, "exact_image_distribution"):
            self.run_image()

    def test_missing_image_interface_remains_incomplete(self):
        (self.package / "utils/binding_utils.py").unlink()
        _, rows, summary = self.run_image()
        self.assertEqual(summary["missing_source_interfaces"], ["robosuite_binding_utils"])
        self.assertTrue(rows)

    def test_metadata_selection_escape_rejected(self):
        self.provenance["allowed_metadata_basenames"] = ["../secrets"]
        with self.assertRaisesRegex(A.ExtractionError, "unsafe_relative_path"):
            self.run_image()

    def test_whole_source_size_cap_includes_archive_and_image(self):
        blob, rows, summary, counts = self.archive
        summary = dict(summary, selected_source_bytes=16 * A.MIB)
        self.archive = (blob, rows, summary, counts)
        with self.assertRaisesRegex(A.ExtractionError, "selected_total_text_cap"):
            self.run_image()

    def test_production_location_cannot_be_retargeted(self):
        blob, rows, summary, _ = self.archive
        with self.assertRaisesRegex(A.ExtractionError, "fixed_immutable_image_location"):
            A.append_image_sources(blob, rows, summary, self.provenance, lambda: None)


if __name__ == "__main__":
    unittest.main()
