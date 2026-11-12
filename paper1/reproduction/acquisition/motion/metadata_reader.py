"""Bounded metadata-only subclass of the frozen Q1 CaseReader.

Reuse original tar/header/Range verification. Cache exact metadata bytes under
index-SHA/member-SHA; never permit an NPZ read in this inventory stage.
"""
import json
from pathlib import Path
import reader
import range_source
import tar_source
from inventory_math import need, sha


class MetadataReader(reader.CaseReader):
    def __init__(self, signer, row, deadline, cache_root):
        self.signer, self.row, self.deadline = signer, row, deadline
        self.scope = row['case_scope']
        self.cache = Path(cache_root) / row['index_sha256']
        self.cache.mkdir(parents=True, exist_ok=True)
        self.metadata_raw, self.read_members = {}, []
        raw_index = self._get(row['index_uri'], row['index_sha256'])
        index = json.loads(raw_index)
        need(len(index['file_manifests']) == 1, 'one_frozen_archive_manifest')
        ref = index['file_manifests'][0]
        need(ref['uri'] == row['manifest_uri'] and ref['sha256'] == row['manifest_sha256']
             and ref['bytes'] <= 2 * 1024 * 1024, 'bound_manifest_identity')
        raw_manifest = self._get(ref['uri'], ref['sha256'])
        need(len(raw_manifest) == ref['bytes'], 'manifest_bytes')
        rows = [json.loads(line) for line in raw_manifest.splitlines() if line.strip()]
        self.mapped = tar_source.build_map(index, rows, self.scope)
        self.range = range_source.RangeReader(signer, index['parts'], deadline)
        self.members = {}
        native_row = self.mapped['NATIVE-FILES-v1.json']['row']
        if row['native_bytes'] is not None:
            need(native_row['bytes'] == row['native_bytes'], 'previous_native_bytes_identity')
        if row['native_sha256'] is not None:
            need(native_row['sha256'] == row['native_sha256'], 'previous_native_SHA_identity')
        raw_native = self.member('NATIVE-FILES-v1.json', limit=4 * 1024 * 1024)
        native = json.loads(raw_native)
        need(len(native['files']) == len({r['path'] for r in native['files']}), 'unique_native_member_schema')
        self.native = {r['path']: r for r in native['files']}
        self.index = index
        self.metadata_raw.update({'INDEX-v1.json': raw_index, 'MANIFEST-v1.jsonl': raw_manifest,
                                  'NATIVE-FILES-v1.json': raw_native})

    def _get(self, uri, expected_sha):
        path = self.cache / (expected_sha + '.metadata')
        if path.is_file():
            raw = path.read_bytes(); need(sha(raw) == expected_sha, 'cached_metadata_SHA')
            return raw
        raw = reader.get_small(self.signer, uri, expected_sha, limit=2 * 1024 * 1024)
        with path.open('xb') as stream: stream.write(raw)
        return raw

    def member(self, name, limit=256 * 1024):
        need(name in ('NATIVE-FILES-v1.json', 'reference/Y/Y-v1.json', 'treatment/Y/Y-v1.json'),
             'inventory_forbids_array_payload_read')
        # Always verify the actual tar header. A cached payload can support a
        # later analysis, but it does not replace this first inventory receipt.
        raw = super().member(name, limit)
        ref = self.mapped[name]['row']
        path = self.cache / (ref['sha256'] + '.member')
        if path.exists(): need(path.read_bytes() == raw, 'cached_member_identity')
        else:
            with path.open('xb') as stream: stream.write(raw)
        self.read_members.append({'member': name, 'bytes': len(raw), 'sha256': sha(raw),
                                  'actual_tar_header_and_payload_verified': True})
        self.metadata_raw[name] = raw
        return raw

    def fields(self, *args, **kwargs):
        raise ValueError('inventory_forbids_array_payload_read')
