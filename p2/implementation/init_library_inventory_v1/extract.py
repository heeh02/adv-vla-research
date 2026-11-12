"""Verify complete two-archive capsule; retain only pinned initialization bytes."""
import hashlib
import io
import tarfile
import archive_reader as A


def extract(stage, inputs, libraries, check):
    refs = {r['path']: r for r in inputs['input_objects']}
    A.need(len(refs) == inputs['input_object_count'] and sum(r['bytes'] for r in refs.values()) == inputs['input_body_bytes'], 'exact_input_population')
    A.need({p.name for p in stage.iterdir()} == set(refs), 'exact_stage_roster')
    wanted = {r['path']: r for r in libraries}
    A.need(len(wanted) == len(libraries), 'unique_library_paths')
    captured, proofs, used = {}, [], []
    for archive in inputs['archives']:
        rows = A.parse_manifest(stage, archive, refs, check)
        offset = 0
        for p in archive['parts']:
            A.need(p['offset'] == offset and all(p[k] == refs[p['stage_path']][k] for k in ('bytes', 'sha256')), 'part_roster_identity')
            offset += p['bytes']
        A.need(offset == archive['archive']['bytes'], 'archive_part_total')
        used += archive['manifest_paths'] + [r['stage_path'] for r in archive['parts']]
        stream = A.Parts(stage, archive['parts'], archive['archive'], check)
        count = 0
        try:
            with tarfile.open(fileobj=stream, mode='r|') as tar:
                for member in tar:
                    check(); A.need(count < len(rows), 'extra_tar_member')
                    row = rows[count]; A.relative(member.name)
                    A.need(member.isfile() and not member.issparse() and member.name == row['path'] and
                           member.size == row['bytes'] and member.mode == row['mode'], 'exact_regular_tar_member')
                    selected = archive['name'] == 'base' and member.name in wanted
                    if selected:
                        A.need(member.size <= 131072 and all(row[k] == wanted[member.name][k] for k in ('bytes', 'sha256')), 'exact_small_pinned_library')
                    digest = hashlib.sha256(); length = 0; body = io.BytesIO() if selected else None
                    src = tar.extractfile(member); A.need(src is not None, 'file_reader')
                    with src:
                        while block := src.read(1024**2):
                            check(); digest.update(block); length += len(block)
                            if body is not None:
                                body.write(block)
                    A.need({'bytes': length, 'sha256': digest.hexdigest()} == {k: row[k] for k in ('bytes', 'sha256')}, 'member_SHA')
                    if selected:
                        A.need(member.name not in captured, 'duplicate_selected_library')
                        captured[member.name] = body.getvalue()
                    count += 1
            A.need(count == len(rows), 'all_manifest_members_seen')
            stream.finish()
            proofs.append({'name': archive['name'], 'parts': stream.verified_parts, 'members': count,
                           'archive': archive['archive'], 'all_member_part_archive_SHA_verified': True})
        finally:
            stream.close()
    A.need(len(used) == len(refs) and set(used) == set(refs), 'inputs_used_exactly_once')
    A.need(set(captured) == set(wanted) and sum(map(len, captured.values())) < 1024**2, 'complete_small_library_roster')
    return captured, proofs
