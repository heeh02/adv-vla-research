#!/usr/bin/env python3
# Export packaging only; no scientific analysis or manuscript edits.
from pathlib import Path
import collections
import hashlib
import html
import json
import re
import subprocess
import zipfile

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'AdvVLA_Paper1_Cloud_Analysis_20261029'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def render_value(v):
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, indent=2)

inventory = ROOT / '02_All_Experiment_Inventory'
reader = ['EXPERIMENT EVIDENCE READER | 2026-10-29',
          '79 audit entries; not 79 independent experiments. Source fields reproduced without new scientific analysis.',
          'Use ../SOURCE-LOCATOR.html or ../SOURCE-LOCATOR.json to resolve source IDs to bundled files.']
for fname, key in [('HISTORICAL.json', 'entries'), ('MECHANISM.json', 'entries'),
                   ('TRAINING-AND-OTHER-LINES.json', 'families')]:
    data = json.loads((inventory / fname).read_text())
    reader += ['\n' + '=' * 72, fname]
    for record in data[key]:
        reader += ['\n' + '-' * 72]
        for k, v in record.items():
            reader.append(k + ':\n' + render_value(v))
(inventory / 'ALL_EXPERIMENTS_READER.txt').write_text('\n\n'.join(reader) + '\n', encoding='utf-8')

ms = json.loads((ROOT / '01_Manuscript/MANUSCRIPT-MAP.json').read_text())
original_to_package = {e['source']: e['file'] for e in ms['source_copies']}
sources = json.loads((ROOT / '03_Experiment_Evidence/SOURCE-MAP.json').read_text())
for e in sources['entries']:
    if e.get('source_original_path') and e.get('packaged_path'):
        original_to_package[e['source_original_path']] = '03_Experiment_Evidence/' + e['packaged_path']
resolved = []
for e in sources['entries']:
    original = e.get('source_original_path') or (e.get('source_original_archive', '') + '::' + e.get('source_archive_member', ''))
    dest = ('03_Experiment_Evidence/' + e['packaged_path']) if e.get('packaged_path') else original_to_package.get(original)
    if dest and not (ROOT / dest).is_file():
        raise RuntimeError('Missing source target: ' + dest)
    if dest and e.get('source_sha256') and sha(ROOT / dest) != e['source_sha256']:
        raise RuntimeError('Source identity mismatch: ' + dest)
    resolved.append({'inventory': e['inventory'], 'source_id': e['source_id'],
                     'file': dest, 'status': 'resolved_in_package' if dest else e['status'],
                     'original_path': original,
                     'sha256': sha(ROOT / dest) if dest else e.get('source_sha256'),
                     'note': e.get('reason', e.get('role', ''))})
save_json(ROOT / 'SOURCE-LOCATOR.json', {'date': '2026-10-29', 'entries': resolved})

css = 'body{font-family:system-ui,sans-serif;max-width:1050px;margin:36px auto;padding:0 24px;color:#183047;line-height:1.65}a{color:#1464a0}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit}small{color:#53677b}'
rows = []
for e in resolved:
    target = ('<a href="' + html.escape(e['file'], quote=True) + '">' + html.escape(e['file']) + '</a>') if e['file'] else html.escape(e['status'])
    rows.append('<tr><td>' + html.escape(e['inventory'] + ':' + e['source_id']) + '</td><td>' + target + '</td></tr>')
(ROOT / 'SOURCE-LOCATOR.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Source locator</title><style>' + css + '</style><body><h1>Inventory to bundled evidence</h1><p>Omitted deployment records remain explicitly marked. Manuscript sources resolve to the manuscript folder.</p><table><tr><th>Inventory source</th><th>Bundled file</th></tr>' + ''.join(rows) + '</table></body></html>', encoding='utf-8')

links = [
    ('r11 main PDF', '01_Manuscript/r11/Paper1_r11_main.pdf'),
    ('r11 supplement PDF', '01_Manuscript/r11/Paper1_r11_supplement.pdf'),
    ('r12 main PDF', '01_Manuscript/r12/Paper1_r12_main.pdf'),
    ('r12 supplement PDF', '01_Manuscript/r12/Paper1_r12_supplement.pdf'),
    ('Experiment atlas / 79 audit entries', '02_All_Experiment_Inventory/EVIDENCE-ATLAS.html'),
    ('All experiment reader / searchable text', '02_All_Experiment_Inventory/ALL_EXPERIMENTS_READER.txt'),
    ('Portable source locator', 'SOURCE-LOCATOR.html'),
    ('Reference papers and reading purposes', '04_Reference_Papers/INDEX.md'),
    ('Cloud analysis request', '00_CLOUD_ANALYSIS_PROMPT.txt'),
]
for _, target in links:
    assert (ROOT / target).is_file(), target
body = '<ul>' + ''.join('<li><a href="' + html.escape(p, quote=True) + '">' + html.escape(label) + '</a></li>' for label, p in links) + '</ul>'
guide = (ROOT / '00_START_HERE.txt').read_text()
(ROOT / '00_START_HERE.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Adv-VLA Cloud Analysis Pack</title><style>' + css + '</style><body><h1>Adv-VLA Paper 1 — 云端深入分析</h1>' + body + '<pre>' + html.escape(guide) + '</pre></body></html>', encoding='utf-8')

files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['MANIFEST.json', 'PACKAGE-QA.json'])
assert not any(p.is_symlink() for p in ROOT.rglob('*'))
assert not any(p.name == '.DS_Store' or '__MACOSX' in p.parts for p in files)
pdfs = []
for p in files:
    if p.suffix == '.json':
        json.loads(p.read_text())
    if p.suffix == '.pdf':
        assert p.read_bytes()[:5] == b'%PDF-', str(p)
        proc = subprocess.run(['pdfinfo', str(p)], capture_output=True, text=True, check=True)
        match = re.search(r'^Pages:\s+(\d+)', proc.stdout, re.M)
        assert match, str(p)
        assert re.search(r'^Encrypted:\s+no', proc.stdout, re.M), str(p)
        pdfs.append({'file': str(p.relative_to(ROOT)), 'pages': int(match.group(1))})

patterns = {
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'aws_access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'github_token': r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    'google_api_key': r'\bAIza[A-Za-z0-9_-]{30,}\b',
    'signed_url': r'(?:X-Amz-Signature|X-Goog-Signature|x-bce-signature)=[A-Fa-f0-9]{20,}',
}
hits = []
for p in files:
    if p.suffix.lower() in ['.md', '.txt', '.json', '.tex', '.bib', '.html', '.py', '.csv', '.tsv']:
        text = p.read_text(errors='replace')
        for name, pattern in patterns.items():
            if re.search(pattern, text):
                hits.append({'file': str(p.relative_to(ROOT)), 'pattern': name})
if hits:
    save_json(BASE / 'SENSITIVE-ITEMS-TO-REVIEW.json', hits)
    raise RuntimeError('Review sensitive pattern hits before packaging')

qa = {'date': '2026-10-29', 'status': 'PASS', 'scope': 'File and reading-package checks, not scientific reproduction',
      'manuscript_edits': False, 'new_experiments': False, 'new_scientific_statistics': False,
      'pdf_count': len(pdfs), 'pdfs': pdfs, 'json_parse': 'all passed', 'sensitive_pattern_hits': [],
      'source_entries': len(resolved), 'source_resolved': sum(e['file'] is not None for e in resolved),
      'unpackaged_source_ids': [e['inventory'] + ':' + e['source_id'] for e in resolved if not e['file']],
      'source_map_hash_checks': 'passed for all resolved entries with supplied source hash',
      'zip_integrity': 'See archive receipt alongside ZIP; verified after compression'}
save_json(ROOT / 'PACKAGE-QA.json', qa)
members = []
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p != ROOT / 'MANIFEST.json':
        members.append({'file': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
save_json(ROOT / 'MANIFEST.json', {'date': '2026-10-29', 'purpose': 'private cloud analysis; not an anonymous submission',
                                 'files': members, 'file_count_excluding_manifest': len(members),
                                 'total_bytes_excluding_manifest': sum(e['bytes'] for e in members)})
archive = BASE / (ROOT.name + '.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():
            z.write(p, str(p.relative_to(BASE)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for e in members:
        data = z.read(ROOT.name + '/' + e['file'])
        assert len(data) == e['bytes'] and hashlib.sha256(data).hexdigest() == e['sha256']
    count = len(z.infolist())
receipt = {'date': '2026-10-29', 'archive': archive.name, 'bytes': archive.stat().st_size,
           'sha256': sha(archive), 'members': count, 'crc_check': 'PASS',
           'member_sha256_check': 'PASS', 'pdf_count': len(pdfs),
           'reference_papers': json.loads((ROOT / '04_Reference_Papers/REFERENCES.json').read_text())['count']}
save_json(BASE / 'ZIP-RECEIPT.json', receipt)
(BASE / (archive.name + '.sha256')).write_text(receipt['sha256'] + '  ' + archive.name + '\n')
print(json.dumps(receipt, ensure_ascii=False, indent=2))
