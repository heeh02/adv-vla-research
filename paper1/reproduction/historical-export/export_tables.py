"""Format existing, small manuscript inputs for Drive; no scientific recomputation."""
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
RELEASE = PROJECT / 'docs/reviews/v7_19-plus-first-consolidation-20261017/delivery'
SOURCE = RELEASE / 'AdvVLA_v7_19_author_review.zip'
DEST = ROOT / 'analysis-ready'
DEST.mkdir(exist_ok=True)
records = []

def save(name, data, member, role):
    path = DEST / name
    path.write_bytes(data)
    item = dict(file=name, path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), source_archive=str(SOURCE), source_member=member, role=role)
    if name.endswith('.csv'):
        reader = csv.reader(io.StringIO(data.decode('utf-8-sig')))
        item['columns'] = next(reader)
        item['rows'] = sum(1 for _ in reader)
    records.append(item)

def csv_bytes(rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':')) if isinstance(v, (dict, list)) else v for k,v in row.items()})
    return stream.getvalue().encode('utf-8')

with zipfile.ZipFile(SOURCE) as archive:
    exact = [
        ('D3_episode_scalars.csv', 'aggregation-replay/data/episode-scalars.csv.gz', 'Original LIBERO; 2400 planned pairs, original two-observation scalar and outcome identities'),
        ('Plus_internal_PWH_cases.csv', 'figure-source/print-supplement-v15/data/PWH-JOIN-v1.csv.gz', 'Earlier Plus diagnostic/native joined scalars; source roles remain distinct'),
        ('Five_arm_command_motion_pairs.csv', 'figure-source/control-visuals-v1/data/C-COMMAND-MOTION-POINTS-v1.csv.gz', 'Two-model noise-pair command and motion plotting records'),
        ('P08_L18_cases.csv', 'figure-source/quantitative-v1/data/A-P08-L18-POINTS-v1.csv.gz', 'Original LIBERO cross-observation L18 geometry'),
        ('P08_all_layers.csv.gz', 'figure-source/supplement-source-v1/two-observation/data/P08-A-EPISODE-LAYER-v1.csv.gz', 'Complete released P08 derived table; retained compressed, not raw activations'),
        ('Attention_generation_summary.csv', 'figure-source/quantitative-v1/data/attention_generation.csv', 'Aggregated generation profiles, not independent cases'),
        ('Attention_command_correlations.csv', 'figure-source/quantitative-v1/data/internal_link.csv', 'Saved correlation estimates'),
        ('Attention_W_minus_P.csv', 'figure-source/quantitative-v1/data/internal_W_minus_P.csv', 'Saved paired association contrasts'),
        ('Prefix_HZV_structure_summary.csv', 'figure-source/quantitative-v1/data/structure_hzv.csv', 'Token/channel derived summaries'),
        ('Prefix_V_structure_summary.csv', 'figure-source/quantitative-v1/data/structure_v.csv', 'Value-space derived summaries'),
        ('Window_exposure_552.csv', 'figure-source/supplement-source-v1/windows/data/exposure_all552.csv', 'Existing randomized-window exposure records'),
        ('Five_arm_outcome_estimates.csv', 'case-analysis/expected-output/OUTCOMES.csv', 'Archived estimates for verification; not estimator inputs'),
        ('Five_arm_motion_estimates.csv', 'case-analysis/expected-output/MOTION.csv', 'Archived estimates for verification; not estimator inputs'),
    ]
    for dest, member, role in exact:
        data = archive.read(member)
        if member.endswith('.gz') and not dest.endswith('.gz'):
            data = gzip.decompress(data)
        save(dest, data, member, role)

    member = 'case-analysis/data/MOTION-C-PLOT-v1.json.gz'
    data = gzip.decompress(archive.read(member))
    save('Five_arm_case_records.json', data, member, 'Exact released 536 model-variant derived records; all unknowns, qualifications, supports and nested fields retained')
    obj = json.loads(data)
    # Explicitly a convenience export; nested fields remain JSON strings, no averages or joins.
    save('Five_arm_cases_flat.csv', csv_bytes(obj['rows']), member, 'Lossless row export; nested fields encoded as JSON; null scalar fields become CSV empty cells; JSON remains authoritative')

    member = 'current-basis-replay/data/task_scores.json.gz'
    obj = json.loads(gzip.decompress(archive.read(member)))
    for key in ['geometry', 'readout']:
        save('Current_basis_'+key+'_task_scores.csv', csv_bytes(obj[key]), member+'#'+key, 'Archived task-component scores, not individual predictions or a PCA refit')

    for dest, member in [
        ('Five_arm_protocol.md', 'case-analysis/README.md'),
        ('Current_basis_protocol.md', 'current-basis-replay/README.md'),
        ('D3_protocol.md', 'aggregation-replay/README.md'),
        ('Glossary.md', 'GLOSSARY.md'),
        ('Claim_source_map.md', 'CLAIM-SOURCE-MAP.md'),
    ]:
        save(dest, archive.read(member), member, 'Exact released documentation')

manifest = {'scope':'Existing v7.19 manuscript-derived inputs; not all project raw data', 'source_archive_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'files':records}
(ROOT/'ANALYSIS-FILE-MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
lines=['# 可分析数据目录与字段字典', '', '本目录提供已有论文的派生输入，保留原统计规则；不是全项目原始神经数组。每张表分别分析，不能沿任务名称跨采集拼接。下表中的行数不是独立样本数。', '']
for item in records:
    lines += ['## '+item['file'], '', item['role'], '', '原件成员：`'+item['source_member']+'`。']
    if 'rows' in item:
        lines += ['行数：'+str(item['rows'])+'。列名：`'+'`, `'.join(item['columns'])+'`。']
    lines += ['']
(ROOT/'DATA_DICTIONARY.md').write_text('\n'.join(lines))
print(json.dumps({'files':len(records),'bytes':sum(x['bytes'] for x in records),'csv_rows':{x['file']:x.get('rows') for x in records if 'rows' in x}},ensure_ascii=False,indent=2))
