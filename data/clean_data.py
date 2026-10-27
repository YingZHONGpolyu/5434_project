"""流式清洗训练集；直接运行 python3 data/clean_data.py，无第三方依赖。"""

import argparse
import csv
import hashlib
import html
import json
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
COLUMNS = ['id', 'title', 'user_id', 'comment', 'prod_id', 'parent_prod_id',
           'time', 'votes', 'purchased', 'rating']
TAGS = re.compile(r'<\s*/?\s*[a-zA-Z][^>]*>')
SPACE = re.compile(r'\s+')
INTEGER = re.compile(r'[+-]?\d+(?:\.0+)?\Z')


def clean_text(value):
    # 先移除实际 HTML 标签，再解码实体，保留实体表示的字面文本。
    value = html.unescape(TAGS.sub(' ', value))
    value = unicodedata.normalize('NFC', value)
    value = ''.join(c for c in value if c in '\n\r\t' or unicodedata.category(c) != 'Cc')
    return SPACE.sub(' ', value).strip()


def integer(value, minimum, maximum=None):
    if not INTEGER.fullmatch(value):
        return ''
    number = int(value.split('.')[0])
    if number < minimum or (maximum is not None and number > maximum):
        return ''
    return str(number)


def clean_dataset(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError('输出文件不能覆盖原始训练集')
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    missing = Counter()
    ratings = Counter()
    # 磁盘索引避免把整个训练集或全部去重键放入内存。
    with tempfile.TemporaryDirectory(prefix='train_clean_', dir=destination.parent) as temp:
        connection = sqlite3.connect(str(Path(temp) / 'seen.sqlite'))
        connection.execute('PRAGMA journal_mode=OFF')
        connection.execute('PRAGMA synchronous=OFF')
        connection.execute('CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID')
        staged = Path(temp) / 'train.csv'
        try:
            with source.open(encoding='utf-8-sig', newline='') as inp, staged.open('w', encoding='utf-8', newline='') as out:
                reader = csv.DictReader(inp)
                if reader.fieldnames != COLUMNS:
                    raise ValueError(f'训练集字段不符合预期: {reader.fieldnames}')
                writer = csv.DictWriter(out, fieldnames=COLUMNS)
                writer.writeheader()
                for raw in reader:
                    counts['input_rows'] += 1
                    if None in raw or any(v is None for v in raw.values()):
                        counts['removed_malformed_rows'] += 1
                        continue
                    row = {k: v.strip() for k, v in raw.items()}
                    for key in ['title', 'comment']:
                        row[key] = clean_text(row[key])
                    row['rating'] = integer(row['rating'], 1, 5)
                    if not row['rating']:
                        counts['removed_invalid_rating'] += 1
                        continue
                    # 无评论正文但有标题时仍保留；两者皆空则无法提供文本信息。
                    if not row['title'] and not row['comment']:
                        counts['removed_empty_text'] += 1
                        continue
                    row['votes'] = integer(row['votes'], 0)
                    row['time'] = integer(row['time'], 0)
                    row['purchased'] = {'true': 'True', 'false': 'False', '1': 'True', '0': 'False'}.get(row['purchased'].lower(), '')
                    # 排除行 ID 后按全部业务字段去重，保留第一次出现的记录。
                    payload = json.dumps([row[k] for k in COLUMNS if k != 'id'], ensure_ascii=False, separators=(',', ':'))
                    digest = hashlib.sha256(payload.encode('utf-8')).digest()
                    cursor = connection.execute('INSERT OR IGNORE INTO seen VALUES (?)', (digest,))
                    if cursor.rowcount == 0:
                        counts['removed_duplicate_rows'] += 1
                        continue
                    counts['modified_retained_rows'] += int(row != raw)
                    for key in ['votes', 'time', 'purchased']:
                        if raw[key].strip() and not row[key]:
                            counts[f'invalid_{key}_set_missing'] += 1
                    missing.update(k for k, v in row.items() if not v)
                    ratings[row['rating']] += 1
                    writer.writerow(row)
                    counts['output_rows'] += 1
                    if counts['input_rows'] % 100000 == 0:
                        connection.commit()
                        print(f"已读取 {counts['input_rows']:,} 行，保留 {counts['output_rows']:,} 行", flush=True)
            connection.commit()
            staged.replace(destination)
        finally:
            connection.close()
    report = {
        'source': str(source), 'output': str(destination),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'counts': dict(counts), 'missing_values_after': {k: missing[k] for k in COLUMNS},
        'rating_distribution_after': dict(sorted(ratings.items())),
        'rules': [
            '保留原始字段、行 ID 和毫秒时间戳单位，原始文件不变',
            '文本移除 HTML 标签、解码 HTML 实体、NFC 规范化、移除控制字符、合并空白',
            '不删除停用词、否定词、标点，不改写评分或按情感猜测标签',
            '移除字段数量异常、评分不是 1–5 整数、标题与正文均空的记录',
            '排除 id 后按清洗后的所有业务字段去重，保留首条',
            'votes 和 time 限非负整数，无法解析则置空；有效极端值保留',
            'purchased 规范为 True/False，无法解析则置空',
            '其余缺失值保留为空，不利用全量数据估计填充值，避免后续验证集泄漏',
        ],
    }
    report_path = destination.with_name(destination.stem + '_cleaning_report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['counts'], ensure_ascii=False, indent=2))
    print(f'清洗结果: {destination}\n统计报告: {report_path}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATA_DIR / 'raw_data' / 'train.csv')
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'cleaned_data' / 'train.csv')
    args = parser.parse_args()
    csv.field_size_limit(100_000_000)
    clean_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
