"""离线端到端测试：小型随机 DeBERTa，检查流程，不代表真实预测效果。"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2Model

from model.common import ROOT, read_json

COLUMNS = ['id', 'title', 'user_id', 'comment', 'prod_id', 'parent_prod_id',
           'time', 'votes', 'purchased', 'rating']


def run_step(name, command, directory, environment):
    print(f'检查 {name} ...', flush=True)
    log = directory / f'{name}.log'
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(encoding='utf-8')[-12000:], flush=True)
        raise RuntimeError(f'{name} 失败；完整日志: {log}')


def check(directory, distributed):
    directory.mkdir(parents=True, exist_ok=True)
    encoder = directory / 'tiny_encoder'
    encoder.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'review', 'good', 'bad', 'title',
             'tea', 'a', 'b', 'c', '!', '"', ',', '<', '>', '/', '&'] + [str(i) for i in range(100)]
    (encoder / 'vocab.txt').write_text('\n'.join(vocab) + '\n', encoding='utf-8')
    tokenizer = BertTokenizerFast(vocab_file=str(encoder / 'vocab.txt'), do_lower_case=True)
    tokenizer.save_pretrained(encoder)
    # 主干与正式路线使用相同的 DeBERTa-v2/v3 实现，只缩小层数/维度用于 CPU 检查。
    config = DebertaV2Config(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32,
                            max_position_embeddings=64, relative_attention=True,
                            position_buckets=8, max_relative_positions=32,
                            position_biased_input=False, pos_att_type=['p2c', 'c2p'],
                            conv_kernel_size=3, pad_token_id=0)
    DebertaV2Model(config).save_pretrained(encoder)
    raw = directory / 'raw'
    raw.mkdir()
    records = []
    for i in range(80):
        row = dict(zip(COLUMNS, [str(i), f'title {i}', f'u{i % 7}',
                                f'good<br />tea &#34;review&#34; {i}\nA, B',
                                f'p{i % 5}', 'parent', '1600000000000', '0', 'True', str(i % 5 + 1)]))
        records.append(row)
    records.append(dict(records[0], id='duplicate'))
    records.append(dict(records[0], id='conflict', rating='2'))
    records.append(dict(records[1], id='invalid', rating='9'))
    with (raw / 'train.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    test_ids = ['009', 'B', 'A', '100', 'X']
    with (raw / 'test.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS[:-1])
        writer.writeheader()
        writer.writerows({k: v for k, v in dict(records[i], id=identifier).items() if k != 'rating'}
                        for i, identifier in enumerate(test_ids))
    with (raw / 'sample.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'rating'])
        writer.writerows((identifier, 3) for identifier in reversed(test_ids))
    environment = os.environ.copy()
    environment.update({'PYSPARK_PYTHON': sys.executable, 'TOKENIZERS_PARALLELISM': 'false',
                        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '1'})
    # 直接运行而非另一个环境的 spark-submit，保证 driver 与 workers 的 Python 相同。
    prepared = directory / 'prepared'
    run_step('prepare', [sys.executable, '-m', 'model.prepare', '--train', str(raw / 'train.csv'),
                        '--test', str(raw / 'test.csv'), '--model-name', str(encoder),
                        '--output', str(prepared), '--spark-master', 'local[2]', '--spark-memory', '1g',
                        '--partitions', '2', '--max-length', '32', '--validation-fraction', '0.2'],
             directory, environment)
    import pyarrow.parquet as pq
    groups = []
    for split in ['train', 'validation']:
        rows = pq.read_table(prepared / 'parquet' / split).to_pydict()
        groups.append(set(zip(rows['title'], rows['comment'])))
    assert not groups[0] & groups[1], '相同文本跨训练/验证集'
    manifest = read_json(prepared / 'manifest.json')
    assert sum(manifest['splits'][s]['rows'] for s in ['train', 'validation']) == 81
    prefix = ([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
              if distributed else [sys.executable, '-m'])
    run = directory / 'run'
    run_step('train', prefix + ['model.train', '--data-dir', str(prepared), '--output', str(run),
                               '--epochs', '1', '--batch-size', '8', '--eval-batch-size', '3',
                               '--accumulation-steps', '2', '--workers', '0', '--device', 'cpu',
                               '--log-every', '2'], directory, environment)
    submission = directory / 'submission.csv'
    run_step('predict', prefix + ['model.predict', '--data-dir', str(prepared),
                                 '--checkpoint', str(run / 'best'), '--sample', str(raw / 'sample.csv'),
                                 '--output', str(submission), '--batch-size', '2',
                                 '--workers', '0', '--device', 'cpu'], directory, environment)
    with submission.open() as stream:
        predictions = list(csv.DictReader(stream))
    assert [row['id'] for row in predictions] == list(reversed(test_ids))
    assert all(1 <= float(row['rating']) <= 5 for row in predictions)
    report = read_json(run / 'training_report.json')
    assert report['epochs'][0]['validation']['samples'] == manifest['splits']['validation']['rows']
    assert report['total_offline_seconds'] >= report['training_total_seconds']
    print(f'端到端检查通过（{"2 进程 CPU" if distributed else "单进程 CPU"}）。'
          '这是随机小模型的流程检查，不是正式训练结果。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, help='保留日志/产物的空目录；默认自动清理临时目录')
    parser.add_argument('--distributed', action='store_true', help='以 2 个 CPU 进程检查 DDP 与分片推理')
    args = parser.parse_args()
    if args.directory:
        directory = args.directory.resolve()
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('smoke-test directory 必须为空')
        check(directory, args.distributed)
    else:
        with tempfile.TemporaryDirectory(prefix='rating-smoke-') as temporary:
            check(Path(temporary), args.distributed)


if __name__ == '__main__':
    main()
