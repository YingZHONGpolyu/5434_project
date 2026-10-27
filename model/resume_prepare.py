"""Resume tokenization from completed Parquet, including legacy runs without a stage report."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
from model.common import ARTIFACTS, TEXT_VERSION, hardware_info, read_json, utc_now, write_json
from model.prepare import tokenize_split_parallel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=ARTIFACTS / 'prepared')
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--original-start-file', type=Path)
    a = p.parse_args()
    root = a.data_dir.resolve()
    if (root / 'manifest.json').exists():
        raise FileExistsError('Preparation is already complete')
    stage = root / 'parquet_report.json'
    started = a.original_start_file.stat().st_mtime if a.original_start_file else time.time()
    if stage.exists():
        report = read_json(stage)
    else:
        report = {'started_at': datetime.fromtimestamp(started, timezone.utc).isoformat(),
                  'hardware': hardware_info(), 'spark_version': '3.5.7', 'spark_master': 'local[8]',
                  'recovered_from_completed_parquet': True, 'splits': {},
                  'parameters': {'max_length': 256, 'token_batch_size': 1024,
                                 'token_workers': a.workers}}
        for split in ['train', 'validation', 'test']:
            directory = root / 'parquet' / split
            if not (directory / '_SUCCESS').exists():
                raise RuntimeError(f'{split} Parquet is incomplete')
            count, label_sum = 0, 0.0
            for path in sorted(directory.glob('part-*.parquet')):
                part = pq.ParquetFile(path)
                count += part.metadata.num_rows
                if split != 'test':
                    for batch in part.iter_batches(columns=['rating']):
                        label_sum += pc.sum(batch.column(0)).as_py()
            report['splits'][split] = {'rows': count}
            if split != 'test':
                report['splits'][split]['mean_rating'] = label_sum / count
        # Original Spark timing was not persisted by the old implementation.
        report['spark_seconds'] = None
        write_json(stage, report)
    token_started = time.perf_counter()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', use_fast=True)
    for split, stats in report['splits'].items():
        target = root / split
        if target.exists():
            backup = root / (split + '.partial-' + str(time.time_ns()))
            target.rename(backup)
            print(f'Preserved interrupted cache: {backup}', flush=True)
        tokenize_split_parallel(root / 'parquet' / split, target, stats['rows'], root / 'tokenizer',
                                report['parameters']['max_length'], 1024, a.workers)
    report.update({'model_name': 'microsoft/deberta-v3-base',
                   'model_revision': '8ccc9b6f36199bec6961081d44eb72fb3f7353f3',
                   'max_length': report['parameters']['max_length'],
                   'pad_token_id': tokenizer.pad_token_id, 'text_version': TEXT_VERSION,
                   'token_workers': a.workers, 'tokenization_seconds': time.perf_counter()-token_started,
                   'preprocessing_seconds': time.time()-started, 'finished_at': utc_now()})
    write_json(root / 'manifest.json', report)
    print('Preparation complete:', root, flush=True)


if __name__ == '__main__':
    main()
