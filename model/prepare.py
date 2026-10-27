"""Spark 清洗/划分数据，再一次性分词并缓存为内存映射数组。"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from pyspark import StorageLevel
from pyspark.sql import SparkSession, functions as F, types as T
from transformers import AutoTokenizer

from model.common import ARTIFACTS, ROOT, TEXT_VERSION, clean_text, hardware_info, utc_now, write_json

COLUMNS = ['id', 'title', 'user_id', 'comment', 'prod_id', 'parent_prod_id',
           'time', 'votes', 'purchased', 'rating']


def read_reviews(spark, path, training):
    """显式字符串 schema，防止 ID 被转换、长文本和双引号被误解析。"""
    columns = COLUMNS if training else COLUMNS[:-1]
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        if next(csv.reader(source)) != columns:
            raise ValueError(f'{path} 的 CSV 表头不符合预期: {columns}')
    schema = T.StructType([T.StructField(c, T.StringType(), True) for c in columns])
    return (spark.read.schema(schema).option('header', True)
            .option('multiLine', True).option('quote', '"').option('escape', '"')
            .option('mode', 'FAILFAST').csv(str(Path(path).resolve())))


def tokenize_split(parquet_dir, target, count, tokenizer, max_length, batch_size):
    """流式读取 Parquet；固定宽度磁盘数组避免将 765 万条样本放入 RAM。

    训练时再按每批的真实最长长度裁剪，因此不会始终计算 256 个位置。
    """
    target.mkdir(parents=True)
    tokens = np.lib.format.open_memmap(target / 'input_ids.npy', mode='w+', dtype=np.int32,
                                       shape=(count, max_length))
    lengths = np.lib.format.open_memmap(target / 'lengths.npy', mode='w+', dtype=np.int32, shape=(count,))
    labels = np.lib.format.open_memmap(target / 'labels.npy', mode='w+', dtype=np.float32, shape=(count,))
    offset = 0
    with (target / 'ids.csv').open('w', encoding='utf-8', newline='') as ids_file:
        writer = csv.writer(ids_file)
        writer.writerow(['id'])
        for path in sorted(parquet_dir.glob('part-*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
                rows = batch.to_pydict()
                size = batch.num_rows
                if offset + size > count:
                    raise RuntimeError('Parquet 行数超过已统计的行数')
                # 标题作为第一段，正文作为第二段；二者都允许被截断。
                encoded = tokenizer(rows['title'], rows['comment'], truncation='longest_first',
                                    max_length=max_length, padding='max_length',
                                    return_attention_mask=True, return_token_type_ids=False,
                                    return_tensors='np')
                tokens[offset:offset + size] = encoded['input_ids']
                lengths[offset:offset + size] = encoded['attention_mask'].sum(axis=1)
                labels[offset:offset + size] = rows.get('rating', [0.0] * size)
                writer.writerows((value,) for value in rows['id'])
                offset += size
    if offset != count:
        raise RuntimeError(f'分词行数错误: expected={count}, actual={offset}')
    for array in (tokens, lengths, labels):
        array.flush()
    return {'rows': count}


def _tokenize_part(task):
    """Each worker writes one disjoint slice; no array/header races."""
    parquet_file, target, offset, tokenizer_path, max_length, batch_size = task
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    target = Path(target)
    arrays = [np.load(target / name, mmap_mode='r+') for name in
              ['input_ids.npy', 'lengths.npy', 'labels.npy']]
    count = 0
    ids_path = target / (Path(parquet_file).stem + '.ids.part')
    with ids_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        for batch in pq.ParquetFile(parquet_file).iter_batches(batch_size=batch_size):
            rows = batch.to_pydict()
            n = batch.num_rows
            encoded = tokenizer(rows['title'], rows['comment'], truncation='longest_first',
                                max_length=max_length, padding='max_length',
                                return_attention_mask=True, return_token_type_ids=False,
                                return_tensors='np')
            start = offset + count
            arrays[0][start:start+n] = encoded['input_ids']
            arrays[1][start:start+n] = encoded['attention_mask'].sum(axis=1)
            arrays[2][start:start+n] = rows.get('rating', [0.0] * n)
            writer.writerows((value,) for value in rows['id'])
            count += n
    for array in arrays:
        array.flush()
    return count


def tokenize_split_parallel(parquet_dir, target, count, tokenizer_path, max_length,
                            batch_size, workers=7):
    import multiprocessing
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    target.mkdir(parents=True)
    for name, dtype, shape in [('input_ids.npy', np.int32, (count, max_length)),
                               ('lengths.npy', np.int32, (count,)),
                               ('labels.npy', np.float32, (count,))]:
        array = np.lib.format.open_memmap(target / name, mode='w+', dtype=dtype, shape=shape)
        array.flush()
        del array
    tasks, offset = [], 0
    for path in sorted(Path(parquet_dir).glob('part-*.parquet')):
        tasks.append((str(path), str(target), offset, str(tokenizer_path), max_length, batch_size))
        offset += pq.ParquetFile(path).metadata.num_rows
    if offset != count:
        raise RuntimeError(f'Parquet metadata rows {offset} != expected {count}')
    # Spawn avoids inheriting Rust tokenizer thread pools and Spark JVM handles.
    os.environ['RAYON_NUM_THREADS'] = '2'
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        done = 0
        for result in pool.map(_tokenize_part, tasks):
            done += result
            print(f'  {Path(target).name}: {done:,}/{count:,} rows ({workers} workers)', flush=True)
    if done != count:
        raise RuntimeError(f'Tokenized rows {done} != expected {count}')
    with (target / 'ids.csv').open('w', encoding='utf-8', newline='') as stream:
        csv.writer(stream).writerow(['id'])
        for task in tasks:
            part = target / (Path(task[0]).stem + '.ids.part')
            with part.open(encoding='utf-8', newline='') as source:
                shutil.copyfileobj(source, stream)
            part.unlink()
    return {'rows': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 默认从原始数据开始，避免对已经清洗的文本进行第二次 HTML 解码。
    parser.add_argument('--train', type=Path, default=ROOT / 'data/raw_data/train.csv')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/raw_data/test.csv')
    parser.add_argument('--output', type=Path, default=ARTIFACTS / 'prepared')
    parser.add_argument('--model-name', default='microsoft/deberta-v3-base')
    parser.add_argument('--revision', default='8ccc9b6f36199bec6961081d44eb72fb3f7353f3',
                        help='默认固定已验证的 DeBERTa commit；更换远程模型时指定对应 revision')
    parser.add_argument('--spark-master', default='local[8]')
    parser.add_argument('--spark-memory', default='8g')
    parser.add_argument('--partitions', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--validation-fraction', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--token-batch-size', type=int, default=1024)
    parser.add_argument('--limit-train', type=int, default=0, help='仅供小规模试运行；0 表示全量')
    parser.add_argument('--token-workers', type=int, default=7)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.max_length < 8:
        parser.error('validation-fraction 必须在 (0,1)，max-length 至少为 8')
    if args.partitions < 1 or args.token_batch_size < 1 or args.limit_train < 0 or args.token_workers < 1:
        parser.error('partitions/token-batch-size 必须为正，limit-train 不能为负')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} 非空；请指定新的 --output，防止覆盖现有数据')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
    os.environ.setdefault('RAYON_NUM_THREADS', '8')
    spark = (SparkSession.builder.master(args.spark_master).appName('review-rating-prepare')
             .config('spark.driver.memory', args.spark_memory)
             .config('spark.sql.shuffle.partitions', args.partitions).getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    report = {'started_at': utc_now(), 'hardware': hardware_info(),
              'spark_version': spark.version, 'spark_master': args.spark_master,
              'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        normalize = F.udf(clean_text, T.StringType())
        raw = read_reviews(spark, args.train, True)
        if args.limit_train:
            raw = raw.orderBy('id').limit(args.limit_train)
        # 单个多行 CSV 的解析并行度有限；读取后重新分区，清洗、去重和写出并行执行。
        train = (raw.repartition(args.partitions)
                 .withColumn('title', normalize('title')).withColumn('comment', normalize('comment'))
                 .withColumn('rating', F.col('rating').cast('double')))
        train = train.filter(F.col('rating').between(1, 5) &
                             (F.col('rating') == F.floor('rating')) &
                             ((F.length('title') + F.length('comment')) > 0))
        # 模型只看文本，因此相同标题+正文必须进入同一集合，包括标签冲突的文本。
        train = train.withColumn('_group', F.sha2(F.to_json(F.struct('title', 'comment')), 256))
        # 去重时仍保留不同用户/商品的独立评论，不按情感或标题改写标签。
        train = train.dropDuplicates([c for c in COLUMNS if c != 'id'])
        train = train.withColumn('_bucket', F.pmod(F.xxhash64('_group', F.lit(args.seed)), F.lit(1000000)))
        train = train.select('id', 'title', 'comment', 'rating', '_bucket').persist(StorageLevel.DISK_ONLY)
        boundary = int(args.validation_fraction * 1000000)
        splits = {
            'train': train.filter(F.col('_bucket') >= boundary).drop('_bucket'),
            'validation': train.filter(F.col('_bucket') < boundary).drop('_bucket'),
        }
        report['splits'] = {}
        for name, frame in splits.items():
            stats = frame.agg(F.count('*').alias('rows'), F.avg('rating').alias('mean_rating')).first()
            if not stats.rows:
                raise ValueError(f'{name} 集合为空；小样例请增加不同文本数量或验证集比例')
            report['splits'][name] = {'rows': stats.rows, 'mean_rating': stats.mean_rating}
            frame.write.mode('error').parquet(str(output / 'parquet' / name))
        train.unpersist()
        test = (read_reviews(spark, args.test, False).repartition(args.partitions)
                .select('id', normalize('title').alias('title'), normalize('comment').alias('comment'))
                .persist(StorageLevel.DISK_ONLY))
        count = test.count()
        if not count or test.filter(F.col('id').isNull() | (F.trim('id') == '')).limit(1).count():
            raise ValueError('测试集为空或含缺失 ID')
        if test.select('id').distinct().count() != count:
            raise ValueError('测试集存在重复 ID，无法安全生成提交')
        report['splits']['test'] = {'rows': count}
        test.write.mode('error').parquet(str(output / 'parquet/test'))
        test.unpersist()
    finally:
        spark.stop()
    report['spark_seconds'] = time.perf_counter() - started
    write_json(output / 'parquet_report.json', report)
    token_started = time.perf_counter()
    model_path = Path(args.model_name).expanduser()
    if model_path.is_dir():
        source = str(model_path.resolve())
        revision = None
        tokenizer_source = source
    else:
        source = args.model_name
        # 固定实际 commit，保证 tokenizer 与训练主干使用完全相同的版本。
        tokenizer_source = snapshot_download(source, revision=args.revision,
                                             allow_patterns=['*.json', '*.model', '*.txt'])
        revision = Path(tokenizer_source).name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer 必须定义 pad_token_id')
    tokenizer.save_pretrained(output / 'tokenizer')
    for name, stats in report['splits'].items():
        print(f'分词并缓存 {name}: {stats["rows"]:,} 行', flush=True)
        tokenize_split_parallel(output / 'parquet' / name, output / name, stats['rows'],
                                output / 'tokenizer', args.max_length, args.token_batch_size,
                                args.token_workers)
    report.update({
        'model_name': source, 'model_revision': revision, 'max_length': args.max_length,
        'pad_token_id': tokenizer.pad_token_id, 'text_version': TEXT_VERSION,
        'tokenization_seconds': time.perf_counter() - token_started,
        'preprocessing_seconds': time.perf_counter() - started,
        'finished_at': utc_now(),
    })
    # 只有所有缓存完整时才写 manifest；训练拒绝使用未完成的数据目录。
    write_json(output / 'manifest.json', report)
    print(f'数据准备完成: {output}', flush=True)


if __name__ == '__main__':
    main()
