"""恢复最佳模型，按提交模板的 ID 顺序输出浮点评分。"""

import argparse
import csv
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from model.common import ARTIFACTS, ROOT, manifest_digest, read_json, utc_now, write_json
from model.dataset import Collator, ExactShardSampler, ReviewDataset
from model.network import RatingModel
from model.runtime import cluster_info, initialize


def read_ids(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or 'id' not in reader.fieldnames:
            raise ValueError(f'{path} 缺少 id 列')
        return [row['id'] for row in reader]


def write_submission(output, cache_ids, predictions, template):
    """不依赖 Spark 行顺序；按真实字符串 ID 关联，严格检查完整性。"""
    template_ids = read_ids(template)
    if len(set(cache_ids)) != len(cache_ids) or len(set(template_ids)) != len(template_ids):
        raise ValueError('测试缓存或提交模板包含重复 ID')
    if set(cache_ids) != set(template_ids) or len(predictions) != len(cache_ids):
        raise ValueError('测试缓存与提交模板的 ID/行数不一致')
    if not np.isfinite(predictions).all() or not ((predictions >= 1) & (predictions <= 5)).all():
        raise ValueError('预测含 NaN/Inf 或超出 [1,5]')
    by_id = dict(zip(cache_ids, predictions))
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as destination:
        writer = csv.writer(destination)
        writer.writerow(['id', 'rating'])
        writer.writerows((identifier, f'{by_id[identifier]:.6f}') for identifier in template_ids)
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ARTIFACTS / 'prepared')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sample', type=Path, default=ROOT / 'data/raw_data/sampleSubmission.csv')
    parser.add_argument('--output', type=Path, default=ARTIFACTS / 'submission.csv')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.cpu_threads) < 1 or args.workers < 0:
        parser.error('batch-size/cpu-threads 必须为正，workers 不能为负')
    started = time.perf_counter()
    runtime = initialize(args.device, cpu_threads=args.cpu_threads)
    runtime.amp = runtime.amp and not args.no_amp
    temporary = None
    try:
        output = args.output.resolve()
        if runtime.primary:
            if output.exists() or output.with_suffix('.report.json').exists():
                raise FileExistsError(f'{output} 或其报告已存在，请指定新的 --output')
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = tempfile.TemporaryDirectory(prefix='prediction-', dir=output.parent)
        shared = [temporary.name if temporary else None]
        if runtime.world_size > 1:
            dist.broadcast_object_list(shared, src=0)
        shard_dir = Path(shared[0])
        manifest = read_json(args.data_dir / 'manifest.json')
        checkpoint = torch.load(args.checkpoint / 'weights.pt', map_location='cpu', weights_only=True)
        if checkpoint['prepared_manifest_sha256'] != manifest_digest(args.data_dir / 'manifest.json'):
            raise ValueError('缓存与训练时 manifest 不一致，不能混用不同模型或重新生成的缓存')
        model = RatingModel.from_checkpoint_config(args.checkpoint / 'encoder_config')
        model.load_state_dict(checkpoint['model_state'], strict=True)
        del checkpoint
        model.to(runtime.device).eval()
        hardware = cluster_info(runtime)
        data = ReviewDataset(args.data_dir, 'test')
        loader = DataLoader(data, batch_size=args.batch_size,
                            sampler=ExactShardSampler(data, runtime.rank, runtime.world_size),
                            num_workers=args.workers, collate_fn=Collator(manifest['pad_token_id']),
                            pin_memory=runtime.device.type == 'cuda')
        indices, values = [], []
        with torch.inference_mode():
            for batch in loader:
                inputs = {k: batch[k].to(runtime.device, non_blocking=True)
                          for k in ['input_ids', 'attention_mask']}
                with runtime.autocast():
                    predictions = model(**inputs).float().argmax(dim=1).add(1)
                indices.extend(batch['indices'].tolist())
                values.extend(predictions.cpu().tolist())
        np.savez(shard_dir / f'{runtime.rank}.npz', indices=np.asarray(indices, dtype=np.int64),
                 values=np.asarray(values, dtype=np.float32))
        runtime.synchronize()
        runtime.barrier()
        if runtime.primary:
            predictions = np.full(len(data), np.nan, dtype=np.float32)
            seen = np.zeros(len(data), dtype=bool)
            for rank in range(runtime.world_size):
                with np.load(shard_dir / f'{rank}.npz') as shard:
                    index = shard['indices']
                    if len(np.unique(index)) != len(index) or seen[index].any():
                        raise ValueError('推理分片出现重复样本')
                    predictions[index] = shard['values']
                    seen[index] = True
            if not seen.all():
                raise ValueError('推理结果缺少样本')
            ids = read_ids(args.data_dir / 'test/ids.csv')
            write_submission(output, ids, predictions, args.sample)
            # 包括权重加载、全部样本预测、分片合并和提交文件写出的端到端耗时。
            report = {'finished_at': utc_now(), 'hardware': hardware, 'test_samples': len(data),
                      'inference_total_seconds': time.perf_counter() - started,
                      'batch_size_per_process': args.batch_size,
                      'checkpoint': str(args.checkpoint.resolve()), 'output': str(output),
                      'prepared_manifest_sha256': manifest_digest(args.data_dir / 'manifest.json')}
            write_json(output.with_suffix('.report.json'), report)
            print(f'提交文件: {output}\n全部 {len(data):,} 条测试样本推理耗时: '
                  f'{report["inference_total_seconds"]:.2f}s', flush=True)
        runtime.barrier()
    finally:
        if temporary is not None:
            temporary.cleanup()
        runtime.close()


if __name__ == '__main__':
    main()
