"""MSE 回归训练；验证指标为全部样本的 RMSE，支持 CPU/单卡/7 卡 DDP。"""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from model.common import ROOT, ARTIFACTS, manifest_digest, read_json, utc_now, write_json
from model.dataset import Collator, ExactShardSampler, ReviewDataset
from model.network import RatingModel
from model.runtime import cluster_info, initialize


@torch.inference_mode()
def evaluate(model, loader, runtime, baseline=None):
    model.eval(); totals=torch.zeros(4,dtype=torch.float64,device=runtime.device)
    for batch in loader:
        labels=batch['labels'].to(runtime.device).float()
        with runtime.autocast(): logits=model(input_ids=batch['input_ids'].to(runtime.device),attention_mask=batch['attention_mask'].to(runtime.device)).float()
        pred=logits.argmax(1).float()+1
        totals[0]+=(pred-labels).double().square().sum(); totals[1]+=(pred==labels).sum(); totals[2]+=len(labels)
        totals[3]+=nn.functional.cross_entropy(logits,labels.long()-1,reduction='sum').double()
    vals=runtime.sum(totals).cpu().tolist(); n=vals[2]
    return {'rmse':math.sqrt(vals[0]/n),'accuracy':vals[1]/n,'correct':int(vals[1]),'cross_entropy':vals[3]/n,'samples':int(n)}


def train_epoch(model, loader, valid_loader, optimizer, scheduler, scaler, runtime, accumulation, log_every, tracker=None, epoch=1, eval_every=50, checkpoint_root=None, manifest_sha256=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(2, dtype=torch.float64, device=runtime.device)
    for step, batch in enumerate(loader):
        end_group = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        # 梯度累积的最后一组可能不足 accumulation 批，按真实样本数归一化。
        group_start = (step // accumulation) * accumulation
        group_samples = min(accumulation * loader.batch_size,
                            len(loader.sampler) - group_start * loader.batch_size)
        sync = nullcontext() if end_group or runtime.world_size == 1 else model.no_sync()
        with sync:
            inputs = {k: batch[k].to(runtime.device, non_blocking=True)
                      for k in ['input_ids', 'attention_mask']}
            labels = batch['labels'].to(runtime.device).long() - 1
            with runtime.autocast():
                logits = model(**inputs).float()
                squared_error = nn.functional.cross_entropy(logits, labels, reduction='sum')
                loss = squared_error / group_samples
            if not torch.isfinite(loss).item():
                raise RuntimeError('训练损失出现 NaN/Inf，请降低学习率或用 --no-amp 诊断')
            scaler.scale(loss).backward()
        totals[0] += squared_error.detach().double()
        totals[1] += len(labels)
        if end_group:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # FP16 溢出时 GradScaler 会跳过更新，此时也不推进学习率。
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        global_step = (epoch - 1) * len(loader) + step + 1
        if (step + 1) % eval_every == 0:
            metrics = evaluate(model.module if hasattr(model, 'module') else model, valid_loader, runtime)
            runtime.barrier()
            if runtime.primary:
                print(f'  step {global_step}: val RMSE={metrics["rmse"]:.6f}, accuracy={metrics["accuracy"]:.6f}', flush=True)
                if checkpoint_root is not None:
                    step_dir = Path(checkpoint_root) / f'step_{global_step:08d}'
                    step_dir.mkdir(parents=True, exist_ok=True)
                    raw = model.module if hasattr(model, 'module') else model
                    raw.encoder.config.save_pretrained(step_dir / 'encoder_config')
                    torch.save({'model_state': {k:v.detach().cpu() for k,v in raw.state_dict().items()}, 'step':global_step, 'epoch':epoch, 'validation':metrics, 'prepared_manifest_sha256':manifest_sha256}, step_dir / 'weights.pt')
                if tracker is not None:
                    tracker.log({'step':global_step,'validation/rmse':metrics['rmse'],'validation/accuracy':metrics['accuracy'],'validation/correct':metrics['correct'],'validation/samples':metrics['samples']})
            runtime.barrier()
            model.train()
        if runtime.primary and (step + 1) % log_every == 0:
            mse = (totals[0] / totals[1]).item()
            if tracker is not None:
                tracker.log({'train/rank0_running_cross_entropy': mse,
                             'train/encoder_lr': optimizer.param_groups[0]['lr'],
                             'train/head_lr': optimizer.param_groups[1]['lr'],
                             'train/batch': global_step})
            print(f'  batch {step + 1}/{len(loader)}, train CE={mse:.5f}', flush=True)
    totals = runtime.sum(totals).cpu().tolist()
    return (totals[0] / totals[1]).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ARTIFACTS / 'prepared')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints' / time.strftime('%Y%m%d-%H%M%S', time.gmtime()))
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=32)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--encoder-lr', type=float, default=2e-5)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--wandb-project', default='5434_project')
    parser.add_argument('--wandb-entity', default=None)
    parser.add_argument('--wandb-name', default=None)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                        default=os.environ.get('WANDB_MODE', 'offline'))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.accumulation_steps,
           args.log_every, args.cpu_threads) < 1 or args.workers < 0:
        parser.error('轮数、batch、累积步数、线程数、日志间隔必须为正，workers 不能为负')
    if not 0 <= args.warmup_ratio < 1 or min(args.encoder_lr, args.head_lr) <= 0 or args.weight_decay < 0:
        parser.error('学习率必须为正，weight-decay 非负，warmup-ratio 在 [0,1)')
    started = time.perf_counter()
    runtime = initialize(args.device, args.seed, args.cpu_threads)
    runtime.amp = runtime.amp and not args.no_amp
    tracker = None
    try:
        manifest_path = args.data_dir / 'manifest.json'
        manifest = read_json(manifest_path)
        shared_output = [str(args.output.resolve()) if runtime.primary else None]
        if runtime.world_size > 1:
            dist.broadcast_object_list(shared_output, src=0)
        output = Path(shared_output[0])
        if runtime.primary:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f'{output} 非空，请指定新的 --output')
            output.mkdir(parents=True, exist_ok=True)
        runtime.barrier()
        report = {'started_at': utc_now(), 'hardware': cluster_info(runtime),
                  'prepared_manifest_sha256': manifest_digest(manifest_path),
                  'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  'world_size': runtime.world_size,
                  'global_batch_size': args.batch_size * args.accumulation_steps * runtime.world_size,
                  'preprocessing_seconds': manifest['preprocessing_seconds'], 'epochs': []}
        if runtime.primary and args.wandb_mode != 'disabled':
            import wandb
            tracker = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                 name=args.wandb_name or output.name, mode=args.wandb_mode,
                                 dir=str(output), config={**report['parameters'],
                                 'world_size': runtime.world_size,
                                 'global_batch_size': report['global_batch_size']})
            report['wandb'] = {'id': tracker.id, 'mode': args.wandb_mode,
                               'url': tracker.url if args.wandb_mode == 'online' else None}
        runtime.barrier()
        train_data = ReviewDataset(args.data_dir, 'train')
        valid_data = ReviewDataset(args.data_dir, 'validation')
        sampler = (DistributedSampler(train_data, num_replicas=runtime.world_size, rank=runtime.rank,
                                      seed=args.seed, shuffle=True, drop_last=False)
                   if runtime.world_size > 1 else RandomSampler(train_data))
        loader_options = {'num_workers': args.workers, 'pin_memory': runtime.device.type == 'cuda',
                          'collate_fn': Collator(manifest['pad_token_id'])}
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, **loader_options)
        valid_loader = DataLoader(valid_data, batch_size=args.eval_batch_size,
                                  sampler=ExactShardSampler(valid_data, runtime.rank, runtime.world_size),
                                  **loader_options)
        baseline = manifest['splits']['train']['mean_rating']
        model = RatingModel.pretrained(manifest['model_name'], manifest['model_revision'], baseline)
        if args.gradient_checkpointing:
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.to(runtime.device)
        raw_model = model
        if runtime.world_size > 1:
            # 兼容带未使用 pooler 的其他 encoder；核心 DeBERTa 路线也可使用。
            model = DistributedDataParallel(model,
                                            device_ids=[runtime.device.index] if runtime.device.type == 'cuda' else None,
                                            find_unused_parameters=True)
        optimizer = torch.optim.AdamW([
            {'params': raw_model.encoder.parameters(), 'lr': args.encoder_lr},
            {'params': raw_model.head.parameters(), 'lr': args.head_lr},
        ], weight_decay=args.weight_decay)
        total_steps = math.ceil(len(train_loader) / args.accumulation_steps) * args.epochs
        warmup = int(total_steps * args.warmup_ratio)

        def lr_factor(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        scaler = torch.amp.GradScaler('cuda', enabled=runtime.amp)
        best = float('inf')
        for epoch in range(1, args.epochs + 1):
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            runtime.barrier()
            runtime.synchronize()
            epoch_started = time.perf_counter()
            training_loss = train_epoch(model, train_loader, valid_loader, optimizer, scheduler, scaler, runtime,
                                        args.accumulation_steps, args.log_every, tracker, epoch, args.eval_every)
            runtime.synchronize()
            train_seconds = runtime.maximum(time.perf_counter() - epoch_started)
            validation_started = time.perf_counter()
            # 直接用底层模型验证：各卡批次数可不同，不触发 DDP forward 的同步。
            metrics = evaluate(raw_model, valid_loader, runtime, baseline)
            if metrics['samples'] != len(valid_data):
                raise RuntimeError('验证样本数不一致')
            runtime.synchronize()
            validation_seconds = runtime.maximum(time.perf_counter() - validation_started)
            improved = metrics['rmse'] < best
            if improved:
                best = metrics['rmse']
                if runtime.primary:
                    checkpoint_dir = output / 'best'
                    checkpoint_dir.mkdir(exist_ok=True)
                    raw_model.encoder.config.save_pretrained(checkpoint_dir / 'encoder_config')
                    state = {'model_state': {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
                             'epoch': epoch, 'validation_rmse': best,
                             'prepared_manifest_sha256': report['prepared_manifest_sha256']}
                    temporary = checkpoint_dir / 'weights.tmp'
                    torch.save(state, temporary)
                    temporary.replace(checkpoint_dir / 'weights.pt')
                    report['best_epoch'] = epoch
            runtime.barrier()
            record = {'epoch': epoch, 'training_loss': training_loss,
                      'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                      'epoch_wall_seconds': runtime.maximum(time.perf_counter() - epoch_started),
                      'validation': metrics}
            report['epochs'].append(record)
            if runtime.primary:
                if tracker is not None:
                    tracker.log({'epoch': epoch, 'train/cross_entropy': training_loss,
                                 'validation/rmse': metrics['rmse'],
                                 'validation/accuracy': metrics['accuracy'],
                                 'validation/cross_entropy': metrics['cross_entropy'],
                                 'validation/samples': metrics['samples'],
                                 'time/train_seconds': train_seconds,
                                 'time/validation_seconds': validation_seconds,
                                 'time/epoch_seconds': record['epoch_wall_seconds']})
                    tracker.summary.update({'best_validation_rmse': best,
                                            'best_epoch': report.get('best_epoch'),
                                            'checkpoint': str(output / 'best')})
                report['best_validation_rmse'] = best
                write_json(output / 'training_report.json', report)
                print(f'Epoch {epoch}: validation RMSE={best if improved else metrics["rmse"]:.6f}, '
                      f'train={train_seconds:.1f}s, best_RMSE={best:.6f}', flush=True)
        elapsed = runtime.maximum(time.perf_counter() - started)
        if runtime.primary:
            report.update({'training_total_seconds': elapsed,
                           'total_offline_seconds': manifest['preprocessing_seconds'] + elapsed,
                           'finished_at': utc_now()})
            write_json(output / 'training_report.json', report)
            print(f'训练完成，最佳验证 RMSE={best:.6f}，权重: {output / "best"}', flush=True)
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            if tracker is not None:
                tracker.finish(exit_code=1 if failed else 0)
        finally:
            runtime.close()


if __name__ == '__main__':
    main()
