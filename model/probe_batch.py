"""Measure worst-case 256-token DDP training memory using the real pretrained encoder."""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model.network import RatingModel
from model.runtime import initialize

p = argparse.ArgumentParser()
p.add_argument('--batch-size', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
r = initialize('cuda')
try:
    model = RatingModel.pretrained('microsoft/deberta-v3-base',
        '8ccc9b6f36199bec6961081d44eb72fb3f7353f3').to(r.device)
    model = DDP(model, device_ids=[r.device.index], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda')
    tokens = torch.randint(10, 1000, (a.batch_size, 256), device=r.device)
    mask = torch.ones_like(tokens)
    torch.cuda.reset_peak_memory_stats(r.device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with r.autocast():
            loss = (model(input_ids=tokens, attention_mask=mask).float() - 3).square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    r.synchronize()
    free, total = torch.cuda.mem_get_info(r.device)
    record = {'rank': r.rank, 'batch_size': a.batch_size,
              'peak_allocated': torch.cuda.max_memory_allocated(r.device),
              'peak_reserved': torch.cuda.max_memory_reserved(r.device),
              'free_after_steps': free, 'total': total}
    records = [None] * r.world_size
    dist.all_gather_object(records, record)
    if r.primary:
        a.output.write_text(json.dumps(records, indent=2))
        print(json.dumps(records), flush=True)
finally:
    r.close()
