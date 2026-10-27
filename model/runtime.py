"""单卡/CPU 与 torchrun 多卡共用的运行环境。"""

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist

from model.common import hardware_info


@dataclass
class Runtime:
    rank: int
    world_size: int
    device: torch.device
    amp: bool

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def sum(self, value):
        if self.world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def maximum(self, value):
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def autocast(self):
        return torch.autocast(device_type='cuda', dtype=torch.float16) if self.amp else nullcontext()

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def initialize(device_name='auto', seed=42, cpu_threads=2):
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    use_cuda = device_name == 'cuda' or (device_name == 'auto' and torch.cuda.is_available())
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；请检查驱动和 PyTorch CUDA 安装，或指定 --device cpu')
    device = torch.device('cuda', local_rank) if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(device)
    torch.set_num_threads(cpu_threads)
    if world > 1:
        dist.init_process_group('nccl' if use_cuda else 'gloo', timeout=timedelta(minutes=30))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return Runtime(rank, world, device, use_cuda)


def cluster_info(runtime):
    host = hardware_info()
    host.update({'rank': runtime.rank, 'torch_version': str(torch.__version__),
                 'cuda_version': torch.version.cuda, 'device': str(runtime.device),
                 'torch_cpu_threads_per_process': torch.get_num_threads()})
    if runtime.device.type == 'cuda':
        properties = torch.cuda.get_device_properties(runtime.device)
        host.update({'gpu_model': properties.name, 'gpu_memory_bytes': properties.total_memory})
    hosts = [None] * runtime.world_size
    if runtime.world_size > 1:
        dist.all_gather_object(hosts, host)
    else:
        hosts[0] = host
    return {'server_count': len({h['hostname'] for h in hosts}),
            'process_count': runtime.world_size, 'processes': hosts}
