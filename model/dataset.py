"""内存映射数据集和无补齐的验证/推理采样器。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from model.common import read_json


class ReviewDataset(Dataset):
    def __init__(self, directory, split):
        self.path = Path(directory) / split
        self.count = read_json(Path(directory) / 'manifest.json')['splits'][split]['rows']
        self.arrays = None

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        # 每个 DataLoader worker 独立打开 mmap，避免复制整个训练集。
        if self.arrays is None:
            self.arrays = [np.load(self.path / name, mmap_mode='r')
                           for name in ['input_ids.npy', 'lengths.npy', 'labels.npy']]
        tokens, lengths, labels = self.arrays
        length = int(lengths[index])
        return index, np.array(tokens[index, :length], dtype=np.int64), float(labels[index])

    def __getstate__(self):
        state = self.__dict__.copy()
        state['arrays'] = None
        return state


@dataclass
class Collator:
    pad_token_id: int

    def __call__(self, rows):
        # 动态 padding，短评论批次只计算其实际需要的序列长度。
        length = max(len(row[1]) for row in rows)
        tokens = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(tokens)
        for i, (_, values, _) in enumerate(rows):
            tokens[i, :len(values)] = torch.from_numpy(values)
            mask[i, :len(values)] = 1
        return {'indices': torch.tensor([row[0] for row in rows]),
                'input_ids': tokens, 'attention_mask': mask,
                'labels': torch.tensor([row[2] for row in rows], dtype=torch.float32)}


class ExactShardSampler(Sampler):
    """每条样本恰好分配一次；允许不同进程的验证批次数不同。"""
    def __init__(self, dataset, rank=0, world_size=1):
        self.size, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))
