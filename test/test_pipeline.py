"""关键算法检查：python -m unittest model.test_pipeline -v。"""

import csv
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.common import clean_text
from model.dataset import Collator, ExactShardSampler
from model.network import RatingModel
from model.predict import write_submission
from model.runtime import Runtime
from model.train import evaluate, train_epoch


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(10, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class LinearRating(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, input_ids, attention_mask):
        return self.linear(input_ids[:, :1].float()).squeeze(-1)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.runtime = Runtime(0, 1, torch.device('cpu'), False)

    def test_text_cleaning(self):
        self.assertEqual(clean_text(' A<br />B &#34;tea&#34;\t\x00 '), 'A B "tea"')
        self.assertEqual(clean_text(None), '')

    def test_exact_shards_including_empty_rank(self):
        for size in [2, 11]:
            parts = [list(ExactShardSampler(range(size), rank, 7)) for rank in range(7)]
            flat = [i for part in parts for i in part]
            self.assertEqual(sorted(flat), list(range(size)))
            self.assertEqual(len(flat), len(set(flat)))

    def test_pooling_ignores_padding(self):
        model = RatingModel(DummyEncoder()).eval()
        with torch.no_grad():
            short = model(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))
            padded = model(torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 1, 0, 0]]))
        torch.testing.assert_close(short, padded)

    def test_global_rmse_not_mean_of_batch_rmse(self):
        class Fixed(nn.Module):
            def forward(self, input_ids, attention_mask):
                return input_ids[:, 0].float()
        rows = [(i, np.array([p]), y) for i, (p, y) in enumerate([(0, 1), (3, 2), (8, 1)])]
        loader = DataLoader(rows, batch_size=2, collate_fn=Collator(0))
        result = evaluate(Fixed(), loader, self.runtime, 3.0)
        self.assertAlmostEqual(result['rmse'], math.sqrt(17 / 3))
        self.assertAlmostEqual(result['raw_rmse'], math.sqrt(51 / 3))
        self.assertEqual(result['samples'], 3)

    def test_accumulation_handles_incomplete_final_batch(self):
        torch.manual_seed(42)
        full = LinearRating()
        micro = LinearRating()
        micro.load_state_dict(full.state_dict())
        rows = [(i, np.array([i + 1]), float(i % 5 + 1)) for i in range(5)]
        for model, batch_size, accumulation in [(full, 5, 1), (micro, 2, 3)]:
            loader = DataLoader(rows, batch_size=batch_size, collate_fn=Collator(0))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            train_epoch(model, loader, optimizer, scheduler, torch.amp.GradScaler('cuda', enabled=False),
                        self.runtime, accumulation, 100)
        for left, right in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)

    def test_submission_alignment_and_reject_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / 'sample.csv'
            sample.write_text('id,rating\nB,1\n001,1\n', encoding='utf-8')
            output = root / 'submission.csv'
            write_submission(output, ['001', 'B'], np.array([2.4, 4.8]), sample)
            with output.open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row['id'] for row in rows], ['B', '001'])
            self.assertEqual([row['rating'] for row in rows], ['4.800000', '2.400000'])
            with self.assertRaises(ValueError):
                write_submission(root / 'invalid.csv', ['001'], np.array([2.4]), sample)


if __name__ == '__main__':
    unittest.main()
