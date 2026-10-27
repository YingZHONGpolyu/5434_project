"""Evaluate a saved regressor on labeled validation data (never on placeholder test labels)."""
import argparse
import math
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from model.common import ARTIFACTS, manifest_digest, read_json, write_json, utc_now
from model.dataset import ReviewDataset, Collator, ExactShardSampler
from model.network import RatingModel
from model.runtime import initialize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ARTIFACTS/'prepared')
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    runtime = initialize('cuda')
    started = time.perf_counter()
    try:
        if a.output.exists():
            raise FileExistsError(a.output)
        manifest_path = a.data_dir/'manifest.json'
        manifest = read_json(manifest_path)
        state = torch.load(a.checkpoint/'weights.pt', map_location='cpu', weights_only=True)
        if state['prepared_manifest_sha256'] != manifest_digest(manifest_path):
            raise ValueError('Checkpoint and data manifest mismatch')
        model = RatingModel.from_checkpoint_config(a.checkpoint/'encoder_config')
        model.load_state_dict(state['model_state'], strict=True)
        epoch = state['epoch']
        del state
        model.to(runtime.device).eval()
        data = ReviewDataset(a.data_dir, 'validation')
        loader = DataLoader(data, batch_size=a.batch_size,
            sampler=ExactShardSampler(data, runtime.rank, runtime.world_size),
            collate_fn=Collator(manifest['pad_token_id']), num_workers=2, pin_memory=True)
        totals = torch.zeros(30, dtype=torch.float64, device=runtime.device)
        with torch.inference_mode():
            for batch in loader:
                labels = batch['labels'].to(runtime.device)
                with runtime.autocast():
                    predictions = model(input_ids=batch['input_ids'].to(runtime.device),
                        attention_mask=batch['attention_mask'].to(runtime.device)).float().clamp(1, 5)
                if not torch.isfinite(predictions).all():
                    raise RuntimeError('Non-finite predictions')
                rounded = predictions.add(.5).floor().long().clamp(1,5)
                error = (predictions.double()-labels.double())
                totals[0] += error.square().sum()
                totals[1] += error.abs().sum()
                totals[2] += (rounded == labels.long()).sum()
                totals[3] += len(labels)
                totals[4] += ((rounded-labels).abs() <= 1).sum()
                bins = (labels.long()-1)*5 + rounded-1
                totals[5:] += torch.bincount(bins, minlength=25)
        values = runtime.sum(totals).cpu().tolist()
        if int(values[3]) != len(data):
            raise RuntimeError('Validation sample count mismatch')
        if runtime.primary:
            report = {'split':'validation', 'samples':int(values[3]), 'checkpoint':str(a.checkpoint.resolve()),
                'checkpoint_epoch':epoch, 'rmse':math.sqrt(values[0]/values[3]),
                'mae':values[1]/values[3], 'rounded_accuracy':values[2]/values[3],
                'rounded_correct':int(values[2]), 'rounded_within_one_accuracy':values[4]/values[3],
                'accuracy_definition':'clip predictions to [1,5], floor(prediction+0.5), exact match with true integer rating',
                'confusion_matrix_true_rows_predicted_columns':[list(map(int,values[5+i*5:10+i*5])) for i in range(5)],
                'note':'Validation set used for model selection; this is not a held-out test score. Raw test.csv has no rating labels.',
                'world_size':runtime.world_size, 'elapsed_seconds':time.perf_counter()-started, 'finished_at':utc_now()}
            write_json(a.output,report)
            print(report,flush=True)
    finally:
        runtime.close()

if __name__ == '__main__':
    main()
