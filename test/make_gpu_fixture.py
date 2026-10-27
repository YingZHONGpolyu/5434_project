"""Create a tiny offline DeBERTa fixture; never use its metrics as model quality."""
import json
from pathlib import Path
import numpy as np
from transformers import DebertaV2Config, DebertaV2Model

root = Path('checkpoints/ddp-smoke-fixture')
root.mkdir(parents=True, exist_ok=False)
encoder = root / 'encoder'
config = DebertaV2Config(vocab_size=128, hidden_size=16, num_hidden_layers=1,
    num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
    relative_attention=True, position_buckets=8, max_relative_positions=32,
    position_biased_input=False, pos_att_type=['p2c', 'c2p'], conv_kernel_size=3, pad_token_id=0)
DebertaV2Model(config).save_pretrained(encoder)
manifest = {'model_name': str(encoder.resolve()), 'model_revision': None,
            'pad_token_id': 0, 'preprocessing_seconds': 0, 'splits': {}}
for split, n in [('train', 57), ('validation', 17)]:
    directory = root / split
    directory.mkdir()
    np.save(directory / 'input_ids.npy', np.full((n, 32), 7, dtype=np.int32))
    np.save(directory / 'lengths.npy', np.full(n, 32, dtype=np.int32))
    np.save(directory / 'labels.npy', (np.arange(n) % 5 + 1).astype(np.float32))
    manifest['splits'][split] = {'rows': n, 'mean_rating': 3.0}
(root / 'manifest.json').write_text(json.dumps(manifest))
print(root)
