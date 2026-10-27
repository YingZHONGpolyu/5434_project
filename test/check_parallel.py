from pathlib import Path
import tempfile
import numpy as np
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from model.prepare import tokenize_split, tokenize_split_parallel

def main():
    root = Path('checkpoints/conda-env-smoke/prepared')
    source = root / 'parquet/train'
    count = sum(pq.ParquetFile(p).metadata.num_rows for p in source.glob('part-*.parquet'))
    tok = AutoTokenizer.from_pretrained(root / 'tokenizer')
    with tempfile.TemporaryDirectory() as temp:
        a, b = Path(temp)/'serial', Path(temp)/'parallel'
        tokenize_split(source, a, count, tok, 32, 4)
        tokenize_split_parallel(source, b, count, root/'tokenizer', 32, 4, 7)
        for name in ['input_ids.npy', 'lengths.npy', 'labels.npy']:
            np.testing.assert_array_equal(np.load(a/name), np.load(b/name))
        assert (a/'ids.csv').read_bytes() == (b/'ids.csv').read_bytes()
    print('PASS: 7-worker tokens, lengths, labels and ID order match serial output')

if __name__ == '__main__':
    main()
