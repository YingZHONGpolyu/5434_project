#!/usr/bin/env bash
set -euo pipefail
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate 5434_project
cd /data2/zls/code/5434_project
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export OMP_NUM_THREADS=2
job_dir="$1"
trap 'printf "failed (exit %s)\n" "$?" > "$job_dir/status"' ERR
printf 'downloading_model\n' > "$job_dir/status"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('microsoft/deberta-v3-base', revision='8ccc9b6f36199bec6961081d44eb72fb3f7353f3',
                  allow_patterns=['config.json', 'pytorch_model.bin', '*.model', 'tokenizer*.json', 'special_tokens_map.json'])
PY
printf 'preparing_and_probing\n' > "$job_dir/status"
python -m model.prepare > "$job_dir/prepare.log" 2>&1 &
prepare_pid=$!
printf '%s\n' "$prepare_pid" > "$job_dir/prepare.pid"
batch=16
python -m torch.distributed.run --standalone --nproc_per_node=7 -m model.probe_batch --batch-size 16 --output "$job_dir/batch16.json" > "$job_dir/batch16.log" 2>&1
# Try 32 only if the worst rank leaves at least 10% of memory unused at batch 16.
if python - "$job_dir/batch16.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
sys.exit(0 if all(x['peak_reserved'] < x['total'] * .9 for x in r) else 1)
PY
then
  if python -m torch.distributed.run --standalone --nproc_per_node=7 -m model.probe_batch --batch-size 32 --output "$job_dir/batch32.json" > "$job_dir/batch32.log" 2>&1; then
    if python - "$job_dir/batch32.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
sys.exit(0 if all(x['free_after_steps'] > 1024**3 for x in r) else 1)
PY
    then batch=32; fi
  fi
fi
printf '%s\n' "$batch" > "$job_dir/selected_batch_size"
printf 'waiting_for_prepared_data\n' > "$job_dir/status"
wait "$prepare_pid"
printf 'training_batch_%s\n' "$batch" > "$job_dir/status"
python -m torch.distributed.run --standalone --nproc_per_node=7 -m model.train \
  --device cuda --batch-size "$batch" --eval-batch-size "$batch" --epochs 2 \
  --workers 2 --log-every 20 --output "$job_dir/run" --wandb-mode offline \
  --wandb-name "$(basename "$job_dir")-bs${batch}" > "$job_dir/train.log" 2>&1
printf 'completed\n' > "$job_dir/status"
