#!/usr/bin/env bash
set -euo pipefail
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate 5434_project
cd /data2/zls/code/5434_project
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 OMP_NUM_THREADS=2
job_dir=checkpoints/train-20261026-225655
trap 'printf "failed (exit %s)\n" "$?" > "$job_dir/status"' ERR
printf 'preparing_tokenization_7_workers\n' > "$job_dir/status"
python -m model.resume_prepare --workers 7 --original-start-file "$job_dir/prepare.pid" > "$job_dir/prepare-parallel.log" 2>&1
printf 'training_batch_32\n' > "$job_dir/status"
python -m torch.distributed.run --standalone --nproc_per_node=7 -m model.train \
  --device cuda --batch-size 32 --eval-batch-size 32 --epochs 2 --workers 2 --log-every 20 \
  --output "$job_dir/run" --wandb-mode offline --wandb-name train-20261026-225655-bs32 \
  > "$job_dir/train.log" 2>&1
printf 'completed\n' > "$job_dir/status"
