#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

mkdir -p data/processed/trajectories/model_ablation_boost
mkdir -p outputs/logs/model_ablation_boost

echo "========== START Qwen3B MATH500 long1024 boost =========="
date
nvidia-smi || true

for SEED in 303 404 505
do
  echo
  echo "========== Qwen3B MATH500 long1024 seed=${SEED} =========="
  date

  python scripts/generate_numeric_trajectories_resume.py \
    --input data/processed/unified/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement.jsonl \
    --output data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed${SEED}.jsonl \
    --generator_config configs/model/generator_qwen25_3b_math500_long1024.yaml \
    --dataset math500 \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens 1024 \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed ${SEED} \
    2>&1 | tee outputs/logs/model_ablation_boost/generate_math500_qwen3b_long1024_extra_seed${SEED}.log

  wc -l data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed${SEED}.jsonl || true
done

echo "========== DONE Qwen3B MATH500 long1024 boost =========="
date
