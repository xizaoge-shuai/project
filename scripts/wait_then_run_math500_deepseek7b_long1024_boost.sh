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

echo "========== MATH500 DS7B long1024 boost =========="
date

echo "===== input target count ====="
wc -l data/processed/unified/model_ablation/math500_deepseek7b_has_disagreement.jsonl

echo "===== wait until no existing VLLM is running ====="
while ps -ef | grep -q "VLLM::EngineCore"
do
  date
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true
  sleep 300
done

echo "===== start long1024 extra generation ====="
date

for SEED in 303 404 505
do
  echo "========== seed=${SEED} =========="
  python scripts/generate_numeric_trajectories_resume.py \
    --input data/processed/unified/model_ablation/math500_deepseek7b_has_disagreement.jsonl \
    --output data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed${SEED}.jsonl \
    --generator_config configs/model/generator_deepseek_r1_distill_qwen7b_math500_long1024.yaml \
    --dataset math500 \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens 1024 \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed ${SEED} \
    2>&1 | tee outputs/logs/model_ablation_boost/generate_math500_deepseek7b_long1024_extra_seed${SEED}.log
done

echo "===== long1024 generation done ====="
for f in data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed*.jsonl
do
  [ -f "$f" ] && printf "%8s  %s\n" "$(wc -l < "$f")" "$f"
done

date
