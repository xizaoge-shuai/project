#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

INPUT=data/processed/unified/model_ablation_qwen7b/asdiv_qwen7b_has_disagreement.jsonl
CFG=configs/model/generator_qwen25_7b_asdiv_ablation.yaml

for SEED in 42 101 202
do
  OUT=data/processed/trajectories/model_ablation_qwen7b/asdiv_qwen7b_extra_seed${SEED}.jsonl
  LOG=outputs/logs/model_ablation_qwen7b/generate_asdiv_qwen7b_extra_seed${SEED}.log

  echo "========== seed ${SEED} =========="
  echo "OUT=${OUT}"
  echo "LOG=${LOG}"

  python scripts/generate_numeric_trajectories_resume.py \
    --input "$INPUT" \
    --output "$OUT" \
    --generator_config "$CFG" \
    --dataset asdiv \
    --n_traj 4 \
    --max_samples 0 \
    --max_new_tokens 384 \
    --temperature 0.95 \
    --top_p 0.95 \
    --seed ${SEED} \
    2>&1 | tee "$LOG"

  echo "rows $(wc -l < "$OUT") $OUT"
done

echo "DONE"
