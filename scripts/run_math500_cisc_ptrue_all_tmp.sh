#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

mkdir -p outputs/predictions/cisc_ptrue
mkdir -p outputs/metrics/cisc_ptrue_compare
mkdir -p outputs/logs/cisc_ptrue_compare

score_case () {
  local NAME="$1"
  local CONFIG="$2"
  local TARGET_JSONL="$3"
  local TARGET_IDS="$4"
  local OUT_JSONL="$5"
  shift 5
  local EXTRAS=("$@")

  echo
  echo "================================================================================"
  echo "[SCORE P(TRUE)] $NAME"
  echo "================================================================================"
  date

  python scripts/score_ptrue_candidates_vllm.py \
    --generator_config "$CONFIG" \
    --target_jsonl "$TARGET_JSONL" \
    --extra_jsonls "${EXTRAS[@]}" \
    --target_ids "$TARGET_IDS" \
    --task_type numeric \
    --out_jsonl "$OUT_JSONL" \
    --batch_size 32 \
    --max_solution_chars 1800 \
    --logprobs 20 \
    --resume 1
}

eval_case () {
  local NAME="$1"
  local BASE_DETAILS="$2"
  local PTRUE_JSONL="$3"
  local TARGET_IDS="$4"
  local BASE_ACC="$5"
  local N_SAMPLES="$6"
  local OURS_JSON="$7"
  local OURS_NAME="$8"

  echo
  echo "================================================================================"
  echo "[EVAL P(TRUE)-CISC] $NAME"
  echo "================================================================================"
  date

  python scripts/eval_cisc_ptrue_from_scores.py \
    --baseline_details "$BASE_DETAILS" \
    --ptrue_jsonl "$PTRUE_JSONL" \
    --target_ids "$TARGET_IDS" \
    --task_type numeric \
    --base_acc "$BASE_ACC" \
    --n_samples "$N_SAMPLES" \
    --out_dir outputs/metrics/cisc_ptrue_compare \
    --prefix "$NAME" \
    --ks 4 8 12 \
    --temps 0.2 0.5 1.0 2.0 \
    --ours_json "$OURS_JSON" \
    --ours_name "$OURS_NAME"
}

# 1. DS7B short
score_case \
  math500_ds7b_short \
  configs/model/generator_deepseek_r1_distill_qwen7b_ptrue4096.yaml \
  data/processed/unified/model_ablation/math500_deepseek7b_has_disagreement.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  outputs/predictions/cisc_ptrue/math500_ds7b_short_ptrue.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed202.jsonl

eval_case \
  math500_ds7b_short \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/predictions/cisc_ptrue/math500_ds7b_short_ptrue.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 \
  500 \
  outputs/metrics/model_ablation/math500_deepseek7b_total2_seed1_margin0.json \
  Confirm-short

# 2. DS7B long1024
score_case \
  math500_ds7b_long1024 \
  configs/model/generator_deepseek_r1_distill_qwen7b_ptrue4096.yaml \
  data/processed/unified/model_ablation/math500_deepseek7b_has_disagreement.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  outputs/predictions/cisc_ptrue/math500_ds7b_long1024_ptrue.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed505.jsonl

eval_case \
  math500_ds7b_long1024 \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/predictions/cisc_ptrue/math500_ds7b_long1024_ptrue.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 \
  500 \
  outputs/metrics/model_ablation_boost/math500_deepseek7b_long1024_total2_seed1_margin0.json \
  Confirm-long1024

# 3. Qwen3B short
score_case \
  math500_qwen3b_short \
  configs/model/generator_qwen25_3b_ptrue4096.yaml \
  data/processed/unified/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  outputs/predictions/cisc_ptrue/math500_qwen3b_short_ptrue.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed202.jsonl

eval_case \
  math500_qwen3b_short \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/predictions/cisc_ptrue/math500_qwen3b_short_ptrue.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 \
  500 \
  outputs/metrics/model_ablation_parallel_qwen3b/math500_qwen3b_total3_seed1_margin0.json \
  Confirm-short

# 4. Qwen3B long1024
score_case \
  math500_qwen3b_long1024 \
  configs/model/generator_qwen25_3b_ptrue4096.yaml \
  data/processed/unified/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  outputs/predictions/cisc_ptrue/math500_qwen3b_long1024_ptrue.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed505.jsonl

eval_case \
  math500_qwen3b_long1024 \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/predictions/cisc_ptrue/math500_qwen3b_long1024_ptrue.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 \
  500 \
  outputs/metrics/model_ablation_boost_qwen3b_long1024/math500_qwen3b_long1024_total2_seed2_margin0.json \
  Confirm-long1024

echo
echo "================================================================================"
echo "DONE ALL P(TRUE)-CISC"
echo "================================================================================"
date

find outputs/metrics/cisc_ptrue_compare -name "*_ptrue_cisc_compare.md" -print
