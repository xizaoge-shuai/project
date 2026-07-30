#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

mkdir -p outputs/logs/fair_baselines
mkdir -p outputs/metrics/fair_baselines

echo "========== FAIR MATH500 AGGREGATION BASELINES =========="
date

run_case () {
  local PREFIX="$1"
  local BASE_DETAILS="$2"
  local TARGET_IDS="$3"
  local BASE_ACC="$4"
  local N_SAMPLES="$5"
  local OURS_JSON="$6"
  local OURS_NAME="$7"
  shift 7
  local EXTRAS=("$@")

  echo
  echo "================================================================================"
  echo "[RUN] ${PREFIX}"
  echo "================================================================================"

  echo "[BASE_DETAILS] ${BASE_DETAILS}"
  echo "[TARGET_IDS]   ${TARGET_IDS}"
  echo "[OURS_JSON]    ${OURS_JSON}"
  echo "[EXTRAS]"
  for f in "${EXTRAS[@]}"; do
    if [ -f "$f" ]; then
      printf "%8s  %s\n" "$(wc -l < "$f")" "$f"
    else
      echo "MISSING  $f"
    fi
  done

  python scripts/eval_tts_baselines_from_candidates.py \
    --baseline_details "$BASE_DETAILS" \
    --extra_jsonls "${EXTRAS[@]}" \
    --target_ids "$TARGET_IDS" \
    --task_type numeric \
    --base_acc "$BASE_ACC" \
    --n_samples "$N_SAMPLES" \
    --out_dir outputs/metrics/fair_baselines \
    --prefix "$PREFIX" \
    --max_candidates 4 8 12 \
    --esc_windows 2 3 4 \
    --cisc_temps 0.2 0.5 1.0 2.0 \
    --gg_lambdas 1,0 1,0.5 1,1 \
    --ours_json "$OURS_JSON" \
    --ours_name "$OURS_NAME"
}

# =========================
# DS7B MATH500 short / old
# =========================
run_case \
  math500_ds7b_short \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 \
  500 \
  outputs/metrics/model_ablation/math500_deepseek7b_total2_seed1_margin0.json \
  Confirm-short \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed202.jsonl

# =========================
# DS7B MATH500 long1024
# =========================
run_case \
  math500_ds7b_long1024 \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 \
  500 \
  outputs/metrics/model_ablation_boost/math500_deepseek7b_long1024_total2_seed1_margin0.json \
  Confirm-long1024 \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed505.jsonl

# =========================
# Qwen3B MATH500 short / old
# =========================
run_case \
  math500_qwen3b_short \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 \
  500 \
  outputs/metrics/model_ablation_parallel_qwen3b/math500_qwen3b_total3_seed1_margin0.json \
  Confirm-short \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed202.jsonl

# =========================
# Qwen3B MATH500 long1024
# =========================
run_case \
  math500_qwen3b_long1024 \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 \
  500 \
  outputs/metrics/model_ablation_boost_qwen3b_long1024/math500_qwen3b_long1024_total2_seed2_margin0.json \
  Confirm-long1024 \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed505.jsonl

echo
echo "========== DONE aggregation baselines =========="
date

find outputs/metrics/fair_baselines -name "*_baseline_compare.md" -print
