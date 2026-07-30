#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

mkdir -p outputs/metrics/hybrid_controller_all
mkdir -p outputs/logs/hybrid_controller_all

run_case () {
  local PREFIX="$1"
  local TASK_TYPE="$2"
  local BASE_DETAILS="$3"
  local TARGET_IDS="$4"
  local BASE_ACC="$5"
  local N_SAMPLES="$6"
  shift 6
  local EXTRAS=("$@")

  echo
  echo "================================================================================"
  echo "[RUN] $PREFIX"
  echo "================================================================================"

  if [ ! -f "$BASE_DETAILS" ]; then echo "[SKIP] missing $BASE_DETAILS"; return 0; fi
  if [ ! -f "$TARGET_IDS" ]; then echo "[SKIP] missing $TARGET_IDS"; return 0; fi

  local OK_EXTRAS=()
  for f in "${EXTRAS[@]}"; do
    if [ -f "$f" ]; then
      OK_EXTRAS+=("$f")
      printf "%8s  %s\n" "$(wc -l < "$f")" "$f"
    else
      echo "MISSING $f"
    fi
  done
  if [ "${#OK_EXTRAS[@]}" -eq 0 ]; then echo "[SKIP] no extras"; return 0; fi

  for W in 2 3 4
  do
    for TOTAL in 2 3 4
    do
      for SEEDSUP in 1 2 3
      do
        for MARGIN in 0 1 2
        do
          OUT_PREFIX="${PREFIX}_w${W}_total${TOTAL}_seed${SEEDSUP}_margin${MARGIN}"
          python scripts/eval_hybrid_esc_confirm.py \
            --baseline_details "$BASE_DETAILS" \
            --extra_jsonls "${OK_EXTRAS[@]}" \
            --target_ids "$TARGET_IDS" \
            --task_type "$TASK_TYPE" \
            --base_acc "$BASE_ACC" \
            --n_samples "$N_SAMPLES" \
            --out_dir outputs/metrics/hybrid_controller_all \
            --prefix "$OUT_PREFIX" \
            --k 12 \
            --esc_w "$W" \
            --min_total_support "$TOTAL" \
            --min_seed_support "$SEEDSUP" \
            --min_margin "$MARGIN" \
            > /dev/null
        done
      done
    done
  done
}

echo "========== DS7B numeric =========="

run_case gsm8k_ds7b numeric \
  outputs/predictions/model_ablation/gsm8k_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/gsm8k_deepseek7b_has_disagreement_ids.txt \
  0.4936 1319 \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed202.jsonl

run_case asdiv_ds7b numeric \
  outputs/predictions/model_ablation/asdiv_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/asdiv_deepseek7b_has_disagreement_ids.txt \
  0.7514 2249 \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed202.jsonl

run_case svamp_ds7b numeric \
  outputs/predictions/model_ablation/svamp_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/svamp_deepseek7b_has_disagreement_ids.txt \
  0.6933 300 \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed202.jsonl

run_case math500_ds7b_short numeric \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 500 \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed202.jsonl

run_case math500_ds7b_long1024 numeric \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 500 \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed505.jsonl

echo "========== Qwen3B numeric =========="

run_case gsm8k_qwen3b numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/gsm8k_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/gsm8k_qwen3b_has_disagreement_ids.txt \
  0.5004 1319 \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed202.jsonl

run_case asdiv_qwen3b numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/asdiv_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/asdiv_qwen3b_has_disagreement_ids.txt \
  0.7835 2249 \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed202.jsonl

run_case svamp_qwen3b numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/svamp_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/svamp_qwen3b_has_disagreement_ids.txt \
  0.8000 300 \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed202.jsonl

run_case math500_qwen3b_short numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 500 \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed202.jsonl

run_case math500_qwen3b_long1024 numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 500 \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed505.jsonl

echo "========== DONE =========="
