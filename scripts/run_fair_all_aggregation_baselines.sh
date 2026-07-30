#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

mkdir -p outputs/logs/fair_baselines
mkdir -p outputs/metrics/fair_baselines

run_case () {
  local PREFIX="$1"
  local TASK_TYPE="$2"
  local BASE_DETAILS="$3"
  local TARGET_IDS="$4"
  local BASE_ACC="$5"
  local N_SAMPLES="$6"
  local OURS_JSON="$7"
  local OURS_NAME="$8"
  shift 8
  local EXTRAS=("$@")

  echo
  echo "================================================================================"
  echo "[RUN] ${PREFIX}"
  echo "================================================================================"
  echo "TASK_TYPE=${TASK_TYPE}"
  echo "BASE_DETAILS=${BASE_DETAILS}"
  echo "TARGET_IDS=${TARGET_IDS}"
  echo "OURS_JSON=${OURS_JSON}"

  if [ ! -f "$BASE_DETAILS" ]; then
    echo "[SKIP] missing base details: $BASE_DETAILS"
    return 0
  fi
  if [ ! -f "$TARGET_IDS" ]; then
    echo "[SKIP] missing target ids: $TARGET_IDS"
    return 0
  fi
  if [ ! -f "$OURS_JSON" ]; then
    echo "[WARN] missing ours json: $OURS_JSON"
  fi

  local OK_EXTRAS=()
  for f in "${EXTRAS[@]}"; do
    if [ -f "$f" ]; then
      OK_EXTRAS+=("$f")
      printf "%8s  %s\n" "$(wc -l < "$f")" "$f"
    else
      echo "MISSING  $f"
    fi
  done

  if [ "${#OK_EXTRAS[@]}" -eq 0 ]; then
    echo "[SKIP] no extras"
    return 0
  fi

  python scripts/eval_tts_baselines_from_candidates.py \
    --baseline_details "$BASE_DETAILS" \
    --extra_jsonls "${OK_EXTRAS[@]}" \
    --target_ids "$TARGET_IDS" \
    --task_type "$TASK_TYPE" \
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

echo "========== DS7B =========="

run_case gsm8k_ds7b numeric \
  outputs/predictions/model_ablation/gsm8k_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/gsm8k_deepseek7b_has_disagreement_ids.txt \
  0.4936 1319 \
  outputs/metrics/model_ablation/gsm8k_deepseek7b_total2_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed202.jsonl

run_case asdiv_ds7b numeric \
  outputs/predictions/model_ablation/asdiv_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/asdiv_deepseek7b_has_disagreement_ids.txt \
  0.7514 2249 \
  outputs/metrics/model_ablation/asdiv_deepseek7b_total3_seed1_margin1.json \
  Confirm \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed202.jsonl

run_case svamp_ds7b numeric \
  outputs/predictions/model_ablation/svamp_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/svamp_deepseek7b_has_disagreement_ids.txt \
  0.6933 300 \
  outputs/metrics/model_ablation/svamp_deepseek7b_total3_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed202.jsonl

run_case math500_ds7b_short numeric \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 500 \
  outputs/metrics/model_ablation/math500_deepseek7b_total2_seed1_margin0.json \
  Confirm-short \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed202.jsonl

run_case math500_ds7b_long1024 numeric \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 500 \
  outputs/metrics/model_ablation_boost/math500_deepseek7b_long1024_total2_seed1_margin0.json \
  Confirm-long1024 \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed505.jsonl

run_case bbh_formal_fallacies_ds7b choice \
  outputs/predictions/model_ablation/bbh_formal_fallacies_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/bbh_formal_fallacies_deepseek7b_has_disagreement_ids.txt \
  0.1500 100 \
  outputs/metrics/model_ablation/bbh_formal_fallacies_deepseek7b_total2_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation/bbh_formal_fallacies_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/bbh_formal_fallacies_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/bbh_formal_fallacies_deepseek7b_extra_seed202.jsonl

run_case bbh_logical_deduction_five_objects_ds7b choice \
  outputs/predictions/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_has_disagreement_ids.txt \
  0.1000 100 \
  outputs/metrics/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_total2_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/bbh_logical_deduction_five_objects_deepseek7b_extra_seed202.jsonl

echo "========== Qwen3B =========="

run_case gsm8k_qwen3b numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/gsm8k_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/gsm8k_qwen3b_has_disagreement_ids.txt \
  0.5004 1319 \
  outputs/metrics/model_ablation_parallel_qwen3b/gsm8k_qwen3b_total3_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed202.jsonl

run_case asdiv_qwen3b numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/asdiv_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/asdiv_qwen3b_has_disagreement_ids.txt \
  0.7835 2249 \
  outputs/metrics/model_ablation_parallel_qwen3b/asdiv_qwen3b_total2_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed202.jsonl

run_case svamp_qwen3b numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/svamp_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/svamp_qwen3b_has_disagreement_ids.txt \
  0.8000 300 \
  outputs/metrics/model_ablation_parallel_qwen3b/svamp_qwen3b_total2_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed202.jsonl

run_case math500_qwen3b_short numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 500 \
  outputs/metrics/model_ablation_parallel_qwen3b/math500_qwen3b_total3_seed1_margin0.json \
  Confirm-short \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed202.jsonl

run_case math500_qwen3b_long1024 numeric \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 500 \
  outputs/metrics/model_ablation_boost_qwen3b_long1024/math500_qwen3b_long1024_total2_seed2_margin0.json \
  Confirm-long1024 \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed505.jsonl

run_case bbh_formal_fallacies_qwen3b choice \
  outputs/predictions/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_has_disagreement_ids.txt \
  0.4500 100 \
  outputs/metrics/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_total2_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_formal_fallacies_qwen3b_extra_seed202.jsonl

run_case bbh_logical_deduction_five_objects_qwen3b choice \
  outputs/predictions/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_has_disagreement_ids.txt \
  0.3400 100 \
  outputs/metrics/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_total2_seed1_margin0.json \
  Confirm \
  data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/bbh_logical_deduction_five_objects_qwen3b_extra_seed202.jsonl

echo
echo "========== DONE ALL AGGREGATION BASELINES =========="
find outputs/metrics/fair_baselines -name "*_baseline_compare.md" -print
