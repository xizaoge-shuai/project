#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

mkdir -p outputs/metrics/baseline_tts_compare
mkdir -p outputs/logs/baseline_tts_compare

run_case () {
  local PREFIX="$1"
  local BASE_DETAILS="$2"
  local TARGET_IDS="$3"
  local BASE_ACC="$4"
  local N_SAMPLES="$5"
  local OURS_JSON="$6"
  shift 6
  local EXTRAS=("$@")

  echo
  echo "================================================================================"
  echo "[RUN] ${PREFIX}"
  echo "================================================================================"

  python scripts/eval_tts_baselines_from_candidates.py \
    --baseline_details "$BASE_DETAILS" \
    --extra_jsonls "${EXTRAS[@]}" \
    --target_ids "$TARGET_IDS" \
    --task_type numeric \
    --base_acc "$BASE_ACC" \
    --n_samples "$N_SAMPLES" \
    --out_dir outputs/metrics/baseline_tts_compare \
    --prefix "$PREFIX" \
    --max_candidates 4 8 12 \
    --esc_windows 2 3 4 \
    --cisc_temps 0.2 0.5 1.0 2.0 \
    --gg_lambdas 1,0 1,0.5 1,1 \
    --ours_json "$OURS_JSON" \
    --ours_name Confirm
}

# ================= DS7B ordinary numeric =================

run_case \
  gsm8k_ds7b \
  outputs/predictions/model_ablation/gsm8k_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/gsm8k_deepseek7b_has_disagreement_ids.txt \
  0.4936 \
  1319 \
  outputs/metrics/model_ablation/gsm8k_deepseek7b_total2_seed1_margin0.json \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/gsm8k_deepseek7b_extra_seed202.jsonl

run_case \
  asdiv_ds7b \
  outputs/predictions/model_ablation/asdiv_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/asdiv_deepseek7b_has_disagreement_ids.txt \
  0.7514 \
  2249 \
  outputs/metrics/model_ablation/asdiv_deepseek7b_total3_seed1_margin1.json \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/asdiv_deepseek7b_extra_seed202.jsonl

run_case \
  svamp_ds7b \
  outputs/predictions/model_ablation/svamp_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/svamp_deepseek7b_has_disagreement_ids.txt \
  0.6933 \
  300 \
  outputs/metrics/model_ablation/svamp_deepseek7b_total3_seed1_margin0.json \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/svamp_deepseek7b_extra_seed202.jsonl

run_case \
  math500_ds7b_old \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 \
  500 \
  outputs/metrics/model_ablation/math500_deepseek7b_total2_seed1_margin0.json \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed202.jsonl

run_case \
  math500_ds7b_long1024 \
  outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  0.3100 \
  500 \
  outputs/metrics/model_ablation_boost/math500_deepseek7b_long1024_total2_seed1_margin0.json \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_deepseek7b_long1024_extra_seed505.jsonl

# ================= Qwen3B ordinary numeric =================

run_case \
  gsm8k_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b/gsm8k_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/gsm8k_qwen3b_has_disagreement_ids.txt \
  0.5004 \
  1319 \
  outputs/metrics/model_ablation_parallel_qwen3b/gsm8k_qwen3b_total3_seed1_margin0.json \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/gsm8k_qwen3b_extra_seed202.jsonl

run_case \
  asdiv_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b/asdiv_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/asdiv_qwen3b_has_disagreement_ids.txt \
  0.7835 \
  2249 \
  outputs/metrics/model_ablation_parallel_qwen3b/asdiv_qwen3b_total2_seed1_margin0.json \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/asdiv_qwen3b_extra_seed202.jsonl

run_case \
  svamp_qwen3b \
  outputs/predictions/model_ablation_parallel_qwen3b/svamp_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/svamp_qwen3b_has_disagreement_ids.txt \
  0.8000 \
  300 \
  outputs/metrics/model_ablation_parallel_qwen3b/svamp_qwen3b_total2_seed1_margin0.json \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/svamp_qwen3b_extra_seed202.jsonl

run_case \
  math500_qwen3b_old \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 \
  500 \
  outputs/metrics/model_ablation_parallel_qwen3b/math500_qwen3b_total3_seed1_margin0.json \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed42.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed101.jsonl \
  data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed202.jsonl

run_case \
  math500_qwen3b_long1024 \
  outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  0.2980 \
  500 \
  outputs/metrics/model_ablation_boost_qwen3b_long1024/math500_qwen3b_long1024_total2_seed2_margin0.json \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed303.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed404.jsonl \
  data/processed/trajectories/model_ablation_boost/math500_qwen3b_long1024_extra_seed505.jsonl

echo
echo "DONE. Results:"
find outputs/metrics/baseline_tts_compare -name "*baseline_compare.md" -print

