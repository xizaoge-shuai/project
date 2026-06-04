#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

mkdir -p outputs/logs/baseline_tts_compare
mkdir -p outputs/metrics/baseline_tts_compare

echo "========== START MATH500 short/old baseline compare =========="
date

echo
echo "========== DS7B MATH500 old/short =========="
python scripts/eval_tts_baselines_from_candidates.py \
  --baseline_details outputs/predictions/model_ablation/math500_deepseek7b_base_details.jsonl \
  --extra_jsonls \
    data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed42.jsonl \
    data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed101.jsonl \
    data/processed/trajectories/model_ablation/math500_deepseek7b_extra_seed202.jsonl \
  --target_ids outputs/targets/model_ablation/math500_deepseek7b_has_disagreement_ids.txt \
  --task_type numeric \
  --base_acc 0.3100 \
  --n_samples 500 \
  --out_dir outputs/metrics/baseline_tts_compare \
  --prefix math500_ds7b_short \
  --max_candidates 4 8 12 \
  --esc_windows 2 3 4 \
  --cisc_temps 0.2 0.5 1.0 2.0 \
  --gg_lambdas 1,0 1,0.5 1,1 \
  --ours_json outputs/metrics/model_ablation/math500_deepseek7b_total2_seed1_margin0.json \
  --ours_name Confirm-short

echo
echo "========== Qwen3B MATH500 old/short =========="
python scripts/eval_tts_baselines_from_candidates.py \
  --baseline_details outputs/predictions/model_ablation_parallel_qwen3b/math500_qwen3b_base_details.jsonl \
  --extra_jsonls \
    data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed42.jsonl \
    data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed101.jsonl \
    data/processed/trajectories/model_ablation_parallel_qwen3b/math500_qwen3b_extra_seed202.jsonl \
  --target_ids outputs/targets/model_ablation_parallel_qwen3b/math500_qwen3b_has_disagreement_ids.txt \
  --task_type numeric \
  --base_acc 0.2980 \
  --n_samples 500 \
  --out_dir outputs/metrics/baseline_tts_compare \
  --prefix math500_qwen3b_short \
  --max_candidates 4 8 12 \
  --esc_windows 2 3 4 \
  --cisc_temps 0.2 0.5 1.0 2.0 \
  --gg_lambdas 1,0 1,0.5 1,1 \
  --ours_json outputs/metrics/model_ablation_parallel_qwen3b/math500_qwen3b_total3_seed1_margin0.json \
  --ours_name Confirm-short

echo
echo "========== SUMMARY =========="
cat outputs/metrics/baseline_tts_compare/math500_ds7b_short_baseline_compare.md
echo
cat outputs/metrics/baseline_tts_compare/math500_qwen3b_short_baseline_compare.md

echo
echo "========== DONE =========="
date
