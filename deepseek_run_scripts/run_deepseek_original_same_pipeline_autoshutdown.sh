#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

NO_SHUTDOWN_FILE=/tmp/NO_DEEPSEEK_ORIGINAL_SHUTDOWN

mkdir -p outputs/logs/model_swap_deepseek /root/autodl-tmp/pce_backups

on_exit() {
  status=$?
  echo "========== EXIT status=$status =========="
  tar --ignore-failed-read -czf /root/autodl-tmp/pce_backups/deepseek_original_same_pipeline_$(date +%Y%m%d_%H%M%S).tar.gz \
    outputs/logs/model_swap_deepseek \
    outputs/metrics \
    outputs/predictions \
    outputs/final_selected_results \
    data/processed/trajectories \
    2>/dev/null || true

  if [ "$status" = "0" ] && [ ! -f "$NO_SHUTDOWN_FILE" ]; then
    echo "[SHUTDOWN] finished successfully."
    sync
    shutdown -h now || poweroff || halt || true
  else
    echo "[NO SHUTDOWN] status=$status or found $NO_SHUTDOWN_FILE"
  fi
}
trap on_exit EXIT

echo "========== DeepSeek original same-pipeline run =========="
date
echo "GEN_CFG=configs/model/generator_deepseek_r1_distill_qwen7b_original.yaml"

echo "========== 1. run_remaining_after_current =========="
bash deepseek_run_scripts/run_remaining_after_current_autoshutdown_deepseek.sh \
  2>&1 | tee outputs/logs/model_swap_deepseek/01_run_remaining_after_current_deepseek.log

echo "========== 2. ASDiv numeric extra confirm v2 =========="
bash deepseek_run_scripts/run_asdiv_numeric_extra_confirm_v2_deepseek.sh \
  2>&1 | tee outputs/logs/model_swap_deepseek/02_asdiv_numeric_extra_confirm_v2_deepseek.log

echo "========== 3. MathQA scale extra confirm =========="
bash deepseek_run_scripts/run_mathqa_scale_extra_confirm_deepseek.sh \
  2>&1 | tee outputs/logs/model_swap_deepseek/03_mathqa_scale_extra_confirm_deepseek.log

echo "========== 4. reasoning QA extra confirm =========="
bash deepseek_run_scripts/run_reasoning_qa_extra_confirm_autoshutdown_deepseek.sh \
  2>&1 | tee outputs/logs/model_swap_deepseek/04_reasoning_qa_extra_confirm_deepseek.log

echo "========== 5. true generation ablation =========="
bash deepseek_run_scripts/run_true_generation_ablation_qwen_autoshutdown_deepseek.sh \
  2>&1 | tee outputs/logs/model_swap_deepseek/05_true_generation_ablation_deepseek.log

echo "========== DONE =========="
