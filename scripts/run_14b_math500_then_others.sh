#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

mkdir -p outputs/logs/model_ablation_14b

echo "================================================================================"
echo "[STAGE 1] Resume / finish 14B MATH500 long1024"
echo "================================================================================"

bash scripts/run_14b_math500_long1024_full.sh \
  2>&1 | tee outputs/logs/model_ablation_14b/run_14b_math500_long1024_full.chain.log

echo
echo "================================================================================"
echo "[CHECK] MATH500 summary"
echo "================================================================================"

if [ -f outputs/metrics/model_ablation_14b/math500_deepseek14b_long1024_summary.md ]; then
  cat outputs/metrics/model_ablation_14b/math500_deepseek14b_long1024_summary.md
else
  echo "[WARN] MATH500 summary missing."
fi

echo
echo "================================================================================"
echo "[STAGE 2] Run remaining numeric datasets: GSM8K / SVAMP / ASDiv"
echo "================================================================================"

bash scripts/run_14b_numeric_other_datasets.sh \
  2>&1 | tee outputs/logs/model_ablation_14b/run_14b_numeric_other_datasets.chain.log

echo
echo "================================================================================"
echo "[DONE] 14B MATH500 + other numeric datasets"
echo "================================================================================"
