#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 该实验不使用 GPU。
export CUDA_VISIBLE_DEVICES=""

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

OUT_ROOT="outputs/metrics/cross_table/offline_stopping"
LOG_ROOT="outputs/logs/cross_table/offline_stopping"

mkdir -p "$OUT_ROOT"
mkdir -p "$LOG_ROOT"

run_one () {
  local DATASET="$1"

  local PREFIXES="data/processed/labels_cross_table_coarse/${DATASET}/test.jsonl"
  local PREDICTIONS="outputs/predictions/cross_table/rollout_fulltest/${DATASET}_rollout_p08_fulltest.jsonl"
  local TRAJECTORIES="data/processed/trajectories_cross_table/${DATASET}/test.jsonl"
  local OUT_DIR="${OUT_ROOT}/${DATASET}"

  test -s "$PREFIXES" || {
    echo "[ERROR] missing prefixes: $PREFIXES"
    exit 1
  }

  test -s "$PREDICTIONS" || {
    echo "[ERROR] missing predictions: $PREDICTIONS"
    exit 1
  }

  test -s "$TRAJECTORIES" || {
    echo "[ERROR] missing trajectories: $TRAJECTORIES"
    exit 1
  }

  echo
  echo "======================================================================"
  echo "[OFFLINE STOPPING] dataset=${DATASET}"
  echo "======================================================================"

  python experiments/eval_offline_trajectory_stopping.py \
    --dataset "$DATASET" \
    --prefixes "$PREFIXES" \
    --predictions "$PREDICTIONS" \
    --trajectories "$TRAJECTORIES" \
    --thresholds \
      0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 \
    --min_progress 0.50 \
    --patience 2 \
    --tokenizer \
      /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --tokenizer_batch_size 256 \
    --out_dir "$OUT_DIR"

  echo "[DONE] ${DATASET}"
}

run_one svamp
run_one asdiv
run_one math500
run_one mathqa

echo
echo "======================================================================"
echo "ALL OFFLINE STOPPING EXPERIMENTS COMPLETED"
echo "======================================================================"
