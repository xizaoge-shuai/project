#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG="configs/model/pce_bert_path_hf_cross.yaml"
LABEL_ROOT="data/processed/labels_cross_table_path_hf"

mkdir -p outputs/checkpoints/cross_table/path_hf
mkdir -p outputs/predictions/cross_table/path_hf
mkdir -p outputs/metrics/cross_table/path_hf
mkdir -p outputs/logs/cross_table

test -s "$CONFIG" || {
  echo "[ERROR] missing config: $CONFIG"
  exit 1
}

HAS_VAL=0
if python -m pce.train_hf --help 2>&1 \
  | grep -q -- "--val_splits"
then
  HAS_VAL=1
fi

run_one () {
  local DATASET="$1"

  local TRAIN="${LABEL_ROOT}/success/path_level/${DATASET}/train.jsonl"
  local VAL="${LABEL_ROOT}/success/path_level/${DATASET}/val.jsonl"
  local TEST="${LABEL_ROOT}/success/path_level/${DATASET}/test.jsonl"

  local OUTDIR="outputs/checkpoints/cross_table/path_hf/${DATASET}_path_hf"
  local TRAIN_METRIC="outputs/metrics/cross_table/path_hf/${DATASET}_path_hf_train.json"
  local PRED="outputs/predictions/cross_table/path_hf/${DATASET}_path_hf_test_raw.jsonl"
  local EVAL="outputs/metrics/cross_table/path_hf/${DATASET}_path_hf_test_raw_eval.json"

  echo
  echo "======================================================================"
  echo "[PATH-HF] dataset=$DATASET"
  echo "======================================================================"

  for F in "$TRAIN" "$VAL" "$TEST"
  do
    if [ ! -s "$F" ]; then
      echo "[ERROR] missing or empty: $F"
      exit 1
    fi
    wc -l "$F"
  done

  if [ "$HAS_VAL" -eq 1 ]; then
    python -m pce.train_hf \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --level path \
      --label_base_dir "$LABEL_ROOT" \
      --train_splits train \
      --val_splits val \
      --test_splits test \
      --feature_set prefix_plus_len_progress \
      --min_prefix_progress 0.0 \
      --out "$OUTDIR" \
      --metrics_out "$TRAIN_METRIC" \
      --test_pred_out "$PRED" \
      --device cuda \
      --seed 42
  else
    python -m pce.train_hf \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --level path \
      --label_base_dir "$LABEL_ROOT" \
      --train_splits train \
      --test_splits test \
      --feature_set prefix_plus_len_progress \
      --min_prefix_progress 0.0 \
      --out "$OUTDIR" \
      --metrics_out "$TRAIN_METRIC" \
      --test_pred_out "$PRED" \
      --device cuda \
      --seed 42
  fi

  test -s "$PRED" || {
    echo "[ERROR] missing prediction: $PRED"
    exit 1
  }

  python scripts/eval_cross_pce_predictions_v2.py \
    --input "$PRED" \
    --output "$EVAL" \
    --bins 10

  echo "[DONE] dataset=$DATASET"
  cat "$EVAL"
}

for DATASET in svamp asdiv math500 mathqa
do
  EVAL="outputs/metrics/cross_table/path_hf/${DATASET}_path_hf_test_raw_eval.json"

  if [ -s "$EVAL" ]; then
    echo "[SKIP] existing result: $EVAL"
    continue
  fi

  run_one "$DATASET"
done

echo
echo "======================================================================"
echo "ALL PATH-HF EXPERIMENTS COMPLETED"
echo "======================================================================"
