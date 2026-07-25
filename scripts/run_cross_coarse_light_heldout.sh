#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

mkdir -p outputs/checkpoints/cross_table/coarse
mkdir -p outputs/predictions/cross_table/coarse
mkdir -p outputs/metrics/cross_table/coarse

for DATASET in svamp asdiv math500 mathqa
do
  TRAIN="data/processed/labels_cross_table_coarse/${DATASET}/train.jsonl"
  VAL="data/processed/labels_cross_table_coarse/${DATASET}/val.jsonl"
  TEST="data/processed/labels_cross_table_coarse/${DATASET}/test.jsonl"

  PRED="outputs/predictions/cross_table/coarse/${DATASET}_coarse_light_test.jsonl"
  MODEL="outputs/checkpoints/cross_table/coarse/${DATASET}_coarse_light.pkl"

  echo
  echo "============================================================"
  echo "[TRAIN/PREDICT] $DATASET Atom-Coarse-Light"
  echo "============================================================"

  python experiments/predict_cross_pce_light_zeroshot.py \
    --train_jsonls "$TRAIN" "$VAL" \
    --test_jsonl "$TEST" \
    --out_jsonl "$PRED" \
    --model_out "$MODEL" \
    --feature_set prefix_plus_len_progress \
    --max_features 100000 \
    --min_df 1 \
    --C 2.0

  test -s "$PRED" || {
    echo "[ERROR] prediction 为空：$PRED"
    exit 1
  }

  wc -l "$PRED"
done
