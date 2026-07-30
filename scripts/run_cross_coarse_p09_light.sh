#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

unset OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for DATASET in svamp asdiv math500 mathqa
do
  TRAIN="data/processed/labels_cross_table_coarse_p09/${DATASET}/train.jsonl"
  VAL="data/processed/labels_cross_table_coarse_p09/${DATASET}/val.jsonl"
  TEST="data/processed/labels_cross_table_coarse_p09/${DATASET}/test.jsonl"

  PRED="outputs/predictions/cross_table/coarse_p09/${DATASET}_coarse_p09_light_test.jsonl"
  MODEL="outputs/checkpoints/cross_table/coarse_p09/${DATASET}_coarse_p09_light.pkl"

  echo
  echo "============================================================"
  echo "[ATOM COARSE 0.9 LIGHT] $DATASET"
  echo "============================================================"

  test -s "$TRAIN"
  test -s "$VAL"
  test -s "$TEST"

  python experiments/predict_cross_pce_light_zeroshot.py \
    --train_jsonls "$TRAIN" "$VAL" \
    --test_jsonl "$TEST" \
    --out_jsonl "$PRED" \
    --model_out "$MODEL" \
    --feature_set prefix_plus_len_progress \
    --max_features 100000 \
    --min_df 1 \
    --C 2.0

  python scripts/eval_cross_pce_predictions_v2.py \
    --input "$PRED" \
    --output \
      "outputs/metrics/cross_table/coarse_p09/${DATASET}_coarse_p09_light_test.json" \
    --bins 10
done
