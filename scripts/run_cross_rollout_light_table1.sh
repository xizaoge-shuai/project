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

mkdir -p outputs/checkpoints/cross_table/rollout
mkdir -p outputs/predictions/cross_table/rollout
mkdir -p outputs/metrics/cross_table/rollout
mkdir -p outputs/logs/cross_table

run_one () {
  local DATASET="$1"
  local TAG="$2"
  local ROOT="$3"

  local TRAIN="${ROOT}/success/atom_level/${DATASET}/train.jsonl"
  local VAL="${ROOT}/success/atom_level/${DATASET}/val.jsonl"
  local TEST="${ROOT}/success/atom_level/${DATASET}/test.jsonl"

  local PRED="outputs/predictions/cross_table/rollout/${DATASET}_${TAG}_light_test.jsonl"
  local MODEL="outputs/checkpoints/cross_table/rollout/${DATASET}_${TAG}_light.pkl"
  local METRIC="outputs/metrics/cross_table/rollout/${DATASET}_${TAG}_light_test.json"

  echo
  echo "======================================================================"
  echo "[TRAIN] dataset=$DATASET setting=$TAG"
  echo "train=$TRAIN"
  echo "val=$VAL"
  echo "test=$TEST"
  echo "======================================================================"

  for F in "$TRAIN" "$VAL" "$TEST"
  do
    if [ ! -s "$F" ]; then
      echo "[ERROR] missing or empty file: $F"
      exit 1
    fi
    wc -l "$F"
  done

  python experiments/predict_cross_pce_light_zeroshot.py \
    --train_jsonls "$TRAIN" "$VAL" \
    --test_jsonl "$TEST" \
    --out_jsonl "$PRED" \
    --model_out "$MODEL" \
    --feature_set prefix_plus_len_progress \
    --max_features 100000 \
    --min_df 1 \
    --C 2.0

  if [ ! -s "$PRED" ]; then
    echo "[ERROR] prediction file missing or empty: $PRED"
    exit 1
  fi

  python scripts/eval_cross_pce_predictions_v2.py \
    --input "$PRED" \
    --output "$METRIC" \
    --bins 10

  echo "[DONE] $DATASET $TAG"
  cat "$METRIC"
}

for DATASET in svamp asdiv math500 mathqa
do
  run_one \
    "$DATASET" \
    atom_rollout_p09 \
    data/processed/labels_cross_table_rollout_p09

  run_one \
    "$DATASET" \
    atom_rollout_p08 \
    data/processed/labels_cross_table_rollout_p08
done

echo
echo "======================================================================"
echo "ALL CROSS-DATASET ROLLOUT LIGHT PCE EXPERIMENTS COMPLETED"
echo "======================================================================"
