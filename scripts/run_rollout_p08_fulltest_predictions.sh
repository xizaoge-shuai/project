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
  TRAIN="data/processed/labels_cross_table_rollout_p08/success/atom_level/${DATASET}/train.jsonl"
  VAL="data/processed/labels_cross_table_rollout_p08/success/atom_level/${DATASET}/val.jsonl"
  TEST="data/processed/labels_cross_table_coarse/${DATASET}/test.jsonl"

  OUT="outputs/predictions/cross_table/rollout_fulltest/${DATASET}_rollout_p08_fulltest.jsonl"
  MODEL="outputs/checkpoints/cross_table/rollout_fulltest/${DATASET}_rollout_p08_fulltest.pkl"

  echo "===== $DATASET Rollout-p08 full-test prediction ====="

  python experiments/predict_cross_pce_light_zeroshot.py \
    --train_jsonls "$TRAIN" "$VAL" \
    --test_jsonl "$TEST" \
    --out_jsonl "$OUT" \
    --model_out "$MODEL" \
    --feature_set prefix_plus_len_progress \
    --max_features 100000 \
    --min_df 1 \
    --C 2.0

  wc -l "$OUT"
done
