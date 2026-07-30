#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

source outputs/tables/cross_dataset/qwen7b_cross_inputs.env

mkdir -p data/processed/labels_cross_table
mkdir -p outputs/predictions/cross_table
mkdir -p outputs/metrics/cross_table
mkdir -p outputs/logs/cross_table

run_one () {
  local DATASET="$1"
  local TRAJ="$2"

  local LABELS="data/processed/labels_cross_table/${DATASET}_atom_coarse.jsonl"
  local PRED="outputs/predictions/cross_table/${DATASET}_atom_coarse_oof5.jsonl"
  local METRIC="outputs/metrics/cross_table/${DATASET}_atom_coarse_oof5.json"
  local WEIGHT_METRIC="outputs/metrics/cross_table/${DATASET}_pce_weighted_tail5.json"
  local WEIGHT_DETAILS="outputs/predictions/cross_table/${DATASET}_pce_weighted_tail5_details.jsonl"

  echo
  echo "=================================================================="
  echo "[BUILD COARSE PREFIX LABELS] $DATASET"
  echo "trajectory=$TRAJ"
  echo "=================================================================="

  python experiments/build_cross_pce_prefix_labels.py \
    --trajectories "$TRAJ" \
    --dataset "$DATASET" \
    --split test \
    --out_jsonl "$LABELS"

  echo
  echo "=================================================================="
  echo "[OOF LIGHT PCE] $DATASET"
  echo "=================================================================="

  python experiments/run_cross_pce_oof.py \
    --target_labels "$LABELS" \
    --dataset "$DATASET" \
    --out_jsonl "$PRED" \
    --out_json "$METRIC" \
    --n_folds 5 \
    --seed 42 \
    --feature_set prefix_plus_len_progress \
    --max_features 100000 \
    --min_df 1 \
    --C 2.0

  echo
  echo "=================================================================="
  echo "[PCE WEIGHTED TAIL5] $DATASET"
  echo "=================================================================="

  python experiments/eval_cross_pce_weighted_selection.py \
    --predictions "$PRED" \
    --trajectories "$TRAJ" \
    --dataset "$DATASET" \
    --tail_k 5 \
    --out_json "$WEIGHT_METRIC" \
    --out_jsonl "$WEIGHT_DETAILS"
}

run_one svamp "$SVAMP_TRJ"
run_one asdiv "$ASDIV_TRJ"
run_one math500 "$MATH500_TRJ"
run_one mathqa "$MATHQA_TRJ"

echo
echo "===== metrics ====="

for f in outputs/metrics/cross_table/*.json
do
  echo
  echo "===== $f ====="
  cat "$f"
done
