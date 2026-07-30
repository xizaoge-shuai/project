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

GEN_CFG="configs/model/qwen2p5_7b_table2.yaml"

METRIC_ROOT="outputs/metrics/cross_table/table2_confirm"
PRED_ROOT="outputs/predictions/cross_table/table2_confirm"

mkdir -p "$METRIC_ROOT"
mkdir -p "$PRED_ROOT"
mkdir -p outputs/logs/cross_table

run_seed () {
  local DATASET="$1"
  local SETTING="$2"
  local MARGIN="$3"
  local SEED="$4"
  local VARIANT="$5"
  local MAXTOK="$6"

  local PRED="outputs/predictions/cross_table/rollout_fulltest_enriched/${DATASET}_rollout_p08_fulltest_enriched.jsonl"
  local TRAJ="data/processed/trajectories_cross_table/${DATASET}/test.jsonl"
  local REPAIR="outputs/predictions/cross_table/table2_valid_v2/${DATASET}_safev3.jsonl"
  local JUDGE="outputs/predictions/cross_table/table2_valid_v2/${DATASET}_judge.jsonl"

  local OUT_JSON="${METRIC_ROOT}/${DATASET}_${SETTING}_seed${SEED}.json"
  local OUT_JSONL="${PRED_ROOT}/${DATASET}_${SETTING}_seed${SEED}.jsonl"

  test -s "$PRED" || {
    echo "[ERROR] missing $PRED"
    exit 1
  }

  test -s "$TRAJ" || {
    echo "[ERROR] missing $TRAJ"
    exit 1
  }

  touch "$REPAIR"
  touch "$JUDGE"

  rm -f "$OUT_JSON" "$OUT_JSONL"

  echo
  echo "======================================================================"
  echo "[RESAMPLING]"
  echo "dataset=${DATASET}"
  echo "setting=${SETTING}"
  echo "margin=${MARGIN}"
  echo "sampling_seed=${SEED}"
  echo "prompt_variant=${VARIANT}"
  echo "======================================================================"

  python experiments/run_selective_resampling.py \
    --predictions "$PRED" \
    --trajectories "$TRAJ" \
    --repair_jsonl "$REPAIR" \
    --current_judge_jsonl "$JUDGE" \
    --generator_config "$GEN_CFG" \
    --dataset "$DATASET" \
    --trigger margin \
    --margin_threshold "$MARGIN" \
    --n_extra 4 \
    --max_new_tokens "$MAXTOK" \
    --max_samples -1 \
    --extra_weight_mode avg_orig \
    --extra_weight 1.0 \
    --temperature 0.8 \
    --top_p 0.95 \
    --sampling_seed "$SEED" \
    --prompt_variant "$VARIANT" \
    --use_judge 1 \
    --out_jsonl "$OUT_JSONL" \
    --out_json "$OUT_JSON"
}

postprocess_setting () {
  local DATASET="$1"
  local SETTING="$2"
  local N="$3"
  local BASE_ACC="$4"

  local S303="${PRED_ROOT}/${DATASET}_${SETTING}_seed303.jsonl"
  local S404="${PRED_ROOT}/${DATASET}_${SETTING}_seed404.jsonl"
  local S505="${PRED_ROOT}/${DATASET}_${SETTING}_seed505.jsonl"

  python scripts/table2_postprocess.py \
    --mode ensemble \
    --inputs "$S303" "$S404" "$S505" \
    --base_acc "$BASE_ACC" \
    --n_samples "$N" \
    --out_json "${METRIC_ROOT}/${DATASET}_${SETTING}.json" \
    --out_jsonl "${PRED_ROOT}/${DATASET}_${SETTING}.jsonl"
}

run_dataset () {
  local DATASET="$1"
  local MAXTOK="$2"

  local JUDGE_JSON="outputs/metrics/cross_table/table2_valid_v2/${DATASET}_judge.json"

  test -s "$JUDGE_JSON" || {
    echo "[ERROR] missing $JUDGE_JSON"
    exit 1
  }

  read -r N BASE_ACC < <(
    python - "$JUDGE_JSON" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))

print(
    int(d["n_samples"]),
    float(d["judge_final_acc"]),
)
PY
  )

  echo
  echo "######################################################################"
  echo "[DATASET] ${DATASET}"
  echo "n_samples=${N}"
  echo "judge_base=${BASE_ACC}"
  echo "######################################################################"

  # Narrow: margin 0.30, 4 extras × 3 independent seeds.
  run_seed "$DATASET" narrow 0.30 303 0 "$MAXTOK"
  run_seed "$DATASET" narrow 0.30 404 1 "$MAXTOK"
  run_seed "$DATASET" narrow 0.30 505 2 "$MAXTOK"

  postprocess_setting \
    "$DATASET" narrow "$N" "$BASE_ACC"

  # Wide: margin 0.40, same 12-call budget per triggered question.
  run_seed "$DATASET" wide_no_guard 0.40 303 0 "$MAXTOK"
  run_seed "$DATASET" wide_no_guard 0.40 404 1 "$MAXTOK"
  run_seed "$DATASET" wide_no_guard 0.40 505 2 "$MAXTOK"

  postprocess_setting \
    "$DATASET" wide_no_guard "$N" "$BASE_ACC"

  # Frequency/original-majority guard.
  python scripts/table2_postprocess.py \
    --mode guard \
    --inputs "${PRED_ROOT}/${DATASET}_wide_no_guard.jsonl" \
    --base_acc "$BASE_ACC" \
    --n_samples "$N" \
    --out_json "${METRIC_ROOT}/${DATASET}_wide_guard.json" \
    --out_jsonl "${PRED_ROOT}/${DATASET}_wide_guard.jsonl"

  echo "[DONE] ${DATASET}"
}

run_dataset svamp 384
run_dataset asdiv 384
run_dataset math500 1024
run_dataset mathqa 768

echo
echo "======================================================================"
echo "ALL CROSS-DATASET CONFIRMATION ABLATIONS COMPLETED"
echo "======================================================================"
