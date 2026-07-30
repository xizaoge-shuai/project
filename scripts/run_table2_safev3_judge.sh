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

GEN_CFG="configs/model/generator_llama_local_rewrite.yaml"

mkdir -p outputs/metrics/cross_table/table2_valid_v2
mkdir -p outputs/predictions/cross_table/table2_valid_v2
mkdir -p outputs/logs/cross_table

run_one () {
  local DATASET="$1"
  local MAXTOK="$2"

  local TRAJ="data/processed/trajectories_cross_table/${DATASET}/test.jsonl"
  local PRED="outputs/predictions/cross_table/rollout_fulltest_enriched/${DATASET}_rollout_p08_fulltest_enriched.jsonl"

  local SAFE_JSON="outputs/metrics/cross_table/table2_valid_v2/${DATASET}_safev3.json"
  local SAFE_JSONL="outputs/predictions/cross_table/table2_valid_v2/${DATASET}_safev3.jsonl"

  local JUDGE_JSON="outputs/metrics/cross_table/table2_valid_v2/${DATASET}_judge.json"
  local JUDGE_JSONL="outputs/predictions/cross_table/table2_valid_v2/${DATASET}_judge.jsonl"

  echo
  echo "======================================================================"
  echo "[SAFEV3] dataset=${DATASET}"
  echo "======================================================================"

  rm -f "$SAFE_JSON" "$SAFE_JSONL" "$JUDGE_JSON" "$JUDGE_JSONL"

  python experiments/run_local_rewrite_backtrack.py \
    --predictions "$PRED" \
    --trajectories "$TRAJ" \
    --generator_config "$GEN_CFG" \
    --dataset "$DATASET" \
    --tau_trigger 0.46 \
    --rewrite_window 1 \
    --max_new_tokens "$MAXTOK" \
    --require_repairable 0 \
    --missing_repairable_policy allow \
    --repair_prob_threshold 0.0 \
    --max_cases -1 \
    --out_json "$SAFE_JSON" \
    --out_jsonl "$SAFE_JSONL" \
    --min_trigger_progress 0.90 \
    --min_trigger_units 2

  # Ensure a valid empty file exists even when no rewrite is accepted.
  touch "$SAFE_JSONL"

  echo
  echo "======================================================================"
  echo "[SELECTIVE JUDGE] dataset=${DATASET}"
  echo "======================================================================"

  python experiments/run_selective_answer_judge.py \
    --predictions "$PRED" \
    --trajectories "$TRAJ" \
    --repair_jsonl "$SAFE_JSONL" \
    --generator_config "$GEN_CFG" \
    --dataset "$DATASET" \
    --trigger all_disagree \
    --margin_threshold 0.30 \
    --max_new_tokens "$MAXTOK" \
    --max_cases -1 \
    --accept_policy final_in_candidates \
    --use_confidence_in_prompt 0 \
    --out_jsonl "$JUDGE_JSONL" \
    --out_json "$JUDGE_JSON"

  python - "$DATASET" "$JUDGE_JSON" <<'PY'
import json
import sys

dataset, fp = sys.argv[1:]
data = json.load(open(fp, encoding="utf-8"))

base = float(data["base_weighted_tail5_acc"])
final = float(data["judge_final_acc"])

if not 0.0 <= base <= 1.0:
    raise SystemExit(f"{dataset}: invalid base accuracy")
if not 0.0 <= final <= 1.0:
    raise SystemExit(f"{dataset}: invalid judge accuracy")

print(
    f"[RESULT] {dataset} "
    f"safev3_pce={base:.4f} "
    f"judge={final:.4f} "
    f"flagged={data['flagged_total']} "
    f"fixed={data['fixed_count']} "
    f"broken={data['broken_count']}"
)
PY

  echo "[DONE] ${DATASET}"
}

run_one svamp 384
run_one asdiv 384
run_one math500 1024
run_one mathqa 768

echo
echo "ALL CROSS-DATASET SAFEV3/JUDGE EXPERIMENTS COMPLETED"
