#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}

mkdir -p outputs/logs/fair_baselines
mkdir -p outputs/metrics/fair_baselines

# 自动检测 qwen7b 目录
TRAJ_DIR=""
PRED_DIR=""
TARGET_DIR=""
METRIC_DIR=""

for d in \
  data/processed/trajectories/model_ablation_parallel_qwen7b \
  data/processed/trajectories/model_ablation_qwen7b \
  data/processed/trajectories/model_ablation/qwen7b
do
  if [ -d "$d" ]; then TRAJ_DIR="$d"; break; fi
done

for d in \
  outputs/predictions/model_ablation_parallel_qwen7b \
  outputs/predictions/model_ablation_qwen7b \
  outputs/predictions/model_ablation/qwen7b
do
  if [ -d "$d" ]; then PRED_DIR="$d"; break; fi
done

for d in \
  outputs/targets/model_ablation_parallel_qwen7b \
  outputs/targets/model_ablation_qwen7b \
  outputs/targets/model_ablation/qwen7b
do
  if [ -d "$d" ]; then TARGET_DIR="$d"; break; fi
done

for d in \
  outputs/metrics/model_ablation_parallel_qwen7b \
  outputs/metrics/model_ablation_qwen7b \
  outputs/metrics/model_ablation/qwen7b
do
  if [ -d "$d" ]; then METRIC_DIR="$d"; break; fi
done

echo "TRAJ_DIR=${TRAJ_DIR}"
echo "PRED_DIR=${PRED_DIR}"
echo "TARGET_DIR=${TARGET_DIR}"
echo "METRIC_DIR=${METRIC_DIR}"

if [ -z "$TRAJ_DIR" ] || [ -z "$PRED_DIR" ] || [ -z "$TARGET_DIR" ] || [ -z "$METRIC_DIR" ]; then
  echo "[STOP] qwen7b directories not found. Run find command first and check names."
  exit 0
fi

run_case () {
  local PREFIX="$1"
  local DATASET="$2"
  local TASK_TYPE="$3"
  local BASE_ACC="$4"
  local N_SAMPLES="$5"
  local OURS_JSON="$6"

  local BASE_DETAILS="${PRED_DIR}/${DATASET}_qwen7b_base_details.jsonl"
  local TARGET_IDS="${TARGET_DIR}/${DATASET}_qwen7b_has_disagreement_ids.txt"

  echo
  echo "================================================================================"
  echo "[QWEN7B RUN] ${PREFIX}"
  echo "================================================================================"

  if [ ! -f "$BASE_DETAILS" ]; then
    echo "[SKIP] missing base details: $BASE_DETAILS"
    return 0
  fi

  if [ ! -f "$TARGET_IDS" ]; then
    echo "[SKIP] missing target ids: $TARGET_IDS"
    return 0
  fi

  if [ ! -f "$OURS_JSON" ]; then
    echo "[SKIP] missing ours json: $OURS_JSON"
    return 0
  fi

  local EXTRAS=()
  for s in 42 101 202; do
    f="${TRAJ_DIR}/${DATASET}_qwen7b_extra_seed${s}.jsonl"
    if [ -f "$f" ]; then
      EXTRAS+=("$f")
    fi
  done

  if [ "${#EXTRAS[@]}" -eq 0 ]; then
    echo "[SKIP] missing extras for ${DATASET}_qwen7b"
    return 0
  fi

  echo "[BASE_DETAILS] $BASE_DETAILS"
  echo "[TARGET_IDS]   $TARGET_IDS"
  echo "[OURS_JSON]    $OURS_JSON"
  echo "[EXTRAS]"
  for f in "${EXTRAS[@]}"; do
    printf "%8s  %s\n" "$(wc -l < "$f")" "$f"
  done

  python scripts/eval_tts_baselines_from_candidates.py \
    --baseline_details "$BASE_DETAILS" \
    --extra_jsonls "${EXTRAS[@]}" \
    --target_ids "$TARGET_IDS" \
    --task_type "$TASK_TYPE" \
    --base_acc "$BASE_ACC" \
    --n_samples "$N_SAMPLES" \
    --out_dir outputs/metrics/fair_baselines \
    --prefix "$PREFIX" \
    --max_candidates 4 8 12 \
    --esc_windows 2 3 4 \
    --cisc_temps 0.2 0.5 1.0 2.0 \
    --gg_lambdas 1,0 1,0.5 1,1 \
    --ours_json "$OURS_JSON" \
    --ours_name Confirm
}

# 这里 base_acc 需要按你 qwen7b 自己的 base 结果改。
# 我先用自动从 best json 里找不到时的占位逻辑：你后面如果已有准确值，直接改这几行。

get_base_acc () {
  local DATASET="$1"
  local DEFAULT="$2"
  local BASE_JSON="${METRIC_DIR}/${DATASET}_qwen7b_base.json"

  if [ -f "$BASE_JSON" ]; then
    python - "$BASE_JSON" "$DEFAULT" <<'PY'
import json, sys
fp, default = sys.argv[1], sys.argv[2]
d = json.load(open(fp, encoding="utf-8"))
print(d.get("majority_acc", d.get("base_acc", d.get("acc", default))))
PY
  else
    echo "$DEFAULT"
  fi
}

# normal numeric datasets
GSM8K_BASE=$(get_base_acc gsm8k 0.0000)
ASDIV_BASE=$(get_base_acc asdiv 0.0000)
SVAMP_BASE=$(get_base_acc svamp 0.0000)
MATH500_BASE=$(get_base_acc math500 0.0000)

run_case gsm8k_qwen7b gsm8k numeric "$GSM8K_BASE" 1319 \
  "${METRIC_DIR}/gsm8k_qwen7b_total2_seed1_margin0.json"

run_case asdiv_qwen7b asdiv numeric "$ASDIV_BASE" 2249 \
  "${METRIC_DIR}/asdiv_qwen7b_total2_seed1_margin0.json"

run_case svamp_qwen7b svamp numeric "$SVAMP_BASE" 300 \
  "${METRIC_DIR}/svamp_qwen7b_total2_seed1_margin0.json"

run_case math500_qwen7b_short math500 numeric "$MATH500_BASE" 500 \
  "${METRIC_DIR}/math500_qwen7b_total2_seed1_margin0.json"

# BBH choice datasets
BBH_F_BASE=$(get_base_acc bbh_formal_fallacies 0.0000)
BBH_L_BASE=$(get_base_acc bbh_logical_deduction_five_objects 0.0000)

run_case bbh_formal_fallacies_qwen7b bbh_formal_fallacies choice "$BBH_F_BASE" 100 \
  "${METRIC_DIR}/bbh_formal_fallacies_qwen7b_total2_seed1_margin0.json"

run_case bbh_logical_deduction_five_objects_qwen7b bbh_logical_deduction_five_objects choice "$BBH_L_BASE" 100 \
  "${METRIC_DIR}/bbh_logical_deduction_five_objects_qwen7b_total2_seed1_margin0.json"

echo
echo "========== DONE QWEN7B AGGREGATION BASELINES =========="
find outputs/metrics/fair_baselines -name "*qwen7b*_baseline_compare.md" -print
