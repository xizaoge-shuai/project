#!/usr/bin/env bash
set -uo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p outputs/logs/own_generation_baselines
mkdir -p outputs/metrics/own_generation_baselines

# ========== configs ==========
# 如果 Qwen7B 的真实 config 不是这个，把这里改成你的 Qwen7B 主线 config。
QWEN7B_CFG=${QWEN7B_CFG:-configs/model/generator_llama_local_rewrite.yaml}

DS7B_CFG=${DS7B_CFG:-configs/model/generator_deepseek_r1_distill_qwen7b_ablation.yaml}
QWEN3B_CFG=${QWEN3B_CFG:-configs/model/generator_qwen25_3b_ptrue4096.yaml}
DS14B_CFG=${DS14B_CFG:-configs/model/generator_deepseek_r1_distill_qwen14b_ablation.yaml}

find_input () {
  local DATASET="$1"
  local TARGET_N="$2"

  python - "$DATASET" "$TARGET_N" <<'PY'
import sys
from pathlib import Path

dataset = sys.argv[1].lower()
target_n = int(sys.argv[2])

patterns = {
    "gsm8k": ["*gsm8k*.jsonl"],
    "svamp": ["*svamp*.jsonl"],
    "asdiv": ["*asdiv*.jsonl"],
    "asdiv_numeric": ["*asdiv*.jsonl"],
    "mathqa": ["*mathqa*.jsonl"],
    "math500": ["*math500*.jsonl"],
}

bad_words = [
    "has_disagreement",
    "extra",
    "trajectory",
    "trajectories",
    "prediction",
    "predictions",
    "baseline_details",
    "confirm",
    "target",
    "targets",
    "decision",
    "support",
]

cands = []
for root in [Path("data/processed/unified"), Path("data/raw"), Path("data/processed")]:
    if not root.exists():
        continue
    for pat in patterns.get(dataset, [f"*{dataset}*.jsonl"]):
        for fp in root.rglob(pat):
            name = str(fp).lower()
            if any(w in name for w in bad_words):
                continue
            try:
                n = sum(1 for _ in open(fp, encoding="utf-8"))
            except Exception:
                continue
            if n >= min(100, target_n // 2):
                cands.append((abs(n - target_n), -n, fp))

if not cands:
    raise SystemExit(f"NO_INPUT_FOUND for {dataset}")

cands.sort()
print(cands[0][2])
PY
}

run_one () {
  local MODEL="$1"
  local DATASET="$2"
  local VARIANT="$3"
  local N="$4"
  local BASE_ACC="$5"
  local CFG="$6"
  local MAX_NEW="$7"

  local INPUT
  INPUT=$(find_input "$DATASET" "$N")

  echo
  echo "================================================================================"
  echo "[RUN] MODEL=$MODEL DATASET=$DATASET VARIANT=$VARIANT"
  echo "[INPUT] $INPUT"
  echo "[N] $N"
  echo "[BASE_ACC] $BASE_ACC"
  echo "[CFG] $CFG"
  echo "[MAX_NEW] $MAX_NEW"
  echo "================================================================================"

  if [ ! -f "$CFG" ]; then
    echo "[ERROR] config not found: $CFG"
    return 1
  fi

  if [ ! -f scripts/run_own_generation_baselines_one_case.sh ]; then
    echo "[ERROR] missing scripts/run_own_generation_baselines_one_case.sh"
    return 1
  fi

  bash scripts/run_own_generation_baselines_one_case.sh \
    "$MODEL" "$DATASET" "$VARIANT" \
    "$INPUT" \
    NONE \
    "$CFG" \
    "$N" \
    "$BASE_ACC" \
    "$MAX_NEW"

  echo "[DONE] $MODEL $DATASET $VARIANT"
}

run_qwen7b () {
  run_one qwen7b asdiv_numeric independent 2249 0.8671 "$QWEN7B_CFG" 512
  run_one qwen7b gsm8k independent 1319 0.8886 "$QWEN7B_CFG" 768
  run_one qwen7b svamp independent 300 0.9000 "$QWEN7B_CFG" 512
  run_one qwen7b mathqa independent 500 0.7780 "$QWEN7B_CFG" 768
  run_one qwen7b math500 independent 500 0.6520 "$QWEN7B_CFG" 768
}

run_ds7b () {
  run_one ds7b asdiv_numeric independent 2249 0.7514 "$DS7B_CFG" 512
  run_one ds7b gsm8k independent 1319 0.4936 "$DS7B_CFG" 768
  run_one ds7b svamp independent 300 0.6933 "$DS7B_CFG" 512
  run_one ds7b mathqa independent 500 0.4900 "$DS7B_CFG" 768
  run_one ds7b math500 independent_short384 500 0.3100 "$DS7B_CFG" 384
  run_one ds7b math500 independent_long1024 500 0.3100 "$DS7B_CFG" 1024
}

run_qwen3b () {
  run_one qwen3b asdiv_numeric independent 2249 0.7835 "$QWEN3B_CFG" 512
  run_one qwen3b gsm8k independent 1319 0.5004 "$QWEN3B_CFG" 768
  run_one qwen3b svamp independent 300 0.8000 "$QWEN3B_CFG" 512
  run_one qwen3b mathqa independent 500 0.4680 "$QWEN3B_CFG" 768
  run_one qwen3b math500 independent_short384 500 0.2980 "$QWEN3B_CFG" 384
  run_one qwen3b math500 independent_long1024 500 0.2980 "$QWEN3B_CFG" 1024
}

run_ds14b () {
  run_one ds14b asdiv_numeric independent 2249 0.8261 "$DS14B_CFG" 512
  run_one ds14b gsm8k independent 1319 0.7536 "$DS14B_CFG" 768
  run_one ds14b svamp independent 300 0.7833 "$DS14B_CFG" 512
  run_one ds14b mathqa independent 500 0.7120 "$DS14B_CFG" 768
  run_one ds14b math500 independent_long1024 500 0.5020 "$DS14B_CFG" 1024
}

MODEL_SET="${MODEL_SET:-qwen7b}"

case "$MODEL_SET" in
  qwen7b)
    run_qwen7b
    ;;
  ds7b)
    run_ds7b
    ;;
  qwen3b)
    run_qwen3b
    ;;
  ds14b)
    run_ds14b
    ;;
  all)
    run_qwen7b
    run_ds7b
    run_qwen3b
    run_ds14b
    ;;
  *)
    echo "[ERROR] unknown MODEL_SET=$MODEL_SET"
    echo "Use MODEL_SET=qwen7b|ds7b|qwen3b|ds14b|all"
    exit 1
    ;;
esac
