#!/usr/bin/env bash
set -euo pipefail

cd /root/pce_reasoning_project/project
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pce

export PYTHONPATH=/root/pce_reasoning_project/project:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p outputs/logs/own_generation_baselines

# ===== configs：如果你的路径不同，在这里改 =====
QWEN7B_CFG=${QWEN7B_CFG:-configs/model/generator_llama_local_rewrite.yaml}
DS7B_CFG=${DS7B_CFG:-configs/model/generator_deepseek_r1_distill_qwen7b_ablation.yaml}
QWEN3B_CFG=${QWEN3B_CFG:-configs/model/generator_qwen25_3b_ptrue4096.yaml}
DS14B_CFG=${DS14B_CFG:-configs/model/generator_deepseek_r1_distill_qwen14b_ablation.yaml}

find_input () {
  local DATASET="$1"
  local N="$2"

  python - "$DATASET" "$N" <<'PY'
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
    "math500_long1024": ["*math500*.jsonl"],
}

bad_words = [
    "has_disagreement", "extra", "trajectory", "trajectories",
    "prediction", "baseline_details", "confirm", "target"
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
  local N="$3"
  local BASE_ACC="$4"
  local CFG="$5"
  local MAX_NEW="$6"

  local INPUT
  INPUT=$(find_input "$DATASET" "$N")

  echo
  echo "================================================================================"
  echo "[RUN] MODEL=$MODEL DATASET=$DATASET N=$N BASE_ACC=$BASE_ACC"
  echo "[INPUT] $INPUT"
  echo "[CFG] $CFG"
  echo "================================================================================"

  local LOG="outputs/logs/own_generation_baselines/run_${MODEL}_${DATASET}_independent.nohup.log"

  nohup bash scripts/run_own_generation_baselines_one_case.sh \
    "$MODEL" "$DATASET" independent \
    "$INPUT" \
    NONE \
    "$CFG" \
    "$N" \
    "$BASE_ACC" \
    "$MAX_NEW" \
    > "$LOG" 2>&1 &

  echo "[PID] $!"
  echo "[LOG] $LOG"
}

# ===== 默认只跑 Qwen7B-MATH500，避免一下子开太多 =====
RUN_QWEN7B=${RUN_QWEN7B:-1}
RUN_DS7B=${RUN_DS7B:-0}
RUN_QWEN3B=${RUN_QWEN3B:-0}
RUN_DS14B=${RUN_DS14B:-0}

if [ "$RUN_QWEN7B" = "1" ]; then
  run_one qwen7b math500 500 0.6520 "$QWEN7B_CFG" 768
  # 想跑全 Qwen7B 再取消注释：
  # run_one qwen7b gsm8k 1319 0.8886 "$QWEN7B_CFG" 768
  # run_one qwen7b svamp 300 0.9000 "$QWEN7B_CFG" 512
  # run_one qwen7b asdiv_numeric 2249 0.8671 "$QWEN7B_CFG" 512
  # run_one qwen7b mathqa 500 0.7780 "$QWEN7B_CFG" 768
fi

if [ "$RUN_DS7B" = "1" ]; then
  run_one ds7b asdiv_numeric 2249 0.7514 "$DS7B_CFG" 768
  run_one ds7b gsm8k 1319 0.4936 "$DS7B_CFG" 768
  run_one ds7b svamp 300 0.6933 "$DS7B_CFG" 512
  run_one ds7b math500_long1024 500 0.3100 "$DS7B_CFG" 1024
  run_one ds7b mathqa 500 0.4900 "$DS7B_CFG" 768
fi

if [ "$RUN_QWEN3B" = "1" ]; then
  run_one qwen3b asdiv_numeric 2249 0.7835 "$QWEN3B_CFG" 768
  run_one qwen3b gsm8k 1319 0.5004 "$QWEN3B_CFG" 768
  run_one qwen3b svamp 300 0.8000 "$QWEN3B_CFG" 512
  run_one qwen3b math500_long1024 500 0.2980 "$QWEN3B_CFG" 1024
  run_one qwen3b mathqa 500 0.4680 "$QWEN3B_CFG" 768
fi

if [ "$RUN_DS14B" = "1" ]; then
  run_one ds14b asdiv_numeric 2249 0.8261 "$DS14B_CFG" 768
  run_one ds14b gsm8k 1319 0.7536 "$DS14B_CFG" 768
  run_one ds14b svamp 300 0.7833 "$DS14B_CFG" 512
  run_one ds14b math500_long1024 500 0.5020 "$DS14B_CFG" 1024
  run_one ds14b mathqa 500 0.7120 "$DS14B_CFG" 768
fi
